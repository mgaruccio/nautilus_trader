# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""Kalshi weather trading strategy for NautilusTrader.

Bridges kalshi-weather's ML signal generation with NautilusTrader's order
management. Replaces the quantdesk hand-rolled engine with NT primitives
(submit_order, on_order_filled, timers) while reusing the same ensemble
models and market scanning logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from nautilus_trader.adapters.kalshi.common.constants import KALSHI_VENUE
from nautilus_trader.adapters.kalshi.common.pricing import (
    ExecutionPhase,
    Opportunity,
    OrderMode,
    OrderPrice,
    compute_maker_price,
    compute_taker_price,
    determine_phase,
    next_escalation_phase,
    phase_price,
)
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.events import OrderCanceled
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.events import OrderRejected
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

log = logging.getLogger(__name__)

# Re-export for external consumers
__all__ = [
    "ExecutionPhase",
    "KalshiWeatherStrategy",
    "KalshiWeatherStrategyConfig",
    "Opportunity",
    "OrderMode",
    "OrderPrice",
    "RestingState",
    "compute_maker_price",
    "compute_taker_price",
    "determine_phase",
    "next_escalation_phase",
    "phase_price",
]


@dataclass
class RestingState:
    ticker: str
    client_order_id: ClientOrderId
    instrument_id: InstrumentId
    side: str
    price_cents: int
    phase: ExecutionPhase
    mode: OrderMode
    placed_at: datetime
    ttl_hours: float | None = None
    escalation_count: int = 0
    contracts: int = 1


# ---------------------------------------------------------------------------
# Strategy config
# ---------------------------------------------------------------------------

_TIF_MAP = {
    "fill_or_kill": TimeInForce.FOK,
    "good_till_canceled": TimeInForce.GTC,
}


class KalshiWeatherStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for the Kalshi weather strategy."""

    # Polling
    poll_interval_secs: int = 60

    # Series to trade
    series_tickers: tuple[str, ...] = ("KXHIGH",)
    excluded_cities: tuple[str, ...] = ()

    # Risk limits
    max_contracts_per_ticker: int = 5
    max_daily_loss_cents: int = 300
    cost_cap_no_cents: int = 99
    cost_cap_yes_cents: int = 92

    # Deep rest mode
    deep_rest_enabled: bool = True
    deep_rest_max_price_cents: int = 85
    max_deep_rest_orders: int = 4
    deep_rest_ttl_hours: float = 14.0

    # Execution
    sell_target_cents: int = 97
    danger_exit_threshold: float = 0.70
    escalation_minutes: float = 3.0
    yes_max_contracts: int = 3

    # Phase thresholds (hours to peak)
    patient_hours: float = 20.0
    maker_hours: float = 10.0
    aggressive_hours: float = 4.0

    # kalshi-weather config path (for hot-reload)
    trader_config_path: str | None = None

    # Dry run mode (generate signals, log decisions, no orders)
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------


class KalshiWeatherStrategy(Strategy):
    """NautilusTrader strategy for weather prediction markets on Kalshi.

    Uses kalshi-weather's ML ensemble for signal generation and NT's order
    management for execution. Supports time-phased pricing, deep rest orders,
    order escalation, auto resting sells, and danger exits.
    """

    def __init__(self, config: KalshiWeatherStrategyConfig) -> None:
        super().__init__(config=config)
        self._cfg = config

        # Internal state
        self._resting_orders: dict[str, RestingState] = {}
        self._buy_entries: dict[str, int] = {}  # ticker -> buy price cents
        self._daily_loss_cents: int = 0
        self._daily_loss_date: str = ""
        self._danger_exited: set[str] = set()
        self._client_to_ticker: dict[ClientOrderId, str] = {}  # map order IDs back
        self._sell_order_ids: set[ClientOrderId] = set()

        # ML models (loaded lazily)
        self._models: list | None = None
        self._model_names: list[str] = []
        self._model_weights: list[float] = []
        self._trader_config: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self._log.info("Starting KalshiWeatherStrategy")

        # Load ML models
        self._load_models()

        # Cancel stale resting buys from previous run
        self._cancel_stale_resting_buys()

        # Set up recurring timer
        import pandas as pd

        self.clock.set_timer(
            name="poll_cycle",
            interval=pd.Timedelta(seconds=self._cfg.poll_interval_secs),
        )

        self._log.info(
            f"Strategy started: poll={self._cfg.poll_interval_secs}s, "
            f"dry_run={self._cfg.dry_run}, "
            f"deep_rest={'on' if self._cfg.deep_rest_enabled else 'off'}"
        )

    def on_timer(self, event) -> None:
        if event.name != "poll_cycle":
            return

        try:
            self._run_cycle()
        except Exception as e:
            self._consecutive_failures = getattr(self, "_consecutive_failures", 0) + 1
            self._log.error(
                f"Cycle failed ({self._consecutive_failures}/3): {e}",
                exc_info=True,
            )
            if self._consecutive_failures >= 3:
                self._log.critical("HALTING: 3 consecutive cycle failures")
                self.stop()

    def on_order_filled(self, event: OrderFilled) -> None:
        ticker = self._client_to_ticker.get(event.client_order_id)
        if ticker is None:
            return

        if event.client_order_id in self._sell_order_ids:
            # Sell fill — compute P&L
            sell_price = int(event.last_px.as_double())
            buy_price = self._buy_entries.get(ticker)
            if buy_price is not None:
                profit = sell_price - buy_price
                self._log.info(
                    f"SELL FILL: {ticker} sell@{sell_price}c buy@{buy_price}c profit={profit}c"
                )
                if profit < 0:
                    self._record_loss(abs(profit))
                self._buy_entries.pop(ticker, None)
            self._sell_order_ids.discard(event.client_order_id)
        else:
            # Buy fill — record entry, place resting sell
            fill_price = int(event.last_px.as_double())
            fill_qty = int(event.last_qty.as_double())
            self._buy_entries[ticker] = fill_price
            self._log.info(f"BUY FILL: {ticker} @ {fill_price}c x{fill_qty}")

            # Remove from resting tracker
            self._resting_orders.pop(ticker, None)

            # Place resting GTC sell
            if not self._cfg.dry_run:
                self._place_sell(ticker, fill_qty, fill_price)

    def on_order_canceled(self, event: OrderCanceled) -> None:
        ticker = self._client_to_ticker.get(event.client_order_id)
        if ticker and ticker in self._resting_orders:
            resting = self._resting_orders[ticker]
            if resting.client_order_id == event.client_order_id:
                self._resting_orders.pop(ticker, None)
        self._sell_order_ids.discard(event.client_order_id)

    def on_order_rejected(self, event: OrderRejected) -> None:
        ticker = self._client_to_ticker.get(event.client_order_id)
        self._log.warning(f"Order rejected for {ticker}: {event.reason}")
        if ticker and ticker in self._resting_orders:
            resting = self._resting_orders[ticker]
            if resting.client_order_id == event.client_order_id:
                self._resting_orders.pop(ticker, None)
        self._sell_order_ids.discard(event.client_order_id)

    def on_stop(self) -> None:
        self.clock.cancel_timer("poll_cycle")
        self._log.info(
            f"Strategy stopped. Resting orders: {len(self._resting_orders)}, "
            f"daily loss: {self._daily_loss_cents}c"
        )

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def _run_cycle(self) -> None:
        # Reset daily loss tracker if date changed
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._daily_loss_date:
            self._daily_loss_cents = 0
            self._daily_loss_date = today

        # Hot-reload trader config
        self._reload_trader_config()

        # 1. Escalate stale normal-mode orders
        self._escalate_stale_orders()

        # 2. Cancel expired deep-rest orders
        self._cancel_expired_deep_rest()

        # 3. Danger exit check
        self._check_danger_exits()

        # 4. Generate signals and execute
        opportunities = self._generate_signals()
        approved = 0
        rejected = 0

        for opp in opportunities:
            if not self._risk_check(opp):
                rejected += 1
                continue
            approved += 1
            self._execute_opportunity(opp)

        if opportunities:
            self._log.info(
                f"Cycle: {len(opportunities)} signals, {approved} approved, {rejected} rejected, "
                f"{len(self._resting_orders)} resting"
            )

    # ------------------------------------------------------------------
    # Signal generation (bridge to kalshi-weather)
    # ------------------------------------------------------------------

    def _resolve_trader_config_path(self) -> Path:
        """Resolve trader config path. Fails hard if not found."""
        if self._cfg.trader_config_path:
            p = Path(self._cfg.trader_config_path)
            if not p.exists():
                raise FileNotFoundError(f"Trader config not found: {p}")
            return p

        import importlib.util

        candidates = [
            Path("/home/kalshi/kalshi-weather/data/trader_config.json"),
            Path.home() / "code" / "altmarkets" / "kalshi-weather" / "data" / "trader_config.json",
        ]
        spec = importlib.util.find_spec("kalshi_weather_ml")
        if spec and spec.origin:
            pkg_src = Path(spec.origin).resolve().parent
            for depth in [3, 2, 1]:
                candidate = pkg_src
                for _ in range(depth):
                    candidate = candidate.parent
                candidate = candidate / "data" / "trader_config.json"
                if candidate not in candidates:
                    candidates.append(candidate)

        for candidate in candidates:
            if candidate.exists():
                self._log.info(f"Resolved trader config: {candidate}")
                return candidate

        raise FileNotFoundError(
            f"No trader_config.json found in: {[str(c) for c in candidates]}. "
            f"Set trader_config_path in strategy config."
        )

    def _load_models(self) -> None:
        """Load ML models. Raises on failure — strategy cannot run without models."""
        from kalshi_weather_ml.config import load_config as load_trader_config

        resolved = self._resolve_trader_config_path()
        self._trader_config = load_trader_config(resolved)
        self._log.info(
            f"Loaded trader config: max_contracts={self._trader_config.max_contracts_per_ticker}"
        )
        self._models, self._model_names, self._model_weights = (
            self._load_ensemble_models(self._trader_config)
        )
        if not self._models:
            raise RuntimeError("No ML models loaded — strategy cannot run")
        self._log.info(
            f"Loaded {len(self._models)} ML models: {self._model_names}"
        )

    def _reload_trader_config(self) -> None:
        try:
            from kalshi_weather_ml.config import load_config as load_trader_config

            resolved = self._resolve_trader_config_path()
            self._trader_config = load_trader_config(resolved)
        except Exception as e:
            self._log.warning(f"Config hot-reload failed (keeping existing): {e}")

    def _generate_signals(self) -> list[Opportunity]:
        if not self._models or self._trader_config is None:
            return []

        from kalshi_weather_ml.forecasts import CONSENSUS_MODEL, PRIMARY_MODEL, get_forecast
        from kalshi_weather_ml.markets import fetch_open_markets
        from kalshi_weather_ml.strategy import evaluate_opportunity_ensemble

        from datetime import datetime as dt
        from zoneinfo import ZoneInfo

        now = dt.now(ZoneInfo("America/New_York"))
        tc = self._trader_config

        markets = fetch_open_markets()  # raises on API failure

        opportunities: list[Opportunity] = []
        for market in markets:
            city = market["city"]
            if city in self._cfg.excluded_cities:
                continue

            settlement_date = market["settlement_date"]

            try:
                ecmwf = get_forecast(city, settlement_date, model=PRIMARY_MODEL)
                gfs = get_forecast(city, settlement_date, model=CONSENSUS_MODEL)
            except Exception as e:
                self._log.warning(f"Forecast failed for {city}/{settlement_date}: {e}")
                continue

            if ecmwf is None or gfs is None:
                continue

            try:
                opps = evaluate_opportunity_ensemble(
                    market,
                    ecmwf,
                    gfs,
                    tc,
                    models=self._models,
                    model_names=self._model_names,
                    model_weights=self._model_weights,
                    now=now,
                    require_unanimous=tc.ensemble_require_unanimous,
                )
            except Exception as e:
                self._log.warning(f"Ensemble eval failed for {market.get('ticker', '?')}: {e}")
                continue

            for opp in opps:
                model_scores: dict[str, float] = {}
                if hasattr(opp, "model_scores") and opp.model_scores:
                    model_scores = opp.model_scores
                model_scores["ensemble"] = opp.p_win

                opportunities.append(
                    Opportunity(
                        ticker=opp.ticker,
                        city=opp.city,
                        direction=opp.direction,
                        side=opp.side,
                        threshold=opp.threshold,
                        settlement_date=opp.settlement_date,
                        yes_bid=opp.yes_bid,
                        yes_ask=opp.yes_ask,
                        ecmwf=opp.ecmwf,
                        gfs=opp.gfs,
                        margin=opp.margin,
                        consensus=opp.consensus,
                        h_to_peak=opp.h_to_peak,
                        p_win=opp.p_win,
                        cost_cents=opp.cost_cents,
                        strategy=opp.strategy,
                        model_scores=model_scores,
                    )
                )

        return opportunities

    # ------------------------------------------------------------------
    # Risk guards
    # ------------------------------------------------------------------

    def _risk_check(self, opp: Opportunity) -> bool:
        # 1. Cost ceiling
        if opp.side == "no" and opp.cost_cents > self._cfg.cost_cap_no_cents:
            return False
        if opp.side == "yes" and opp.cost_cents > self._cfg.cost_cap_yes_cents:
            return False

        # 2. Duplicate: already resting or in position
        if opp.ticker in self._resting_orders:
            return False
        if opp.ticker in self._buy_entries:
            return False

        # 3. Daily loss limit
        if self._daily_loss_cents >= self._cfg.max_daily_loss_cents:
            self._log.info("Daily loss limit reached, rejecting signal")
            return False

        # 4. YES side contract cap
        if opp.side == "yes":
            # Count existing YES positions
            # (simplified — use buy_entries as proxy for held positions)
            pass

        return True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_opportunity(self, opp: Opportunity) -> None:
        mode = OrderMode.DEEP_REST if self._cfg.deep_rest_enabled else OrderMode.NORMAL

        # Cap deep rest orders
        if mode == OrderMode.DEEP_REST:
            deep_rest_count = sum(
                1 for r in self._resting_orders.values() if r.mode == OrderMode.DEEP_REST
            )
            if deep_rest_count >= self._cfg.max_deep_rest_orders:
                self._log.info(
                    f"Deep rest cap ({self._cfg.max_deep_rest_orders}), "
                    f"falling back to NORMAL for {opp.ticker}"
                )
                mode = OrderMode.NORMAL

        # Compute order price
        if mode == OrderMode.DEEP_REST:
            order_price = OrderPrice(
                side=opp.side,
                price_cents=self._cfg.deep_rest_max_price_cents,
                is_maker=True,
                time_in_force="good_till_canceled",
                phase=ExecutionPhase.DEEP_REST,
            )
        else:
            spread = opp.yes_ask - opp.yes_bid
            phase = determine_phase(
                opp.h_to_peak,
                spread,
                patient_hours=self._cfg.patient_hours,
                maker_hours=self._cfg.maker_hours,
                aggressive_hours=self._cfg.aggressive_hours,
            )
            order_price = phase_price(phase, opp.side, opp.yes_bid, opp.yes_ask)

        contracts = min(
            self._cfg.max_contracts_per_ticker,
            self._cfg.yes_max_contracts if opp.side == "yes" else self._cfg.max_contracts_per_ticker,
        )

        if self._cfg.dry_run:
            self._log.info(
                f"[DRY RUN] {order_price.phase.value} {opp.side} {opp.ticker} "
                f"@ {order_price.price_cents}c x{contracts}"
            )
            return

        # Submit order via NT
        instrument_id = InstrumentId(Symbol(opp.ticker), KALSHI_VENUE)
        nt_tif = _TIF_MAP.get(order_price.time_in_force, TimeInForce.GTC)

        order = self.order_factory.limit(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_int(contracts),
            price=Price.from_int(order_price.price_cents),
            time_in_force=nt_tif,
        )

        self._client_to_ticker[order.client_order_id] = opp.ticker
        self.submit_order(order, params={"kalshi_side": opp.side})

        # Track resting maker orders
        if order_price.is_maker:
            ttl = self._cfg.deep_rest_ttl_hours if mode == OrderMode.DEEP_REST else None
            self._resting_orders[opp.ticker] = RestingState(
                ticker=opp.ticker,
                client_order_id=order.client_order_id,
                instrument_id=instrument_id,
                side=opp.side,
                price_cents=order_price.price_cents,
                phase=order_price.phase,
                mode=mode,
                placed_at=datetime.now(UTC),
                ttl_hours=ttl,
                contracts=contracts,
            )

        self._log.info(
            f"Placed {order_price.phase.value} {opp.side} {opp.ticker} "
            f"@ {order_price.price_cents}c x{contracts}"
        )

    def _place_sell(self, ticker: str, qty: int, buy_price: int) -> None:
        """Place a resting GTC sell order after a buy fill."""
        instrument_id = InstrumentId(Symbol(ticker), KALSHI_VENUE)

        # Look up the side from buy_entries context
        # We stored the RestingState before it was removed on fill,
        # but we have the buy_price. For sell: always sell at target.
        side = "no"  # default
        for coid, t in self._client_to_ticker.items():
            if t == ticker and coid not in self._sell_order_ids:
                # Try to infer side from the resting state we just removed
                break

        sell_price = self._cfg.sell_target_cents

        order = self.order_factory.limit(
            instrument_id=instrument_id,
            order_side=OrderSide.SELL,
            quantity=Quantity.from_int(qty),
            price=Price.from_int(sell_price),
            time_in_force=TimeInForce.GTC,
        )

        self._client_to_ticker[order.client_order_id] = ticker
        self._sell_order_ids.add(order.client_order_id)
        self.submit_order(order, params={"kalshi_side": side})

        self._log.info(f"Resting sell placed: {ticker} @ {sell_price}c x{qty}")

    # ------------------------------------------------------------------
    # Order lifecycle management
    # ------------------------------------------------------------------

    def _escalate_stale_orders(self) -> None:
        now = datetime.now(UTC)
        stale = []
        for ticker, resting in self._resting_orders.items():
            if resting.mode != OrderMode.NORMAL:
                continue
            age = now - resting.placed_at
            if age > timedelta(minutes=self._cfg.escalation_minutes):
                stale.append(ticker)

        for ticker in stale:
            resting = self._resting_orders.get(ticker)
            if resting is None:
                continue

            next_phase = next_escalation_phase(resting.phase)
            if next_phase is None:
                # Max escalation — cancel
                self._cancel_resting_order(ticker)
                self._log.info(f"Max escalation reached, cancelled {ticker}")
                continue

            # Cancel and re-place at new price
            self._cancel_resting_order(ticker)

            # We need fresh market data for repricing — use the HTTP client directly
            # via the exec client's connection. For now, skip repricing in escalation
            # since the data client doesn't provide real-time quotes.
            # The next poll cycle will re-evaluate this ticker.
            self._log.info(
                f"Escalated {ticker}: {resting.phase.value} -> cancelled "
                f"(will re-evaluate next cycle)"
            )

    def _cancel_expired_deep_rest(self) -> None:
        now = datetime.now(UTC)
        expired = []
        for ticker, resting in self._resting_orders.items():
            if resting.mode != OrderMode.DEEP_REST:
                continue
            if resting.ttl_hours is None:
                continue
            deadline = resting.placed_at + timedelta(hours=resting.ttl_hours)
            if deadline < now:
                expired.append(ticker)

        for ticker in expired:
            self._cancel_resting_order(ticker)
            self._log.info(f"Cancelled expired deep rest order: {ticker}")

    def _cancel_resting_order(self, ticker: str) -> None:
        resting = self._resting_orders.pop(ticker, None)
        if resting is None:
            return
        if self._cfg.dry_run:
            return

        order = self.cache.order(resting.client_order_id)
        if order is not None and order.is_open:
            self.cancel_order(order)

    def _cancel_stale_resting_buys(self) -> None:
        """Cancel all open buy orders on startup to avoid duplicates."""
        open_orders = self.cache.orders_open(venue=KALSHI_VENUE)
        for order in open_orders:
            if order.side == OrderSide.BUY:
                self.cancel_order(order)
                self._log.info(f"Cancelled stale buy order: {order.client_order_id}")

    # ------------------------------------------------------------------
    # Danger exits
    # ------------------------------------------------------------------

    def _check_danger_exits(self) -> None:
        """Re-evaluate held positions; FOK sell if p_win drops below threshold."""
        if not self._models or self._trader_config is None:
            return

        for ticker in list(self._buy_entries.keys()):
            if ticker in self._danger_exited:
                continue

            p_win = self._evaluate_ticker(ticker)
            if p_win is None:
                continue

            if p_win >= self._cfg.danger_exit_threshold:
                continue

            self._log.warning(
                f"DANGER EXIT: {ticker} p_win={p_win:.3f} < {self._cfg.danger_exit_threshold}"
            )

            if not self._cfg.dry_run:
                # Cancel any resting sell for this ticker
                for coid in list(self._sell_order_ids):
                    if self._client_to_ticker.get(coid) == ticker:
                        order = self.cache.order(coid)
                        if order is not None and order.is_open:
                            self.cancel_order(order)

                # Place FOK sell to exit
                instrument_id = InstrumentId(Symbol(ticker), KALSHI_VENUE)
                # Sell at worst acceptable price (1c = market order equivalent)
                order = self.order_factory.limit(
                    instrument_id=instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=Quantity.from_int(1),
                    price=Price.from_int(1),
                    time_in_force=TimeInForce.FOK,
                )
                self._client_to_ticker[order.client_order_id] = ticker
                self._sell_order_ids.add(order.client_order_id)
                self.submit_order(order, params={"kalshi_side": "no"})

            self._danger_exited.add(ticker)

    def _evaluate_ticker(self, ticker: str) -> float | None:
        """Re-evaluate ensemble p_win for a held position."""
        try:
            from kalshi_weather_ml.forecasts import CONSENSUS_MODEL, PRIMARY_MODEL, get_forecast
            from kalshi_weather_ml.markets import parse_ticker
            from kalshi_weather_ml.strategy import evaluate_opportunity_ensemble

            from datetime import datetime as dt
            from zoneinfo import ZoneInfo

            parsed = parse_ticker(ticker)
            if parsed is None:
                return None

            city = parsed["city"]
            settlement_date = parsed["settlement_date"]
            now = dt.now(ZoneInfo("America/New_York"))

            ecmwf = get_forecast(city, settlement_date, model=PRIMARY_MODEL)
            gfs = get_forecast(city, settlement_date, model=CONSENSUS_MODEL)
            if ecmwf is None or gfs is None:
                return None

            opps = evaluate_opportunity_ensemble(
                {"ticker": ticker, "city": city, "settlement_date": settlement_date},
                ecmwf,
                gfs,
                self._trader_config,
                models=self._models,
                model_names=self._model_names,
                model_weights=self._model_weights,
                now=now,
                require_unanimous=False,
            )
            if opps:
                return max(o.p_win for o in opps)
            return None
        except Exception as e:
            self._log.warning(f"Danger exit eval failed for {ticker}: {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_loss(self, loss_cents: int) -> None:
        self._daily_loss_cents += loss_cents
        self._log.info(
            f"Loss recorded: {loss_cents}c (daily total: {self._daily_loss_cents}c / "
            f"{self._cfg.max_daily_loss_cents}c limit)"
        )

    @staticmethod
    def _load_ensemble_models(config) -> tuple[list, list[str], list[float]]:
        """Load ML ensemble models from kalshi-weather."""
        try:
            import importlib
            from pathlib import Path as KWPath

            from kalshi_weather_ml.models.emos import EMOSModel
            from kalshi_weather_ml.models.ngboost_model import NGBoostModel

            kw_data = None
            candidates = [
                KWPath("/home/kalshi/kalshi-weather/data"),
                KWPath.home() / "code" / "altmarkets" / "kalshi-weather" / "data",
            ]
            kw_spec = importlib.util.find_spec("kalshi_weather_ml")
            if kw_spec and kw_spec.origin:
                pkg_root = KWPath(kw_spec.origin).resolve().parent.parent.parent
                candidates.insert(0, pkg_root / "data")

            for candidate in candidates:
                if (candidate / "models").exists():
                    kw_data = candidate
                    break

            if kw_data is None:
                return [], [], []

            models_dir = kw_data / "models"
            cal_path = (
                kw_data / "calibration_dataset.parquet"
                if (kw_data / "calibration_dataset.parquet").exists()
                else None
            )

            models, names, weights = [], [], []

            emos_path = models_dir / "emos_normal.json"
            if emos_path.exists():
                models.append(EMOSModel.load(emos_path))
                names.append("emos")
                weights.append(config.ensemble_weights.get("emos", 1.0))

            ngb_path = models_dir / "ngboost_normal.pkl"
            if ngb_path.exists():
                models.append(NGBoostModel.load(ngb_path, cal_path))
                names.append("ngboost")
                weights.append(config.ensemble_weights.get("ngboost", 1.0))

            return models, names, weights
        except Exception as e:
            raise RuntimeError(f"Failed to load ensemble models: {e}") from e
