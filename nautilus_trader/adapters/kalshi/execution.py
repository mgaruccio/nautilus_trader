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

"""Kalshi execution client for NautilusTrader.

Implements order submission, cancellation, fill detection, and position tracking
via the Kalshi REST API with RSA-PSS authentication.
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from nautilus_trader.adapters.kalshi.common.constants import KALSHI_VENUE
from nautilus_trader.adapters.kalshi.common.enums import KalshiOrderStatus
from nautilus_trader.adapters.kalshi.common.parsing import kalshi_side_to_order_side
from nautilus_trader.adapters.kalshi.http.client import KalshiHttpError
from nautilus_trader.adapters.kalshi.common.parsing import kalshi_status_to_nautilus
from nautilus_trader.adapters.kalshi.common.parsing import kalshi_tif_to_nautilus
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.execution.messages import CancelAllOrders
from nautilus_trader.execution.messages import CancelOrder
from nautilus_trader.execution.messages import GenerateFillReports
from nautilus_trader.execution.messages import GenerateOrderStatusReports
from nautilus_trader.execution.messages import GeneratePositionStatusReports
from nautilus_trader.execution.messages import ModifyOrder
from nautilus_trader.execution.messages import SubmitOrder
from nautilus_trader.execution.reports import FillReport
from nautilus_trader.execution.reports import OrderStatusReport
from nautilus_trader.execution.reports import PositionStatusReport
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import ContingencyType
from nautilus_trader.model.enums import LiquiditySide
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import OrderStatus
from nautilus_trader.model.enums import OrderType
from nautilus_trader.model.enums import PositionSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import AccountId
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.identifiers import VenueOrderId
from nautilus_trader.model.objects import AccountBalance
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

if TYPE_CHECKING:
    from nautilus_trader.adapters.kalshi.config import KalshiExecClientConfig
    from nautilus_trader.adapters.kalshi.http.client import KalshiHttpClient
    from nautilus_trader.adapters.kalshi.providers import KalshiInstrumentProvider
    from nautilus_trader.cache.cache import Cache
    from nautilus_trader.common.component import LiveClock
    from nautilus_trader.common.component import MessageBus

USD = Currency.from_str("USD")

# Kalshi TIF mapping from NautilusTrader to Kalshi API
_TIF_MAP = {
    TimeInForce.FOK: "fill_or_kill",
    TimeInForce.GTC: "good_till_canceled",
    TimeInForce.IOC: "immediate_or_cancel",
}


def _compute_taker_fee(count: int, price_cents: int) -> float:
    """Compute Kalshi taker fee: 0.07 * C * P * (100-P) / 100, capped at 2c/contract."""
    p = float(price_cents)
    fee_per = min(0.07 * p * (100.0 - p) / 100.0, 2.0)
    return fee_per * count


class KalshiExecutionClient(LiveExecutionClient):
    """
    Provides an execution client for the Kalshi prediction market.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        The event loop for the client.
    http_client : KalshiHttpClient
        The Kalshi HTTP client.
    msgbus : MessageBus
        The message bus for the client.
    cache : Cache
        The cache for the client.
    clock : LiveClock
        The clock for the client.
    instrument_provider : KalshiInstrumentProvider
        The instrument provider for the client.
    config : KalshiExecClientConfig
        The configuration for the client.
    name : str, optional
        The custom client name.

    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        http_client: KalshiHttpClient,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
        instrument_provider: KalshiInstrumentProvider,
        config: KalshiExecClientConfig,
        name: str | None = None,
    ):
        super().__init__(
            loop=loop,
            client_id=ClientId(name or KALSHI_VENUE.value),
            venue=KALSHI_VENUE,
            oms_type=OmsType.NETTING,
            instrument_provider=instrument_provider,
            account_type=AccountType.CASH,
            base_currency=USD,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )
        self._http_client = http_client
        self._config = config

        # Set account ID (required before generate_account_state)
        account_id = AccountId(f"{name or KALSHI_VENUE.value}-001")
        self._set_account_id(account_id)

    # -- Connection ------------------------------------------------------------

    async def _connect(self) -> None:
        """Connect to the Kalshi API — load instruments and fetch balance."""
        await self._instrument_provider.load_all_async()
        self._log.info(
            f"Loaded {len(self._instrument_provider.list_all())} instruments",
        )
        await self._update_account_state()

    async def _disconnect(self) -> None:
        """Disconnect from the Kalshi API."""
        self._log.info("Kalshi execution client disconnected")

    # -- Account state ---------------------------------------------------------

    async def _update_account_state(self) -> None:
        """Fetch balance from Kalshi and emit account state."""
        resp = await asyncio.to_thread(self._http_client.get_balance)
        balance_cents = resp.get("balance", 0)
        balance_dollars = balance_cents / 100.0

        self.generate_account_state(
            balances=[
                AccountBalance(
                    total=Money(balance_dollars, USD),
                    locked=Money(0, USD),
                    free=Money(balance_dollars, USD),
                ),
            ],
            margins=[],
            reported=True,
            ts_event=self._clock.timestamp_ns(),
        )
        self._log.info(f"Account balance: ${balance_dollars:.2f} ({balance_cents}c)")

    # -- Order operations ------------------------------------------------------

    async def _submit_order(self, command: SubmitOrder) -> None:
        """Submit an order to Kalshi."""
        order = self._cache.order(command.client_order_id)
        if order is None:
            self._log.error(f"Order not found in cache: {command.client_order_id}")
            return

        if order.order_type != OrderType.LIMIT:
            self._generate_order_submitted_event(command, order)
            self._generate_order_rejected_event(
                command,
                order,
                f"Unsupported order type for Kalshi: {order.order_type}",
            )
            return

        price = order.price
        if price is None:
            self._generate_order_submitted_event(command, order)
            self._generate_order_rejected_event(
                command,
                order,
                "Limit orders require a price",
            )
            return

        tif = order.time_in_force
        kalshi_tif = _TIF_MAP.get(tif)
        if kalshi_tif is None:
            self._generate_order_submitted_event(command, order)
            self._generate_order_rejected_event(
                command,
                order,
                f"Unsupported time-in-force for Kalshi: {tif}",
            )
            return

        # Resolve Kalshi side from command params or default to "no"
        kalshi_side = "no"
        if command.params and command.params.get("kalshi_side"):
            kalshi_side = command.params["kalshi_side"]

        ticker = order.instrument_id.symbol.value
        price_cents = int(price.as_double())
        count = int(order.quantity.as_double())

        self._generate_order_submitted_event(command, order)

        try:
            kalshi_order = await asyncio.to_thread(
                self._http_client.place_order,
                ticker=ticker,
                side=kalshi_side,
                action="buy",
                count=count,
                price_cents=price_cents,
                time_in_force=kalshi_tif,
                client_order_id=str(order.client_order_id),
            )
        except KalshiHttpError as e:
            self._generate_order_rejected_event(
                command,
                order,
                f"Kalshi API rejected order: {e}",
            )
            return
        except Exception as e:
            self._log.exception(f"Unexpected error submitting order: {e}")
            self._generate_order_rejected_event(
                command,
                order,
                f"Unexpected error: {e}",
            )
            return

        order_id = kalshi_order.get("order_id", "")
        if not order_id:
            self._generate_order_rejected_event(
                command,
                order,
                "Kalshi returned order with missing order_id",
            )
            return

        venue_order_id = VenueOrderId(order_id)
        self._generate_order_accepted_event(command, order, venue_order_id)

        # Check for immediate fill (FOK orders)
        fill_count = kalshi_order.get("fill_count", 0)
        if fill_count > 0:
            fill_price = kalshi_order.get(
                "no_price" if kalshi_side == "no" else "yes_price",
                price_cents,
            )
            fee = _compute_taker_fee(fill_count, fill_price)

            self.generate_order_filled(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=order.client_order_id,
                venue_order_id=venue_order_id,
                venue_position_id=None,
                trade_id=TradeId(kalshi_order.get("order_id", str(uuid.uuid4()))),
                order_side=order.side,
                order_type=order.order_type,
                last_qty=Quantity.from_int(fill_count),
                last_px=Price.from_int(fill_price),
                quote_currency=USD,
                commission=Money(fee, USD),
                liquidity_side=LiquiditySide.TAKER if tif == TimeInForce.FOK else LiquiditySide.MAKER,
                ts_event=self._clock.timestamp_ns(),
            )

    async def _modify_order(self, command: ModifyOrder) -> None:
        """Kalshi doesn't support order modification — reject."""
        self._log.warning(
            f"Order modification not supported on Kalshi: {command.client_order_id}",
        )

    async def _cancel_order(self, command: CancelOrder) -> None:
        """Cancel an order on Kalshi."""
        order = self._cache.order(command.client_order_id)
        if order is None:
            self._log.warning(f"Order not found for cancel: {command.client_order_id}")
            return

        venue_order_id = order.venue_order_id
        if venue_order_id is None:
            self._log.warning(f"No venue_order_id for cancel: {command.client_order_id}")
            return

        try:
            await asyncio.to_thread(
                self._http_client.cancel_order,
                str(venue_order_id),
            )
            self.generate_order_canceled(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=order.client_order_id,
                venue_order_id=venue_order_id,
                ts_event=self._clock.timestamp_ns(),
            )
        except KalshiHttpError as e:
            if e.status == 404:
                self._log.warning(
                    f"Order {venue_order_id} not found on exchange (404), canceling locally"
                )
                self.generate_order_canceled(
                    strategy_id=order.strategy_id,
                    instrument_id=order.instrument_id,
                    client_order_id=order.client_order_id,
                    venue_order_id=venue_order_id,
                    ts_event=self._clock.timestamp_ns(),
                )
            else:
                self._log.error(f"Failed to cancel order {venue_order_id}: {e}")
        except Exception as e:
            self._log.exception(f"Unexpected error canceling order {venue_order_id}: {e}")

    async def _cancel_all_orders(self, command: CancelAllOrders) -> None:
        """Cancel all open orders on Kalshi."""
        open_orders = self._cache.orders_open(
            venue=KALSHI_VENUE,
            instrument_id=command.instrument_id,
        )
        for order in open_orders:
            if order.venue_order_id is not None:
                try:
                    await asyncio.to_thread(
                        self._http_client.cancel_order,
                        str(order.venue_order_id),
                    )
                    self.generate_order_canceled(
                        strategy_id=order.strategy_id,
                        instrument_id=order.instrument_id,
                        client_order_id=order.client_order_id,
                        venue_order_id=order.venue_order_id,
                        ts_event=self._clock.timestamp_ns(),
                    )
                except KalshiHttpError as e:
                    if e.status == 404:
                        self._log.warning(
                            f"Order {order.venue_order_id} not found on exchange (404), canceling locally"
                        )
                        self.generate_order_canceled(
                            strategy_id=order.strategy_id,
                            instrument_id=order.instrument_id,
                            client_order_id=order.client_order_id,
                            venue_order_id=order.venue_order_id,
                            ts_event=self._clock.timestamp_ns(),
                        )
                    else:
                        self._log.warning(f"Failed to cancel {order.venue_order_id}: {e}")
                except Exception as e:
                    self._log.exception(f"Unexpected error canceling {order.venue_order_id}: {e}")

    # -- Reports ---------------------------------------------------------------

    async def generate_order_status_reports(
        self,
        command: GenerateOrderStatusReports,
    ) -> list[OrderStatusReport]:
        """Generate order status reports from Kalshi API."""
        self._log.debug("Requesting OrderStatusReports...")
        reports: list[OrderStatusReport] = []

        orders = await asyncio.to_thread(self._http_client.get_orders)
        ts_init = self._clock.timestamp_ns()

        for kalshi_order in orders:
            order_id = kalshi_order.get("order_id", "")
            if not order_id:
                self._log.warning("Skipping order with missing order_id")
                continue

            ticker = kalshi_order.get("ticker", "")
            instrument = self._instrument_provider.get_by_ticker(ticker)
            if instrument is None:
                continue

            side = kalshi_order.get("side", "no")
            action = kalshi_order.get("action", "buy")
            order_side = kalshi_side_to_order_side(side, action)
            status = kalshi_status_to_nautilus(kalshi_order.get("status", "pending"))
            tif = kalshi_tif_to_nautilus(kalshi_order.get("time_in_force", "fill_or_kill"))

            initial_count = kalshi_order.get("initial_count", 0)
            fill_count = kalshi_order.get("fill_count", 0)
            price_cents = kalshi_order.get(
                "no_price" if side == "no" else "yes_price", 0,
            )

            client_order_id = None
            if kalshi_order.get("client_order_id"):
                client_order_id = ClientOrderId(kalshi_order["client_order_id"])

            report = OrderStatusReport(
                account_id=self.account_id,
                instrument_id=instrument.id,
                client_order_id=client_order_id,
                venue_order_id=VenueOrderId(order_id),
                order_side=order_side,
                order_type=OrderType.LIMIT,
                contingency_type=ContingencyType.NO_CONTINGENCY,
                time_in_force=tif,
                order_status=status,
                price=Price.from_int(price_cents),
                quantity=Quantity.from_int(initial_count),
                filled_qty=Quantity.from_int(fill_count),
                ts_accepted=ts_init,
                ts_last=ts_init,
                report_id=UUID4(),
                ts_init=ts_init,
            )
            reports.append(report)

        self._log.info(f"Generated {len(reports)} order status reports")
        return reports

    async def generate_fill_reports(
        self,
        command: GenerateFillReports,
    ) -> list[FillReport]:
        """Generate fill reports from Kalshi API."""
        self._log.debug("Requesting FillReports...")
        reports: list[FillReport] = []

        orders = await asyncio.to_thread(
            self._http_client.get_orders,
            status="executed",
        )
        ts_init = self._clock.timestamp_ns()

        for kalshi_order in orders:
            order_id = kalshi_order.get("order_id", "")
            if not order_id:
                self._log.warning("Skipping order with missing order_id")
                continue

            fill_count = kalshi_order.get("fill_count", 0)
            if fill_count == 0:
                continue

            ticker = kalshi_order.get("ticker", "")
            instrument = self._instrument_provider.get_by_ticker(ticker)
            if instrument is None:
                continue

            side = kalshi_order.get("side", "no")
            action = kalshi_order.get("action", "buy")
            order_side = kalshi_side_to_order_side(side, action)
            price_cents = kalshi_order.get(
                "no_price" if side == "no" else "yes_price", 0,
            )

            tif_str = kalshi_order.get("time_in_force", "fill_or_kill")
            liquidity_side = (
                LiquiditySide.MAKER if tif_str == "good_till_canceled"
                else LiquiditySide.TAKER
            )

            fee = _compute_taker_fee(fill_count, price_cents)

            client_order_id = None
            if kalshi_order.get("client_order_id"):
                client_order_id = ClientOrderId(kalshi_order["client_order_id"])

            report = FillReport(
                account_id=self.account_id,
                instrument_id=instrument.id,
                client_order_id=client_order_id,
                venue_order_id=VenueOrderId(order_id),
                trade_id=TradeId(order_id),
                order_side=order_side,
                last_qty=Quantity.from_int(fill_count),
                last_px=Price.from_int(price_cents),
                commission=Money(fee, USD),
                liquidity_side=liquidity_side,
                report_id=UUID4(),
                ts_event=ts_init,
                ts_init=ts_init,
            )
            reports.append(report)

        self._log.info(f"Generated {len(reports)} fill reports")
        return reports

    async def generate_position_status_reports(
        self,
        command: GeneratePositionStatusReports,
    ) -> list[PositionStatusReport]:
        """Generate position status reports from Kalshi API."""
        self._log.debug("Requesting PositionStatusReports...")
        reports: list[PositionStatusReport] = []

        positions = await asyncio.to_thread(self._http_client.get_positions)
        ts_init = self._clock.timestamp_ns()

        for pos in positions:
            position_val = pos.get("position", 0)
            if position_val == 0:
                continue

            ticker = pos.get("ticker", "")
            instrument = self._instrument_provider.get_by_ticker(ticker)
            if instrument is None:
                continue

            if position_val > 0:
                position_side = PositionSide.LONG
                qty = position_val
            else:
                position_side = PositionSide.SHORT
                qty = abs(position_val)

            report = PositionStatusReport(
                account_id=self.account_id,
                instrument_id=instrument.id,
                position_side=position_side,
                quantity=Quantity.from_int(qty),
                report_id=UUID4(),
                ts_last=ts_init,
                ts_init=ts_init,
            )
            reports.append(report)

        self._log.info(f"Generated {len(reports)} position status reports")
        return reports

    # -- Helpers ---------------------------------------------------------------

    def _generate_order_submitted_event(self, command, order) -> None:
        """Emit OrderSubmitted event."""
        self.generate_order_submitted(
            strategy_id=order.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=order.client_order_id,
            ts_event=self._clock.timestamp_ns(),
        )

    def _generate_order_accepted_event(self, command, order, venue_order_id) -> None:
        """Emit OrderAccepted event."""
        self.generate_order_accepted(
            strategy_id=order.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=order.client_order_id,
            venue_order_id=venue_order_id,
            ts_event=self._clock.timestamp_ns(),
        )

    def _generate_order_rejected_event(self, command, order, reason: str) -> None:
        """Emit OrderRejected event."""
        self.generate_order_rejected(
            strategy_id=order.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=order.client_order_id,
            reason=reason,
            ts_event=self._clock.timestamp_ns(),
        )
