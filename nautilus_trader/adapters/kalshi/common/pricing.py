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

"""Pure pricing functions and execution phase logic for Kalshi weather strategy.

Zero NautilusTrader dependencies — fully testable without the Cython build.
Ported from quantdesk/src/quantdesk/execution/kalshi/order_manager.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ExecutionPhase(str, Enum):
    PATIENT = "patient"
    MAKER = "maker"
    AGGRESSIVE = "aggressive"
    TAKER = "taker"
    DEEP_REST = "deep_rest"


class OrderMode(str, Enum):
    NORMAL = "normal"
    DEEP_REST = "deep_rest"


@dataclass
class OrderPrice:
    side: str  # "yes" or "no"
    price_cents: int
    is_maker: bool
    time_in_force: str  # "fill_or_kill" or "good_till_canceled"
    phase: ExecutionPhase


@dataclass
class Opportunity:
    """Minimal opportunity representation for the strategy layer."""

    ticker: str
    city: str
    direction: str
    side: str
    threshold: float
    settlement_date: str
    yes_bid: int
    yes_ask: int
    ecmwf: float
    gfs: float
    margin: float
    consensus: float
    h_to_peak: float
    p_win: float
    cost_cents: int
    strategy: str
    model_scores: dict[str, float] = field(default_factory=dict)


def compute_maker_price(side: str, yes_bid: int, yes_ask: int, improvement: int = 0) -> int:
    """Side-correct maker price.

    NO side: 100 - yes_ask + improvement (join/improve NO bid).
    YES side: yes_bid + improvement (join/improve YES bid).
    """
    if side == "no":
        return 100 - yes_ask + improvement
    return yes_bid + improvement


def compute_taker_price(side: str, yes_bid: int, yes_ask: int) -> int:
    """Side-correct taker price (crossing the spread).

    NO side: 100 - yes_bid (cross YES bid).
    YES side: yes_ask (cross YES ask).
    """
    if side == "no":
        return 100 - yes_bid
    return yes_ask


def determine_phase(
    h_to_peak: float,
    spread: int,
    *,
    min_spread_for_maker: int = 2,
    profit_cents: int = 99,
    patient_hours: float = 20.0,
    maker_hours: float = 10.0,
    aggressive_hours: float = 4.0,
) -> ExecutionPhase:
    """Determine execution phase based on urgency and market conditions."""
    if spread < min_spread_for_maker:
        return ExecutionPhase.TAKER
    if profit_cents < 8:
        return ExecutionPhase.MAKER
    if h_to_peak > patient_hours:
        return ExecutionPhase.PATIENT
    if h_to_peak > maker_hours:
        return ExecutionPhase.MAKER
    if h_to_peak > aggressive_hours:
        return ExecutionPhase.AGGRESSIVE
    return ExecutionPhase.TAKER


def phase_price(phase: ExecutionPhase, side: str, yes_bid: int, yes_ask: int) -> OrderPrice:
    """Compute OrderPrice for a given phase and side."""
    spread = yes_ask - yes_bid
    if phase == ExecutionPhase.TAKER:
        return OrderPrice(
            side=side,
            price_cents=compute_taker_price(side, yes_bid, yes_ask),
            is_maker=False,
            time_in_force="fill_or_kill",
            phase=phase,
        )
    improvement = {
        ExecutionPhase.PATIENT: 0,
        ExecutionPhase.MAKER: 1,
        ExecutionPhase.AGGRESSIVE: max(2, spread // 3),
    }.get(phase, 0)
    return OrderPrice(
        side=side,
        price_cents=compute_maker_price(side, yes_bid, yes_ask, improvement),
        is_maker=True,
        time_in_force="good_till_canceled",
        phase=phase,
    )


def next_escalation_phase(current: ExecutionPhase) -> ExecutionPhase | None:
    """Return the next escalation phase, or None if at max (AGGRESSIVE)."""
    order = [ExecutionPhase.PATIENT, ExecutionPhase.MAKER, ExecutionPhase.AGGRESSIVE]
    try:
        idx = order.index(current)
        if idx + 1 < len(order):
            return order[idx + 1]
    except ValueError:
        pass
    return None
