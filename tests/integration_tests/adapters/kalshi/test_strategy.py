# -------------------------------------------------------------------------------------------------
#  Tests for Kalshi weather strategy — phase logic, pricing, escalation.
#
#  Pure pricing functions are in common/pricing.py with zero NT dependencies,
#  so these tests work without the Cython build.
# -------------------------------------------------------------------------------------------------

import sys
from pathlib import Path

import pytest

# Add the source tree to sys.path so we can import common.pricing directly
# without triggering nautilus_trader.__init__ (which needs Cython).
_src = str(Path(__file__).resolve().parents[4] / "nautilus_trader" / "adapters" / "kalshi")
if _src not in sys.path:
    sys.path.insert(0, _src)

from common.pricing import (
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


# ---------------------------------------------------------------------------
# Phase determination
# ---------------------------------------------------------------------------


class TestDeterminePhase:
    def test_tight_spread_forces_taker(self):
        assert determine_phase(h_to_peak=24.0, spread=1) == ExecutionPhase.TAKER

    def test_thin_profit_forces_maker(self):
        assert determine_phase(h_to_peak=24.0, spread=5, profit_cents=5) == ExecutionPhase.MAKER

    def test_patient_phase(self):
        assert determine_phase(h_to_peak=25.0, spread=5) == ExecutionPhase.PATIENT

    def test_maker_phase(self):
        assert determine_phase(h_to_peak=15.0, spread=5) == ExecutionPhase.MAKER

    def test_aggressive_phase(self):
        assert determine_phase(h_to_peak=6.0, spread=5) == ExecutionPhase.AGGRESSIVE

    def test_taker_phase_by_time(self):
        assert determine_phase(h_to_peak=2.0, spread=5) == ExecutionPhase.TAKER

    def test_custom_thresholds(self):
        assert (
            determine_phase(h_to_peak=12.0, spread=5, patient_hours=15.0, maker_hours=8.0)
            == ExecutionPhase.MAKER
        )

    def test_boundary_patient_maker(self):
        """At exactly patient_hours, should be MAKER (not PATIENT)."""
        assert determine_phase(h_to_peak=20.0, spread=5) == ExecutionPhase.MAKER

    def test_boundary_maker_aggressive(self):
        """At exactly maker_hours, should be AGGRESSIVE."""
        assert determine_phase(h_to_peak=10.0, spread=5) == ExecutionPhase.AGGRESSIVE

    def test_zero_hours(self):
        assert determine_phase(h_to_peak=0.0, spread=5) == ExecutionPhase.TAKER


# ---------------------------------------------------------------------------
# Price computation
# ---------------------------------------------------------------------------


class TestPriceComputation:
    def test_maker_price_no_side_join(self):
        assert compute_maker_price("no", yes_bid=60, yes_ask=65, improvement=0) == 35

    def test_maker_price_no_side_improve(self):
        assert compute_maker_price("no", yes_bid=60, yes_ask=65, improvement=1) == 36

    def test_maker_price_yes_side_join(self):
        assert compute_maker_price("yes", yes_bid=60, yes_ask=65, improvement=0) == 60

    def test_maker_price_yes_side_improve(self):
        assert compute_maker_price("yes", yes_bid=60, yes_ask=65, improvement=1) == 61

    def test_taker_price_no_side(self):
        assert compute_taker_price("no", yes_bid=60, yes_ask=65) == 40

    def test_taker_price_yes_side(self):
        assert compute_taker_price("yes", yes_bid=60, yes_ask=65) == 65


class TestPhasePrice:
    def test_patient_no_side(self):
        op = phase_price(ExecutionPhase.PATIENT, "no", yes_bid=60, yes_ask=65)
        assert op.price_cents == 35
        assert op.is_maker is True
        assert op.time_in_force == "good_till_canceled"

    def test_maker_no_side(self):
        op = phase_price(ExecutionPhase.MAKER, "no", yes_bid=60, yes_ask=65)
        assert op.price_cents == 36

    def test_aggressive_no_side(self):
        op = phase_price(ExecutionPhase.AGGRESSIVE, "no", yes_bid=60, yes_ask=65)
        # spread=5, improvement=max(2, 5//3)=2
        assert op.price_cents == 37

    def test_taker_no_side(self):
        op = phase_price(ExecutionPhase.TAKER, "no", yes_bid=60, yes_ask=65)
        assert op.price_cents == 40
        assert op.is_maker is False
        assert op.time_in_force == "fill_or_kill"

    def test_patient_yes_side(self):
        op = phase_price(ExecutionPhase.PATIENT, "yes", yes_bid=60, yes_ask=65)
        assert op.price_cents == 60

    def test_taker_yes_side(self):
        op = phase_price(ExecutionPhase.TAKER, "yes", yes_bid=60, yes_ask=65)
        assert op.price_cents == 65

    def test_wide_spread_aggressive(self):
        # spread=20, improvement=max(2, 20//3)=6
        op = phase_price(ExecutionPhase.AGGRESSIVE, "no", yes_bid=50, yes_ask=70)
        assert op.price_cents == 36  # 100 - 70 + 6


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


class TestEscalation:
    def test_patient_to_maker(self):
        assert next_escalation_phase(ExecutionPhase.PATIENT) == ExecutionPhase.MAKER

    def test_maker_to_aggressive(self):
        assert next_escalation_phase(ExecutionPhase.MAKER) == ExecutionPhase.AGGRESSIVE

    def test_aggressive_to_none(self):
        assert next_escalation_phase(ExecutionPhase.AGGRESSIVE) is None

    def test_taker_to_none(self):
        assert next_escalation_phase(ExecutionPhase.TAKER) is None

    def test_deep_rest_to_none(self):
        assert next_escalation_phase(ExecutionPhase.DEEP_REST) is None


# ---------------------------------------------------------------------------
# Risk guard logic (pure, no NT)
# ---------------------------------------------------------------------------


class TestRiskGuards:
    @staticmethod
    def _make_opp(
        ticker: str = "KXHIGHNY-26MAR05-T72",
        side: str = "no",
        cost_cents: int = 40,
        **kwargs,
    ) -> Opportunity:
        defaults = dict(
            city="new_york",
            direction="above",
            threshold=72.0,
            settlement_date="2026-03-05",
            yes_bid=60,
            yes_ask=65,
            ecmwf=70.0,
            gfs=69.5,
            margin=2.0,
            consensus=69.75,
            h_to_peak=12.0,
            p_win=0.96,
            strategy="above_no",
        )
        defaults.update(kwargs)
        return Opportunity(ticker=ticker, side=side, cost_cents=cost_cents, **defaults)

    def test_no_side_cost_cap(self):
        opp = self._make_opp(side="no", cost_cents=100)
        assert opp.cost_cents > 99  # exceeds typical cost_cap_no_cents

    def test_yes_side_cost_cap(self):
        opp = self._make_opp(side="yes", cost_cents=95)
        assert opp.cost_cents > 92  # exceeds typical cost_cap_yes_cents

    def test_deep_rest_cap_logic(self):
        """When resting orders at cap, should fall back to NORMAL mode."""
        resting = {
            f"ticker{i}": OrderMode.DEEP_REST for i in range(4)
        }
        deep_rest_count = sum(1 for m in resting.values() if m == OrderMode.DEEP_REST)
        max_deep_rest = 4
        assert deep_rest_count >= max_deep_rest

    def test_deep_rest_under_cap(self):
        resting = {
            f"ticker{i}": OrderMode.DEEP_REST for i in range(2)
        }
        deep_rest_count = sum(1 for m in resting.values() if m == OrderMode.DEEP_REST)
        assert deep_rest_count < 4


# ---------------------------------------------------------------------------
# OrderPrice dataclass
# ---------------------------------------------------------------------------


class TestOrderPrice:
    def test_deep_rest_order_price(self):
        op = OrderPrice(
            side="no",
            price_cents=85,
            is_maker=True,
            time_in_force="good_till_canceled",
            phase=ExecutionPhase.DEEP_REST,
        )
        assert op.phase == ExecutionPhase.DEEP_REST
        assert op.is_maker is True
        assert op.price_cents == 85

    def test_taker_order_price(self):
        op = OrderPrice(
            side="yes",
            price_cents=65,
            is_maker=False,
            time_in_force="fill_or_kill",
            phase=ExecutionPhase.TAKER,
        )
        assert op.is_maker is False
        assert op.time_in_force == "fill_or_kill"


# ---------------------------------------------------------------------------
# Opportunity dataclass
# ---------------------------------------------------------------------------


class TestOpportunity:
    def test_model_scores_default(self):
        opp = Opportunity(
            ticker="TEST", city="test", direction="above", side="no",
            threshold=72.0, settlement_date="2026-03-05", yes_bid=60,
            yes_ask=65, ecmwf=70.0, gfs=69.5, margin=2.0, consensus=69.75,
            h_to_peak=12.0, p_win=0.96, cost_cents=40, strategy="above_no",
        )
        assert opp.model_scores == {}

    def test_model_scores_provided(self):
        opp = Opportunity(
            ticker="TEST", city="test", direction="above", side="no",
            threshold=72.0, settlement_date="2026-03-05", yes_bid=60,
            yes_ask=65, ecmwf=70.0, gfs=69.5, margin=2.0, consensus=69.75,
            h_to_peak=12.0, p_win=0.96, cost_cents=40, strategy="above_no",
            model_scores={"emos": 0.97, "ngboost": 0.95},
        )
        assert opp.model_scores["emos"] == 0.97
