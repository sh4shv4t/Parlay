"""Tests for parlay_env/game_theory.py."""
import pytest
import math
from parlay_env.game_theory import (
    compute_zopa,
    compute_nash_bargaining_solution,
    compute_pareto_frontier,
    compute_shapley_value,
    offer_anchoring_effect,
    compute_rubinstein_deadline_advantage,
)


class TestComputeZOPA:
    def test_zopa_exists(self):
        result = compute_zopa(batna_buyer=165_000, batna_seller=125_000)
        assert result is not None, "Expected ZOPA to exist"
        assert result == (125_000, 165_000), f"Expected (125000, 165000), got {result}"

    def test_no_zopa(self):
        result = compute_zopa(batna_buyer=100_000, batna_seller=150_000)
        assert result is None, f"Expected None, got {result}"

    def test_equal_batna_no_zopa(self):
        result = compute_zopa(batna_buyer=100_000, batna_seller=100_000)
        assert result is None, f"Expected None for equal BATNAs, got {result}"

    def test_zopa_ordering(self):
        lower, upper = compute_zopa(batna_buyer=200, batna_seller=100)
        assert lower < upper, f"Expected lower < upper: {lower} < {upper}"


class TestComputeNashBargainingSolution:
    def test_midpoint(self):
        result = compute_nash_bargaining_solution(batna_buyer=200, batna_seller=100)
        assert result == 150.0, f"Expected 150.0, got {result}"

    def test_symmetric(self):
        a = compute_nash_bargaining_solution(batna_buyer=300, batna_seller=100)
        b = compute_nash_bargaining_solution(batna_buyer=100, batna_seller=300)
        assert a == b, f"Expected symmetric result: {a} vs {b}"

    def test_large_values(self):
        result = compute_nash_bargaining_solution(16_000_000, 10_500_000)
        assert result == 13_250_000.0, f"Expected 13250000, got {result}"


class TestComputeParetoFrontier:
    def test_empty_input(self):
        result = compute_pareto_frontier([])
        assert result == [], f"Expected empty list, got {result}"

    def test_single_point_is_pareto(self):
        result = compute_pareto_frontier([(5.0, 5.0)])
        assert result == [(5.0, 5.0)], f"Expected [(5,5)], got {result}"

    def test_dominated_point_excluded(self):
        result = compute_pareto_frontier([(1.0, 1.0), (2.0, 2.0)])
        assert (1.0, 1.0) not in result, f"Dominated point should be excluded: {result}"
        assert (2.0, 2.0) in result, f"Dominant point should be included: {result}"

    def test_pareto_trade_off(self):
        result = compute_pareto_frontier([(3.0, 1.0), (1.0, 3.0), (2.0, 2.0)])
        assert len(result) >= 2, f"Expected at least 2 Pareto points, got {result}"


class TestComputeShapleyValue:
    def test_two_players_symmetric(self):
        values = {
            frozenset(): 0.0,
            frozenset(["A"]): 10.0,
            frozenset(["B"]): 10.0,
            frozenset(["A", "B"]): 30.0,
        }
        result = compute_shapley_value(values)
        assert abs(result["A"] - result["B"]) < 1e-9, f"Expected symmetric: {result}"

    def test_shapley_values_sum_to_grand_coalition(self):
        values = {
            frozenset(): 0.0,
            frozenset(["A"]): 5.0,
            frozenset(["B"]): 8.0,
            frozenset(["A", "B"]): 20.0,
        }
        result = compute_shapley_value(values)
        total = sum(result.values())
        assert abs(total - 20.0) < 1e-9, f"Expected sum=20, got {total}"


class TestOfferAnchoringEffect:
    def test_anchoring_reduces_adjustment(self):
        result = offer_anchoring_effect(anchor=100, adjustment=50)
        assert result < 150, f"Expected < 150 (anchoring), got {result}"
        assert result > 100, f"Expected > 100 (some adjustment), got {result}"

    def test_zero_adjustment(self):
        result = offer_anchoring_effect(anchor=100, adjustment=0)
        assert result == 100.0, f"Expected 100, got {result}"


class TestRubinsteinDeadlineAdvantage:
    def test_returns_value_in_range(self):
        result = compute_rubinstein_deadline_advantage(turns_remaining=10)
        assert 0.0 <= result <= 1.0, f"Expected [0,1], got {result}"

    def test_zero_turns_returns_zero(self):
        result = compute_rubinstein_deadline_advantage(turns_remaining=0)
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_more_patient_opponent_advantages_first_mover(self):
        patient_opp = compute_rubinstein_deadline_advantage(
            turns_remaining=5, discount_rate_opponent=0.90
        )
        impatient_opp = compute_rubinstein_deadline_advantage(
            turns_remaining=5, discount_rate_opponent=0.50
        )
        assert patient_opp != impatient_opp, "Different discount rates should give different shares"

    def test_invalid_discount_rate_raises(self):
        with pytest.raises(ValueError):
            compute_rubinstein_deadline_advantage(turns_remaining=5, discount_rate_self=1.5)
