"""Tests for parlay_env/reward.py constants."""
import pytest
from parlay_env.reward import (
    ALPHA, BETA, DELTA, THETA,
    GAMMA, EPSILON, ZETA, ETA, OMEGA,
    MAX_TURNS, CP_START, CP_REGEN,
)


class TestRewardConstants:
    def test_all_constants_positive(self):
        for name, val in [
            ("ALPHA", ALPHA), ("BETA", BETA), ("DELTA", DELTA), ("THETA", THETA),
            ("GAMMA", GAMMA), ("EPSILON", EPSILON), ("ZETA", ZETA), ("ETA", ETA),
            ("OMEGA", OMEGA),
        ]:
            assert val > 0, f"Expected {name} > 0, got {val}"

    def test_omega_is_largest_terminal(self):
        assert OMEGA > GAMMA, f"Expected OMEGA({OMEGA}) > GAMMA({GAMMA})"

    def test_gamma_is_primary_terminal_reward(self):
        assert GAMMA == 100.0, f"Expected GAMMA=100.0, got {GAMMA}"

    def test_max_acts_bonus_is_30(self):
        assert ETA * 3 == 30.0, f"Expected ETA*3=30, got {ETA*3}"

    def test_cp_start_and_regen(self):
        assert CP_START == 100, f"Expected CP_START=100, got {CP_START}"
        assert CP_REGEN == 5, f"Expected CP_REGEN=5, got {CP_REGEN}"
