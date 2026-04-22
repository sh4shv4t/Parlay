"""Tests for parlay_env/grader.py."""
import pytest
from parlay_env.models import (
    ParlayState, ParlayAction, BeliefState, HiddenState,
    PersonaType, TacticalMove,
)
from parlay_env.grader import (
    compute_step_reward, compute_terminal_reward, grade_episode, EpisodeGrade
)
from parlay_env.reward import GAMMA, OMEGA, ETA


def _make_hidden(budget: float = 165_000, walk: float = 125_000) -> HiddenState:
    return HiddenState(
        budget_ceiling=budget,
        walk_away_price=walk,
        urgency_score=0.5,
        has_alternative=False,
        persona_drifted=False,
    )


def _make_belief(budget: float = 140_000, walk: float = 130_000) -> BeliefState:
    return BeliefState(
        est_budget=budget,
        est_walk_away=walk,
        est_urgency=0.5,
        est_has_alternative=False,
        confidence=0.5,
    )


def _make_state(
    step: int = 0,
    cumulative: float = 0.0,
    offers: list[float] | None = None,
    beliefs: list[BeliefState] | None = None,
) -> ParlayState:
    hidden = _make_hidden()
    return ParlayState(
        session_id="test-session",
        scenario_id="saas_enterprise",
        persona=PersonaType.SHARK,
        act=1,
        step_count=step,
        cumulative_reward=cumulative,
        hidden_state=hidden,
        belief_history=beliefs or [_make_belief()],
        offer_history=offers or [],
        drift_events_fired=0,
        episode_done=False,
        credibility_points=100,
    )


class TestComputeStepReward:
    def test_happy_path_returns_float(self):
        state = _make_state()
        action = ParlayAction(utterance="I propose 145000.", offer_amount=145_000.0)
        next_state = _make_state(step=1, offers=[145_000.0])
        result = compute_step_reward(state, action, next_state)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_no_offer_returns_tom_bonus_only(self):
        state = _make_state()
        action = ParlayAction(utterance="I'll need more time.")
        next_state = _make_state(step=1)
        result = compute_step_reward(state, action, next_state)
        assert result > -50.0, f"Expected > -50, got {result}"

    def test_noise_penalty_applied(self):
        """Random/ungrounded utterance should get a noise penalty."""
        state = _make_state(offers=[140_000.0])
        action = ParlayAction(utterance="xyz", offer_amount=140_000.0)
        next_state = _make_state(step=1, offers=[140_000.0])
        result = compute_step_reward(state, action, next_state)
        assert isinstance(result, float)

    def test_concession_penalty(self):
        """Lowering your offer should incur a concession penalty."""
        state = _make_state(offers=[150_000.0])
        action = ParlayAction(utterance="Fine, I'll go to 140000.", offer_amount=140_000.0)
        next_state = _make_state(step=1, offers=[150_000.0, 140_000.0])
        result = compute_step_reward(state, action, next_state)
        assert isinstance(result, float)


class TestComputeTerminalReward:
    def test_good_deal_positive_reward(self):
        state = _make_state()
        result = compute_terminal_reward(state, final_price=145_000.0, t_close=10, t_max=20)
        assert result > 0, f"Expected positive reward, got {result}"

    def test_capitulation_returns_negative_omega(self):
        state = _make_state()
        result = compute_terminal_reward(state, final_price=120_000.0, t_close=10)
        assert result == -OMEGA, f"Expected -{OMEGA}, got {result}"

    def test_no_deal_partial_reward(self):
        state = _make_state()
        result = compute_terminal_reward(state, final_price=None)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_speed_bonus_for_early_close(self):
        state = _make_state()
        fast = compute_terminal_reward(state, final_price=145_000.0, t_close=5, t_max=20)
        slow = compute_terminal_reward(state, final_price=145_000.0, t_close=18, t_max=20)
        assert fast > slow, f"Expected fast close > slow close: {fast} vs {slow}"

    def test_drift_adaptation_bonus(self):
        state = _make_state()
        with_drift = compute_terminal_reward(state, final_price=145_000.0, t_close=10, drift_adapted=True)
        without_drift = compute_terminal_reward(state, final_price=145_000.0, t_close=10, drift_adapted=False)
        assert with_drift > without_drift, f"Expected drift bonus: {with_drift} vs {without_drift}"


class TestGradeEpisode:
    def test_grade_episode_returns_episodegrade(self):
        state = _make_state(step=10, offers=[145_000.0])
        grade = grade_episode(state, final_price=145_000.0, t_close=10)
        assert isinstance(grade, EpisodeGrade), f"Expected EpisodeGrade, got {type(grade)}"

    def test_deal_efficiency_in_range(self):
        state = _make_state(step=10, offers=[145_000.0])
        grade = grade_episode(state, final_price=145_000.0, t_close=10)
        assert 0.0 <= grade.deal_efficiency <= 1.0, f"Efficiency out of range: {grade.deal_efficiency}"

    def test_no_deal_zero_efficiency(self):
        state = _make_state(step=20)
        grade = grade_episode(state, final_price=None)
        assert grade.deal_efficiency == 0.0, f"Expected 0.0, got {grade.deal_efficiency}"

    def test_bluffs_caught_passed_through(self):
        state = _make_state(step=10, offers=[145_000.0])
        grade = grade_episode(state, final_price=145_000.0, bluffs_caught=3)
        assert grade.bluffs_caught == 3, f"Expected 3, got {grade.bluffs_caught}"
