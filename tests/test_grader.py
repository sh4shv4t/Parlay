"""Tests for parlay_env/grader.py."""
from dashboard.api import _apply_zopa_erosion
from parlay_env.grader import (
    EpisodeGrade,
    compute_step_reward,
    compute_terminal_reward,
    detect_bluff_challenge,
    grade_episode,
)
from parlay_env.models import BeliefState, HiddenState, ParlayAction, ParlayState, PersonaType
from parlay_env.reward import OMEGA, PSI


def _make_hidden(
    budget: float = 165_000,
    walk: float = 125_000,
    last_stated_batna: float | None = None,
) -> HiddenState:
    return HiddenState(
        budget_ceiling=budget,
        walk_away_price=walk,
        urgency_score=0.5,
        has_alternative=False,
        persona_drifted=False,
        last_stated_batna=last_stated_batna,
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
    hidden: HiddenState | None = None,
) -> ParlayState:
    actual_hidden = hidden or _make_hidden()
    return ParlayState(
        session_id="test-session",
        scenario_id="saas_enterprise",
        persona=PersonaType.SHARK,
        step_count=step,
        cumulative_reward=cumulative,
        hidden_state=actual_hidden,
        belief_history=beliefs or [_make_belief()],
        offer_history=offers or [],
        drift_events_fired=0,
        episode_done=False,
        credibility_points=100,
        original_zopa_width=actual_hidden.budget_ceiling - actual_hidden.walk_away_price,
    )


class TestComputeStepReward:
    def test_happy_path_returns_float(self):
        state = _make_state()
        action = ParlayAction(utterance="I propose 145000.", offer_amount=145_000.0)
        next_state = _make_state(step=1, offers=[145_000.0])
        result = compute_step_reward(state, action, next_state)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_noise_penalty_applied(self):
        state = _make_state(offers=[140_000.0])
        action = ParlayAction(utterance="xyz", offer_amount=140_000.0)
        next_state = _make_state(step=1, offers=[140_000.0, 140_000.0])
        result = compute_step_reward(state, action, next_state)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_bluff_detection_awards_psi(self):
        hidden = _make_hidden(last_stated_batna=198_000.0)
        state = _make_state(hidden=hidden)
        next_state = _make_state(step=1, hidden=hidden)
        action = ParlayAction(
            utterance="I don't believe that's your walk-away.",
            offer_amount=None,
            tactical_move=None,
        )

        caught = detect_bluff_challenge(
            utterance=action.utterance,
            opponent_stated_batna=198_000.0,
            opponent_true_batna=165_000.0,
        )
        reward = compute_step_reward(state, action, next_state)

        assert caught is True, f"Expected True, got {caught}"
        assert reward >= PSI, f"Expected at least PSI={PSI}, got {reward}"


class TestComputeTerminalReward:
    def test_good_deal_positive_reward(self):
        state = _make_state()
        result = compute_terminal_reward(state, final_price=145_000.0, t_close=10, t_max=20)
        assert result > 0, f"Expected positive reward, got {result}"

    def test_capitulation_returns_negative_omega(self):
        state = _make_state()
        result = compute_terminal_reward(state, final_price=120_000.0, t_close=10)
        assert result == -OMEGA, f"Expected -{OMEGA}, got {result}"

    def test_speed_bonus_for_early_close(self):
        state = _make_state()
        fast = compute_terminal_reward(state, final_price=145_000.0, t_close=5, t_max=20)
        slow = compute_terminal_reward(state, final_price=145_000.0, t_close=18, t_max=20)
        assert fast > slow, f"Expected fast close > slow close: {fast} vs {slow}"


class TestGradeEpisode:
    def test_grade_episode_returns_episodegrade(self):
        state = _make_state(step=10, offers=[145_000.0])
        grade = grade_episode(state, final_price=145_000.0, t_close=10)
        assert isinstance(grade, EpisodeGrade), f"Expected EpisodeGrade, got {type(grade)}"

    def test_deal_efficiency_in_range(self):
        state = _make_state(step=10, offers=[145_000.0])
        grade = grade_episode(state, final_price=145_000.0, t_close=10)
        assert 0.0 <= grade.deal_efficiency <= 1.0, f"Expected [0,1], got {grade.deal_efficiency}"

    def test_no_deal_zero_efficiency(self):
        state = _make_state(step=20)
        grade = grade_episode(state, final_price=None)
        assert grade.deal_efficiency == 0.0, f"Expected 0.0, got {grade.deal_efficiency}"

    def test_bluffs_caught_passed_through(self):
        state = _make_state(step=10, offers=[145_000.0])
        grade = grade_episode(state, final_price=145_000.0, bluffs_caught=3)
        assert grade.bluffs_caught == 3, f"Expected 3, got {grade.bluffs_caught}"

    def test_zopa_collapse_walk_away(self):
        hidden = _make_hidden(budget=103.0, walk=100.0)
        state = _make_state(hidden=hidden)

        for _ in range(3):
            state.tension_score = 80.0
            state.high_tension_streak = 2
            _apply_zopa_erosion(state)

        assert state.zopa_erosion_ticks >= 1, f"Expected >=1, got {state.zopa_erosion_ticks}"

        while not state.walk_away and state.zopa_erosion_ticks < 100:
            state.tension_score = 80.0
            state.high_tension_streak = 2
            _apply_zopa_erosion(state)

        assert state.termination_reason == "zopa_collapsed", f"Expected zopa_collapsed, got {state.termination_reason}"
