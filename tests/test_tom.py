"""Tests for agent/tom_tracker.py."""
import pytest
from parlay_env.models import BeliefState, HiddenState, PersonaType, TacticalMove
from agent.tom_tracker import ToMTracker


def _initial_belief() -> BeliefState:
    return BeliefState(
        est_budget=130_000, est_walk_away=135_000,
        est_urgency=0.50, est_has_alternative=False, confidence=0.30,
    )


def _hidden() -> HiddenState:
    return HiddenState(
        budget_ceiling=165_000, walk_away_price=125_000,
        urgency_score=0.6, has_alternative=True, persona_drifted=False,
    )


class TestToMTracker:
    def test_initial_belief_stored(self):
        tracker = ToMTracker(_initial_belief(), PersonaType.SHARK)
        assert len(tracker.history) == 1, f"Expected 1, got {len(tracker.history)}"
        assert tracker.current_belief.confidence == 0.30

    def test_update_grows_history(self):
        tracker = ToMTracker(_initial_belief(), PersonaType.SHARK)
        tracker.update(observed_offer=140_000, observed_move=None, utterance="test", turn=1)
        assert len(tracker.history) == 2, f"Expected 2, got {len(tracker.history)}"

    def test_alternative_signal_in_utterance(self):
        tracker = ToMTracker(_initial_belief(), PersonaType.DIPLOMAT)
        tracker.update(
            observed_offer=None,
            observed_move=None,
            utterance="We have a competitor offer on the table.",
            turn=1,
        )
        assert tracker.current_belief.est_has_alternative is True, \
            "Expected alternative to be detected from utterance"

    def test_confidence_increases_over_turns(self):
        tracker = ToMTracker(_initial_belief(), PersonaType.ANALYST)
        first_confidence = tracker.current_belief.confidence
        tracker.update(140_000, None, "Let's review the data.", 1)
        tracker.update(138_000, None, "What metrics support this?", 2)
        assert tracker.current_belief.confidence > first_confidence, \
            f"Expected confidence to grow: {first_confidence} -> {tracker.current_belief.confidence}"

    def test_drift_event_reduces_confidence(self):
        tracker = ToMTracker(_initial_belief(), PersonaType.WILDCARD)
        pre_confidence = tracker.current_belief.confidence
        tracker.drift_event(effect_on_urgency=0.3, effect_on_has_alternative=True)
        assert tracker.current_belief.confidence < pre_confidence, \
            "Expected confidence to drop after drift"

    def test_accuracy_returns_zero_to_one(self):
        tracker = ToMTracker(_initial_belief(), PersonaType.VETERAN)
        acc = tracker.accuracy_against(_hidden())
        assert 0.0 <= acc <= 1.0, f"Expected [0,1], got {acc}"

    def test_bluffs_detected_increments(self):
        tracker = ToMTracker(_initial_belief(), PersonaType.SHARK)
        tracker.update(
            observed_offer=None,
            observed_move=TacticalMove.BATNA_REVEAL,
            utterance="We have a competitor offering less.",
            turn=1,
        )
        assert tracker.bluffs_detected >= 1, \
            f"Expected bluff to be detected, got {tracker.bluffs_detected}"
