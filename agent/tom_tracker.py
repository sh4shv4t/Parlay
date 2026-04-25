"""
Theory of Mind belief tracker for Parlay.
Tracks and updates agent beliefs about opponent hidden state.
"""
import logging
import sys
from typing import Optional

from parlay_env.models import BeliefState, HiddenState, PersonaType, TacticalMove

logger = logging.getLogger(__name__)

# NOTE: ToMTracker is used in two paths:
# (1) agent/runner.py self-play — full update each turn;
# (2) parlay_env/server.py WebSocket server — also uses ToMTracker after Task 1 fix.
# Both paths now produce comparable belief_history for grader._tom_accuracy.


class ToMTracker:
    """
    Tracks Theory of Mind beliefs about the opponent's hidden state.

    Maintains a belief history and updates beliefs based on:
    - Observed offers
    - Tactical move signals
    - Utterance content
    - Drift event triggers
    """

    def __init__(
        self,
        initial_belief: BeliefState,
        persona: PersonaType,
    ) -> None:
        """
        Args:
            initial_belief: Starting belief state (imprecise prior).
            persona:        Opponent's persona type (known to player).
        """
        self.history: list[BeliefState] = [initial_belief]
        self.persona = persona
        self._bluffs_detected: int = 0

    @property
    def current_belief(self) -> BeliefState:
        """Most recent belief state."""
        return self.history[-1]

    @property
    def bluffs_detected(self) -> int:
        """Count of detected bluffs this session."""
        return self._bluffs_detected

    def log_belief_snapshot(self, turn: int) -> None:
        """Print current belief estimates to stderr (diagnostic; training / live debug)."""
        b = self.current_belief
        print(
            f"[ToM turn={turn}] budget={b.est_budget:.3f}  "
            f"urgency={b.est_urgency:.3f}  walkaway={b.est_walk_away:.3f}",
            file=sys.stderr,
        )

    def update(
        self,
        observed_offer: Optional[float],
        observed_move: Optional[TacticalMove],
        utterance: str,
        turn: int,
    ) -> BeliefState:
        """
        Update beliefs based on latest opponent action.

        Args:
            observed_offer: Opponent's counter-offer (None if no offer made).
            observed_move:  Tactical move used (if any).
            utterance:      Opponent's latest utterance.
            turn:           Current turn number.

        Returns:
            Updated BeliefState.
        """
        last = self.current_belief

        est_budget = last.est_budget
        if observed_offer is not None:
            est_budget = max(est_budget, observed_offer * 1.05)

        est_walk_away = last.est_walk_away
        if observed_move == TacticalMove.BATNA_REVEAL:
            est_walk_away = last.est_walk_away * 0.95
            logger.debug("ToM: BATNA_REVEAL detected — hedging walk-away estimate")

        est_urgency = last.est_urgency
        if observed_offer is not None and last.est_budget > 0:
            offer_ratio = observed_offer / last.est_budget
            if offer_ratio < 0.85:
                est_urgency = min(1.0, est_urgency + 0.05)
            elif offer_ratio > 0.95:
                est_urgency = max(0.0, est_urgency - 0.03)

        est_has_alternative = last.est_has_alternative
        alternative_signals = ["competitor", "alternative", "other offer", "another bid"]
        if any(sig in utterance.lower() for sig in alternative_signals):
            est_has_alternative = True
            logger.debug("ToM: alternative signal detected in utterance")

        confidence = min(1.0, last.confidence + 0.04)

        if (
            self.persona == PersonaType.SHARK
            and observed_move == TacticalMove.BATNA_REVEAL
            and "competitor" in utterance.lower()
        ):
            self._bluffs_detected += 1
            logger.info(f"ToM: bluff detected (total: {self._bluffs_detected})")

        updated = BeliefState(
            est_budget=round(est_budget, 2),
            est_walk_away=round(est_walk_away, 2),
            est_urgency=round(est_urgency, 4),
            est_has_alternative=est_has_alternative,
            confidence=round(confidence, 4),
        )
        self.history.append(updated)
        logger.debug(
            f"ToM update turn={turn}: "
            f"budget={est_budget:,.0f}, walk={est_walk_away:,.0f}, "
            f"urgency={est_urgency:.2f}, alt={est_has_alternative}, "
            f"confidence={confidence:.2f}"
        )
        return updated

    def drift_event(
        self,
        effect_on_urgency: float,
        effect_on_has_alternative: bool,
        event_description: str = "",
    ) -> BeliefState:
        """
        Apply a drift event to beliefs.

        Args:
            effect_on_urgency:         Signed delta to urgency estimate.
            effect_on_has_alternative: Override for has_alternative belief.
            event_description:         Human-readable scenario event string
                                       (e.g. "Competitor drops price 15%").

        Returns:
            Updated BeliefState post-drift.
        """
        last = self.current_belief
        new_urgency = float(max(0.0, min(1.0, last.est_urgency + effect_on_urgency)))
        updated = BeliefState(
            est_budget=last.est_budget,
            est_walk_away=last.est_walk_away,
            est_urgency=round(new_urgency, 4),
            est_has_alternative=effect_on_has_alternative,
            confidence=max(0.0, last.confidence - 0.15),  # drift reduces confidence
        )
        self.history.append(updated)
        desc_part = f" | event={event_description!r}" if event_description else ""
        logger.info(
            f"ToM drift applied{desc_part}: "
            f"urgency_delta={effect_on_urgency:+.2f} → {new_urgency:.2f}, "
            f"alt={effect_on_has_alternative}"
        )
        return updated

    def brier_scores(self, hidden: HiddenState) -> dict[str, float]:
        """
        Compute per-field Brier scores over the full belief history.

        Brier score = (1/N) Σ (predicted - actual)²
        Lower is better; 0 = perfect.

        Fields scored:
          - urgency:      est_urgency (continuous 0–1) vs hidden.urgency_score
          - has_alt:      est_has_alternative (0/1 probability) vs hidden.has_alternative

        Args:
            hidden: The true hidden state revealed at episode end.

        Returns:
            Dict with keys "urgency" and "has_alt", each a float in [0, 1].
        """
        if not self.history:
            return {"urgency": 1.0, "has_alt": 1.0}

        actual_urgency = hidden.urgency_score
        actual_alt = float(hidden.has_alternative)

        urgency_sq_err = sum(
            (b.est_urgency - actual_urgency) ** 2 for b in self.history
        )
        alt_sq_err = sum(
            (float(b.est_has_alternative) - actual_alt) ** 2 for b in self.history
        )
        n = len(self.history)
        return {
            "urgency": round(urgency_sq_err / n, 6),
            "has_alt": round(alt_sq_err / n, 6),
        }

    def accuracy_against(self, hidden: HiddenState) -> float:
        """
        Compute current belief accuracy against true hidden state.

        Args:
            hidden: The true hidden state.

        Returns:
            Accuracy score in [0, 1].
        """
        b = self.current_belief
        budget_range = max(hidden.budget_ceiling * 0.5, 1.0)
        walk_range   = max(hidden.walk_away_price * 0.5, 1.0)

        budget_err  = abs(b.est_budget       - hidden.budget_ceiling)  / budget_range
        walk_err    = abs(b.est_walk_away    - hidden.walk_away_price) / walk_range
        urgency_err = abs(b.est_urgency      - hidden.urgency_score)
        alt_err     = 0.0 if b.est_has_alternative == hidden.has_alternative else 1.0

        mean_err = (budget_err + walk_err + urgency_err + alt_err) / 4.0
        return max(0.0, 1.0 - mean_err)
