"""
Bayesian Theory-of-Mind belief tracker for Parlay.

Drop-in replacement for ToMTracker that uses Kalman-filter-style Gaussian
belief updates instead of hand-tuned arithmetic nudges.

Key insight
-----------
The opponent has hidden variables (budget_ceiling, walk_away_price, urgency,
has_alternative). Each observed offer is a noisy signal about these.
We model each continuous variable as a Gaussian (mean, variance) and update
using the standard Bayesian update for Gaussian conjugate priors:

    posterior_mean = (prior_mean / prior_var + obs / obs_var) /
                     (1 / prior_var + 1 / obs_var)
    posterior_var  = 1 / (1 / prior_var + 1 / obs_var)

`confidence` is derived from the posterior variance:
    confidence = 1 / (1 + sqrt(budget_var / budget_mean²))

Usage (as feature-flag alternative to ToMTracker):
    from agent.tom_tracker_bayesian import BayesianToMTracker as ToMTracker

    # Then use exactly the same API as ToMTracker — all method signatures match.
"""
import logging
import math
import sys
from typing import Optional

from parlay_env.models import BeliefState, HiddenState, PersonaType, TacticalMove

logger = logging.getLogger(__name__)


class BayesianToMTracker:
    """
    Gaussian-posterior belief tracker for the opponent's hidden state.

    Extends the original ToMTracker API with proper Bayesian updating.
    The same public methods (update, drift_event, accuracy_against,
    brier_scores, log_belief_snapshot) are preserved for drop-in use.

    Internal state:
        _budget_mean, _budget_var    — Gaussian over opponent's budget ceiling.
        _walk_mean,   _walk_var      — Gaussian over opponent's walk-away price.
        _urgency_mean, _urgency_var  — Gaussian over urgency [0, 1].
        _alt_prob                    — Bernoulli probability of has_alternative.
    """

    # Observation noise variances (tuned for B2B negotiation scale).
    # Budget/walk-away: observed offer is a noisy signal; high variance because
    # opponents rarely reveal their true limits.
    _OBS_BUDGET_VAR_FRAC = 0.10   # 10% of current mean estimate as std
    _OBS_URGENCY_VAR = 0.05       # small update per offer-ratio signal

    def __init__(
        self,
        initial_belief: BeliefState,
        persona: PersonaType,
    ) -> None:
        """
        Args:
            initial_belief: Starting BeliefState (imprecise prior).
            persona:        Opponent persona (known to the player).
        """
        self.persona = persona
        self._bluffs_detected: int = 0

        # Initialise Gaussian priors from the initial belief
        self._budget_mean = float(initial_belief.est_budget)
        self._walk_mean   = float(initial_belief.est_walk_away)
        self._urgency_mean = float(initial_belief.est_urgency)
        self._alt_prob    = 0.3   # prior: 30% chance opponent has an alternative

        # Initial variances — large uncertainty at the start
        self._budget_var  = (self._budget_mean * 0.30) ** 2   # ±30% std
        self._walk_var    = (self._walk_mean   * 0.30) ** 2
        self._urgency_var = 0.08   # std ≈ 0.28 over [0, 1]

        self.history: list[BeliefState] = [self._snapshot()]
        logger.debug(
            "BayesianToMTracker init: budget_mean=%.0f walk_mean=%.0f urgency_mean=%.2f",
            self._budget_mean, self._walk_mean, self._urgency_mean,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _snapshot(self) -> BeliefState:
        """Convert current Gaussian state to a BeliefState snapshot."""
        confidence = self._compute_confidence()
        return BeliefState(
            est_budget=round(self._budget_mean, 2),
            est_walk_away=round(self._walk_mean, 2),
            est_urgency=round(max(0.0, min(1.0, self._urgency_mean)), 4),
            est_has_alternative=self._alt_prob >= 0.5,
            confidence=round(confidence, 4),
        )

    def _compute_confidence(self) -> float:
        """
        Confidence = 1 - mean relative std across all variables.
        Shrinks variance → higher confidence.
        """
        budget_rel_std = math.sqrt(self._budget_var) / max(abs(self._budget_mean), 1.0)
        walk_rel_std   = math.sqrt(self._walk_var)   / max(abs(self._walk_mean), 1.0)
        urgency_std    = math.sqrt(self._urgency_var)
        alt_std        = math.sqrt(self._alt_prob * (1.0 - self._alt_prob))
        mean_uncertainty = (budget_rel_std + walk_rel_std + urgency_std + alt_std) / 4.0
        return max(0.0, min(1.0, 1.0 - mean_uncertainty))

    @staticmethod
    def _gaussian_update(
        prior_mean: float,
        prior_var: float,
        obs: float,
        obs_var: float,
    ) -> tuple[float, float]:
        """
        Closed-form Bayesian update for Gaussian conjugate prior.

        posterior_mean = (prior_mean / prior_var + obs / obs_var) /
                         (1 / prior_var + 1 / obs_var)
        posterior_var  = 1 / (1 / prior_var + 1 / obs_var)
        """
        prec_prior = 1.0 / max(prior_var, 1e-10)
        prec_obs   = 1.0 / max(obs_var, 1e-10)
        posterior_prec = prec_prior + prec_obs
        posterior_mean = (prec_prior * prior_mean + prec_obs * obs) / posterior_prec
        posterior_var  = 1.0 / posterior_prec
        return posterior_mean, posterior_var

    # ── Public API (matches ToMTracker) ──────────────────────────────────────

    @property
    def current_belief(self) -> BeliefState:
        return self.history[-1]

    @property
    def bluffs_detected(self) -> int:
        return self._bluffs_detected

    def log_belief_snapshot(self, turn: int) -> None:
        b = self.current_belief
        print(
            f"[BayesToM turn={turn}] "
            f"budget={b.est_budget:.0f}±{math.sqrt(self._budget_var):.0f}  "
            f"urgency={b.est_urgency:.3f}±{math.sqrt(self._urgency_var):.3f}  "
            f"alt_prob={self._alt_prob:.2f}  conf={b.confidence:.2f}",
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
        Bayesian update of all beliefs from one observed opponent action.

        Budget update: if we see an offer O, the true budget is likely > O.
            We treat O as a lower-bound signal: observation = O * 1.05
            with variance proportional to the current mean.
        Urgency update: offer-ratio below 0.85 → urgency signal 0.7;
            above 0.95 → urgency signal 0.3. Both with moderate obs variance.
        has_alternative: updated as Bernoulli likelihood ratio (keyword match).
        """
        # ── Budget Bayesian update ──────────────────────────────────────────
        if observed_offer is not None and observed_offer > 0:
            budget_obs = observed_offer * 1.05
            obs_budget_var = (self._budget_mean * self._OBS_BUDGET_VAR_FRAC) ** 2
            self._budget_mean, self._budget_var = self._gaussian_update(
                self._budget_mean, self._budget_var,
                budget_obs, obs_budget_var,
            )
            logger.debug(
                "Bayesian budget update: obs=%.0f → mean=%.0f std=%.0f",
                budget_obs, self._budget_mean, math.sqrt(self._budget_var),
            )

        # ── Walk-away update: BATNA_REVEAL is a noisy signal ───────────────
        if observed_move == TacticalMove.BATNA_REVEAL:
            if observed_offer is not None:
                walk_obs = observed_offer * 0.95
                obs_walk_var = (self._walk_mean * 0.15) ** 2
                self._walk_mean, self._walk_var = self._gaussian_update(
                    self._walk_mean, self._walk_var,
                    walk_obs, obs_walk_var,
                )
            logger.debug("Bayesian walk-away update via BATNA_REVEAL")

        # ── Urgency Bayesian update via offer-ratio signal ─────────────────
        if observed_offer is not None and self._budget_mean > 0:
            offer_ratio = observed_offer / self._budget_mean
            if offer_ratio < 0.85:
                urgency_obs = 0.70   # low offer → opponent likely more urgent
            elif offer_ratio > 0.95:
                urgency_obs = 0.30   # high offer → opponent comfortable
            else:
                urgency_obs = 0.50   # neutral
            self._urgency_mean, self._urgency_var = self._gaussian_update(
                self._urgency_mean, self._urgency_var,
                urgency_obs, self._OBS_URGENCY_VAR,
            )
            self._urgency_mean = max(0.0, min(1.0, self._urgency_mean))

        # ── has_alternative Bernoulli update (likelihood ratio) ────────────
        alt_signals = ["competitor", "alternative", "other offer", "another bid"]
        if any(sig in utterance.lower() for sig in alt_signals):
            self._alt_prob = min(0.95, self._alt_prob + (1.0 - self._alt_prob) * 0.35)
            logger.debug("Alternative signal detected → alt_prob=%.2f", self._alt_prob)
        else:
            self._alt_prob = max(0.05, self._alt_prob * 0.98)   # small decay

        # ── Bluff detection (shark persona + BATNA_REVEAL + "competitor") ──
        if (
            self.persona == PersonaType.SHARK
            and observed_move == TacticalMove.BATNA_REVEAL
            and "competitor" in utterance.lower()
        ):
            self._bluffs_detected += 1
            logger.info("BayesToM: bluff detected (total: %d)", self._bluffs_detected)

        updated = self._snapshot()
        self.history.append(updated)
        logger.debug(
            "BayesToM update turn=%d: budget=%.0f walk=%.0f urgency=%.2f alt_prob=%.2f conf=%.2f",
            turn, self._budget_mean, self._walk_mean, self._urgency_mean,
            self._alt_prob, updated.confidence,
        )
        return updated

    def drift_event(
        self,
        effect_on_urgency: float,
        effect_on_has_alternative: bool,
        event_description: str = "",
    ) -> BeliefState:
        """
        Apply a market/scenario drift event.

        Nudges the urgency mean and resets alt_prob based on the drift direction.
        Also inflates all variances (drift = increased uncertainty).
        """
        self._urgency_mean = float(max(0.0, min(1.0, self._urgency_mean + effect_on_urgency)))
        self._urgency_var  = min(0.1, self._urgency_var * 1.5)   # inflate uncertainty

        # Drift shifts alt belief
        if effect_on_has_alternative:
            self._alt_prob = min(0.9, self._alt_prob + 0.25)
        else:
            self._alt_prob = max(0.1, self._alt_prob - 0.1)

        # Inflate budget/walk variances — drift reduces confidence
        self._budget_var *= 1.3
        self._walk_var   *= 1.3

        updated = self._snapshot()
        self.history.append(updated)
        desc_part = f" | event={event_description!r}" if event_description else ""
        logger.info(
            "BayesToM drift applied%s: urgency_delta=%+.2f → %.2f, alt_prob=%.2f, conf=%.2f",
            desc_part, effect_on_urgency, self._urgency_mean, self._alt_prob, updated.confidence,
        )
        return updated

    def accuracy_against(self, hidden: HiddenState) -> float:
        """
        Compute current belief accuracy against true hidden state.
        Same formula as ToMTracker for comparability.
        """
        b = self.current_belief
        budget_range = max(hidden.budget_ceiling * 0.5, 1.0)
        walk_range   = max(hidden.walk_away_price * 0.5, 1.0)
        budget_err   = abs(b.est_budget - hidden.budget_ceiling) / budget_range
        walk_err     = abs(b.est_walk_away - hidden.walk_away_price) / walk_range
        urgency_err  = abs(b.est_urgency - hidden.urgency_score)
        alt_err      = 0.0 if b.est_has_alternative == hidden.has_alternative else 1.0
        mean_err = (budget_err + walk_err + urgency_err + alt_err) / 4.0
        return max(0.0, 1.0 - mean_err)

    def brier_scores(self, hidden: HiddenState) -> dict[str, float]:
        """Brier scores for urgency and has_alternative over full belief history."""
        if not self.history:
            return {"urgency": 1.0, "has_alt": 1.0}
        actual_urgency = hidden.urgency_score
        actual_alt = float(hidden.has_alternative)
        n = len(self.history)
        brier_urgency = sum((b.est_urgency - actual_urgency) ** 2 for b in self.history) / n
        brier_alt     = sum((float(b.est_has_alternative) - actual_alt) ** 2 for b in self.history) / n
        return {
            "urgency": round(brier_urgency, 6),
            "has_alt": round(brier_alt, 6),
        }
