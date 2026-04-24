"""
Reward computation and episode grading for Parlay.
Pure functions — no I/O, no side effects, no API calls.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from .models import BeliefState, HiddenState, ParlayAction, ParlayState
from .reward import ALPHA, BETA, DELTA, EPSILON, GAMMA, OMEGA, PSI, THETA, ZETA

logger = logging.getLogger(__name__)

SKEPTICISM_PHRASES = [
    "i don't believe",
    "that seems",
    "i'm skeptical",
    "hard to accept",
    "walk away",
    "bluffing",
    "not credible",
    "prove it",
    "show me",
    "really your",
    "actually your",
    "doubt that",
    "questionable",
    "i question",
    "that can't be",
    "unrealistic",
    "implausible",
]


@dataclass
class EpisodeGrade:
    """Summary metrics for a completed episode."""

    total_reward: float
    deal_efficiency: float
    tom_accuracy_avg: float
    bluffs_caught: int
    termination_reason: Optional[str]
    drift_adapted: bool


def _tom_accuracy(belief: BeliefState, hidden: HiddenState) -> float:
    budget_range = max(hidden.budget_ceiling * 0.5, 1.0)
    walk_range = max(hidden.walk_away_price * 0.5, 1.0)

    budget_err = abs(belief.est_budget - hidden.budget_ceiling) / budget_range
    walk_err = abs(belief.est_walk_away - hidden.walk_away_price) / walk_range
    urgency_err = abs(belief.est_urgency - hidden.urgency_score)
    alt_err = 0.0 if belief.est_has_alternative == hidden.has_alternative else 1.0

    mean_err = (budget_err + walk_err + urgency_err + alt_err) / 4.0
    return max(0.0, 1.0 - mean_err)


def _cosine_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / (len(tokens_a) ** 0.5 * len(tokens_b) ** 0.5)


def detect_bluff_challenge(
    utterance: str,
    opponent_stated_batna: float,
    opponent_true_batna: float,
) -> bool:
    """
    Return True when the utterance challenges a BATNA bluff.
    """
    if opponent_stated_batna is None or opponent_true_batna is None:
        return False

    bluff_threshold = abs(opponent_true_batna) * 0.15
    is_bluffing = abs(opponent_stated_batna - opponent_true_batna) > bluff_threshold
    if not is_bluffing:
        return False

    utterance_lower = utterance.lower()
    return any(phrase in utterance_lower for phrase in SKEPTICISM_PHRASES)


def compute_step_reward(
    state: ParlayState,
    action: ParlayAction,
    next_state: ParlayState,
) -> float:
    """
    Compute per-step reward R_t.
    """
    delta_v = 0.0
    if action.offer_amount is not None:
        prev_offer = state.offer_history[-1] if state.offer_history else action.offer_amount
        if action.offer_amount > prev_offer:
            delta_v = max(0.0, (action.offer_amount - prev_offer) / 1000.0)

    tom_t = 0.0
    if next_state.belief_history:
        tom_t = _tom_accuracy(next_state.belief_history[-1], next_state.hidden_state)

    concession_t = 0.0
    if state.offer_history and action.offer_amount is not None:
        prev_offer = state.offer_history[-1]
        if prev_offer > 0:
            concession_t = max(0.0, (prev_offer - action.offer_amount) / prev_offer)

    noise_t = 0.0
    if action.utterance and len(next_state.offer_history) > 1:
        history_text = " ".join(str(offer) for offer in next_state.offer_history[:-1])
        noise_t = 1.0 if _cosine_similarity(action.utterance, history_text) < 0.3 else 0.0

    bluff_bonus = 0.0
    if (
        action.tactical_move is None
        and state.hidden_state.last_stated_batna is not None
        and detect_bluff_challenge(
            utterance=action.utterance,
            opponent_stated_batna=state.hidden_state.last_stated_batna,
            opponent_true_batna=state.hidden_state.budget_ceiling,
        )
    ):
        bluff_bonus = PSI

    reward = (
        ALPHA * delta_v
        + BETA * tom_t
        - DELTA * concession_t
        - THETA * noise_t
        + bluff_bonus
    )
    logger.debug(
        "Step reward: total=%.3f (dv=%.3f, tom=%.3f, concession=%.3f, noise=%.0f, bluff=%.3f)",
        reward,
        delta_v,
        tom_t,
        concession_t,
        noise_t,
        bluff_bonus,
    )
    return reward


def compute_terminal_reward(
    state: ParlayState,
    final_price: Optional[float] = None,
    t_close: Optional[int] = None,
    t_max: int = 20,
    drift_adapted: bool = False,
) -> float:
    """
    Compute terminal reward:
        R_T = γ·E + ε·S + ζ·D − ω·1[deal_price < BATNA_self]
    """
    batna_seller = state.hidden_state.walk_away_price
    batna_buyer = state.hidden_state.budget_ceiling
    zopa_width = max(1.0, batna_buyer - batna_seller)

    if final_price is None:
        return -OMEGA * 0.5

    if final_price < batna_seller:
        logger.warning("Capitulation detected: %.2f < %.2f", final_price, batna_seller)
        return -OMEGA

    deal_efficiency = max(0.0, min(1.0, (final_price - batna_seller) / zopa_width))
    speed_bonus = max(0.0, (t_max - (t_close or t_max)) / t_max) if t_close is not None else 0.0
    drift_bonus = 1.0 if drift_adapted else 0.0

    reward = GAMMA * deal_efficiency + EPSILON * speed_bonus + ZETA * drift_bonus
    logger.info(
        "Terminal reward: %.2f (E=%.3f, S=%.3f, D=%.0f)",
        reward,
        deal_efficiency,
        speed_bonus,
        drift_bonus,
    )
    return reward


def grade_episode(
    session: ParlayState,
    final_price: Optional[float] = None,
    t_close: Optional[int] = None,
    t_max: int = 20,
    drift_adapted: bool = False,
    bluffs_caught: Optional[int] = None,
) -> EpisodeGrade:
    """
    Produce a full EpisodeGrade summary for a completed session.
    """
    batna_seller = session.hidden_state.walk_away_price
    batna_buyer = session.hidden_state.budget_ceiling
    zopa_width = max(1.0, batna_buyer - batna_seller)

    if final_price is not None and final_price >= batna_seller:
        deal_efficiency = max(0.0, min(1.0, (final_price - batna_seller) / zopa_width))
    else:
        deal_efficiency = 0.0

    tom_scores = [_tom_accuracy(belief, session.hidden_state) for belief in session.belief_history]
    tom_accuracy_avg = sum(tom_scores) / len(tom_scores) if tom_scores else 0.0
    terminal = compute_terminal_reward(session, final_price, t_close, t_max, drift_adapted)

    return EpisodeGrade(
        total_reward=session.cumulative_reward + terminal,
        deal_efficiency=deal_efficiency,
        tom_accuracy_avg=tom_accuracy_avg,
        bluffs_caught=session.bluffs_caught if bluffs_caught is None else bluffs_caught,
        termination_reason=session.termination_reason,
        drift_adapted=drift_adapted,
    )
