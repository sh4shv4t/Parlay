"""
Reward computation and episode grading for Parlay.
Pure functions — no I/O, no side effects, no API calls.
Exports: compute_step_reward, compute_terminal_reward, grade_episode, EpisodeGrade.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from .models import BeliefState, HiddenState, ParlayAction, ParlayState
from .reward import ALPHA, BETA, DELTA, ETA, EPSILON, GAMMA, OMEGA, THETA, ZETA

logger = logging.getLogger(__name__)


@dataclass
class EpisodeGrade:
    """Summary metrics for a completed episode."""

    total_reward:       float
    deal_efficiency:    float          # (final_price - batna_seller) / zopa_width ∈ [0,1]
    tom_accuracy_avg:   float          # mean ToM accuracy across all turns ∈ [0,1]
    bluffs_caught:      int            # number of opponent bluffs the agent detected
    acts_completed:     int            # 1, 2, or 3
    termination_reason: Optional[str]
    drift_adapted:      bool           # True if adapted within 2 turns of a drift event


def _tom_accuracy(belief: BeliefState, hidden: HiddenState) -> float:
    """
    Compute Theory-of-Mind accuracy for a single turn.

    Returns a value in [0, 1]; 1.0 = perfect belief.
    """
    budget_range  = max(hidden.budget_ceiling  * 0.5, 1.0)
    walk_range    = max(hidden.walk_away_price * 0.5, 1.0)
    urgency_range = 1.0

    budget_err  = abs(belief.est_budget    - hidden.budget_ceiling)  / budget_range
    walk_err    = abs(belief.est_walk_away - hidden.walk_away_price) / walk_range
    urgency_err = abs(belief.est_urgency   - hidden.urgency_score)   / urgency_range
    alt_err     = 0.0 if belief.est_has_alternative == hidden.has_alternative else 1.0

    mean_err = (budget_err + walk_err + urgency_err + alt_err) / 4.0
    return max(0.0, 1.0 - mean_err)


def _cosine_similarity(a: str, b: str) -> float:
    """Simple bag-of-words cosine similarity for noise detection."""
    if not a or not b:
        return 0.0
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / (len(tokens_a) ** 0.5 * len(tokens_b) ** 0.5)


def compute_step_reward(
    state: ParlayState,
    action: ParlayAction,
    next_state: ParlayState,
) -> float:
    """
    Compute per-step reward R_t.

    Formula:
        R_t = α·ΔV_t + β·ToM_t − δ·C_t − θ·noise_t

    Where:
        ΔV_t  = max(0, move toward ZOPA midpoint / 1000)
        ToM_t = 1 − mean(|belief_i − true_i| / range_i)
        C_t   = max(0, (prev_offer − curr_offer) / prev_offer)
        noise_t = 1 if cosine_sim(utterance, history) < 0.3 else 0

    Args:
        state:      State before the action.
        action:     The action taken.
        next_state: State after the action.

    Returns:
        Scalar step reward.
    """
    # ΔV_t — offer improvement toward ZOPA midpoint
    delta_v = 0.0
    if action.offer_amount is not None and next_state.offer_history:
        prev_offer = state.offer_history[-1] if state.offer_history else action.offer_amount
        if action.offer_amount > prev_offer:
            delta_v = max(0.0, (action.offer_amount - prev_offer) / 1000.0)

    # ToM_t — belief accuracy
    tom_t = 0.0
    if next_state.belief_history:
        current_belief = next_state.belief_history[-1]
        tom_t = _tom_accuracy(current_belief, next_state.hidden_state)

    # C_t — concession rate penalty
    concession_t = 0.0
    if state.offer_history and action.offer_amount is not None:
        prev = state.offer_history[-1]
        if prev > 0:
            concession_t = max(0.0, (prev - action.offer_amount) / prev)

    # noise_t — ungrounded utterance penalty
    noise_t = 0.0
    if action.utterance and next_state.offer_history and len(next_state.offer_history) > 1:
        history_text = " ".join(str(o) for o in next_state.offer_history[:-1])
        sim = _cosine_similarity(action.utterance, history_text)
        noise_t = 1.0 if sim < 0.3 else 0.0

    reward = (
        ALPHA * delta_v
        + BETA * tom_t
        - DELTA * concession_t
        - THETA * noise_t
    )

    logger.debug(
        f"Step reward: total={reward:.3f} "
        f"(Δv={delta_v:.3f}, tom={tom_t:.3f}, "
        f"concession={concession_t:.3f}, noise={noise_t:.0f})"
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
    Compute terminal reward R_T.

    Formula:
        R_T = γ·E + ε·S + ζ·D + η·K − ω·𝟙[deal < BATNA_self]

    Where:
        E = (final_price − batna_seller) / zopa_width  ∈ [0,1]
        S = max(0, (T_max − T_close) / T_max)
        D = 1 if adapted within 2 turns of drift event
        K = number of acts completed

    Args:
        state:        Final episode state.
        final_price:  Agreed deal price (None if no deal).
        t_close:      Turn the deal closed (None if no deal).
        t_max:        Maximum allowed turns.
        drift_adapted: True if agent adapted within 2 turns of drift.

    Returns:
        Scalar terminal reward (can be negative for capitulation).
    """
    batna_seller = state.hidden_state.walk_away_price
    batna_buyer  = state.hidden_state.budget_ceiling
    zopa_width   = max(1.0, batna_buyer - batna_seller)

    if final_price is None:
        # No deal — partial credit for acts completed, partial penalty
        k_bonus = ETA * state.act
        return k_bonus - OMEGA * 0.5

    # Capitulation cliff — hard discontinuous
    if final_price < batna_seller:
        logger.warning(f"Capitulation detected: {final_price} < {batna_seller}")
        return -OMEGA

    # Deal efficiency E
    E = max(0.0, min(1.0, (final_price - batna_seller) / zopa_width))

    # Speed bonus S
    S = max(0.0, (t_max - (t_close or t_max)) / t_max) if t_close is not None else 0.0

    # Drift adaptation bonus D
    D = 1.0 if drift_adapted else 0.0

    # Act completion bonus K
    K = float(state.act)

    reward = GAMMA * E + EPSILON * S + ZETA * D + ETA * K
    logger.info(
        f"Terminal reward: {reward:.2f} "
        f"(E={E:.3f}, S={S:.3f}, D={D}, K={K})"
    )
    return reward


def grade_episode(
    session: ParlayState,
    final_price: Optional[float] = None,
    t_close: Optional[int] = None,
    t_max: int = 20,
    drift_adapted: bool = False,
    bluffs_caught: int = 0,
) -> EpisodeGrade:
    """
    Produce a full EpisodeGrade summary for a completed session.

    Args:
        session:       Final ParlayState.
        final_price:   Agreed price, or None if no deal.
        t_close:       Turn deal closed, or None.
        t_max:         Max turns allowed.
        drift_adapted: Whether agent adapted within 2 turns of drift.
        bluffs_caught: Number of opponent bluffs detected.

    Returns:
        EpisodeGrade with all summary metrics.
    """
    batna_seller = session.hidden_state.walk_away_price
    batna_buyer  = session.hidden_state.budget_ceiling
    zopa_width   = max(1.0, batna_buyer - batna_seller)

    # Deal efficiency
    if final_price is not None and final_price >= batna_seller:
        deal_efficiency = max(0.0, min(1.0, (final_price - batna_seller) / zopa_width))
    else:
        deal_efficiency = 0.0

    # Average ToM accuracy
    tom_scores = [
        _tom_accuracy(belief, session.hidden_state)
        for belief in session.belief_history
    ]
    tom_accuracy_avg = sum(tom_scores) / len(tom_scores) if tom_scores else 0.0

    # Total reward = cumulative step rewards + terminal reward
    terminal = compute_terminal_reward(session, final_price, t_close, t_max, drift_adapted)
    total_reward = session.cumulative_reward + terminal

    return EpisodeGrade(
        total_reward=total_reward,
        deal_efficiency=deal_efficiency,
        tom_accuracy_avg=tom_accuracy_avg,
        bluffs_caught=bluffs_caught,
        acts_completed=session.act,
        termination_reason=session.termination_reason,
        drift_adapted=drift_adapted,
    )
