"""
Reward functions for GRPOTrainer.
Signature: fn(completions: list[str], **kwargs) -> list[float]
All functions wrap parlay_env/grader.py logic.
"""
import json
import re
import logging
from parlay_env.reward import GAMMA, OMEGA

logger = logging.getLogger(__name__)


def _clean_json(text: str) -> str:
    """Strip markdown code fences and surrounding whitespace."""
    return re.sub(r"```(?:json)?|```", "", text).strip()


def negotiation_efficiency_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Primary reward: fraction of ZOPA captured.

    Parses offer_amount from each completion JSON.
    E = (offer - batna_seller) / zopa_width  ∈ [0, 1]
    Returns value in [0, GAMMA] = [0, 100].

    Args:
        completions: List of G=8 model outputs (JSON strings).
        **kwargs:    Must contain batna_seller (float) and zopa_width (float).

    Returns:
        List of float rewards, same length as completions.
    """
    rewards = []
    batna = float(kwargs.get("batna_seller", 0))
    zopa_width = float(kwargs.get("zopa_width", 1))
    if zopa_width <= 0:
        zopa_width = 1.0

    for completion in completions:
        try:
            data = json.loads(_clean_json(completion))
            offer = float(data.get("offer_amount") or 0)
            if offer > 0:
                E = max(0.0, min(1.0, (offer - batna) / zopa_width))
                rewards.append(float(E * GAMMA))
            else:
                rewards.append(0.0)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.debug(f"negotiation_efficiency_reward parse error: {exc}")
            rewards.append(0.0)
    return rewards


def tom_accuracy_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Theory-of-Mind reward: utterance reflects accurate opponent beliefs.

    Uses keyword matching against persona-specific signals as a lightweight proxy.
    Full accuracy computed by grader.py; this is used for fast training feedback.

    Args:
        completions: List of G=8 model outputs.
        **kwargs:    Must contain persona (str).

    Returns:
        List of float rewards in [0, 7.5].
    """
    persona = str(kwargs.get("persona", ""))
    tom_signals: dict[str, list[str]] = {
        "shark":    ["deadline", "competitor", "alternative", "pressure", "offer expires"],
        "diplomat": ["relationship", "partnership", "mutual", "together", "trust"],
        "analyst":  ["data", "evidence", "justif", "metric", "roi", "benchmark"],
        "wildcard": ["feel", "sense", "flexible", "open to", "spontan"],
        "veteran":  ["experience", "seen this", "long-term", "trust", "patience"],
    }
    signals = tom_signals.get(persona.lower(), [])
    rewards = []
    for completion in completions:
        text = completion.lower()
        signal_count = sum(1 for s in signals if s in text)
        rewards.append(float(min(7.5, signal_count * 2.5)))
    return rewards


def anti_capitulation_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Hard penalty if the agent's offer falls below its own BATNA.

    The agent plays as the SELLER. The seller's own walk-away price is
    batna_seller — the minimum price the agent will accept. Any offer
    below batna_seller means the agent is capitulating below its floor.

    Returns -OMEGA (= -200) for capitulation, 0 otherwise.
    This is a hard cliff — no smoothing.

    Args:
        completions: List of G=8 model outputs.
        **kwargs:    Must contain batna_seller (float) — the seller-agent's
                     own walk-away price (minimum acceptable price).

    Returns:
        List of float rewards: -OMEGA or 0.
    """
    batna_self = float(kwargs.get("batna_seller", 0.0))
    rewards = []
    for completion in completions:
        try:
            data = json.loads(_clean_json(completion))
            offer = float(data.get("offer_amount") or float("inf"))
            if offer < batna_self:
                rewards.append(-float(OMEGA))
                logger.debug(f"Capitulation detected: offer={offer} < batna={batna_self}")
            else:
                rewards.append(0.0)
        except (json.JSONDecodeError, ValueError):
            rewards.append(0.0)
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Structural reward: encourages valid JSON output with required fields.

    +2.0 for valid JSON with non-empty utterance field.
    +0.5 for valid JSON but missing utterance.
    -1.0 for invalid JSON.

    Args:
        completions: List of G=8 model outputs.

    Returns:
        List of float rewards.
    """
    rewards = []
    for completion in completions:
        try:
            data = json.loads(_clean_json(completion))
            has_utterance = bool(str(data.get("utterance", "")).strip())
            rewards.append(2.0 if has_utterance else 0.5)
        except json.JSONDecodeError:
            rewards.append(-1.0)
    return rewards
