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

# Scenarios where the AI plays as a BUYER (pushes offers DOWN).
# For these, ZOPA efficiency is measured from the buyer's side.
_BUYER_AI_SCENARIOS = frozenset({"hiring_package", "acquisition_term_sheet"})


def _clean_json(text: str) -> str:
    """Strip markdown code fences and surrounding whitespace."""
    return re.sub(r"```(?:json)?|```", "", text).strip()


def _kw_first(v, default=0.0) -> "float | str":
    """TRL GRPO may pass a scalar or a 1-item list of dataset fields; normalize for reward math."""
    if v is None:
        return default
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) else default
    return v


def _kw_float(v, default: float = 0.0) -> float:
    v = _kw_first(v, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _kw_str(v, default: str = "") -> str:
    v = _kw_first(v, default)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return str(default)
    return str(v)


def negotiation_efficiency_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Primary reward: fraction of ZOPA captured by the AI agent.

    For seller-AI scenarios (saas_enterprise):
        E = (offer - batna_seller) / zopa_width  ∈ [0, 1]
    For buyer-AI scenarios (hiring_package, acquisition_term_sheet):
        E = (batna_buyer - offer) / zopa_width   ∈ [0, 1]
    Returns value in [0, GAMMA] = [0, 100].

    Args:
        completions: List of G=8 model outputs (JSON strings).
        **kwargs:    Must contain batna_seller (float), batna_buyer (float),
                     zopa_width (float), and optionally scenario_id (str).

    Returns:
        List of float rewards, same length as completions.
    """
    rewards = []
    batna_seller = _kw_float(kwargs.get("batna_seller", 0), 0.0)
    batna_buyer = _kw_float(kwargs.get("batna_buyer", batna_seller), batna_seller)
    zopa_width = _kw_float(kwargs.get("zopa_width", 1), 1.0)
    scenario_id = _kw_str(kwargs.get("scenario_id", ""), "")
    is_buyer_ai  = scenario_id in _BUYER_AI_SCENARIOS

    if zopa_width <= 0:
        zopa_width = 1.0

    for completion in completions:
        try:
            data = json.loads(_clean_json(completion))
            offer = float(data.get("offer_amount") or 0)
            if offer > 0:
                if is_buyer_ai:
                    # AI is buyer: lower offers are better; score = (buyer_batna - offer) / width
                    E = max(0.0, min(1.0, (batna_buyer - offer) / zopa_width))
                else:
                    # AI is seller: higher offers are better; score = (offer - seller_batna) / width
                    E = max(0.0, min(1.0, (offer - batna_seller) / zopa_width))
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

    Signal lists are disjoint across personas (no double-counting).

    Args:
        completions: List of G=8 model outputs.
        **kwargs:    Must contain persona (str).

    Returns:
        List of float rewards in [0, 7.5].
    """
    persona = _kw_str(kwargs.get("persona", ""), "")
    tom_signals: dict[str, list[str]] = {
        "shark":    ["deadline", "competitor", "alternative", "pressure", "offer expires"],
        "diplomat": ["relationship", "partnership", "mutual", "together", "trust"],
        # "trust" removed from veteran to avoid double-counting with diplomat
        "veteran":  ["experience", "seen this", "long-term", "patience", "seasoned"],
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
    Hard penalty if the agent's offer crosses its own BATNA floor.

    For seller-AI: offer < batna_seller is capitulation.
    For buyer-AI:  offer > batna_buyer  is capitulation (paying too much).

    Returns -OMEGA (= -200) for capitulation, 0 otherwise.
    Parse errors are logged and treated as 0 (no false penalty for malformed output).

    Args:
        completions: List of G=8 model outputs.
        **kwargs:    Must contain batna_seller (float).
                     Optionally batna_buyer (float) and scenario_id (str).

    Returns:
        List of float rewards: -OMEGA or 0.
    """
    batna_seller = _kw_float(kwargs.get("batna_seller", 0.0), 0.0)
    batna_buyer = _kw_float(kwargs.get("batna_buyer", float("inf")), float("inf"))
    scenario_id = _kw_str(kwargs.get("scenario_id", ""), "")
    is_buyer_ai  = scenario_id in _BUYER_AI_SCENARIOS

    rewards = []
    for completion in completions:
        try:
            data = json.loads(_clean_json(completion))
            offer = float(data.get("offer_amount") or float("inf"))
            if is_buyer_ai:
                capitulated = offer > batna_buyer
            else:
                capitulated = offer < batna_seller
            if capitulated:
                rewards.append(-float(OMEGA))
                logger.debug(
                    f"Capitulation: offer={offer} {'>' if is_buyer_ai else '<'} "
                    f"batna={'buyer=' + str(batna_buyer) if is_buyer_ai else 'seller=' + str(batna_seller)}"
                )
            else:
                rewards.append(0.0)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"anti_capitulation_reward parse error (treated as 0): {exc}")
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
