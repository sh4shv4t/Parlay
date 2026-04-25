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


# Patterns like "offer_amount": 50000 or "offer": 50000 in partial / non-JSON text
_OFFER_AMOUNT_RE = re.compile(r'["\']offer_amount["\']\s*:\s*([0-9eE+.-]+)', re.IGNORECASE)
_OFFER_RE = re.compile(r'["\']offer["\']\s*:\s*([0-9eE+.-]+)', re.IGNORECASE)


def _parse_offer_anti_capitulation(completion: str) -> float | None:
    try:
        data = json.loads(completion)
        if not isinstance(data, dict):
            return None
        oa = data.get("offer_amount", None)
        if oa is None:
            return None
        return float(oa)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    m = re.search(r'"offer_amount"\s*:\s*([\d.]+)', completion)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    text = _clean_json(completion)
    m2 = _OFFER_AMOUNT_RE.search(text) or _OFFER_RE.search(text)
    if m2:
        try:
            return float(m2.group(1))
        except ValueError:
            return None
    return None


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
    Unparseable output yields 0.0; missing-offer details are logged at DEBUG only.

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
        offer = _parse_offer_anti_capitulation(completion)
        if offer is None:
            logger.debug("anti_capitulation_reward: no offer parsed, reward=0.0")
            rewards.append(0.0)
            continue
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
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Structural reward: encourages valid JSON output with a smooth gradient
    (not a cliff for invalid output).

    Returns:
        1.0 if valid JSON with both ``utterance`` and ``offer_amount`` keys;
        0.5 if valid JSON with at least ``utterance`` key;
        0.3 if not valid JSON but the string contains the word ``utterance``;
        0.0 otherwise.
    """
    rewards = []
    word_utterance = re.compile(r"(?<![\w])utterance(?![\w])", re.IGNORECASE)

    for completion in completions:
        try:
            data = json.loads(_clean_json(completion))
        except json.JSONDecodeError:
            rewards.append(0.3 if word_utterance.search(completion) else 0.0)
            continue
        if not isinstance(data, dict):
            rewards.append(0.3 if word_utterance.search(completion) else 0.0)
            continue
        has_u = "utterance" in data
        has_oa = "offer_amount" in data
        if has_u and has_oa:
            rewards.append(1.0)
        elif has_u:
            rewards.append(0.5)
        else:
            rewards.append(0.3 if word_utterance.search(completion) else 0.0)
    return rewards
