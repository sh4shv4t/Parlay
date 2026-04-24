"""
Episode quality filters and SFT label transforms for Parlay training data.

- Filters drop rows with broken offer fields or outlier rewards (post-collection).
- Label transforms clip rewards and log-scale deal_efficiency for stabler SFT targets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Fallback text from agent.gemini_client.SYNTHETIC_RESPONSE
_SYNTHETIC_UTTERANCE = "I need a moment to consider your proposal."


@dataclass
class SFTFilterConfig:
    """Thresholds for dropping episodes and for clipping values used in SFT text."""

    reward_drop_min: float = -400.0
    reward_drop_max: float = 400.0
    clip_reward_min: float = -200.0
    clip_reward_max: float = 200.0


def efficiency_sft_label(deal_efficiency: float | None) -> float:
    """
    Map deal_efficiency ∈ [0, 1] to a log-spread scalar in [0, 1] for SFT metadata.

    Emphasizes low-to-mid efficiency differences (stabler than raw for ranking).
    """
    e = 0.0 if deal_efficiency is None else float(deal_efficiency)
    e = max(0.0, min(1.0, e))
    return math.log1p(e) / math.log1p(1.0)


def clip_reward_for_label(reward: float | None, cfg: SFTFilterConfig) -> float:
    """Clip total episode reward for inclusion in SFT auxiliary metadata."""
    r = 0.0 if reward is None else float(reward)
    return max(cfg.clip_reward_min, min(cfg.clip_reward_max, r))


def _is_numeric_offer(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def episode_has_offer_field_failures(record: dict) -> bool:
    """
    True if negotiator/opponent turns are missing offer keys or look like API failures.

    Drops episodes where the model never produced a numeric price on the agent side,
    or where structured fields are absent.
    """
    conv = record.get("conversation") or []
    negotiator_turns = [t for t in conv if t.get("role") == "negotiator"]
    opponent_turns = [t for t in conv if t.get("role") == "opponent"]

    for t in negotiator_turns + opponent_turns:
        if "offer" not in t:
            return True
        utterance = (t.get("content") or "").strip()
        if utterance == _SYNTHETIC_UTTERANCE.strip():
            return True

    if not negotiator_turns:
        return True

    if not any(_is_numeric_offer(t.get("offer")) for t in negotiator_turns):
        return True

    if record.get("deal_reached"):
        if not any(_is_numeric_offer(t.get("offer")) for t in opponent_turns):
            return True

    return False


def episode_has_extreme_reward(record: dict, cfg: SFTFilterConfig) -> bool:
    """True if episode total reward should be excluded as an outlier."""
    r = record.get("reward")
    if r is None:
        return True
    r = float(r)
    return r < cfg.reward_drop_min or r > cfg.reward_drop_max


def episode_passes_sft_filters(
    record: dict,
    cfg: SFTFilterConfig,
) -> tuple[bool, str]:
    """
    Return (keep, reason_if_dropped).

    reason_if_dropped is "" when keep is True.
    """
    if episode_has_offer_field_failures(record):
        return False, "offer_field_failure"
    if episode_has_extreme_reward(record, cfg):
        return False, "extreme_reward"
    return True, ""


def filter_records(
    records: list[dict],
    cfg: SFTFilterConfig,
) -> tuple[list[dict], dict[str, Any]]:
    """Filter a list of episode dicts; return (kept, stats dict)."""
    kept: list[dict] = []
    drops: dict[str, int] = {}
    for rec in records:
        ok, reason = episode_passes_sft_filters(rec, cfg)
        if ok:
            kept.append(rec)
        else:
            drops[reason] = drops.get(reason, 0) + 1
    stats = {"total_in": len(records), "kept": len(kept), "dropped": drops}
    return kept, stats
