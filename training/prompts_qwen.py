"""
Qwen2.5-Instruct chat formatting for SFT/GRPO from JSONL (system + first user + assistant).

Falls back to the native im_start / im_end string layout if the tokenizer cannot be loaded.
"""
from __future__ import annotations

import os
from typing import Any

_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# When there is no user/negotiator turn, match training expectations for a JSON reply.
_SYNTHETIC_USER = (
    "Please make your opening offer or response. Reply in valid JSON format: "
    '{"utterance": "...", "offer_amount": <number or null>, "tactical_move": <string or null>}'
)

# Qwen2.5 chat template end-of-turn (same token as agent/hf_opponent.py)
_IM_END = str(bytes((60, 124, 105, 109, 95, 101, 110, 100, 124, 62)), "ascii")


def _first_user_content(conversation: list[Any] | None) -> str:
    """
    First human-side message: JSONL uses role 'user' or 'negotiator' (player turn).
    """
    if not isinstance(conversation, list):
        return _SYNTHETIC_USER
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        if role in ("user", "negotiator"):
            c = str(turn.get("content", "")).strip()
            if c:
                return c
    return _SYNTHETIC_USER


def _manual_grpo_prefix(system_msg: str, user_msg: str) -> str:
    """Qwen2.5 chat: prefix ending at assistant turn so the model writes JSON next."""
    return (
        f"<|im_start|>system\n{system_msg}{_IM_END}\n"
        f"<|im_start|>user\n{user_msg}{_IM_END}\n"
        f"<|im_start|>assistant\n"
    )


def _manual_sft_full(system_msg: str, user_msg: str, assistant_msg: str) -> str:
    return (
        f"<|im_start|>system\n{system_msg}{_IM_END}\n"
        f"<|im_start|>user\n{user_msg}{_IM_END}\n"
        f"<|im_start|>assistant\n{assistant_msg}{_IM_END}\n"
    )


def load_tokenizer_for_chat(model_id: str | None = None) -> Any | None:
    mid = (model_id or os.environ.get("BASE_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL).strip()
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    except Exception:
        return None


def format_grpo_prompt(
    rec: dict,
    tokenizer: Any | None = None,
) -> str:
    """
    System + first user/negotiator + open assistant: ready for generation (JSON policy output).
    """
    system_msg = str(rec.get("prompt", "")).strip()
    user_msg = _first_user_content(rec.get("conversation", []))
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return _manual_grpo_prefix(system_msg, user_msg)


def format_sft_text(
    rec: dict,
    assistant_content: str,
    tokenizer: Any | None = None,
) -> str:
    """Full supervised example: system, user, assistant (completion)."""
    system_msg = str(rec.get("prompt", "")).strip()
    user_msg = _first_user_content(rec.get("conversation", []))
    a = str(assistant_content).strip()
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": a},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return _manual_sft_full(system_msg, user_msg, a)
