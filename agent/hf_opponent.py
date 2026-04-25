"""
Optional Hugging Face causal-LM path for the dashboard opponent.
Used when OPPONENT_MODE=trained and HF_MODEL_REPO is set.
Falls back to Gemini if this module fails to load or generate.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_lock = "hf_opponent_lock"
_model = None
_tokenizer = None
_loaded_repo: str | None = None


def _get_lock():
    import threading

    g = globals()
    if g[_lock] == "hf_opponent_lock":
        g[_lock] = threading.Lock()
    return g[_lock]


def _parse_json_block(text: str) -> dict[str, Any]:
    text = (text or "").replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"utterance": text[:300] or "…", "offer_amount": None, "tactical_move": None}
    try:
        out = json.loads(m.group(0))
        if not isinstance(out, dict):
            raise ValueError("not a dict")
        out.setdefault("utterance", m.group(0)[:200])
        out.setdefault("offer_amount", None)
        out.setdefault("tactical_move", None)
        return out
    except Exception:  # noqa: BLE001
        return {"utterance": text[:300], "offer_amount": None, "tactical_move": None}


def _load(repo_id: str) -> None:
    global _model, _tokenizer, _loaded_repo
    with _get_lock():
        if _loaded_repo == repo_id and _model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from pathlib import Path
        from huggingface_hub import hf_hub_download

        tok = AutoTokenizer.from_pretrained(
            repo_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN", "")
        )
        is_adapter = False
        try:
            hf_hub_download(repo_id=repo_id, filename="adapter_config.json")
            is_adapter = True
        except Exception:
            is_adapter = False

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if is_adapter:
            from peft import PeftModel
            from huggingface_hub import hf_hub_download
            import json as _j

            cfg_p = Path(hf_hub_download(repo_id=repo_id, filename="adapter_config.json"))
            base = _j.loads(cfg_p.read_text(encoding="utf-8")).get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
            base_m = AutoModelForCausalLM.from_pretrained(
                base,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN", "") or None,
            )
            _model = PeftModel.from_pretrained(
                base_m, repo_id, token=os.environ.get("HF_TOKEN", "") or None
            )
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN", "") or None,
            )
        _tokenizer = tok
        _loaded_repo = repo_id
        logger.info("HF opponent model loaded: %s (adapter=%s)", repo_id, is_adapter)


def _build_prompt(system_prompt: str, messages: list[dict]) -> str:
    """Qwen2.5 Instruct / ChatML-style string for causal LM."""
    eot = str(bytes((60, 124, 105, 109, 95, 101, 110, 100, 124, 62)), "ascii")
    parts: list[str] = [f"<|im_start|>system\n{system_prompt}\n{eot}\n"]
    for m in messages:
        role = m.get("role", "user")
        r = "user" if role == "user" else "assistant"
        part_text = m.get("parts", [""])[0] if m.get("parts") else ""
        parts.append(f"<|im_start|>{r}\n{part_text}\n{eot}\n")
    parts.append(
        "<|im_start|>assistant\n"
        "Respond ONLY with valid JSON: "
        '{"utterance": "...", "offer_amount": <number or null>, "tactical_move": <string or null>}\n'
    )
    return "".join(parts)


def _sync_generate(
    system_prompt: str, messages: list[dict], max_new_tokens: int
) -> dict[str, Any]:
    import torch

    repo = (os.environ.get("HF_MODEL_REPO") or "").strip()
    if not repo:
        raise RuntimeError("HF_MODEL_REPO not set")

    _load(repo)
    assert _tokenizer is not None and _model is not None

    prompt = _build_prompt(system_prompt, messages)
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    dev = next(_model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _parse_json_block(text)


async def call_hf_opponent(
    system_prompt: str,
    messages: list[dict],
    *,
    max_tokens: int = 500,
    persona: str = "shark",
    scenario_id: str | None = None,
) -> dict[str, Any]:
    if not (os.environ.get("HF_MODEL_REPO") or "").strip():
        raise RuntimeError("HF_MODEL_REPO not set")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _sync_generate(system_prompt, messages, min(max_tokens, 512)),
    )
