#!/usr/bin/env python3
"""
Validate JSONL rows against what training/sft_train.py will actually use.
Read-only; does not run training.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Mirror training/sft_train._extract_completions (do not import to avoid heavy deps at import time)
def _extract_completions(rec: dict) -> list[str]:
    completion = rec.get("completion")
    if isinstance(completion, str) and completion.strip():
        return [completion.strip()]

    conversation = rec.get("conversation", [])
    candidates: list[str] = []
    if isinstance(conversation, list):
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).lower()
            content = str(turn.get("content", "")).strip()
            if role == "negotiator" and content:
                candidates.append(content)
    return candidates


def _approx_tokens(text: str) -> float:
    """Rough token estimate without tokenizer (good enough for preflight OOM risk)."""
    if not text:
        return 0.0
    return len(text) / 4.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SFT JSONL against sft_train.py expectations")
    parser.add_argument("--data", type=str, default="data/episodes.jsonl", help="Path to JSONL file")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.is_file():
        print(f"File not found: {path.resolve()}")
        print("Run: python -m training.generate_data --episodes 80 --output data/episodes.jsonl")
        return

    usable_rows = 0
    skipped = 0
    prompt_tok: list[float] = []
    completion_tok: list[float] = []
    first_bad_line: int | None = None

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if first_bad_line is None:
                    first_bad_line = line_no
                continue
            if not isinstance(rec, dict):
                skipped += 1
                continue

            prompt = str(rec.get("prompt", "")).strip()
            if not prompt:
                skipped += 1
                continue

            completions = _extract_completions(rec)
            if not completions:
                skipped += 1
                continue

            usable_rows += 1
            p_t = _approx_tokens(prompt)
            prompt_tok.append(p_t)
            for c in completions:
                completion_tok.append(_approx_tokens(c))

    sft_trains_one_row_per_completion = "sft_train.py expands one dataset row per negotiator line"
    print("SFT data validator (vs training/sft_train.py load_sft_dataset / _extract_completions)")
    print(f"  File: {path.resolve()}")
    print(f"  Note: {sft_trains_one_row_per_completion} when 'completion' is absent.")
    print()
    print(f"  JSONL records usable (has non-empty 'prompt' and completion or negotiator text): {usable_rows}")
    print(f"  Records / rows SKIPPED: {skipped}")
    if first_bad_line is not None:
        print(f"  (includes malformed JSONL starting around line {first_bad_line} if any)")
    print()

    if not prompt_tok and not completion_tok:
        print("No prompt/completion lengths to summarize (all skipped).")
    else:
        def _summary(vals: list[float], label: str) -> None:
            if not vals:
                print(f"  {label}: (empty)")
                return
            print(
                f"  {label} (approx. tokens, len/4): "
                f"min={min(vals):.1f}  max={max(vals):.1f}  mean={statistics.mean(vals):.1f}  "
                f"std={(statistics.pstdev(vals) if len(vals) > 1 else 0.0):.1f}"
            )

        _summary(prompt_tok, "Prompt length")
        _summary(completion_tok, "Completion length (each negotiator / completion string)")
        if prompt_tok and statistics.mean(prompt_tok) > 2048.0:
            print(
                "  FLAG: Mean prompt length > 2048 (approx. tokens) - may OOM or truncate with "
                "SFTConfig max_length=2048 in sft_train.py on small GPUs."
            )

    print()
    print(f"  {usable_rows} records usable for SFT, {skipped} will be skipped (at record level; negotiator")
    print("  expansion in sft_train can still multiply rows for usable records).")
    if usable_rows < 50:
        print("  WARNING: May be insufficient for SFT. Generate more data first.")

    if usable_rows == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
