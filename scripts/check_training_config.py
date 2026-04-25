#!/usr/bin/env python3
"""
Pre-flight training configuration checklist (SFT + GRPO).
Read-only: inspects training/sft_train.py and training/grpo_train.py; does not start training.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-flight SFT/GRPO training config checklist")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Project root (default: parent of scripts/)",
    )
    args = parser.parse_args()

    root = (args.repo_root or Path(__file__).resolve().parent.parent).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    sft_path = root / "training" / "sft_train.py"
    grpo_path = root / "training" / "grpo_train.py"
    if not sft_path.is_file() or not grpo_path.is_file():
        print(f"Missing {sft_path} or {grpo_path}")
        return

    import training.sft_train as sft
    import training.grpo_train as grpo

    sft_text = sft_path.read_text(encoding="utf-8")
    grpo_text = grpo_path.read_text(encoding="utf-8")
    sft_fn = inspect.getsource(sft.train_sft)

    checks: list[tuple[str, bool, str]] = []

    # Base model
    want_model = "Qwen/Qwen2.5-1.5B-Instruct"
    ok_model = sft.DEFAULT_MODEL == want_model
    checks.append(("[ ] Base model: Qwen/Qwen2.5-1.5B-Instruct", ok_model, f"found {sft.DEFAULT_MODEL!r}"))

    # LoRA
    ok_lora = "r=16" in sft_fn and "lora_alpha=32" in sft_fn
    checks.append(("[ ] SFT LoRA r=16, alpha=32", ok_lora, "in train_sft()"))

    # SFT training args
    ok_epochs = "num_train_epochs=3" in sft_text
    ok_b = "per_device_train_batch_size=4" in sft_text
    ok_g = "gradient_accumulation_steps=4" in sft_text
    eff = 4 * 4
    ok_sft = ok_epochs and ok_b and ok_g
    checks.append(
        (
            f"[ ] SFT epochs=3, batch=4, grad_accum=4 (effective ~{eff})",
            ok_sft,
            f"epochs={ok_epochs} batch={ok_b} grad={ok_g}",
        )
    )

    # Output dir
    want_out = "checkpoints/sft_1.5b/"
    ok_out = sft.DEFAULT_OUTPUT == want_out
    checks.append(("[ ] SFT output: checkpoints/sft_1.5b/", ok_out, f"default={sft.DEFAULT_OUTPUT!r}"))

    # GRPO BASE_MODEL (read at import time in grpo_train)
    base = os.getenv("BASE_MODEL", "")
    grpo_default = grpo.BASE_MODEL
    if not base:
        grpo_brief = f"BASE_MODEL env not set - will use module default {grpo_default!r}"
    else:
        grpo_brief = f"set to {base!r}"
    checks.append(
        (
            "[ ] GRPO reads BASE_MODEL from env",
            True,
            grpo_brief,
        )
    )

    # GRPO reward weights
    want_line = "reward_weights=[3.0, 1.5, 2.0, 0.5]"
    in_rw = want_line in grpo_text
    w_line = next((ln.strip() for ln in grpo_text.splitlines() if "reward_weights" in ln), "")
    checks.append(
        (
            "[ ] GRPO reward weights [efficiency, tom, anti-cap, format] = [3.0, 1.5, 2.0, 0.5]",
            in_rw,
            w_line or "not found",
        )
    )

    # GRPO data path
    d_ok = 'default="data/episodes.jsonl"' in grpo_text
    checks.append(
        (
            '[ ] GRPO --data default: data/episodes.jsonl',
            d_ok,
            "see grpo_train.main argparse" if d_ok else "check grpo_train.py",
        )
    )

    checks.append(
        (
            "[ ] Estimated VRAM note (1.5B + LoRA r=16 ~6-8GB SFT; more for GRPO)",
            True,
            "informational (not a failure if you skip the box)",
        )
    )

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ok_hf = bool(hf)
    checks.append(
        (
            "[ ] HF token for push (HF_TOKEN or HUGGING_FACE_HUB_TOKEN)",
            ok_hf,
            "set" if ok_hf else "not set - needed to push checkpoints",
        )
    )

    print("Training config pre-flight (read from training/sft_train.py, training/grpo_train.py)\n")
    for line, ok, note in checks:
        mark = "x" if ok else " "
        display = line.replace("[ ]", f"[{mark}]", 1) if line.startswith("[ ]") else line
        print(display)
        if note:
            print(f"  -> {note}")
    print()

    core_ok = ok_model and ok_lora and ok_sft and ok_out and in_rw and d_ok
    if core_ok and ok_hf:
        print("\nREADY FOR TRAINING (SFT + GRPO config strings match; HF token present for hub).")
    elif core_ok:
        print(
            "\nMOSTLY READY: fix missing HF token if you need push_to_hub; verify BASE_MODEL for GRPO stage."
        )
    else:
        print("\nNEEDS FIXING: see failed [ ] items above (model path, LoRA, SFT args, or output dir).")


if __name__ == "__main__":
    main()
