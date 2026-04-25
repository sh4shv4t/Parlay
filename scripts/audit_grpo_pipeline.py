#!/usr/bin/env python3
"""
Smoke test for ParlayGRPOEnvWrapper against one JSONL prompt (keyless / mock path).
Read-only: does not modify training or env.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


class _StubTrainer:
    """Minimal object satisfying ParlayGRPOEnvWrapper's trainer attribute interface."""

    def train(self) -> None:
        return None

    def save_model(self, _out: str) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO env wrapper smoke test (reset + play_turn, JSON completion handling)"
    )
    parser.add_argument("--data", type=str, default="data/episodes.jsonl", help="Path to JSONL (first row)")
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

    from training.grpo_env_wrapper import ParlayGRPOEnvWrapper

    path = Path(args.data)
    if not path.is_file():
        print(f"File not found: {path.resolve()}")
        print("Run: python -m training.generate_data --episodes 80 --output data/episodes.jsonl")
        return

    with path.open("r", encoding="utf-8") as f:
        first = next((ln for ln in f if ln.strip()), None)
    if not first:
        print("Empty JSONL.")
        return
    try:
        row = json.loads(first.strip())
    except json.JSONDecodeError as e:
        print(f"First row is not valid JSON: {e}")
        return

    scenario_id = str(row.get("scenario_id") or "saas_enterprise")
    persona = str(row.get("persona") or "diplomat")

    wrapper = ParlayGRPOEnvWrapper(_StubTrainer())
    print("ParlayGRPOEnvWrapper smoke test")
    print(f"  JSONL: {path.resolve()}")
    print(f"  Using scenario_id={scenario_id!r} persona={persona!r} from first row (defaults if missing)")

    entries: list[tuple[str, str, str]] = []

    # 1) reset
    try:
        obs = wrapper.reset(scenario_id=scenario_id, persona=persona, seed=42)
        entries.append(
            (
                "reset() completes",
                "PASS",
                f"ok; scenario_id in obs: {obs.get('scenario_id')!r}",
            )
        )
    except Exception:
        entries.append(("reset() completes", "FAIL", traceback.format_exc()[:500]))
        _print_checks(entries)
        return

    # 2) play_turn with valid parsed completion
    sample_json = '{"utterance": "I propose 50000", "offer_amount": 50000}'
    try:
        action = json.loads(sample_json)
        out = wrapper.play_turn(action)
        reward = float(out.get("reward", 0.0))
    except Exception:
        entries.append(
            (
                "play_turn(valid JSON → dict with offer)",
                "FAIL",
                traceback.format_exc()[:500],
            )
        )
        _print_checks(entries)
        return

    print(f"  Sample model completion: {sample_json}")
    print(f"  play_turn reward (wrapper): {reward}")
    print(
        "  Note: play_turn() returns result.grade.total_reward when offer is set (full episode total), not "
        "the GRPO weighted reward_fn. GRPO training uses training/reward_fn.py on generated strings."
    )

    lo, hi = -10.0, 50.0
    in_range = lo <= reward <= hi
    entries.append(
        (
            f"Reward in [{lo}, {hi}] (heuristic single-step window)",
            "PASS" if in_range else "FAIL",
            (
                f"reward={reward} inside range"
                if in_range
                else f"reward={reward} - expected often OUTSIDE range: wrapper total_reward can be large"
            ),
        )
    )

    # 3) Malformed JSON: must not be passed to play_turn as a string from a correct pipeline
    bad = '{"utterance": "hello"'
    try:
        json.loads(bad)
        par_mal = "UNEXPECTED: bad JSON parsed"
    except json.JSONDecodeError:
        err_line = None
        try:
            wrapper.play_turn(bad)  # type: ignore[arg-type]
        except Exception as e:
            err_line = f"json.loads fails; play_turn(str) -> {type(e).__name__}: {e!s}"[:200]
        par_mal = err_line or "play_turn(str) did not raise"
    entries.append(
        (
            "Malformed JSON string mishandled at play_turn",
            "FAIL" if par_mal.startswith("UNEXPECTED") else "PASS",
            "Correct pipeline: json.loads first; " + par_mal,
        )
    )

    # 4) Empty string
    empty_explain = []
    try:
        json.loads("")
    except json.JSONDecodeError as e0:
        empty_explain.append(f"json.loads('') -> {e0!s}"[:100])
    try:
        wrapper.play_turn("")
    except Exception as e1:
        empty_explain.append(f"play_turn('') -> {type(e1).__name__}")
    else:
        empty_explain.append("play_turn('') did not raise (unexpected)")
    entries.append(
        (
            "Empty string completion / action",
            "PASS" if "did not raise" not in str(empty_explain[-1]) else "FAIL",
            " | ".join(empty_explain),
        )
    )

    _print_checks(entries)


def _print_checks(rows: list[tuple[str, str, str]]) -> None:
    print()
    print("CHECKS")
    for name, status, detail in rows:
        print(f"  [{status}] {name}")
        for line in detail.split("\n"):
            print(f"        {line}")


if __name__ == "__main__":
    main()
