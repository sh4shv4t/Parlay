#!/usr/bin/env python3
"""
Pre-training data quality inspector for Parlay JSONL episode files.
Read-only: loads JSONL and prints statistics and RED FLAGS.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _percentile_sorted(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _outcome_bucket(rec: dict[str, Any]) -> str:
    tr = (rec.get("termination_reason") or "") or ""
    tr_l = tr.lower()
    if rec.get("deal_reached") is True or tr_l in ("deal_reached", "deal", "agreement"):
        return "deal"
    if "zopa_collapsed" in tr_l or tr_l == "zopa_collapsed":
        return "zopa_collapsed"
    if "walk" in tr_l or tr_l in ("walk_away", "walkaway"):
        return "walk_away"
    if tr_l in ("max_turns",) or (rec.get("deal_reached") is False and "max" in tr_l):
        return "max_turns"
    if tr_l:
        return f"other:{tr_l}"
    if rec.get("deal_reached") is False and rec.get("final_price") is None:
        return "no_deal_or_unknown"
    return "unknown"


def _utterance_lengths(conversation: Any) -> list[int]:
    if not isinstance(conversation, list):
        return []
    out: list[int] = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        content = turn.get("content", "")
        if isinstance(content, str) and content.strip():
            out.append(len(content))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Parlay episode JSONL data quality")
    parser.add_argument("--data", type=str, default="data/episodes.jsonl", help="Path to JSONL file")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.is_file():
        print(f"File not found: {path.resolve()}")
        print("Run: python -m training.generate_data --episodes 80 --output data/episodes.jsonl")
        return

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping line {line_no}: invalid JSON ({e})")

    n = len(records)
    print(f"=== Parlay data inspector: {path} ===")
    print(f"Total episode records: {n}\n")

    if n == 0:
        print("No records to analyze.")
        return

    # SCHEMA
    missing_prompt = sum(1 for r in records if not str(r.get("prompt", "")).strip())
    missing_scenario = sum(1 for r in records if not str(r.get("scenario_id", "")).strip())
    missing_persona = sum(1 for r in records if not str(r.get("persona", "")).strip())
    missing_metadata = sum(1 for r in records if "metadata" not in r)
    print("SCHEMA")
    print(f"  prompt present:        {n - missing_prompt}/{n}")
    print(f"  scenario_id present:  {n - missing_scenario}/{n}")
    print(f"  persona present:      {n - missing_persona}/{n}")
    print(f"  metadata key present:  {n - missing_metadata}/{n}  (audit checklist; generate_data may omit)")
    print()

    def cum_reward(r: dict[str, Any]) -> float:
        if "cumulative_reward" in r:
            return _safe_float(r.get("cumulative_reward"))
        return _safe_float(r.get("reward"))

    rews = [cum_reward(r) for r in records]
    rews_sorted = sorted(rews)

    print("REWARD (total / cumulative - field 'reward' or 'cumulative_reward')")
    print(f"  min:    {min(rews):.4f}")
    print(f"  max:    {max(rews):.4f}")
    print(f"  mean:   {statistics.mean(rews):.4f}")
    print(f"  std:    {statistics.stdev(rews) if len(rews) > 1 else 0.0:.4f}")
    print(f"  p10:    {_percentile_sorted(rews_sorted, 10):.4f}")
    print(f"  p90:    {_percentile_sorted(rews_sorted, 90):.4f}")
    print()

    outcomes = [_outcome_bucket(r) for r in records]
    oc = Counter(outcomes)
    print("EPISODE OUTCOMES (best-effort from termination_reason + deal_reached)")
    for k, v in sorted(oc.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({100.0 * v / n:.1f}%)")
    print()

    effs = [_safe_float(r.get("deal_efficiency"), 0.0) for r in records]
    toms = []
    for r in records:
        t = r.get("tom_accuracy_avg", r.get("tom_accuracy"))
        toms.append(_safe_float(t, 0.0))

    print("EFFICIENCY (deal_efficiency, 0-1)")
    if effs:
        print(f"  mean: {statistics.mean(effs):.4f}  min: {min(effs):.4f}  max: {max(effs):.4f}")
    print()

    print("TOM (tom_accuracy_avg or tom_accuracy)")
    if toms:
        print(f"  mean: {statistics.mean(toms):.4f}  min: {min(toms):.4f}  max: {max(toms):.4f}")
    print()

    all_lens: list[int] = []
    degenerate_turns = 0
    total_turns = 0
    for r in records:
        lens = _utterance_lengths(r.get("conversation"))
        all_lens.extend(lens)
        for L in lens:
            total_turns += 1
            if L < 10:
                degenerate_turns += 1

    print("UTTERANCE LENGTH (conversation[*].content)")
    if all_lens:
        print(f"  mean chars/turn: {statistics.mean(all_lens):.1f}")
        print(f"  turns < 10 chars: {degenerate_turns}/{total_turns} ({100.0 * degenerate_turns / max(1, total_turns):.1f}%)")
    else:
        print("  (no conversation utterances found)")
    print()

    bluff_pos = sum(1 for r in records if int(r.get("bluffs_caught", 0) or 0) > 0)
    drift_yes = sum(1 for r in records if r.get("drift_adapted") is True)

    print("BLUFF RATE: episodes with bluffs_caught > 0")
    print(f"  {bluff_pos}/{n} ({100.0 * bluff_pos / n:.1f}%)  (field may be missing in JSONL -> counted as 0)")
    print()
    print("DRIFT ADAPTATION: drift_adapted == True")
    print(f"  {drift_yes}/{n} ({100.0 * drift_yes / n:.1f}%)")
    print()

    by_persona: dict[str, list[dict]] = defaultdict(list)
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        p = str(r.get("persona", "??"))
        s = str(r.get("scenario_id", "??"))
        by_persona[p].append(r)
        by_scenario[s].append(r)

    print("=== PER-PERSONA ===")
    for p in sorted(by_persona.keys()):
        grp = by_persona[p]
        m = len(grp)
        pr = [cum_reward(x) for x in grp]
        pe = [_safe_float(x.get("deal_efficiency"), 0) for x in grp]
        pt = [_safe_float(x.get("tom_accuracy_avg", x.get("tom_accuracy")), 0) for x in grp]
        po = [_outcome_bucket(x) for x in grp]
        dr = sum(1 for f in po if f == "deal") / m
        print(
            f"  {p}: n={m}  mean_reward={statistics.mean(pr) if pr else 0:.2f}  "
            f"mean_eff={statistics.mean(pe) if pe else 0:.3f}  mean_tom={statistics.mean(pt) if pt else 0:.3f}  deal_rate={dr:.2%}"
        )
    print()

    print("=== PER-SCENARIO ===")
    for s in sorted(by_scenario.keys()):
        grp = by_scenario[s]
        m = len(grp)
        pr = [cum_reward(x) for x in grp]
        pe = [_safe_float(x.get("deal_efficiency"), 0) for x in grp]
        pt = [_safe_float(x.get("tom_accuracy_avg", x.get("tom_accuracy")), 0) for x in grp]
        po = [_outcome_bucket(x) for x in grp]
        dr = sum(1 for f in po if f == "deal") / m
        print(
            f"  {s}: n={m}  mean_reward={statistics.mean(pr) if pr else 0:.2f}  "
            f"mean_eff={statistics.mean(pe) if pe else 0:.3f}  mean_tom={statistics.mean(pt) if pt else 0:.3f}  deal_rate={dr:.2%}"
        )
    print()

    # RED FLAGS
    print("=== RED FLAGS ===")
    flags: list[str] = []

    bad_rew = sum(1 for x in rews if x < -50) / n
    if bad_rew > 0.30:
        flags.append(f"> 30% episodes with total reward < -50 ({100 * bad_rew:.1f}%)")

    max_turns_rate = sum(1 for o in outcomes if o == "max_turns") / n
    if max_turns_rate > 0.40:
        flags.append(f"> 40% ending in max_turns ({100 * max_turns_rate:.1f}%)")

    drift_rate = drift_yes / n
    if drift_rate < 0.10:
        flags.append(f"< 10% drift_adapted ({100 * drift_rate:.1f}%)")

    for p, grp in by_persona.items():
        po = [_outcome_bucket(x) for x in grp]
        dr = sum(1 for f in po if f == "deal") / len(grp) if grp else 0.0
        if dr == 0.0 and len(grp) >= 3:
            flags.append(f"Persona {p!r} has 0% deal rate (n={len(grp)})")

    for s, grp in by_scenario.items():
        po = [_outcome_bucket(x) for x in grp]
        dr = sum(1 for f in po if f == "deal") / len(grp) if grp else 0.0
        if dr == 0.0 and len(grp) >= 3:
            flags.append(f"Scenario {s!r} has 0% deal rate (n={len(grp)})")

    if all_lens and statistics.mean(all_lens) < 20.0:
        flags.append(f"Mean utterance length {statistics.mean(all_lens):.1f} chars < 20 (possibly degenerate)")

    if max(rews) > 400:
        flags.append(f"At least one episode with total reward > 400 (max={max(rews):.2f}) - check for scale bugs or rare combo")

    if missing_metadata == n:
        flags.append("No record has a top-level 'metadata' key (optional for training; audit asked for it)")

    if not flags:
        print("  (none triggered)")
    else:
        for f in flags:
            print(f"  * {f}")
    print()


if __name__ == "__main__":
    main()
