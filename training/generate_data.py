"""Generate quality-filtered negotiation episodes via Gemini self-play."""
import argparse
import asyncio
import json
import logging
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Repo root on path when run as `python training/generate_data.py` (script dir is training/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from agent.gemini_client import get_and_reset_counts
from agent.runner import EpisodeResult, run_episode
from game.scenarios import SCENARIOS
from parlay_env.models import PersonaType

logger = logging.getLogger(__name__)

DIVERSITY_CONFIG = {
    "noise_injection_rate": 0.3,
    "drift_force_rate": 0.4,
}

REQUIRED_COMBINATIONS = [
    (persona, scenario)
    for persona in ["shark", "diplomat", "veteran"]
    for scenario in ["saas_enterprise", "hiring_package", "acquisition_term_sheet"]
]


def is_quality_episode(grade, args) -> tuple[bool, str]:
    """
    Returns (keep: bool, reason: str).
    """
    if not args.quality_filter:
        return True, "no_filter"
    if grade.deal_efficiency >= args.min_efficiency:
        return True, "deal_efficiency"
    if grade.termination_reason == "walk_away" and grade.total_reward > -200:
        return True, "principled_walkaway"
    if grade.drift_adapted:
        return True, "drift_adapted"
    if grade.tom_accuracy_avg >= 0.5:
        return True, "good_tom"
    return False, f"low_quality (eff={grade.deal_efficiency:.2f}, tom={grade.tom_accuracy_avg:.2f})"


def _grade_proxy_from_record(record: dict) -> object:
    return type(
        "GradeProxy",
        (),
        {
            "deal_efficiency": record["deal_efficiency"],
            "termination_reason": record["termination_reason"],
            "total_reward": record["reward"],
            "drift_adapted": record["drift_adapted"],
            "tom_accuracy_avg": record["tom_accuracy_avg"],
        },
    )()


def _record_from_result(persona: str, scenario_id: str, result: EpisodeResult) -> dict:
    return {
        "prompt": result.system_prompt,
        "conversation": [{k: v for k, v in msg.items()} for msg in result.conversation],
        "reward": result.grade.total_reward,
        "deal_efficiency": result.grade.deal_efficiency,
        "persona": persona,
        "scenario_id": scenario_id,
        "acts_completed": 1,
        "tom_accuracy": result.grade.tom_accuracy_avg,
        "tom_accuracy_avg": result.grade.tom_accuracy_avg,
        "drift_adapted": result.grade.drift_adapted,
        "split": "train" if random.random() < 0.9 else "eval",
        "deal_reached": result.final_price is not None,
        "episode_id": result.session.session_id,
        "termination_reason": result.grade.termination_reason,
        "batna_seller": result.session.hidden_state.walk_away_price,
        "batna_buyer": result.session.hidden_state.budget_ceiling,
    }


async def _run_episode_full(
    persona: str, scenario_id: str, seed: int, max_turns: int
) -> tuple[dict | None, EpisodeResult | None]:
    try:
        result = await run_episode(
            persona=PersonaType(persona),
            scenario_id=scenario_id,
            inject_noise=random.random() < DIVERSITY_CONFIG["noise_injection_rate"],
            force_drift=random.random() < DIVERSITY_CONFIG["drift_force_rate"],
            seed=seed,
            max_turns=max_turns,
        )
    except Exception as exc:
        logger.warning("Episode failed (%s, %s): %s", persona, scenario_id, exc)
        return None, None

    return _record_from_result(persona, scenario_id, result), result


async def _run_one(persona: str, scenario_id: str, seed: int, max_turns: int) -> dict | None:
    record, _ = await _run_episode_full(persona, scenario_id, seed, max_turns)
    return record


def _classify_discard(grade, args) -> str:
    """Single bucket per discarded episode (mutually exclusive)."""
    if grade.deal_efficiency < args.min_efficiency:
        return "low_efficiency_no_deal"
    if grade.tom_accuracy_avg < 0.5:
        return "tom_accuracy_below_threshold"
    return "other"


def _conversation_mentions_market(conversation: list[dict]) -> bool:
    for msg in conversation:
        for v in msg.values():
            if isinstance(v, str) and "market" in v.lower():
                return True
    return False


def _print_inspect_report(
    coverage: dict[tuple[str, str], int],
    total_pre: int,
    kept: int,
    discarded: int,
    keep_reason_counts: Counter[str],
    kept_records: list[dict],
    kept_tom: list[bool],
    kept_rewards: list[float],
    kept_eff: list[float],
    kept_tom_acc: list[float],
    kept_turns: list[float],
    div_drift: int,
    div_market: int,
    div_bluff: int,
    div_zopa: int,
    discard_by_label: Counter[str],
) -> None:
    n_k = max(kept, 1)
    pct = lambda x: 100.0 * x / n_k
    d_rate = 100.0 * discarded / max(total_pre, 1)

    def st(values: list[float]) -> str:
        if len(values) < 2:
            return "0.00"
        return f"{statistics.stdev(values):.2f}"

    def mean_t(values: list[float]) -> str:
        if not values:
            return "0.00"
        return f"{statistics.mean(values):.2f}"

    mean_turns = statistics.mean(kept_turns) if kept_turns else 0.0

    deal_n = sum(1 for r in kept_records if r.get("deal_reached"))
    walk_n = keep_reason_counts.get("principled_walkaway", 0)
    drift_n = keep_reason_counts.get("drift_adapted", 0)
    tom5_n = sum(1 for t in kept_tom if t)

    r1, r2, r3 = (
        discard_by_label.get("low_efficiency_no_deal", 0),
        discard_by_label.get("tom_accuracy_below_threshold", 0),
        discard_by_label.get("other", 0),
    )

    _lw = 31
    print()
    print("=== QUALITY REPORT (60 episodes) ===")
    print()
    print("Coverage (persona × scenario):")
    for persona, scenario_id in REQUIRED_COMBINATIONS:
        n = coverage.get((persona, scenario_id), 0)
        print(f"  {persona:8s} × {scenario_id:30s} : {n} episodes")
    print()
    print("Quality filter:")
    print(f"  {'Total generated (before filter)':<{_lw}}: {total_pre}")
    print(f"  {'Kept after filter':<{_lw}}: {kept}")
    print(f"  {'Discarded':<{_lw}}: {discarded}")
    print(f"  {'Discard rate':<{_lw}}: {d_rate:.1f}%")
    print()
    print("Kept episode breakdown:")
    print(f"  Deal reached          : {deal_n:3d}  ({pct(deal_n):.1f}%)")
    print(f"  Principled walkaway   : {walk_n:3d}  ({pct(walk_n):.1f}%)")
    print(f"  Drift adapted         : {drift_n:3d}  ({pct(drift_n):.1f}%)")
    print(f"  ToM accuracy >= 0.5   : {tom5_n:3d}  ({pct(tom5_n):.1f}%)")
    print()
    print("Reward stats (kept episodes only):")
    print(f"  Mean cumulative reward : {mean_t(kept_rewards)}")
    print(f"  Std cumulative reward  : {st(kept_rewards)}")
    print(f"  Min                    : {min(kept_rewards) if kept_rewards else 0.0:.2f}")
    print(f"  Max                    : {max(kept_rewards) if kept_rewards else 0.0:.2f}")
    print(f"  Mean deal efficiency   : {mean_t(kept_eff)}")
    print(f"  Mean ToM accuracy      : {mean_t(kept_tom_acc)}")
    print(f"  Mean turns to close    : {mean_turns:.1f}")
    print()
    print("Diversity flags (kept episodes):")
    print(
        f"  {'Episodes with drift event':<{_lw}}: {div_drift:3d}  ({100.0 * div_drift / n_k:.1f}%)"
    )
    print(
        f"  {'Episodes with market event':<{_lw}}: {div_market:3d}  ({100.0 * div_market / n_k:.1f}%)"
    )
    print(
        f"  {'Episodes with bluff caught':<{_lw}}: {div_bluff:3d}  ({100.0 * div_bluff / n_k:.1f}%)"
    )
    print(
        f"  {'Episodes with ZOPA erosion':<{_lw}}: {div_zopa:3d}  ({100.0 * div_zopa / n_k:.1f}%)"
    )
    print()
    print("Top 3 discard reasons:")
    print(f"  1. low_efficiency_no_deal      : {r1}")
    print(f"  2. tom_accuracy_below_threshold: {r2}")
    print(f"  3. other                       : {r3}")


async def run_inspect_mode(args) -> None:
    out_path = Path("training/data/inspect_run.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coverage: dict[tuple[str, str], int] = defaultdict(int)
    keep_reason_counts: Counter[str] = Counter()
    kept_records: list[dict] = []
    kept_tom: list[bool] = []
    kept_rewards: list[float] = []
    kept_eff: list[float] = []
    kept_tom_acc: list[float] = []
    kept_turns: list[float] = []
    div_drift = div_market = div_bluff = div_zopa = 0
    discard_by_label: Counter[str] = Counter()
    total_pre = 60
    discarded = 0
    seed = 0
    n_inspect = 60

    for i in range(n_inspect):
        persona, scenario_id = REQUIRED_COMBINATIONS[i % len(REQUIRED_COMBINATIONS)]
        record, res = await _run_episode_full(
            persona, scenario_id, seed=seed, max_turns=args.max_turns
        )
        seed += 1
        coverage[(persona, scenario_id)] += 1

        if record is None or res is None:
            discarded += 1
            discard_by_label["other"] += 1
            continue

        g = res.grade
        proxy = _grade_proxy_from_record(record)
        keep, reason = is_quality_episode(proxy, args)
        if not keep:
            discarded += 1
            discard_by_label[_classify_discard(g, args)] += 1
            continue

        keep_reason_counts[reason] += 1
        kept_rewards.append(record["reward"])
        kept_eff.append(record["deal_efficiency"])
        kept_tom_acc.append(record["tom_accuracy_avg"])
        kept_turns.append(float(res.session.step_count))
        kept_tom.append(record["tom_accuracy_avg"] >= 0.5)
        kept_records.append(record)

        s = res.session
        if record["drift_adapted"]:
            div_drift += 1
        if _conversation_mentions_market(res.conversation):
            div_market += 1
        if s.bluffs_caught > 0 or g.bluffs_caught > 0:
            div_bluff += 1
        if s.zopa_erosion_ticks > 0:
            div_zopa += 1

    with open(out_path, "w", encoding="utf-8") as out_f:
        for r in kept_records:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _print_inspect_report(
        coverage,
        total_pre=total_pre,
        kept=len(kept_records),
        discarded=discarded,
        keep_reason_counts=keep_reason_counts,
        kept_records=kept_records,
        kept_tom=kept_tom,
        kept_rewards=kept_rewards,
        kept_eff=kept_eff,
        kept_tom_acc=kept_tom_acc,
        kept_turns=kept_turns,
        div_drift=div_drift,
        div_market=div_market,
        div_bluff=div_bluff,
        div_zopa=div_zopa,
        discard_by_label=discard_by_label,
    )
    print()
    print(f"Kept episodes written to: {out_path.resolve()}")


async def run_diversity_pass(args, output_path: Path) -> None:
    """
    Generate a quality-filtered dataset with guaranteed persona x scenario coverage.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage: dict[tuple[str, str], int] = defaultdict(int)
    kept_reason_counts: Counter[str] = Counter()
    kept_records: list[dict] = []
    generated = 0
    discarded = 0
    seed = 0
    min_per_combo = max(2, args.episodes // len(REQUIRED_COMBINATIONS))
    total_live_calls: int = 0
    total_fallback_calls: int = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        while len(kept_records) < args.episodes:
            progress_made = False
            for persona, scenario_id in REQUIRED_COMBINATIONS:
                if len(kept_records) >= args.episodes:
                    break
                if coverage[(persona, scenario_id)] >= min_per_combo:
                    continue

                record = await _run_one(persona, scenario_id, seed=seed, max_turns=args.max_turns)
                seed += 1
                generated += 1
                if record is None:
                    _live_n, _fall_n = get_and_reset_counts()
                    total_live_calls += _live_n
                    total_fallback_calls += _fall_n
                    continue

                keep, reason = is_quality_episode(
                    _grade_proxy_from_record(record),
                    args,
                )
                if not keep:
                    discarded += 1
                    _live_d, _fall_d = get_and_reset_counts()
                    total_live_calls += _live_d
                    total_fallback_calls += _fall_d
                    print(
                        f"[EP --/{args.episodes:02d}] "
                        f"{persona}×{scenario_id:<27s} | "
                        f"reward={record.get('reward', 0.0):+.2f} | "
                        f"eff={record.get('deal_efficiency', 0.0):.3f} | "
                        f"kept=NO  | "
                        f"total_kept={len(kept_records)}/{generated} | "
                        f"gemini_live={_live_d} fallback={_fall_d}",
                        file=sys.stderr,
                    )
                    continue

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept_records.append(record)
                _live, _fall = get_and_reset_counts()
                total_live_calls += _live
                total_fallback_calls += _fall
                _ep_num = len(kept_records)
                _reward = record.get("reward", 0.0)
                _eff = record.get("deal_efficiency", 0.0)
                _combo = f"{record['persona']}×{record['scenario_id']}"
                print(
                    f"[EP {_ep_num:02d}/{args.episodes:02d}] "
                    f"{_combo:<35s} | "
                    f"reward={_reward:+.2f} | "
                    f"eff={_eff:.3f} | "
                    f"kept=YES | "
                    f"total_kept={_ep_num}/{generated} | "
                    f"gemini_live={_live} fallback={_fall}",
                    file=sys.stderr,
                )
                if _ep_num in (20, 40, 60):
                    _all_rewards = [r.get("reward", 0.0) for r in kept_records]
                    _all_eff = [r.get("deal_efficiency", 0.0) for r in kept_records]
                    _combos_covered = len({(r["persona"], r["scenario_id"]) for r in kept_records})
                    print(f"\n{'━' * 40}", file=sys.stderr)
                    print(f"[CHECKPOINT {_ep_num}/{args.episodes}]", file=sys.stderr)
                    print(
                        f"  Kept so far     : {_ep_num}/{generated}  ({100 * _ep_num / max(generated, 1):.1f}%)",
                        file=sys.stderr,
                    )
                    print(f"  Mean reward     : {statistics.mean(_all_rewards):.2f}", file=sys.stderr)
                    print(f"  Mean efficiency : {statistics.mean(_all_eff):.3f}", file=sys.stderr)
                    print(f"  Combos covered  : {_combos_covered}/9", file=sys.stderr)
                    print(f"  Live calls total: {total_live_calls}", file=sys.stderr)
                    print(f"  Fallback total  : {total_fallback_calls}", file=sys.stderr)
                    print(f"{'━' * 40}\n", file=sys.stderr)
                coverage[(persona, scenario_id)] += 1
                kept_reason_counts[reason] += 1
                progress_made = True

            if len(kept_records) >= args.episodes:
                break

            if not progress_made:
                persona, scenario_id = random.choice(REQUIRED_COMBINATIONS)
                record = await _run_one(persona, scenario_id, seed=seed, max_turns=args.max_turns)
                seed += 1
                generated += 1
                if record is None:
                    _live_n, _fall_n = get_and_reset_counts()
                    total_live_calls += _live_n
                    total_fallback_calls += _fall_n
                    continue
                keep, reason = is_quality_episode(
                    _grade_proxy_from_record(record),
                    args,
                )
                if keep:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept_records.append(record)
                    _live, _fall = get_and_reset_counts()
                    total_live_calls += _live
                    total_fallback_calls += _fall
                    _ep_num = len(kept_records)
                    _reward = record.get("reward", 0.0)
                    _eff = record.get("deal_efficiency", 0.0)
                    _combo = f"{record['persona']}×{record['scenario_id']}"
                    print(
                        f"[EP {_ep_num:02d}/{args.episodes:02d}] "
                        f"{_combo:<35s} | "
                        f"reward={_reward:+.2f} | "
                        f"eff={_eff:.3f} | "
                        f"kept=YES | "
                        f"total_kept={_ep_num}/{generated} | "
                        f"gemini_live={_live} fallback={_fall}",
                        file=sys.stderr,
                    )
                    if _ep_num in (20, 40, 60):
                        _all_rewards = [r.get("reward", 0.0) for r in kept_records]
                        _all_eff = [r.get("deal_efficiency", 0.0) for r in kept_records]
                        _combos_covered = len({(r["persona"], r["scenario_id"]) for r in kept_records})
                        print(f"\n{'━' * 40}", file=sys.stderr)
                        print(f"[CHECKPOINT {_ep_num}/{args.episodes}]", file=sys.stderr)
                        print(
                            f"  Kept so far     : {_ep_num}/{generated}  ({100 * _ep_num / max(generated, 1):.1f}%)",
                            file=sys.stderr,
                        )
                        print(f"  Mean reward     : {statistics.mean(_all_rewards):.2f}", file=sys.stderr)
                        print(f"  Mean efficiency : {statistics.mean(_all_eff):.3f}", file=sys.stderr)
                        print(f"  Combos covered  : {_combos_covered}/9", file=sys.stderr)
                        print(f"  Live calls total: {total_live_calls}", file=sys.stderr)
                        print(f"  Fallback total  : {total_fallback_calls}", file=sys.stderr)
                        print(f"{'━' * 40}\n", file=sys.stderr)
                    coverage[(persona, scenario_id)] += 1
                    kept_reason_counts[reason] += 1
                else:
                    discarded += 1
                    _live_d, _fall_d = get_and_reset_counts()
                    total_live_calls += _live_d
                    total_fallback_calls += _fall_d
                    print(
                        f"[EP --/{args.episodes:02d}] "
                        f"{persona}×{scenario_id:<27s} | "
                        f"reward={record.get('reward', 0.0):+.2f} | "
                        f"eff={record.get('deal_efficiency', 0.0):.3f} | "
                        f"kept=NO  | "
                        f"total_kept={len(kept_records)}/{generated} | "
                        f"gemini_live={_live_d} fallback={_fall_d}",
                        file=sys.stderr,
                    )

    discard_pct = (discarded / max(generated, 1)) * 100.0
    print(
        f"Generated: {generated} episodes | Kept: {len(kept_records)} | "
        f"Discarded: {discarded} ({discard_pct:.0f}%)"
    )
    reasons_str = ", ".join(f"{reason}={count}" for reason, count in sorted(kept_reason_counts.items()))
    print(f"Reasons kept: {reasons_str or 'none'}")
    print("\nCoverage:")
    for persona, scenario_id in REQUIRED_COMBINATIONS:
        print(f"  {persona:9s} x {scenario_id:24s} -> {coverage[(persona, scenario_id)]}")

    _fallback_rate = 100.0 * total_fallback_calls / max(total_live_calls + total_fallback_calls, 1)
    _verdict = (
        "ALL CALLS LIVE — data is real"
        if _fallback_rate < 5.0
        else "WARNING: fallback rate high — check API key and rate limits"
    )
    print(f"\nGemini API health:")
    print(f"  Total live calls  : {total_live_calls}")
    print(f"  Total fallback    : {total_fallback_calls}")
    print(f"  Fallback rate     : {_fallback_rate:.1f}%")
    print(f"  VERDICT: {_verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Parlay training data")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--output", type=str, default="data/episodes.jsonl")
    parser.add_argument(
        "--quality_filter",
        action="store_true",
        help="Discard low-quality episodes instead of writing them",
    )
    parser.add_argument(
        "--min_efficiency",
        type=float,
        default=0.25,
        help="Min deal_efficiency to keep episode (if quality_filter enabled)",
    )
    parser.add_argument("--google_api_key", type=str, default="")
    parser.add_argument("--max-turns", type=int, default=14)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run a fixed 60-episode quality diagnostic; writes training/data/inspect_run.jsonl",
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)

    if args.google_api_key:
        os.environ["GOOGLE_API_KEY"] = args.google_api_key

    if args.inspect:
        asyncio.run(run_inspect_mode(args))
        return

    output_path = Path(args.output)
    asyncio.run(run_diversity_pass(args, output_path))

    records = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    deals = sum(1 for record in records if record.get("deal_efficiency", 0) > 0)
    avg_reward = sum(record.get("reward", 0.0) for record in records) / max(total, 1)

    print(f"\n{'=' * 50}")
    print("  GENERATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Episodes in file : {total}")
    print(f"  Deal rate        : {deals / max(total, 1) * 100:.1f}% ({deals}/{total})")
    print(f"  Avg total reward : {avg_reward:.2f}")
    print(f"  max_turns used   : {args.max_turns}")
    print(f"  Output file      : {output_path.resolve()}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
