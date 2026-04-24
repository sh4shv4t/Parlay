"""Generate quality-filtered negotiation episodes via Gemini self-play."""
import argparse
import asyncio
import json
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from agent.runner import run_episode
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


async def _run_one(persona: str, scenario_id: str, seed: int, max_turns: int) -> dict | None:
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
        return None

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
                    continue

                keep, reason = is_quality_episode(
                    type(
                        "GradeProxy",
                        (),
                        {
                            "deal_efficiency": record["deal_efficiency"],
                            "termination_reason": record["termination_reason"],
                            "total_reward": record["reward"],
                            "drift_adapted": record["drift_adapted"],
                            "tom_accuracy_avg": record["tom_accuracy_avg"],
                        },
                    )(),
                    args,
                )
                if not keep:
                    discarded += 1
                    continue

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept_records.append(record)
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
                    continue
                keep, reason = is_quality_episode(
                    type(
                        "GradeProxy",
                        (),
                        {
                            "deal_efficiency": record["deal_efficiency"],
                            "termination_reason": record["termination_reason"],
                            "total_reward": record["reward"],
                            "drift_adapted": record["drift_adapted"],
                            "tom_accuracy_avg": record["tom_accuracy_avg"],
                        },
                    )(),
                    args,
                )
                if keep:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept_records.append(record)
                    coverage[(persona, scenario_id)] += 1
                    kept_reason_counts[reason] += 1
                else:
                    discarded += 1

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
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)

    if args.google_api_key:
        os.environ["GOOGLE_API_KEY"] = args.google_api_key

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
