"""
Random-policy baseline for Parlay.
Runs N episodes with purely random move selection (no Gemini API — always
uses mock mode) and writes a summary JSON that the training notebook uses
to benchmark SFT / GRPO improvement.

Usage:
    python training/random_baseline.py
    python training/random_baseline.py --episodes 20 --output data/random_baseline.json
"""
import argparse
import asyncio
import json
import logging
import os
import random
import statistics
import sys
from pathlib import Path

# Repo root on sys.path when run as `python training/random_baseline.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force mock mode — random baseline never calls the real Gemini API
os.environ.pop("GOOGLE_API_KEY", None)

from agent.runner import run_episode
from game.scenarios import SCENARIOS
from parlay_env.models import PersonaType

logger = logging.getLogger(__name__)

REQUIRED_COMBINATIONS = [
    (persona, scenario)
    for persona in ["shark", "diplomat", "veteran"]
    for scenario in ["saas_enterprise", "hiring_package", "acquisition_term_sheet"]
]


async def _run_baseline(episodes: int) -> list[dict]:
    """Run `episodes` random-policy episodes and return per-episode stats."""
    results = []
    seed = 0
    for i in range(episodes):
        persona_str, scenario_id = REQUIRED_COMBINATIONS[i % len(REQUIRED_COMBINATIONS)]
        try:
            result = await run_episode(
                persona=PersonaType(persona_str),
                scenario_id=scenario_id,
                inject_noise=True,   # random noise simulates random policy
                force_drift=random.random() < 0.4,
                seed=seed,
                max_turns=14,
            )
            results.append({
                "persona": persona_str,
                "scenario_id": scenario_id,
                "reward": result.grade.total_reward,
                "deal_efficiency": result.grade.deal_efficiency,
                "deal_reached": result.final_price is not None,
                "tom_accuracy_avg": result.grade.tom_accuracy_avg,
                "drift_adapted": result.grade.drift_adapted,
                "termination_reason": result.grade.termination_reason,
            })
        except Exception as exc:
            logger.warning("Baseline episode %d failed (%s, %s): %s", i, persona_str, scenario_id, exc)
        seed += 1
    return results


def _summarise(results: list[dict]) -> dict:
    if not results:
        return {"error": "no episodes completed", "n_episodes": 0}

    rewards = [r["reward"] for r in results]
    efficiencies = [r["deal_efficiency"] for r in results]
    deal_count = sum(1 for r in results if r["deal_reached"])
    drift_count = sum(1 for r in results if r["drift_adapted"])

    return {
        "n_episodes": len(results),
        "mean_reward": round(statistics.mean(rewards), 3),
        "std_reward": round(statistics.stdev(rewards) if len(rewards) > 1 else 0.0, 3),
        "min_reward": round(min(rewards), 3),
        "max_reward": round(max(rewards), 3),
        "mean_efficiency": round(statistics.mean(efficiencies), 4),
        "deal_rate": round(deal_count / len(results), 4),
        "drift_adapted_rate": round(drift_count / len(results), 4),
        "policy": "random_mock",
        "note": (
            "Baseline uses Parlay mock responses (no real Gemini API). "
            "Compare mean_reward and mean_efficiency against SFT/GRPO runs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parlay random-policy baseline")
    parser.add_argument("--episodes", type=int, default=27,
                        help="Number of baseline episodes (default: 27 = 3 per combo)")
    parser.add_argument("--output", type=str, default="data/random_baseline.json",
                        help="Output path for the baseline JSON summary")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print(f"Running {args.episodes} random-policy episodes (mock mode, no API key)…")
    results = asyncio.run(_run_baseline(args.episodes))

    summary = _summarise(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBaseline complete ({summary['n_episodes']} episodes):")
    print(f"  Mean reward     : {summary.get('mean_reward', 'N/A')}")
    print(f"  Mean efficiency : {summary.get('mean_efficiency', 'N/A')}")
    print(f"  Deal rate       : {summary.get('deal_rate', 'N/A'):.1%}")
    print(f"  Written to      : {out_path.resolve()}")


if __name__ == "__main__":
    main()
