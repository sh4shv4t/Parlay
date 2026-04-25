"""Run Gemini self-play baseline and save summary JSON."""
import argparse
import asyncio
import json
import logging
from pathlib import Path

from agent.runner import run_episode
from parlay_env.models import PersonaType

PERSONAS = [PersonaType.SHARK, PersonaType.DIPLOMAT, PersonaType.VETERAN]
SCENARIOS = ["saas_enterprise", "hiring_package", "acquisition_term_sheet"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def _run(episodes: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(episodes):
        persona = PERSONAS[i % len(PERSONAS)]
        scenario_id = SCENARIOS[(i // len(PERSONAS)) % len(SCENARIOS)]
        result = await run_episode(
            persona=persona,
            scenario_id=scenario_id,
            inject_noise=False,
            force_drift=True,
            seed=i + 100,
            max_turns=20,
        )
        rows.append(
            {
                "avg_reward": float(result.grade.total_reward),
                "deal_rate": 1.0 if result.final_price is not None else 0.0,
                "avg_efficiency": float(result.grade.deal_efficiency),
                "avg_tom_accuracy": float(result.grade.tom_accuracy_avg),
                "bluffs_caught": int(result.grade.bluffs_caught),
            }
        )
    return rows


def _summarise(rows: list[dict], episodes_requested: int) -> dict:
    return {
        "episodes_requested": episodes_requested,
        "episodes_completed": len(rows),
        "avg_reward": round(_mean([r["avg_reward"] for r in rows]), 4),
        "deal_rate": round(_mean([r["deal_rate"] for r in rows]), 4),
        "avg_efficiency": round(_mean([r["avg_efficiency"] for r in rows]), 4),
        "avg_tom_accuracy": round(_mean([r["avg_tom_accuracy"] for r in rows]), 4),
        "bluffs_caught": int(sum(r["bluffs_caught"] for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini self-play baseline")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="results/gemini_baseline.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rows = asyncio.run(_run(args.episodes))
    summary = _summarise(rows, args.episodes)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved Gemini baseline to {out.resolve()}")


if __name__ == "__main__":
    main()
