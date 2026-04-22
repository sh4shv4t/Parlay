"""
Generate diverse negotiation episodes via Gemini self-play.
Writes to data/episodes.jsonl

Usage:
    python -m training.generate_data --episodes 2000 --output data/episodes.jsonl
"""
import argparse
import asyncio
import json
import logging
import os
import random
from pathlib import Path

import numpy as np

from parlay_env.models import PersonaType
from game.scenarios import SCENARIOS
from agent.runner import run_episode

logger = logging.getLogger(__name__)

TOP_PLAYER_THRESHOLD = float(os.getenv("TOP_PLAYER_THRESHOLD", "0.60"))

DIVERSITY_CONFIG = {
    "min_per_combination": 20,   # at least 20 episodes per (persona, scenario) pair
    "noise_injection_rate": 0.3, # 30% inject random early moves
    "drift_force_rate": 0.4,     # 40% force drift events
    "act3_rate": 0.25,           # 25% target Act 3 scenarios
}


async def generate_dataset(
    n_episodes: int,
    output_path: Path,
    top_threshold: float = TOP_PLAYER_THRESHOLD,
) -> dict:
    """
    Generate diverse self-play episodes and write to JSONL.

    Args:
        n_episodes:     Total target episode count.
        output_path:    Where to write the JSONL file.
        top_threshold:  Efficiency threshold for the 'above_threshold' stat.

    Returns:
        Stats dict: {total, above_threshold, persona_counts, scenario_counts,
                     mean_efficiency, mean_reward}
    """
    records: list[dict] = []
    persona_counts: dict[str, int] = {p.value: 0 for p in PersonaType}
    scenario_counts: dict[str, int] = {s.id: 0 for s in SCENARIOS}
    seed = 0

    async def _run_one(
        persona: PersonaType,
        scenario_id: str,
        inject_noise: bool,
        force_drift: bool,
        s: int,
    ) -> dict | None:
        nonlocal seed
        try:
            result = await run_episode(
                persona=persona,
                scenario_id=scenario_id,
                inject_noise=inject_noise,
                force_drift=force_drift,
                seed=s,
            )
            conv = [
                {k: v for k, v in msg.items()}
                for msg in result.conversation
            ]
            return {
                "prompt": result.system_prompt,
                "conversation": conv,
                "reward": result.grade.total_reward,
                "deal_efficiency": result.grade.deal_efficiency,
                "persona": persona.value,
                "scenario_id": scenario_id,
                "acts_completed": result.grade.acts_completed,
                "tom_accuracy": result.grade.tom_accuracy_avg,
                "drift_adapted": result.grade.drift_adapted,
                "split": "train" if random.random() < 0.9 else "eval",
            }
        except Exception as exc:
            logger.warning(f"Episode failed (persona={persona.value}, scenario={scenario_id}): {exc}")
            return None

    logger.info(
        f"Generating {n_episodes} episodes "
        f"(min {DIVERSITY_CONFIG['min_per_combination']} per combination)"
    )

    # Pass 1: ensure minimum coverage per (persona × scenario)
    for persona in PersonaType:
        for scenario in SCENARIOS:
            for i in range(DIVERSITY_CONFIG["min_per_combination"]):
                inject_noise = random.random() < DIVERSITY_CONFIG["noise_injection_rate"]
                force_drift  = random.random() < DIVERSITY_CONFIG["drift_force_rate"]
                record = await _run_one(persona, scenario.id, inject_noise, force_drift, seed)
                seed += 1
                if record:
                    records.append(record)
                    persona_counts[persona.value] += 1
                    scenario_counts[scenario.id] += 1

    logger.info(f"Coverage pass done: {len(records)} episodes")

    # Pass 2: fill remaining to n_episodes
    remaining = n_episodes - len(records)
    for i in range(max(0, remaining)):
        persona = random.choice(list(PersonaType))
        scenario = random.choice(SCENARIOS)
        inject_noise = random.random() < DIVERSITY_CONFIG["noise_injection_rate"]
        force_drift  = random.random() < DIVERSITY_CONFIG["drift_force_rate"]
        record = await _run_one(persona, scenario.id, inject_noise, force_drift, seed)
        seed += 1
        if record:
            records.append(record)
            persona_counts[persona.value] += 1
            scenario_counts[scenario.id] += 1

    # Write all records (filtering happens at load time in SFT/GRPO)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    efficiencies = [r["deal_efficiency"] for r in records]
    rewards = [r["reward"] for r in records]
    stats = {
        "total": len(records),
        "above_threshold": sum(1 for e in efficiencies if e >= top_threshold),
        "persona_counts": persona_counts,
        "scenario_counts": scenario_counts,
        "mean_efficiency": sum(efficiencies) / max(len(efficiencies), 1),
        "mean_reward": sum(rewards) / max(len(rewards), 1),
    }
    logger.info(f"Dataset stats: {stats}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Parlay training data")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--output", default="data/episodes.jsonl")
    parser.add_argument("--threshold", type=float, default=TOP_PLAYER_THRESHOLD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    stats = asyncio.run(generate_dataset(args.episodes, Path(args.output), args.threshold))
    print(f"\nGeneration complete:")
    print(f"  Total episodes:     {stats['total']}")
    print(
        f"  Above threshold:    {stats['above_threshold']} "
        f"({stats['above_threshold'] / max(stats['total'], 1) * 100:.1f}%)"
    )
    print(f"  Mean efficiency:    {stats['mean_efficiency']:.3f}")
    print(f"  Mean reward:        {stats['mean_reward']:.2f}")


if __name__ == "__main__":
    main()
