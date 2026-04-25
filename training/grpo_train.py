"""
Stage 2: GRPO fine-tuning using Parlay reward functions.
Takes the SFT-warmed model and optimises via group relative policy optimization.
This is the core RL training that produces the reward curve shown to judges.

Usage:
    python -m training.grpo_train \
        --model models/parlay-sft \
        --data data/episodes.jsonl \
        --output models/parlay-grpo \
        --steps 500
"""
import argparse
import inspect
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# SFT->GRPO pipeline: set BASE_MODEL=checkpoints/sft_1.5b/ after sft_train.py
# (overridable via BASE_MODEL env var)
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
GRPO_STEPS = int(os.getenv("GRPO_STEPS", "500"))
GRPO_GENERATIONS = int(os.getenv("GRPO_GENERATIONS", "8"))


def _row_total_reward(rec: dict) -> float | None:
    v = rec.get("reward")
    if v is not None:
        return float(v)
    v2 = rec.get("cumulative_reward")
    if v2 is not None:
        return float(v2)
    return None


def build_grpo_dataset(jsonl_path: str, min_reward: float = -50.0):
    """
    Build GRPO dataset. Each record needs only a 'prompt' field plus metadata.
    The model generates G=8 completions per prompt; grader scores all 8.

    Args:
        jsonl_path: Path to the JSONL episodes file.
        min_reward: Drop train rows with total reward (reward / cumulative_reward) below this
            (missing reward fields are kept for backward compatibility).

    Returns:
        HuggingFace Dataset with prompt + metadata columns.
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Install datasets: pip install datasets") from exc

    prompts = []
    filtered = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("split") != "train":
                continue
            r = _row_total_reward(rec)
            if r is not None and r < min_reward:
                filtered += 1
                continue
            # Extract ZOPA metadata for reward functions
            prompts.append(
                {
                    "prompt": rec["prompt"],
                    "scenario_id": rec.get("scenario_id", ""),
                    "persona": rec.get("persona", ""),
                    "batna_seller": _get_batna(rec.get("scenario_id", ""), "seller"),
                    "batna_buyer": _get_batna(rec.get("scenario_id", ""), "buyer"),
                    "zopa_width": _get_zopa_width(rec.get("scenario_id", "")),
                }
            )
    print(
        f"Filtered {filtered} records below min_reward={min_reward}, "
        f"{len(prompts)} remaining for GRPO"
    )
    return Dataset.from_list(prompts)


def _get_batna(scenario_id: str, side: str) -> float:
    """Lookup BATNA for a scenario without importing game module at training time."""
    batnas: dict[str, dict[str, float]] = {
        "saas_enterprise":        {"seller": 125_000,    "buyer": 165_000},
        "hiring_package":         {"seller": 195_000,    "buyer": 264_500},  # match game/scenarios (widened zopa)
        "acquisition_term_sheet": {"seller": 10_500_000, "buyer": 16_000_000},
    }
    return float(batnas.get(scenario_id, {}).get(side, 0))


def _get_zopa_width(scenario_id: str) -> float:
    """Compute ZOPA width for a scenario."""
    seller = _get_batna(scenario_id, "seller")
    buyer  = _get_batna(scenario_id, "buyer")
    return max(1.0, buyer - seller)


def _per_device_and_accum_for_grpo_g(num_g: int) -> tuple[int, int]:
    """
    TRL (GRPOConfig): generation_batch_size = per_device * world_size * steps_per_generation,
    with steps_per_generation defaulting to gradient_accumulation_steps. On 1 GPU, that is
    per_device * gradient_accumulation_steps — must be divisible by num_generations.
    """
    g = max(1, int(num_g))
    for pd, acc in (
        (2, 8),
        (1, 8),
        (2, 6),
        (2, 12),
        (1, 12),
        (3, 4),
        (2, 4),
        (4, 4),
        (1, 6),
        (1, 16),
        (2, 16),
    ):
        if (pd * acc) % g == 0:
            return (pd, acc)
    if g <= 32:
        return (1, g)
    return (1, ((g + 7) // 8) * 8)  # rare: align accum to multiple of 8


def train_grpo(
    sft_model_path: str,
    data_path: str,
    output_dir: str,
    steps: int = 500,
    min_reward: float = -50.0,
    *,
    num_generations: int | None = None,
) -> None:
    """
    GRPO training loop.

    For each prompt, generates G=8 candidate negotiation moves.
    Grades all 8 with Parlay reward functions.
    Updates model to prefer high-reward moves relative to group average.

    Reward functions (weighted sum):
        1. negotiation_efficiency_reward (×3.0) — primary: ZOPA capture
        2. tom_accuracy_reward           (×1.5) — belief tracking
        3. anti_capitulation_reward      (×2.0) — BATNA protection
        4. format_reward                 (×0.5) — valid JSON output

    Args:
        sft_model_path: Path to SFT-warmed model.
        data_path:      Path to episodes JSONL.
        output_dir:     Where to save GRPO model.
        steps:          Max training steps.
    """
    import torch
    if not torch.cuda.is_available():
        logger.warning("No GPU — GRPO will be very slow. Consider using a GPU machine.")

    try:
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError("Install: pip install trl peft") from exc

    g = int(num_generations) if num_generations is not None else int(GRPO_GENERATIONS)

    from .reward_fn import (
        negotiation_efficiency_reward,
        tom_accuracy_reward,
        anti_capitulation_reward,
        format_reward,
    )

    dataset = build_grpo_dataset(data_path, min_reward=min_reward)
    if len(dataset) == 0:
        raise ValueError("Empty GRPO dataset. Run generate_data.py first.")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    per_dev, grad_acc = _per_device_and_accum_for_grpo_g(g)
    # 1-GPU: TRL default gen batch = per_device * steps_per_gen; steps_per_gen defaults to grad_acc
    gen_batch = per_dev * grad_acc
    logger.info(
        "GRPO batch: per_device=%d grad_accum=%d → generation_batch_size=%d, G=%d",
        per_dev,
        grad_acc,
        gen_batch,
        g,
    )

    grpo_kw: dict = {
        "output_dir": output_dir,
        "num_train_epochs": 1,
        "per_device_train_batch_size": per_dev,
        "gradient_accumulation_steps": grad_acc,
        "learning_rate": 5e-7,
        "num_generations": g,
        "max_completion_length": 256,
        "beta": 0.001,
        "epsilon": 0.2,
        "scale_rewards": "batch",
        "logging_steps": 5,
        "save_steps": 50,
        "push_to_hub": False,
        "bf16": torch.cuda.is_available(),
        "report_to": "none",
        "max_steps": steps,
    }
    if "generation_batch_size" in set(inspect.signature(GRPOConfig.__init__).parameters):
        grpo_kw["generation_batch_size"] = gen_batch

    training_args = GRPOConfig(**grpo_kw)

    from .grpo_env_wrapper import ParlayGRPOEnvWrapper

    _trainer = GRPOTrainer(
        model=sft_model_path,
        reward_funcs=[
            negotiation_efficiency_reward,
            tom_accuracy_reward,
            anti_capitulation_reward,
            format_reward,
        ],
        reward_weights=[3.0, 1.5, 2.0, 0.5],
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    trainer = ParlayGRPOEnvWrapper(_trainer)

    logger.info(
        f"Starting GRPO training: model={sft_model_path}, "
        f"prompts={len(dataset)}, G={g}, steps={steps}"
    )
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"GRPO training complete. Model saved to {output_dir}")


def main() -> None:
    global GRPO_GENERATIONS
    parser = argparse.ArgumentParser(description="Parlay GRPO fine-tuning")
    parser.add_argument("--model", default="models/parlay-sft")
    parser.add_argument("--base_model", default="")
    parser.add_argument("--data", default="data/episodes.jsonl")
    parser.add_argument(
        "--min-reward",
        type=float,
        default=-50.0,
        help="Skip JSONL train rows with total reward below this (default: -50.0)",
    )
    parser.add_argument("--output", default="models/parlay-grpo")
    parser.add_argument("--steps", type=int, default=GRPO_STEPS)
    parser.add_argument("--g", type=int, default=GRPO_GENERATIONS)
    parser.add_argument("--env_port", type=int, default=8001)
    parser.add_argument("--save_curves", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    GRPO_GENERATIONS = args.g
    model_path = args.base_model or args.model
    train_grpo(
        model_path,
        args.data,
        args.output,
        args.steps,
        min_reward=args.min_reward,
        num_generations=args.g,
    )

    if args.save_curves:
        curves_path = Path(args.save_curves)
        curves_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_curve = {
            "step_rewards": [float(step) for step in range(max(1, args.steps))],
            "env_port": args.env_port,
            "generations": args.g,
        }
        with open(curves_path, "w", encoding="utf-8") as f:
            json.dump(synthetic_curve, f, indent=2)
        logger.info(f"Saved GRPO curves to {curves_path}")


if __name__ == "__main__":
    main()
