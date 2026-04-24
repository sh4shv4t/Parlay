"""
Stage 1: SFT warmup on best episodes (efficiency >= threshold).
Fine-tunes Qwen2.5-7B-Instruct on demonstrations of successful negotiation.

Applies episode quality filters (offers + reward outliers) and stable SFT target
metadata (log-scaled efficiency, clipped reward) when building training text.

Usage:
    python -m training.sft_train \
        --data data/episodes.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output models/parlay-sft \
        --threshold 0.30
"""
import argparse
import json
import logging
import os
from pathlib import Path

from .episode_filters import (
    SFTFilterConfig,
    clip_reward_for_label,
    efficiency_sft_label,
    episode_passes_sft_filters,
)

logger = logging.getLogger(__name__)

TOP_PLAYER_THRESHOLD = float(os.getenv("TOP_PLAYER_THRESHOLD", "0.30"))
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def load_sft_dataset(
    jsonl_path: Path,
    threshold: float = 0.30,
    filter_cfg: SFTFilterConfig | None = None,
    include_sft_targets: bool = True,
):
    """
    Load episodes above efficiency threshold and format for SFT.

    Only 'train' split episodes above the threshold are included.
    Rows failing quality filters (broken offers, extreme rewards) are skipped.
    Each agent turn becomes one training example.

    Args:
        jsonl_path: Path to the JSONL episodes file.
        threshold:  Minimum deal_efficiency to include.
        filter_cfg: Drop/clip thresholds; default SFTFilterConfig().
        include_sft_targets: If True, embed eff_log and reward_clip in the example text.

    Returns:
        HuggingFace Dataset with 'text' column.
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Install datasets: pip install datasets") from exc

    filter_cfg = filter_cfg or SFTFilterConfig()
    records = []
    skipped_filter = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            ok, _reason = episode_passes_sft_filters(rec, filter_cfg)
            if not ok:
                skipped_filter += 1
                continue
            if rec.get("deal_efficiency", 0) >= threshold and rec.get("split") == "train":
                conversation = rec.get("conversation", [])
                eff_l = efficiency_sft_label(rec.get("deal_efficiency"))
                r_clip = clip_reward_for_label(rec.get("reward"), filter_cfg)
                for i, turn in enumerate(conversation[:-1]):
                    if turn.get("role") == "negotiator":
                        context = conversation[:i]
                        records.append({
                            "text": _format_sft_example(
                                system=rec["prompt"],
                                context=context,
                                response=turn["content"],
                                efficiency_label=eff_l,
                                reward_clip=r_clip,
                                include_sft_targets=include_sft_targets,
                            )
                        })

    logger.info(
        f"SFT dataset: {len(records)} training examples from {jsonl_path} "
        f"(skipped {skipped_filter} episodes by quality filter)"
    )
    return Dataset.from_list(records)


def _format_sft_example(
    system: str,
    context: list[dict],
    response: str,
    efficiency_label: float,
    reward_clip: float,
    include_sft_targets: bool = True,
) -> str:
    """Format a single SFT training example in chat template format."""
    history_lines = []
    for turn in context:
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "")
        history_lines.append(f"{role}: {content}")
    history = "\n".join(history_lines)

    targets_block = ""
    if include_sft_targets:
        targets_block = (
            f"<|sft_targets|>eff_log={efficiency_label:.4f} "
            f"reward_clip={reward_clip:.2f}</s>\n"
        )

    return (
        f"<|system|>{system}</s>\n"
        f"{targets_block}"
        f"<|negotiation_history|>{history}</s>\n"
        f"<|assistant|>{response}</s>"
    )


def train_sft(
    data_path: Path,
    model_id: str,
    output_dir: Path,
    threshold: float = 0.30,
    filter_cfg: SFTFilterConfig | None = None,
    include_sft_targets: bool = True,
) -> Path:
    """
    Run SFT fine-tuning.

    Args:
        data_path:   Path to episodes JSONL.
        model_id:    HuggingFace model ID or local path.
        output_dir:  Where to save the trained model.
        threshold:   Efficiency filter for training data.
        filter_cfg:  Quality filter / clip config.
        include_sft_targets: Embed normalized targets in training strings.

    Returns:
        output_dir path.
    """
    import torch
    if not torch.cuda.is_available():
        logger.warning("No GPU detected — SFT will be very slow on CPU")

    try:
        from peft import LoraConfig
        from trl import SFTTrainer, SFTConfig
    except ImportError as exc:
        raise ImportError("Install: pip install trl peft") from exc

    filter_cfg = filter_cfg or SFTFilterConfig()
    dataset = load_sft_dataset(
        data_path, threshold, filter_cfg, include_sft_targets=include_sft_targets
    )
    if len(dataset) == 0 and threshold > 0.0:
        logger.warning(
            f"No episodes above threshold {threshold}. Lowering to 0.0 (all train rows)."
        )
        dataset = load_sft_dataset(
            data_path, 0.0, filter_cfg, include_sft_targets=include_sft_targets
        )
    if len(dataset) == 0:
        raise RuntimeError(
            "SFT dataset is empty. "
            "Run generate_data.py first with --episodes >= 200"
        )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=False,
        bf16=torch.cuda.is_available(),
        max_seq_length=2048,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model_id,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    logger.info(f"Starting SFT training: model={model_id}, examples={len(dataset)}, epochs=3")
    trainer.train()
    trainer.save_model(str(output_dir))
    logger.info(f"SFT training complete. Model saved to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Parlay SFT warmup training")
    parser.add_argument("--data", default="data/episodes.jsonl")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--output", default="models/parlay-sft")
    parser.add_argument("--steps", type=int, default=0, help="Notebook compatibility flag")
    parser.add_argument("--threshold", type=float, default=TOP_PLAYER_THRESHOLD)
    parser.add_argument("--reward-drop-min", type=float, default=-400.0)
    parser.add_argument("--reward-drop-max", type=float, default=400.0)
    parser.add_argument("--clip-reward-min", type=float, default=-200.0)
    parser.add_argument("--clip-reward-max", type=float, default=200.0)
    parser.add_argument(
        "--no-sft-targets",
        action="store_true",
        help="Omit eff_log / reward_clip block from training text",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = SFTFilterConfig(
        reward_drop_min=args.reward_drop_min,
        reward_drop_max=args.reward_drop_max,
        clip_reward_min=args.clip_reward_min,
        clip_reward_max=args.clip_reward_max,
    )
    train_sft(
        Path(args.data),
        args.model,
        Path(args.output),
        args.threshold,
        filter_cfg=cfg,
        include_sft_targets=not args.no_sft_targets,
    )


if __name__ == "__main__":
    main()
