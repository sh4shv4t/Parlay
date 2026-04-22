"""
Stage 1: SFT warmup on best episodes (efficiency >= threshold).
Fine-tunes Qwen2.5-7B-Instruct on demonstrations of successful negotiation.

Usage:
    python -m training.sft_train \
        --data data/episodes.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output models/parlay-sft \
        --threshold 0.60
"""
import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

TOP_PLAYER_THRESHOLD = float(os.getenv("TOP_PLAYER_THRESHOLD", "0.60"))
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def load_sft_dataset(jsonl_path: Path, threshold: float = 0.60):
    """
    Load episodes above efficiency threshold and format for SFT.

    Only 'train' split episodes above the threshold are included.
    Each agent turn becomes one training example.

    Args:
        jsonl_path: Path to the JSONL episodes file.
        threshold:  Minimum deal_efficiency to include.

    Returns:
        HuggingFace Dataset with 'text' column.
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Install datasets: pip install datasets") from exc

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("deal_efficiency", 0) >= threshold and rec.get("split") == "train":
                conversation = rec.get("conversation", [])
                for i, turn in enumerate(conversation[:-1]):
                    if turn.get("role") == "negotiator":
                        context = conversation[:i]
                        records.append({
                            "text": _format_sft_example(
                                system=rec["prompt"],
                                context=context,
                                response=turn["content"],
                                efficiency=rec["deal_efficiency"],
                            )
                        })

    logger.info(f"SFT dataset: {len(records)} training examples from {jsonl_path}")
    return Dataset.from_list(records)


def _format_sft_example(
    system: str,
    context: list[dict],
    response: str,
    efficiency: float,
) -> str:
    """Format a single SFT training example in chat template format."""
    history_lines = []
    for turn in context:
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "")
        history_lines.append(f"{role}: {content}")
    history = "\n".join(history_lines)

    return (
        f"<|system|>{system}</s>\n"
        f"<|negotiation_history|>{history}</s>\n"
        f"<|assistant|>{response}</s>"
    )


def train_sft(
    data_path: Path,
    model_id: str,
    output_dir: Path,
    threshold: float = 0.60,
) -> Path:
    """
    Run SFT fine-tuning.

    Args:
        data_path:   Path to episodes JSONL.
        model_id:    HuggingFace model ID or local path.
        output_dir:  Where to save the trained model.
        threshold:   Efficiency filter for training data.

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

    dataset = load_sft_dataset(data_path, threshold)
    if len(dataset) == 0:
        raise ValueError(
            f"No training examples above threshold={threshold}. "
            "Lower the threshold or generate more data."
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
    parser.add_argument("--threshold", type=float, default=TOP_PLAYER_THRESHOLD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train_sft(Path(args.data), args.model, Path(args.output), args.threshold)


if __name__ == "__main__":
    main()
