"""
Run before grpo_train.py for SFT→GRPO pipeline. Pass checkpoint path as
BASE_MODEL env var to grpo_train.py.
"""
import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT = "checkpoints/sft_1.5b/"


def _extract_completions(rec: dict) -> list[str]:
    """Return candidate completion texts from a record."""
    completion = rec.get("completion")
    if isinstance(completion, str) and completion.strip():
        return [completion.strip()]

    conversation = rec.get("conversation", [])
    candidates: list[str] = []
    if isinstance(conversation, list):
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).lower()
            content = str(turn.get("content", "")).strip()
            if role == "negotiator" and content:
                candidates.append(content)
    return candidates


def _row_total_reward(rec: dict) -> float | None:
    v = rec.get("reward")
    if v is not None:
        return float(v)
    v2 = rec.get("cumulative_reward")
    if v2 is not None:
        return float(v2)
    return None


def load_sft_dataset(data_path: Path, min_reward: float = -50.0):
    """Build a text dataset from JSONL prompt/completion pairs."""
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Install datasets: pip install datasets") from exc

    rows: list[dict[str, str]] = []
    skipped = 0
    reward_filtered = 0
    remaining_records = 0
    with data_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL row %d", line_no)
                skipped += 1
                continue

            r = _row_total_reward(rec)
            if r is not None and r < min_reward:
                reward_filtered += 1
                continue

            prompt = str(rec.get("prompt", "")).strip()
            if not prompt:
                logger.warning("Skipping row %d: missing prompt", line_no)
                skipped += 1
                continue

            completions = _extract_completions(rec)
            if not completions:
                logger.warning("Skipping row %d: missing completion and negotiator turns", line_no)
                skipped += 1
                continue

            remaining_records += 1
            for completion in completions:
                rows.append(
                    {
                        "text": (
                            f"<|system|>{prompt}</s>\n"
                            f"<|assistant|>{completion}</s>"
                        )
                    }
                )

    print(
        f"Filtered {reward_filtered} records below min_reward={min_reward}, "
        f"{remaining_records} remaining for SFT"
    )
    if skipped:
        logger.info("Also skipped %d malformed/empty JSONL rows; expanded to %d text rows", skipped, len(rows))
    if not rows:
        raise RuntimeError("No valid SFT examples found in dataset.")
    return Dataset.from_list(rows)


def train_sft(
    data_path: Path, model_id: str, output_dir: Path, min_reward: float = -50.0
) -> None:
    """Fine-tune a base model with LoRA via TRL SFTTrainer."""
    import torch
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dataset = load_sft_dataset(data_path, min_reward=min_reward)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
        max_seq_length=2048,
    )

    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU detected; training may be very slow.")

    trainer = SFTTrainer(
        model=model_id,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    logger.info("Starting SFT: model=%s, examples=%d", model_id, len(dataset))
    trainer.train()
    trainer.save_model(str(output_dir))
    logger.info("Saved SFT checkpoint to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parlay SFT training")
    parser.add_argument("--data", default="data/episodes.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-reward",
        type=float,
        default=-50.0,
        help="Skip JSONL records with total reward below this (default: -50.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train_sft(Path(args.data), args.model, Path(args.output), min_reward=args.min_reward)


if __name__ == "__main__":
    main()
