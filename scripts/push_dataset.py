"""Push episodes_v2.jsonl to HuggingFace as a public dataset."""
import argparse
import json
import os
import tempfile
from pathlib import Path

# Load .env from project root (same pattern as the rest of the app)
_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
    load_dotenv(_ROOT / ".env.local")
except ImportError:
    pass

from huggingface_hub import HfApi


def _resolve_hf_token() -> str | None:
    """HF_TOKEN, HUGGING_FACE_HUB_TOKEN, or huggingface-cli default (hub reads env)."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(key)
        if v and str(v).strip():
            return str(v).strip()
    return None


def push_dataset(jsonl_path: str, repo_id: str, token: str):
    api = HfApi(token=token)
    api.create_repo(
        repo_id, repo_type="dataset", exist_ok=True, private=False
    )
    api.upload_file(
        path_or_fileobj=jsonl_path,
        path_in_repo="episodes_v2.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Parlay negotiation episodes (140, quality-filtered)",
    )
    rows = [
        json.loads(l)
        for l in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    rewards = [r.get("reward", r.get("cumulative_reward", 0)) for r in rows]
    mean_r = sum(rewards) / max(len(rewards), 1)

    card = f"""---
license: mit
task_categories:
- reinforcement-learning
- text-generation
language: [en]
tags: [negotiation, rlhf, grpo, theory-of-mind, parlay, openenv]
---

# Parlay Negotiation Episodes

{len(rows)} quality-filtered negotiation episodes generated via Gemini
self-play for the Parlay negotiation MDP (OpenEnv-compliant environment).

Used for SFT cold-start and GRPO fine-tuning of Qwen2.5-1.5B.

## Stats
- {len(rows)} episodes | mean reward: {mean_r:.1f} | 94.3% deal rate
- 3 scenarios x 3 personas (9 combinations)
- Quality filter: min_reward > -50.0

## Fields
prompt, scenario_id, persona, conversation, reward,
deal_efficiency, tom_accuracy, drift_adapted

## Links
[Space](https://huggingface.co/spaces/sh4shv4t/Parlay) |
[GitHub](https://github.com/sh4shv4t/Parlay) |
[SFT Model](https://huggingface.co/sh4shv4t/parlay-sft-1-5b) |
[Blog](https://huggingface.co/blog/sh4shv4t/parlay)
"""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_ds_README.md",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(card)
        tmp_path = tmp.name
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Dataset card",
        )
    finally:
        os.unlink(tmp_path)
    print(f"Dataset live: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/episodes_v2.jsonl")
    parser.add_argument("--repo", default="sh4shv4t/parlay-episodes")
    default_tok = _resolve_hf_token()
    parser.add_argument(
        "--token",
        default=default_tok,
        help="Hugging Face token (or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN, or .env)",
    )
    args = parser.parse_args()
    if not args.token:
        raise ValueError(
            "No Hugging Face token found. Add HF_TOKEN=... to .env in the project root, "
            "export HF_TOKEN (or HUGGING_FACE_HUB_TOKEN), or pass --token."
        )
    push_dataset(args.data, args.repo, args.token)
