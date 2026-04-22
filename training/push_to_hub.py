"""
Upload the trained GRPO model to HuggingFace Hub.

Usage:
    python -m training.push_to_hub \
        --model models/parlay-grpo \
        --repo your-username/parlay-negotiator
"""
import argparse
import logging
import os

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO  = os.getenv("HF_REPO_ID", "")


def push_to_hub(model_path: str, repo_id: str, private: bool = False) -> str:
    """
    Push trained model to HuggingFace Hub.

    Args:
        model_path: Local path to the trained model directory.
        repo_id:    HF Hub repo ID (e.g. "username/parlay-negotiator").
        private:    Whether to create a private repo.

    Returns:
        URL of the uploaded model.
    """
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set.")
    if not repo_id:
        raise ValueError("repo_id is required (or set HF_REPO_ID env var).")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import HfApi
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Install: pip install transformers huggingface-hub peft") from exc

    api = HfApi(token=HF_TOKEN)

    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        logger.info(f"Repo {repo_id} ready")
    except Exception as exc:
        logger.warning(f"Repo creation skipped: {exc}")

    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Parlay GRPO-trained negotiation model",
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Model uploaded: {url}")
    print(f"\nModel uploaded successfully!")
    print(f"  URL: {url}")
    print(f"  Repo: {repo_id}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Push Parlay model to HF Hub")
    parser.add_argument("--model", default="models/parlay-grpo")
    parser.add_argument("--repo", default=HF_REPO, help="HF Hub repo ID")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.repo:
        print("ERROR: Provide --repo or set HF_REPO_ID environment variable.")
        return

    push_to_hub(args.model, args.repo, args.private)


if __name__ == "__main__":
    main()
