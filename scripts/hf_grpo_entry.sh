#!/usr/bin/env bash
# Run from the Parlay repository root (the folder that contains training/ and parlay_env/).
# Intended for Linux GPU jobs (Hugging Face Jobs, RunPod, etc.).
#
# Usage (after: git clone ... && cd Parlay && pip install -r requirements-train.txt):
#   export HF_TOKEN=...          # read private assets + push (required if PUSH_TO_HF=1)
#   export GRPO_STEPS=120 GRPO_G=4
#   bash scripts/hf_grpo_entry.sh
#
# See training/GRPO_HF_RUNBOOK.md for a full walkthrough.
set -euo pipefail
export PYTHONUNBUFFERED=1

: "${DATASET_ID:=sh4shv4t/parlay-episodes}"
: "${EPISODE_FILE:=episodes_v2.jsonl}"
: "${SFT_MODEL:=sh4shv4t/parlay-sft-1-5b}"
: "${GRPO_STEPS:=120}"
: "${GRPO_G:=4}"
: "${MIN_REWARD:=-50.0}"
: "${OUTPUT_DIR:=outputs/grpo_run}"
# Set to 0 to skip push (e.g. smoke test)
: "${PUSH_TO_HF:=1}"
# Model repo to upload the GRPO output folder to
: "${HF_GRPO_REPO:=sh4shv4t/parlay-grpo-1-5b}"

if [[ ! -f "training/grpo_train.py" ]]; then
  echo "Run this script from the Parlay repo root (training/grpo_train.py not found). pwd=$(pwd)" >&2
  exit 1
fi

echo "==> Downloading ${EPISODE_FILE} from dataset ${DATASET_ID} ..."
export DATASET_ID EPISODE_FILE
JSONL_PATH=$(
  python -c "import os
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id=os.environ['DATASET_ID'],
    filename=os.environ['EPISODE_FILE'],
    repo_type='dataset',
))"
)
echo "    JSONL: ${JSONL_PATH}"

mkdir -p "$(dirname "$OUTPUT_DIR")"
OUT_ABS="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")"

echo "==> GRPO: SFT=${SFT_MODEL} steps=${GRPO_STEPS} G=${GRPO_G} out=${OUT_ABS}"
python -m training.grpo_train \
  --model "${SFT_MODEL}" \
  --data "${JSONL_PATH}" \
  --output "${OUT_ABS}" \
  --steps "${GRPO_STEPS}" \
  --g "${GRPO_G}" \
  --min-reward "${MIN_REWARD}"

# Bundle Matplotlib curves + TRL log JSON into the model folder so one Hub upload includes visualizations.
echo "==> Collecting training plots under ${OUT_ABS}/training_plots/ ..."
TP="${OUT_ABS}/training_plots"
mkdir -p "${TP}"
for f in results/grpo_reward_curve.png results/grpo_loss_curve.png; do
  if [[ -f "$f" ]]; then
    cp -f "$f" "${TP}/"
    echo "    + ${f}"
  fi
done
if [[ -d "${OUT_ABS}/plots" ]]; then
  shopt -s nullglob
  for f in "${OUT_ABS}/plots/"*.png "${OUT_ABS}/plots/"*.json; do
    [[ -e "$f" ]] || continue
    cp -f "$f" "${TP}/"
    echo "    + ${f}"
  done
  shopt -u nullglob
fi
if [[ ! -f "${TP}/grpo_reward_curve.png" && ! -f "${TP}/grpo_reward.png" ]]; then
  echo "    (warning: no reward plot in training_plots — check logs for empty log_history or plot errors)"
fi

if [[ "${PUSH_TO_HF}" == "1" || "${PUSH_TO_HF}" == "true" ]]; then
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    echo "PUSH_TO_HF is set but neither HF_TOKEN nor HUGGINGFACE_HUB_TOKEN is set." >&2
    exit 1
  fi
  # push_to_hub.py reads HF_TOKEN; Jobs often set HUGGINGFACE_HUB_TOKEN
  export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
  echo "==> Pushing to https://huggingface.co/${HF_GRPO_REPO} ..."
  export HF_REPO_ID="${HF_GRPO_REPO}"
  python -m training.push_to_hub --model "${OUT_ABS}" --repo "${HF_GRPO_REPO}"
else
  echo "==> PUSH_TO_HF disabled; model saved at ${OUT_ABS}"
fi

echo "==> Done."
