# Run Parlay GRPO on Hugging Face (Jobs + your credits)

## Checklist — what you actually do

1. **On your PC:** install the HF CLI → `hf auth login` (token with read + write).
2. **Money:** confirm [pre-paid credits / billing](https://huggingface.co/settings/billing) ($30 is enough for a modest A100 run if you cap time and steps).
3. **Repo:** commit and push this repo (including `scripts/hf_grpo_entry.sh`) to GitHub — or set `GITHUB_CLONE` in the job to a URL/branch that has the script.
4. **Start the job** from your terminal using the **`hf jobs run ... huggingface/trl ...`** command in [§3.1](#31-command-template). This runbook uses **`--timeout 6h`** on **`l4x1`** (see [§3.2](#32-is-6-hours-on-l4-enough)).
5. **Watch** [huggingface.co/jobs](https://huggingface.co/jobs) until it finishes or errors.
6. **Artifacts:** the trained adapter lands in `HF_GRPO_REPO` (default `sh4shv4t/parlay-grpo-1-5b`). **Plots** are generated during training and copied into **`training_plots/`** inside that upload (see [Visualizations](#visualizations--curves)).
7. **Optional — local app / README:** download `grpo_reward_curve.png` / `grpo_loss_curve.png` from the model repo's `training_plots/` (or from the job's `results/` folder) and put them in your clone under `results/` so the Space **Training Results** page can show them ([`dashboard/api.py`](../dashboard/api.py) looks for `results/grpo_reward_curve.png`).

---

This walkthrough uses assets that are **already on the Hub**:

| What | Hub ID |
|------|--------|
| Episodes JSONL (dataset) | `sh4shv4t/parlay-episodes` (file: `episodes_v2.jsonl`) |
| SFT LoRA (starting point) | `sh4shv4t/parlay-sft-1-5b` |
| Optional: upload GRPO output here | `sh4shv4t/parlay-grpo-1-5b` (default in `scripts/hf_grpo_entry.sh`) |

`training/grpo_train.py` loads the SFT **adapter** from the Hub (it looks for `adapter_config.json` and fetches the **base** model name from that file, usually `Qwen/Qwen2.5-1.5B-Instruct`).

**Requirements:** a Hugging Face account with [pre-paid credits](https://huggingface.co/pricing) and a [fine-grained or classic token](https://huggingface.co/settings/tokens) with **read** (dataset + models) and **write** (if you push the trained adapter to your model repo).

---

## 0. What you are paying for

- [Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing) is **per minute** while the job is **starting** or **running**. 1× L4 ≈ $0.80/hr. **$30** covers **~37 h** of L4 if used nonstop—always check the current table.
- The default timeout for Jobs is **short**; you must set **`--timeout`**. This guide uses **`6h`** for L4 with `GRPO_STEPS=80` and `G=2`.

---

## 1. One-time setup on your laptop

1. **Install the HF CLI** (see [Quickstart](https://huggingface.co/docs/hub/jobs-quickstart)):
   - e.g. `curl -LsSf https://hf.co/cli/install.sh | bash` (macOS/Linux) or the Windows installer.
2. **Log in**:
   - `hf auth login`
   - Paste a token with **read** and **write** to the Hub.
3. **Confirm credits** at [huggingface.co/settings/billing](https://huggingface.co/settings/billing).
4. **Open the Jobs UI** to watch runs: [huggingface.co/jobs](https://huggingface.co/jobs).

---

## 2. What the repo provides

From the **repository root** after `git clone`:

- **`scripts/hf_grpo_entry.sh`**
  - Downloads `episodes_v2.jsonl` from `sh4shv4t/parlay-episodes`
  - Runs `python -m training.grpo_train` with the Hub SFT adapter
  - Optionally runs `python -m training.push_to_hub` to upload the output folder

You only need a **GPU Linux** environment with **git**, **Python**, and **pip**; the rest is installed from `requirements-train.txt`.

---

## 3. Recommended: Hugging Face Job with the TRL image

Hugging Face documents a ready image for TRL workflows: `huggingface/trl` (see [Popular images](https://huggingface.co/docs/hub/jobs-popular-images)).

### 3.1 Command template

Run this from **your** machine (Git Bash / WSL / macOS). It starts the job in the cloud.

```bash
hf jobs run \
  --flavor l4x1 \
  --timeout 6h \
  --secrets HF_TOKEN \
  --env GITHUB_CLONE=https://github.com/sh4shv4t/Parlay.git \
  --env GRPO_STEPS=80 \
  --env GRPO_G=2 \
  --env PUSH_TO_HF=1 \
  --env HF_GRPO_REPO=sh4shv4t/parlay-grpo-1-5b \
  huggingface/trl \
  sh -c 'apt-get update -qq && apt-get install -y -qq git && git clone --depth 1 "$GITHUB_CLONE" /work && test -f /work/scripts/hf_grpo_entry.sh && cd /work && pip install -r requirements-train.txt && bash /work/scripts/hf_grpo_entry.sh'
```

---

### ⚠️ Windows / Git Bash: why `sh -c` and NOT `bash` or `/bin/bash`

| What you type | What goes wrong | Result |
|---|---|---|
| `bash -lc '...'` | Git Bash drops `-lc`; container gets `bash 'set -e;...'` | **exit 127** — treats the whole string as a filename |
| `/bin/bash -c '...'` | Git Bash converts `/bin/bash` → `C:/Program Files/Git/usr/bin/bash` | **exit 128** — OCI runtime can't find Windows path on Linux |
| **`sh -c '...'`** | Bare word — Git Bash does NOT path-resolve it; container uses its own `/bin/sh` | ✅ works |

`"$GITHUB_CLONE"` inside the single-quoted string expands **inside the container** (where `--env` set it). Your local shell never sees it — that is correct.

---

### 3.2 Is 6 hours on L4 enough?

**Yes** for these defaults: **Qwen2.5-1.5B** + SFT LoRA, **`GRPO_STEPS=80`**, **`GRPO_G=2`**. Training typically finishes in **1–3h** wall time; 6h is a safety ceiling for installs, model downloads, and plotting. If you raise steps to 120+ you can still keep 6h; only very long runs (500+ steps) need more.

### 3.3 If the job OOMs

Lower **`GRPO_G`** to `1`, then **`GRPO_STEPS`**. L4 (24 GB) handles 1.5B + LoRA with G=2 comfortably.

### 3.4 If the job times out

Increase **`--timeout`** or lower `GRPO_STEPS`.

### 3.5 If job stays Scheduling > 45 min

Cancel and retry. L4 is usually available faster than A100. Try at a different time of day.

### 3.6 After the run

- **Model:** [huggingface.co/sh4shv4t/parlay-grpo-1-5b](https://huggingface.co/sh4shv4t/parlay-grpo-1-5b) (if you pushed to that repo).
- **Curves:** also under that repo in **`training_plots/`** (bundled by `scripts/hf_grpo_entry.sh` before upload).

---

## Visualizations & curves

GRPO already **builds charts in code** (`training/grpo_train.py` → `_save_training_plots`):

| Output | Where it is written during training | What it shows |
|--------|-------------------------------------|----------------|
| Reward curve | `results/grpo_reward_curve.png` and `<output>/plots/grpo_reward.png` | Mean reward vs step (blue) |
| Loss curve | `results/grpo_loss_curve.png` and `<output>/plots/grpo_loss.png` | Training loss vs step |
| Raw TRL logs | `<output>/plots/grpo_log.json` | Full `log_history` for your own plotting |

**`scripts/hf_grpo_entry.sh`** copies those into **`<OUTPUT_DIR>/training_plots/`** before `push_to_hub`, so after a successful job you get **PNGs + JSON on the model card** next to the adapter (no extra step).

**Training Results page (`/train`):** the API exposes `plots_available.reward_curve` when **`results/grpo_reward_curve.png`** exists in the deployed app's repo. After you download the PNGs from the Hub (or copy from a job artifact), add them under `results/` in the git repo you deploy to Spaces and redeploy.

**Optional extras (not wired in repo by default):**

- **W&B / TensorBoard:** set `report_to` in `GRPOConfig` inside `training/grpo_train.py` if you want live dashboards (adds setup and secrets).
- **Eval bar chart:** after GRPO, run `python -m training.evaluate ...` locally or in a small job to regenerate `results/eval_results.json` and comparison figures (see main README).

---

## 4. Alternative: Colab (no Jobs)

Use `training/notebooks/parlay_grpo_colab.ipynb`. In the **config** cell, set:

```python
JSONL_VIA_HF = ("sh4shv4t/parlay-episodes", "episodes_v2.jsonl")
SFT_MODEL_HF = "sh4shv4t/parlay-sft-1-5b"
```

---

## 5. Environment variables reference

| Variable | Default (in `hf_grpo_entry.sh`) | Purpose |
|----------|---------------------------------|---------|
| `DATASET_ID` | `sh4shv4t/parlay-episodes` | JSONL source |
| `EPISODE_FILE` | `episodes_v2.jsonl` | File inside the dataset repo |
| `SFT_MODEL` | `sh4shv4t/parlay-sft-1-5b` | Hub LoRA to continue from |
| `GRPO_STEPS` | `120` | Training steps |
| `GRPO_G` | `4` | Group size (lower if OOM) |
| `MIN_REWARD` | `-50.0` | Skips very bad train rows |
| `OUTPUT_DIR` | `outputs/grpo_run` | Local output in the job |
| `PUSH_TO_HF` | `1` | Set to `0` to skip upload |
| `HF_GRPO_REPO` | `sh4shv4t/parlay-grpo-1-5b` | Push target |

---

## 6. Sanity checks before spending credits

- **Non-empty train split:** `grpo_train` only uses JSONL lines with `"split": "train"`. If the script prints `0 remaining for GRPO`, fix the JSONL or filters (`MIN_REWARD`).
- **Token:** `python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('sh4shv4t/parlay-episodes','episodes_v2.jsonl', repo_type='dataset'))"` on your machine should print a local path.
- **Config pre-flight:** `python scripts/check_training_config.py` (read-only) reviews env defaults for SFT/GRPO.

If anything in this file drifts (repo names, file names), align with `README.md` and `scripts/push_dataset.py` (`episodes_v2.jsonl`).
