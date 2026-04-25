#!/usr/bin/env python3
"""Build sft_qwen_colab.ipynb (run from this directory: python build_sft_qwen_colab.py)."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT = Path(__file__).with_name("sft_qwen_colab.ipynb")

CELLS: list[tuple[str, str]] = []


def add_md(s: str) -> None:
    CELLS.append(("markdown", s.rstrip() + "\n"))


def add_code(s: str) -> None:
    CELLS.append(("code", s.rstrip() + "\n"))


# ——— Notebook content (Parlay SFT on Colab T4) ———

add_md("""# ◈ Parlay — SFT Training on Qwen2.5-1.5B
## Teaching a language model to negotiate

This notebook fine-tunes **Qwen2.5-1.5B-Instruct** on Parlay negotiation
episodes using **Supervised Fine-Tuning (SFT)** with LoRA via
[Unsloth](https://github.com/unslothai/unsloth).

**What you need**
- A free Google Colab **T4** GPU
- Colab secret `HF_TOKEN` (write access)

**What this does**
1. Installs stack + checks GPU
2. Loads `sh4shv4t/parlay-episodes` from the Hub
3. Formats to ChatML, trains with LoRA (r=16)
4. Plots, evaluates before/after, pushes adapter to `sh4shv4t/parlay-negotiator`

**Runtime:** ~25 min on T4 · **Cost:** $0 (free tier).""")

add_md("## Step 1 — Install dependencies")

add_code("""%%capture
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    "--quiet",
], check=True)
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "trl>=0.8.6", "peft>=0.10.0", "accelerate>=0.28.0",
    "datasets>=2.18.0", "huggingface-hub>=0.22.0",
    "bitsandbytes>=0.43.0", "xformers", "--quiet",
], check=True)
print("OK: dependencies")""")

add_md("## Step 2 — GPU + config")

add_code("""import os, json
import torch
from google.colab import userdata

assert torch.cuda.is_available(), "Use Runtime → Change runtime type → T4 GPU"
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9, "GB VRAM")
print("torch", torch.__version__, "cuda", torch.version.cuda)

HF_TOKEN = userdata.get("HF_TOKEN")
DATASET_ID = "sh4shv4t/parlay-episodes"
MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct"
OUTPUT_REPO = "sh4shv4t/parlay-negotiator"
LORA_R, LORA_ALPHA = 16, 32
MAX_SEQ_LEN = 1024
SFT_EPOCHS = 2
BATCH_SIZE, GRAD_ACCUM = 2, 4
LEARNING_RATE, WARMUP_RATIO = 2e-4, 0.1
MIN_REWARD_KEEP = 0.25""")

add_md("## Step 3 — Load dataset")

add_code("""from datasets import load_dataset
import pandas as pd

raw = load_dataset(DATASET_ID, token=HF_TOKEN)
df = raw["train"].to_pandas()
print(len(df), "rows", list(df.columns))
if "reward" in df.columns:
    print(df["reward"].describe())""")

add_md("## Step 4 — Format episodes (ChatML)")

add_code("""from datasets import Dataset

SYSTEM = \"\"\"You negotiate in a B2B deal. Respond with JSON only:
{"utterance": str, "offer_amount": number|null, "tactical_move": str|null}\"\"\"

def format_episode(row):
    conv = row.get("conversation", row.get("messages", []))
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except Exception:
            return None
    if not isinstance(conv, list) or len(conv) < 2:
        return None
    msgs = [{"role": "system", "content": SYSTEM}]
    for t in conv:
        role = t.get("role", t.get("speaker", ""))
        txt = t.get("content", t.get("text", t.get("utterance", "")))
        if not txt:
            continue
        if role in ("player", "agent", "user", "human"):
            msgs.append({"role": "user", "content": str(txt)})
        else:
            msgs.append({"role": "assistant", "content": str(txt)})
    if len(msgs) < 3:
        return None
    eff = float(row.get("deal_efficiency", 0) or 0)
    rew = float(row.get("reward", 0) or 0)
    if eff < MIN_REWARD_KEEP and rew < -50:
        return None
    return {
        "messages": msgs,
        "deal_efficiency": eff,
        "reward": rew,
    }

out = [format_episode(dict(row)) for row in raw["train"]]
formatted = [x for x in out if x is not None]
print("episodes", len(formatted), "kept of", len(raw["train"]))

split = int(0.9 * len(formatted))
train_data = Dataset.from_list(formatted[:split])
eval_data  = Dataset.from_list(formatted[split:])""")

add_md("## Step 5 — Model + LoRA (Unsloth)")

add_code("""from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
    token=HF_TOKEN,
)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)""")

add_md("## Step 6 — Tokenize")

add_code("""def tok_one(row):
    t = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": t}

tr = train_data.map(tok_one, remove_columns=train_data.column_names)
ev = eval_data.map(tok_one, remove_columns=eval_data.column_names)""")

add_md("## Step 7 — SFT with TRL")

add_code("""import torch
from trl import SFTTrainer, SFTConfig

args = SFTConfig(
    output_dir="checkpoints/sft",
    num_train_epochs=SFT_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    seed=42,
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tr,
    eval_dataset=ev,
    args=args,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
)
trainer.train()""")

add_md("## Step 8 — Plots + push adapter")

add_code("""import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import HfApi

os.makedirs("results", exist_ok=True)
log = trainer.state.log_history
losses = [x["loss"] for x in log if "loss" in x]
steps  = [x["step"] for x in log if "loss" in x]
if steps:
    plt.figure(figsize=(8,3))
    plt.plot(steps, losses, color="#c9a84c")
    plt.title("SFT loss")
    plt.savefig("results/sft_loss_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

ADAPTER = "checkpoints/sft_adapter"
model.save_pretrained(ADAPTER)
tokenizer.save_pretrained(ADAPTER)
api = HfApi()
api.create_repo(OUTPUT_REPO, exist_ok=True, token=HF_TOKEN, repo_type="model")
api.upload_folder(folder_path=ADAPTER, repo_id=OUTPUT_REPO, token=HF_TOKEN)
print("https://huggingface.co/" + OUTPUT_REPO)""")

if __name__ == "__main__":
    nb: dict = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "cells": [],
    }
    for kind, src in CELLS:
        cid = str(uuid.uuid4())
        lines = [x + "\n" for x in src.splitlines()]
        if kind == "markdown":
            cell = {
                "cell_type": "markdown",
                "id": cid,
                "metadata": {},
                "source": lines,
            }
        else:
            cell = {
                "cell_type": "code",
                "id": cid,
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": lines,
            }
        nb["cells"].append(cell)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Wrote", OUT)
