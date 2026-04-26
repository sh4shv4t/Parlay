"""
Replot GRPO curves from a Hugging Face Job log (logging every 5 steps).
Data source: sh4shv4t/parlay-sft-1-5b → GRPO, 80 steps, G=2, L4, 2026-04-26.
Run: python scripts/plot_grpo_hf_job_curves.py
Outputs: results/grpo_reward_curve.png, results/grpo_loss_curve.png, results/grpo_train_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Steps 5,10,...,80 (TRL logs every 5 steps)
STEPS = list(range(5, 81, 5))
# 'reward' and 'loss' from job log (mean reward is GRPO combined reward)
REWARDS = [
    -4.659,
    -4.604,
    -4.764,
    -9.603,
    -4.233,
    0.5344,
    0.8288,
    -4.509,
    0.675,
    -4.456,
    -9.468,
    -9.338,
    0.3431,
    0.4913,
    -4.505,
    -9.64,
]
LOSSES = [
    8.714e-05,
    0.0001001,
    4.062e-05,
    7.433e-05,
    0.0001185,
    0.0002067,
    0.0002253,
    5.42e-05,
    4.912e-05,
    0.0001332,
    0.0001032,
    4.481e-05,
    5.264e-05,
    0.0001981,
    0.0001187,
    7.517e-05,
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [
        {"step": s, "reward": r, "loss": L} for s, r, L in zip(STEPS, REWARDS, LOSSES, strict=True)
    ]
    (out_dir / "grpo_train_metrics.json").write_text(
        json.dumps(
            {
                "meta": {
                    "source": "Hugging Face Job",
                    "sft_model": "sh4shv4t/parlay-sft-1-5b",
                    "grpo_steps": 80,
                    "grpo_g": 2,
                    "flavor": "l4x1",
                    "train_loss_final": 0.0001051,
                    "repo": "sh4shv4t/parlay-grpo-1-5b",
                },
                "points": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(STEPS, REWARDS, color="#2e6f95", marker="o", ms=4, lw=1.5)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Mean batch reward")
    ax.set_title("GRPO — mean reward (HF Job, 80 steps, G=2, L4)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 85)
    fig.tight_layout()
    fig.savefig(out_dir / "grpo_reward_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(STEPS, LOSSES, color="#8b4513", marker="o", ms=4, lw=1.5)
    ax2.set_xlabel("Global step")
    ax2.set_ylabel("Policy loss")
    ax2.set_title("GRPO — training loss (HF Job, 80 steps, G=2, L4)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 85)
    fig2.tight_layout()
    fig2.savefig(out_dir / "grpo_loss_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote {out_dir / 'grpo_reward_curve.png'}")
    print(f"Wrote {out_dir / 'grpo_loss_curve.png'}")
    print(f"Wrote {out_dir / 'grpo_train_metrics.json'}")


if __name__ == "__main__":
    main()
