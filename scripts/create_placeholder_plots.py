"""Generate placeholder PNG files so README image links don't break."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

results = Path("results")
results.mkdir(exist_ok=True)

for fname, title in [
    ("grpo_reward_curve.png", "GRPO Reward Curve — replace after training"),
    ("grpo_loss_curve.png", "GRPO Loss Curve — replace after training"),
    ("comparison.png", "Baseline vs Trained — replace after eval"),
    ("sft_loss_curve.png", "SFT Loss Curve — replace after training"),
]:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(
        0.5,
        0.5,
        title,
        ha="center",
        va="center",
        fontsize=13,
        color="gray",
        transform=ax.transAxes,
    )
    ax.set_title(f"Parlay ◈ — {title}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    plt.tight_layout()
    plt.savefig(results / fname, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Created: results/{fname}")

print("Done. Replace with real plots after training.")
