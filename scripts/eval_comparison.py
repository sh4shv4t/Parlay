"""Compare baseline, Gemini, and GRPO JSON summaries."""
import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _row(label: str, data: dict) -> str:
    return (
        f"| {label} | {data.get('avg_reward', 0):.3f} | "
        f"{_fmt_pct(float(data.get('deal_rate', 0)))} | "
        f"{float(data.get('avg_efficiency', 0)):.3f} | "
        f"{float(data.get('avg_tom_accuracy', 0)):.3f} | "
        f"{int(data.get('bluffs_caught', 0))} |"
    )


def _save_chart(baseline: dict, gemini: dict, grpo: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["avg_reward", "deal_rate", "avg_efficiency", "avg_tom_accuracy"]
    names = ["Random", "Gemini", "GRPO"]
    series = [baseline, gemini, grpo]

    x = range(len(labels))
    width = 0.22

    plt.figure(figsize=(10, 5))
    for idx, name in enumerate(names):
        vals = [float(series[idx].get(k, 0.0)) for k in labels]
        plt.bar([p + (idx - 1) * width for p in x], vals, width=width, label=name)

    plt.xticks(list(x), labels)
    plt.ylabel("Metric value")
    plt.title("Parlay Baseline vs Gemini vs GRPO")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evaluation result JSON files")
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--gemini-results", required=True)
    parser.add_argument("--grpo-results", required=True)
    args = parser.parse_args()

    baseline = _load(args.baseline_results)
    gemini = _load(args.gemini_results)
    grpo = _load(args.grpo_results)

    lines = [
        "| Model | avg_reward | deal_rate | avg_efficiency | avg_tom_accuracy | bluffs_caught |",
        "|---|---:|---:|---:|---:|---:|",
        _row("Random baseline", baseline),
        _row("Gemini baseline", gemini),
        _row("GRPO", grpo),
    ]
    table = "\n".join(lines)
    print(table)

    chart_path = Path("results/comparison.png")
    _save_chart(baseline, gemini, grpo, chart_path)
    print(f"\nSaved chart: {chart_path.resolve()}")


if __name__ == "__main__":
    main()
