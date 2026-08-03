"""Plot helpers: accuracy/ASR vs. rounds for one or more saved experiment runs."""
import json
import os
from typing import List

import matplotlib.pyplot as plt


def load_run(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_accuracy_and_asr(run_paths: List[str], labels: List[str], out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for path, label in zip(run_paths, labels):
        run = load_run(path)
        rounds = [r["round"] for r in run["rounds"]]
        acc = [r["accuracy"] for r in run["rounds"]]
        asr = [r["asr"] for r in run["rounds"]]
        axes[0].plot(rounds, acc, marker="o", markersize=3, label=label)
        axes[1].plot(rounds, asr, marker="o", markersize=3, label=label)

    axes[0].set_title("Main-task accuracy vs. rounds")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Attack success rate vs. rounds")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("ASR")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    import glob

    metrics_dir = "./results/metrics"
    plots_dir = "./results/plots"
    paths = sorted(glob.glob(os.path.join(metrics_dir, "*.json")))
    paths = [p for p in paths if "smoke_test" not in p]
    labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if paths:
        plot_accuracy_and_asr(paths, labels, os.path.join(plots_dir, "accuracy_asr_comparison.png"))
    else:
        print("No metrics found yet.")
