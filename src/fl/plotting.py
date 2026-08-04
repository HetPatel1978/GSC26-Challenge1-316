"""Plot helpers: accuracy/ASR vs. rounds, and a final-round summary bar chart,
across one or more saved experiment runs."""
import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


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


def plot_final_bars(run_paths: List[str], labels: List[str], out_path: str):
    final_acc = []
    final_asr = []
    for path in run_paths:
        run = load_run(path)
        final_acc.append(run["rounds"][-1]["accuracy"])
        final_asr.append(run["rounds"][-1]["asr"])

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 5))
    ax.bar(x - width / 2, final_acc, width, label="Final accuracy")
    ax.bar(x + width / 2, final_asr, width, label="Final ASR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Final-round accuracy vs. attack success rate, by attack/defense combo")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    import glob

    metrics_dir = "./results/metrics"
    plots_dir = "./results/plots"
    excluded_markers = ("smoke", "check")
    paths = sorted(glob.glob(os.path.join(metrics_dir, "*.json")))
    paths = [p for p in paths if not any(m in os.path.basename(p) for m in excluded_markers)]
    labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if paths:
        plot_accuracy_and_asr(paths, labels, os.path.join(plots_dir, "accuracy_asr_comparison.png"))
        plot_final_bars(paths, labels, os.path.join(plots_dir, "final_round_comparison.png"))
    else:
        print("No metrics found yet.")
