"""Plot helpers: accuracy/ASR vs. rounds, and a final-round summary bar
chart, across one or more saved experiment runs. Grouped by attack family
with consistent per-defense colors (DEFENSE_COLORS below) so a judge can
read the comparison at a glance instead of parsing snake_case filenames off
one crowded legend."""
import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_run(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_accuracy_and_asr(
    run_paths: List[str], labels: List[str], out_path: str, colors: Optional[List[Optional[str]]] = None
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = colors or [None] * len(run_paths)

    for path, label, color in zip(run_paths, labels, colors):
        run = load_run(path)
        rounds = [r["round"] for r in run["rounds"]]
        acc = [r["accuracy"] for r in run["rounds"]]
        asr = [r["asr"] for r in run["rounds"]]
        axes[0].plot(rounds, acc, marker="o", markersize=3, label=label, color=color)
        axes[1].plot(rounds, asr, marker="o", markersize=3, label=label, color=color)

    axes[0].set_title("Main-task accuracy vs. rounds")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Attack success rate vs. rounds")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("ASR")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def plot_final_bars(
    run_paths: List[str], labels: List[str], out_path: str, title: str = "Final-round accuracy vs. attack success rate"
):
    final_acc = []
    final_asr = []
    for path in run_paths:
        run = load_run(path)
        final_acc.append(run["rounds"][-1]["accuracy"])
        final_asr.append(run["rounds"][-1]["asr"])

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(1.7 * len(labels), 8), 5.5))
    ax.bar(x - width / 2, final_acc, width, label="Final accuracy", color="#4c72b0")
    ax.bar(x + width / 2, final_asr, width, label="Final ASR", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


# Defense -> color, held constant across every figure below so the same
# defense reads as the same color whether it's fighting BadNets, DBA, or
# the adaptive attacker.
DEFENSE_COLORS = {
    "FedAvg (no defense)": "#7f7f7f",
    "Krum": "#1f77b4",
    "Multi-Krum": "#17becf",
    "FLAME": "#9467bd",
    "FLTrust": "#2ca02c",
    "ATA (ours)": "#d62728",
}


def _defense_color(label: str) -> Optional[str]:
    return DEFENSE_COLORS.get(label.split(" + ")[-1])


# (metrics filename stem, human-readable label) per figure. A stem missing
# from results/metrics/ (e.g. a run not evaluated yet) is skipped rather
# than erroring, so this file works incrementally as runs land.
BADNETS_RUNS = [
    ("fedavg_baseline", "BadNets + FedAvg (no defense)"),
    ("krum_defense", "BadNets + Krum"),
    ("multikrum_defense", "BadNets + Multi-Krum"),
    ("flame_badnets", "BadNets + FLAME"),
    ("badnets_fltrust", "BadNets + FLTrust"),
    ("ata_badnets", "BadNets + ATA (ours)"),
]

DBA_RUNS = [
    ("dba_fedavg", "DBA + FedAvg (no defense)"),
    ("dba_fltrust", "DBA + FLTrust"),
    ("flame_dba", "DBA + FLAME"),
    ("ata_dba", "DBA + ATA (ours)"),
]

# The key stress test: naive vs. defense-aware attacker, FLTrust vs. ATA.
ADAPTIVE_RUNS = [
    ("badnets_fltrust", "Naive attacker + FLTrust"),
    ("adaptive_badnets_fltrust", "Adaptive attacker + FLTrust"),
    ("ata_badnets", "Naive attacker + ATA (ours)"),
    ("adaptive_badnets_ata", "Adaptive attacker + ATA (ours)"),
]

SUMMARY_RUNS = BADNETS_RUNS + DBA_RUNS + [
    ("adaptive_badnets_fltrust", "Adaptive attacker + FLTrust"),
    ("adaptive_badnets_ata", "Adaptive attacker + ATA (ours)"),
]


def _build(metrics_dir: str, runs):
    paths, labels, colors = [], [], []
    for stem, label in runs:
        path = os.path.join(metrics_dir, f"{stem}.json")
        if not os.path.exists(path):
            continue
        paths.append(path)
        labels.append(label)
        colors.append(_defense_color(label))
    return paths, labels, colors


if __name__ == "__main__":
    metrics_dir = "./results/metrics"
    plots_dir = "./results/plots"

    badnets_paths, badnets_labels, badnets_colors = _build(metrics_dir, BADNETS_RUNS)
    dba_paths, dba_labels, dba_colors = _build(metrics_dir, DBA_RUNS)
    adaptive_paths, adaptive_labels, adaptive_colors = _build(metrics_dir, ADAPTIVE_RUNS)
    summary_paths, summary_labels, _ = _build(metrics_dir, SUMMARY_RUNS)

    if badnets_paths:
        plot_accuracy_and_asr(
            badnets_paths, badnets_labels, os.path.join(plots_dir, "badnets_comparison.png"), badnets_colors
        )
    if dba_paths:
        plot_accuracy_and_asr(dba_paths, dba_labels, os.path.join(plots_dir, "dba_comparison.png"), dba_colors)
    if adaptive_paths:
        plot_accuracy_and_asr(
            adaptive_paths,
            adaptive_labels,
            os.path.join(plots_dir, "adaptive_attacker_comparison.png"),
            adaptive_colors,
        )
    if summary_paths:
        plot_final_bars(summary_paths, summary_labels, os.path.join(plots_dir, "final_summary_bars.png"))
    if not (badnets_paths or dba_paths):
        print("No metrics found yet.")
