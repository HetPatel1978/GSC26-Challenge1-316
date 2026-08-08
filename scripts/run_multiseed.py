"""Multi-seed robustness check: BadNets vs FedAvg and BadNets vs ATA, 3 seeds
each (30 rounds), reporting mean +/- std of final-round accuracy and ASR.
Every other result in this repo is a single-seed point estimate; this is
the one comparison -- the accuracy/robustness claim ATA is meant to
establish -- that gets an actual error bar instead. Requires
src.fl.experiment.run_experiment to seed torch's RNG (added alongside ATA)
so each seed varies model init and DataLoader shuffling too, not just data
partitioning and attacker/client selection."""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.fl.data import load_cifar10, reserve_root_set, IndexedSubset
from src.fl.experiment import ExperimentConfig, run_experiment, get_device
from src.fl.models import CNNCifar
from src.defenses.aggregation import fedavg_aggregate
from src.defenses.ata import make_ata_aggregate

SEEDS = [42, 43, 44]
RESULTS_DIR = "./results/multiseed_metrics"


def run_fedavg(seed: int):
    cfg = ExperimentConfig(
        name=f"fedavg_badnets_seed{seed}",
        num_clients=20,
        dirichlet_alpha=0.5,
        fraction_fit=0.5,
        num_rounds=30,
        local_epochs=2,
        attack_type="badnets",
        fraction_malicious=0.2,
        poison_rate=0.5,
        target_label=0,
        seed=seed,
        results_dir=RESULTS_DIR,
    )
    return run_experiment(cfg, fedavg_aggregate)


def run_ata(seed: int):
    cfg = ExperimentConfig(
        name=f"ata_badnets_seed{seed}",
        num_clients=20,
        dirichlet_alpha=0.5,
        fraction_fit=0.5,
        num_rounds=30,
        local_epochs=2,
        attack_type="badnets",
        fraction_malicious=0.2,
        poison_rate=0.5,
        target_label=0,
        seed=seed,
        root_size=500,
        results_dir=RESULTS_DIR,
    )
    train_set, _ = load_cifar10(cfg.data_root)
    root_idx, _ = reserve_root_set(len(train_set), cfg.root_size, cfg.seed)
    root_loader = DataLoader(IndexedSubset(train_set, root_idx), batch_size=cfg.batch_size, shuffle=True)
    device = get_device()
    aggregate_fn = make_ata_aggregate(root_loader, CNNCifar, device, local_epochs=cfg.local_epochs, lr=cfg.lr)
    return run_experiment(cfg, aggregate_fn)


def summarize(histories):
    accs = [h[-1]["accuracy"] for h in histories]
    asrs = [h[-1]["asr"] for h in histories]
    return {
        "acc_mean": statistics.mean(accs),
        "acc_std": statistics.stdev(accs),
        "asr_mean": statistics.mean(asrs),
        "asr_std": statistics.stdev(asrs),
        "acc_per_seed": accs,
        "asr_per_seed": asrs,
    }


if __name__ == "__main__":
    print(f"Running BadNets vs FedAvg over seeds {SEEDS}...")
    fedavg_histories = [run_fedavg(s) for s in SEEDS]

    print(f"Running BadNets vs ATA over seeds {SEEDS}...")
    ata_histories = [run_ata(s) for s in SEEDS]

    summary = {
        "seeds": SEEDS,
        "fedavg_badnets": summarize(fedavg_histories),
        "ata_badnets": summarize(ata_histories),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 64)
    for label, s in [("BadNets vs FedAvg", summary["fedavg_badnets"]), ("BadNets vs ATA", summary["ata_badnets"])]:
        print(
            f"{label}: accuracy = {s['acc_mean']:.4f} +/- {s['acc_std']:.4f}, "
            f"ASR = {s['asr_mean']:.4f} +/- {s['asr_std']:.4f}  (seeds {SEEDS})"
        )
    print(f"Saved summary to {out_path}")
    print("=" * 64)
