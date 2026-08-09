"""Multi-seed check for DBA vs ATA and the adaptive attacker vs ATA, 3 seeds
each (30 rounds) -- same treatment run_multiseed.py already gave BadNets vs
FedAvg/ATA, extended to the other two key ATA results in the README so every
headline ATA number has an error bar, not just BadNets."""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.fl.data import load_cifar10, reserve_root_set, IndexedSubset
from src.fl.experiment import ExperimentConfig, run_experiment, get_device
from src.fl.models import CNNCifar
from src.defenses.ata import make_ata_aggregate
from src.attacks.adaptive import make_adaptive_fltrust_adapt_fn_factory

SEEDS = [42, 43, 44]
RESULTS_DIR = "./results/multiseed_metrics"


def run_dba_ata(seed: int):
    cfg = ExperimentConfig(
        name=f"ata_dba_seed{seed}",
        num_clients=20,
        dirichlet_alpha=0.5,
        fraction_fit=0.5,
        num_rounds=30,
        local_epochs=2,
        attack_type="dba",
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


def run_adaptive_ata(seed: int):
    cfg = ExperimentConfig(
        name=f"adaptive_ata_seed{seed}",
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
    adapt_fn_factory = make_adaptive_fltrust_adapt_fn_factory(
        CNNCifar, device, local_epochs=cfg.local_epochs, lr=cfg.lr, batch_size=cfg.batch_size, target_cos_sim=0.9
    )
    return run_experiment(cfg, aggregate_fn, adapt_fn_factory=adapt_fn_factory)


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
    print(f"Running DBA vs ATA over seeds {SEEDS}...")
    dba_histories = [run_dba_ata(s) for s in SEEDS]

    print(f"Running adaptive attacker vs ATA over seeds {SEEDS}...")
    adaptive_histories = [run_adaptive_ata(s) for s in SEEDS]

    summary = {
        "seeds": SEEDS,
        "dba_ata": summarize(dba_histories),
        "adaptive_ata": summarize(adaptive_histories),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "summary_dba_adaptive.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 64)
    for label, s in [("DBA vs ATA", summary["dba_ata"]), ("Adaptive attacker vs ATA", summary["adaptive_ata"])]:
        print(
            f"{label}: accuracy = {s['acc_mean']:.4f} +/- {s['acc_std']:.4f}, "
            f"ASR = {s['asr_mean']:.4f} +/- {s['asr_std']:.4f}  (seeds {SEEDS})"
        )
    print(f"Saved summary to {out_path}")
    print("=" * 64)
