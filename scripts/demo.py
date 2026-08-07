"""Quick demo for judges: a scaled-down 5-round BadNets-vs-FLTrust simulation
that prints round-by-round accuracy/ASR straight to the terminal and saves a
plot, in well under 2 minutes. For the full 30-round, 20-client results this
is a fast preview of, see the README's results table and
scripts/run_fedavg_baseline.py / run_dba_fltrust.py etc."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.fl.data import load_cifar10, reserve_root_set, IndexedSubset
from src.fl.experiment import ExperimentConfig, run_experiment, get_device
from src.fl.models import CNNCifar
from src.defenses.aggregation import make_fltrust_aggregate
from src.fl.plotting import plot_accuracy_and_asr

cfg = ExperimentConfig(
    name="demo_badnets_fltrust",
    num_clients=10,
    dirichlet_alpha=0.5,
    fraction_fit=0.5,
    num_rounds=5,
    local_epochs=1,
    attack_type="badnets",
    fraction_malicious=0.2,
    poison_rate=0.5,
    target_label=0,
    root_size=200,
    results_dir="./results/demo_metrics",
)

if __name__ == "__main__":
    start = time.time()
    print("=" * 64)
    print("QUICK DEMO -- BadNets attack vs FLTrust defense (5 rounds, 10 clients)")
    print("=" * 64)

    train_set, _ = load_cifar10(cfg.data_root)
    root_idx, _ = reserve_root_set(len(train_set), cfg.root_size, cfg.seed)
    root_loader = DataLoader(IndexedSubset(train_set, root_idx), batch_size=cfg.batch_size, shuffle=True)
    device = get_device()
    print(f"Device: {device}")

    aggregate_fn = make_fltrust_aggregate(root_loader, CNNCifar, device, local_epochs=cfg.local_epochs, lr=cfg.lr)
    history = run_experiment(cfg, aggregate_fn)

    out_path = os.path.join("./results/plots", "demo.png")
    plot_accuracy_and_asr(
        [os.path.join(cfg.results_dir, f"{cfg.name}.json")],
        ["BadNets vs FLTrust (demo)"],
        out_path,
    )

    elapsed = time.time() - start
    print("=" * 64)
    print(f"Done in {elapsed:.1f}s -- final round: accuracy={history[-1]['accuracy']:.4f} asr={history[-1]['asr']:.4f}")
    print(f"Plot saved to {out_path}")
    print("For the full 30-round results across every attack/defense combo, see the README.")
    print("=" * 64)
