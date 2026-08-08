"""Quick demo for judges: a scaled-down 15-round BadNets attack run
side-by-side under FedAvg (no defense) and ATA (this repo's own combined
defense), on the identical malicious-client setup, so the contrast between
"backdoor compounding unchecked" and "backdoor getting suppressed" is
visible without waiting for the full 30-round/20-client results. Prints a
one-line explanation after every round for both runs, not just raw numbers.
For the full-scale results across every attack/defense combo, see the
README's results table and scripts/run_*.py."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.fl.data import load_cifar10, reserve_root_set, IndexedSubset
from src.fl.experiment import ExperimentConfig, run_experiment, get_device
from src.fl.models import CNNCifar
from src.defenses.aggregation import fedavg_aggregate
from src.defenses.ata import make_ata_aggregate
from src.fl.plotting import plot_accuracy_and_asr

NUM_ROUNDS = 15

cfg_fedavg = ExperimentConfig(
    name="demo_badnets_fedavg",
    num_clients=10,
    dirichlet_alpha=0.5,
    fraction_fit=0.5,
    num_rounds=NUM_ROUNDS,
    local_epochs=1,
    attack_type="badnets",
    fraction_malicious=0.2,
    poison_rate=0.5,
    target_label=0,
    results_dir="./results/demo_metrics",
)

cfg_ata = ExperimentConfig(
    name="demo_badnets_ata",
    num_clients=10,
    dirichlet_alpha=0.5,
    fraction_fit=0.5,
    num_rounds=NUM_ROUNDS,
    local_epochs=1,
    attack_type="badnets",
    fraction_malicious=0.2,
    poison_rate=0.5,
    target_label=0,
    root_size=200,
    results_dir="./results/demo_metrics",
)


def annotate_fedavg(round_record: dict) -> str:
    asr = round_record["asr"]
    if asr < 0.1:
        return "backdoor not triggered yet -- too early for the trigger pattern to have been learned"
    if asr < 0.5:
        return "backdoor taking hold -- FedAvg has no way to tell this update apart from a clean one"
    return "backdoor fully compounding -- nothing in FedAvg ever filters or dampens the malicious updates"


def annotate_ata(round_record: dict) -> str:
    asr = round_record["asr"]
    if asr < 0.15:
        return "ATA's trust scoring + sign correction is holding the line -- suppressed"
    if asr < 0.4:
        return "some malicious signal getting through -- ATA is dampening it, not eliminating it"
    return "ATA under pressure this round -- still far below FedAvg's undefended trajectory (see below)"


if __name__ == "__main__":
    start = time.time()
    print("=" * 72)
    print(f"QUICK DEMO -- BadNets attack, FedAvg vs ATA ({NUM_ROUNDS} rounds, 10 clients)")
    print("=" * 72)

    device = get_device()
    print(f"Device: {device}")

    print("\n--- Run 1/2: FedAvg (no defense) -----------------------------------")
    fedavg_history = run_experiment(cfg_fedavg, fedavg_aggregate)
    print("\nAnnotated recap -- FedAvg:")
    for r in fedavg_history:
        print(f"  round {r['round']:02d}: accuracy={r['accuracy']:.3f}  ASR={r['asr']:.3f}  -- {annotate_fedavg(r)}")

    print("\n--- Run 2/2: ATA (this repo's combined defense) ---------------------")
    train_set, _ = load_cifar10(cfg_ata.data_root)
    root_idx, _ = reserve_root_set(len(train_set), cfg_ata.root_size, cfg_ata.seed)
    root_loader = DataLoader(IndexedSubset(train_set, root_idx), batch_size=cfg_ata.batch_size, shuffle=True)
    ata_aggregate_fn = make_ata_aggregate(root_loader, CNNCifar, device, local_epochs=cfg_ata.local_epochs, lr=cfg_ata.lr)
    ata_history = run_experiment(cfg_ata, ata_aggregate_fn)
    print("\nAnnotated recap -- ATA:")
    for r in ata_history:
        print(f"  round {r['round']:02d}: accuracy={r['accuracy']:.3f}  ASR={r['asr']:.3f}  -- {annotate_ata(r)}")

    out_path = os.path.join("./results/plots", "demo.png")
    plot_accuracy_and_asr(
        [
            os.path.join(cfg_fedavg.results_dir, f"{cfg_fedavg.name}.json"),
            os.path.join(cfg_ata.results_dir, f"{cfg_ata.name}.json"),
        ],
        ["BadNets + FedAvg (no defense)", "BadNets + ATA (ours)"],
        out_path,
        colors=["#7f7f7f", "#d62728"],
    )

    elapsed = time.time() - start
    print("=" * 72)
    print(
        f"Done in {elapsed:.1f}s -- final round: "
        f"FedAvg acc={fedavg_history[-1]['accuracy']:.3f} asr={fedavg_history[-1]['asr']:.3f}  |  "
        f"ATA acc={ata_history[-1]['accuracy']:.3f} asr={ata_history[-1]['asr']:.3f}"
    )
    print(f"Plot saved to {out_path}")
    print("For the full 30-round, 20-client results across every attack/defense combo, see the README.")
    print("=" * 72)
