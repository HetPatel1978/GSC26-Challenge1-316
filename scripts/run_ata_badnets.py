"""ATA (this repo's own combined defense) against BadNets. Same setup as
run_badnets_fltrust.py so it's directly comparable to FLTrust alone."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.fl.data import load_cifar10, reserve_root_set, IndexedSubset
from src.fl.experiment import ExperimentConfig, run_experiment, get_device
from src.fl.models import CNNCifar
from src.defenses.ata import make_ata_aggregate

cfg = ExperimentConfig(
    name="ata_badnets",
    num_clients=20,
    dirichlet_alpha=0.5,
    fraction_fit=0.5,
    num_rounds=30,
    local_epochs=2,
    attack_type="badnets",
    fraction_malicious=0.2,
    poison_rate=0.5,
    target_label=0,
    root_size=500,
)

if __name__ == "__main__":
    train_set, _ = load_cifar10(cfg.data_root)
    root_idx, _ = reserve_root_set(len(train_set), cfg.root_size, cfg.seed)
    root_loader = DataLoader(IndexedSubset(train_set, root_idx), batch_size=cfg.batch_size, shuffle=True)
    device = get_device()

    aggregate_fn = make_ata_aggregate(root_loader, CNNCifar, device, local_epochs=cfg.local_epochs, lr=cfg.lr)
    run_experiment(cfg, aggregate_fn)
