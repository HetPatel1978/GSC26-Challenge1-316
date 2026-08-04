"""DBA under FLTrust: server holds a small disjoint trusted root set (200
clean samples, carved out of the training pool before client partitioning)
and trust-weights + norm-rescales client updates by cosine similarity to its
own root-trained update."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.fl.data import load_cifar10, reserve_root_set, IndexedSubset
from src.fl.experiment import ExperimentConfig, run_experiment, get_device
from src.fl.models import CNNCifar
from src.defenses.aggregation import make_fltrust_aggregate

cfg = ExperimentConfig(
    name="dba_fltrust",
    num_clients=20,
    dirichlet_alpha=0.5,
    fraction_fit=0.5,
    num_rounds=30,
    local_epochs=2,
    attack_type="dba",
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

    aggregate_fn = make_fltrust_aggregate(root_loader, CNNCifar, device, local_epochs=cfg.local_epochs, lr=cfg.lr)
    run_experiment(cfg, aggregate_fn)
