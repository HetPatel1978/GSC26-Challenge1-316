"""The key stress test: the defense-aware adaptive attacker (constrain-and-
scale toward a self-trained clean-data reference direction, same mechanism
as run_adaptive_vs_fltrust.py) against ATA instead of plain FLTrust. FLTrust
alone broke against this same setup at 0.624 ASR (and against naive BadNets
at 0.807 -- see run_badnets_fltrust.py); this answers whether ATA's added
sign-gating + median clipping holds where FLTrust's cosine check alone
didn't. The attacker's own logic doesn't change: it has no knowledge of
which aggregator is running, only that *some* cosine-similarity trust check
is -- reusing it unchanged against ATA is the fair test."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.fl.data import load_cifar10, reserve_root_set, IndexedSubset
from src.fl.experiment import ExperimentConfig, run_experiment, get_device
from src.fl.models import CNNCifar
from src.defenses.ata import make_ata_aggregate
from src.attacks.adaptive import make_adaptive_fltrust_adapt_fn_factory

cfg = ExperimentConfig(
    name="adaptive_badnets_ata",
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
    adapt_fn_factory = make_adaptive_fltrust_adapt_fn_factory(
        CNNCifar, device, local_epochs=cfg.local_epochs, lr=cfg.lr, batch_size=cfg.batch_size, target_cos_sim=0.9
    )
    run_experiment(cfg, aggregate_fn, adapt_fn_factory=adapt_fn_factory)
