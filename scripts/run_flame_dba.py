"""FLAME defense against DBA: same clustering+clipping+noise pipeline as
run_flame_badnets.py, tested against the harder-to-detect distributed trigger."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fl.experiment import ExperimentConfig, run_experiment
from src.defenses.aggregation import flame_aggregate

cfg = ExperimentConfig(
    name="flame_dba",
    num_clients=20,
    dirichlet_alpha=0.5,
    fraction_fit=0.5,
    num_rounds=30,
    local_epochs=2,
    attack_type="dba",
    fraction_malicious=0.2,
    poison_rate=0.5,
    target_label=0,
)

if __name__ == "__main__":
    run_experiment(cfg, flame_aggregate, aggregate_kwargs={"seed": cfg.seed})
