"""Krum (single-Krum) defense against BadNets, same setup as the FedAvg baseline
so the two runs are directly comparable."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fl.experiment import ExperimentConfig, run_experiment
from src.defenses.aggregation import krum_aggregate

cfg = ExperimentConfig(
    name="krum_defense",
    num_clients=20,
    dirichlet_alpha=0.5,
    fraction_fit=0.5,
    num_rounds=30,
    local_epochs=2,
    fraction_malicious=0.2,
    poison_rate=0.5,
    target_label=0,
)

if __name__ == "__main__":
    num_fit = max(2, round(cfg.num_clients * cfg.fraction_fit))
    # Defender's assumed number of Byzantine clients among those selected per round.
    num_malicious = max(1, math.ceil(cfg.fraction_malicious * num_fit))
    run_experiment(cfg, krum_aggregate, aggregate_kwargs={"num_malicious": num_malicious, "to_keep": 0})
