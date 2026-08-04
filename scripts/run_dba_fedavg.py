"""DBA (distributed backdoor attack) under FedAvg: no defense, tracks accuracy + ASR.
Same client/round/attack-strength setup as the BadNets baseline so the two
attacks are directly comparable."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fl.experiment import ExperimentConfig, run_experiment
from src.defenses.aggregation import fedavg_aggregate

cfg = ExperimentConfig(
    name="dba_fedavg",
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
    run_experiment(cfg, fedavg_aggregate)
