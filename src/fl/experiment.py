"""Reusable FL experiment harness.

Runs a sequential (non-Ray) FL simulation loop that still uses Flower's real
`NumPyClient` for local training and Flower's own `aggregate` / `aggregate_krum`
implementations for server-side aggregation. `flwr.simulation.start_simulation`
(Ray-backed) is not used because Ray currently ships no wheel for Python 3.13 on
this machine; this driver reproduces the same client/round/aggregate/evaluate
loop by hand instead of spinning up Ray actors. See README limitations.
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.fl.data import load_cifar10, dirichlet_partition, IndexedSubset
from src.fl.models import CNNCifar
from src.fl.client import FlowerClient, get_parameters, set_parameters, test
from src.attacks.badnets import BadNetsPoisonedDataset, TriggerAllDataset


@dataclass
class ExperimentConfig:
    name: str
    num_clients: int = 20
    dirichlet_alpha: float = 0.5
    fraction_fit: float = 0.5
    num_rounds: int = 30
    local_epochs: int = 2
    lr: float = 0.01
    batch_size: int = 32
    seed: int = 42

    # Attack params (BadNets). Set fraction_malicious=0.0 for a clean run.
    fraction_malicious: float = 0.2
    poison_rate: float = 0.5
    target_label: int = 0

    data_root: str = "./data"
    results_dir: str = "./results/metrics"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_malicious_ids(cfg: ExperimentConfig) -> set:
    rng = np.random.default_rng(cfg.seed)
    n_malicious = int(round(cfg.num_clients * cfg.fraction_malicious))
    malicious = rng.choice(cfg.num_clients, size=n_malicious, replace=False)
    return set(int(m) for m in malicious)


def build_clients(
    cfg: ExperimentConfig,
    client_indices: List[np.ndarray],
    malicious_ids: set,
    shared_model: torch.nn.Module,
    device: torch.device,
) -> Dict[int, FlowerClient]:
    train_set, _ = load_cifar10(cfg.data_root)
    clients: Dict[int, FlowerClient] = {}
    for cid in range(cfg.num_clients):
        local_ds = IndexedSubset(train_set, client_indices[cid])
        is_malicious = cid in malicious_ids
        if is_malicious:
            local_ds = BadNetsPoisonedDataset(
                local_ds, poison_rate=cfg.poison_rate, target_label=cfg.target_label, seed=cfg.seed + cid
            )
        trainloader = DataLoader(local_ds, batch_size=cfg.batch_size, shuffle=True)
        clients[cid] = FlowerClient(
            cid=str(cid),
            model=shared_model,  # sequential execution -> safe to share one instance
            trainloader=trainloader,
            valloader=trainloader,
            device=device,
            local_epochs=cfg.local_epochs,
            lr=cfg.lr,
            is_malicious=is_malicious,
        )
    return clients


def make_evaluate_fn(cfg: ExperimentConfig, history_out: List[Dict], device: torch.device):
    _, test_set = load_cifar10(cfg.data_root)
    backdoor_test_set = TriggerAllDataset(test_set, target_label=cfg.target_label)
    clean_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    backdoor_loader = DataLoader(backdoor_test_set, batch_size=256, shuffle=False)
    eval_model = CNNCifar()

    def evaluate_fn(server_round: int, parameters):
        set_parameters(eval_model, parameters)
        clean_loss, clean_acc = test(eval_model, clean_loader, device)
        _, asr = test(eval_model, backdoor_loader, device)  # "accuracy" on this loader = attack success rate

        record = {"round": server_round, "accuracy": clean_acc, "loss": clean_loss, "asr": asr}
        history_out.append(record)
        print(f"[round {server_round:03d}] acc={clean_acc:.4f} loss={clean_loss:.4f} asr={asr:.4f}")
        return record

    return evaluate_fn


def run_experiment(
    cfg: ExperimentConfig,
    aggregate_fn: Callable,
    aggregate_kwargs: Optional[Dict] = None,
):
    aggregate_kwargs = aggregate_kwargs or {}
    os.makedirs(cfg.results_dir, exist_ok=True)
    device = get_device()

    train_set, _ = load_cifar10(cfg.data_root)
    client_indices = dirichlet_partition(train_set.targets, cfg.num_clients, cfg.dirichlet_alpha, seed=cfg.seed)
    malicious_ids = build_malicious_ids(cfg)

    shared_model = CNNCifar()
    clients = build_clients(cfg, client_indices, malicious_ids, shared_model, device)

    round_history: List[Dict] = []
    evaluate_fn = make_evaluate_fn(cfg, round_history, device)

    global_model = CNNCifar()
    global_params = get_parameters(global_model)

    rng = np.random.default_rng(cfg.seed + 1000)
    num_fit = max(2, round(cfg.num_clients * cfg.fraction_fit))

    for rnd in range(1, cfg.num_rounds + 1):
        sampled_ids = rng.choice(cfg.num_clients, size=num_fit, replace=False)
        results = []
        n_malicious_sampled = 0
        for cid in sampled_ids:
            params, num_examples, metrics = clients[int(cid)].fit(global_params, {})
            results.append((params, num_examples))
            n_malicious_sampled += int(metrics["is_malicious"])

        global_params = aggregate_fn(results, **aggregate_kwargs)
        evaluate_fn(rnd, global_params)

    out_path = os.path.join(cfg.results_dir, f"{cfg.name}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "malicious_ids": sorted(malicious_ids),
                "rounds": round_history,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to {out_path}")
    return round_history
