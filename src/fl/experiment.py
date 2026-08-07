"""Reusable FL experiment harness.

Runs a sequential (non-Ray) FL simulation loop that still uses Flower's real
`NumPyClient` for local training and Flower's own `aggregate` / `aggregate_krum`
implementations (or the FLTrust implementation in src/defenses/aggregation.py)
for server-side aggregation. `flwr.simulation.start_simulation` (Ray-backed)
is not used because Ray currently ships no wheel for Python 3.13 on this
machine; this driver reproduces the same client/round/aggregate/evaluate loop
by hand instead of spinning up Ray actors. See README limitations.
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.fl.data import load_cifar10, dirichlet_partition, reserve_root_set, IndexedSubset
from src.fl.models import CNNCifar
from src.fl.client import FlowerClient, get_parameters, set_parameters, test
from src.attacks.badnets import BadNetsPoisonedDataset, TriggerAllDataset, square_trigger
from src.attacks.dba import make_local_trigger_fn, make_combined_trigger_fn, DBA_NUM_PARTS


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

    # Attack params. attack_type: "badnets" (single trigger, every malicious
    # client poisons with the full pattern) or "dba" (trigger decomposed into
    # DBA_NUM_PARTS pieces, each malicious client only ever poisons with its
    # own piece; ASR is measured with the reassembled full trigger).
    attack_type: str = "badnets"
    fraction_malicious: float = 0.2
    poison_rate: float = 0.5
    target_label: int = 0

    # FLTrust only: size of the server's disjoint trusted root set, carved out
    # of the training pool before client partitioning. 0 = unused.
    root_size: int = 0

    data_root: str = "./data"
    results_dir: str = "./results/metrics"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_malicious_ids(cfg: ExperimentConfig) -> set:
    rng = np.random.default_rng(cfg.seed)
    n_malicious = int(round(cfg.num_clients * cfg.fraction_malicious))
    malicious = rng.choice(cfg.num_clients, size=n_malicious, replace=False)
    return set(int(m) for m in malicious)


def _client_trigger_fn(cfg: ExperimentConfig, cid: int, malicious_ids: set):
    """The trigger_fn a given malicious client poisons its local data with."""
    if cfg.attack_type == "badnets":
        return square_trigger
    if cfg.attack_type == "dba":
        rank = sorted(malicious_ids).index(cid)
        part_idx = rank % DBA_NUM_PARTS
        return make_local_trigger_fn(part_idx)
    raise ValueError(f"unknown attack_type {cfg.attack_type}")


def _eval_trigger_fn(cfg: ExperimentConfig):
    """The trigger_fn used to build the ASR evaluation set -- always the FULL
    trigger, since that's what a backdoor-triggering input actually looks like
    at inference time regardless of how it was distributed during training."""
    if cfg.attack_type == "badnets":
        return square_trigger
    if cfg.attack_type == "dba":
        return make_combined_trigger_fn()
    raise ValueError(f"unknown attack_type {cfg.attack_type}")


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
                local_ds,
                poison_rate=cfg.poison_rate,
                target_label=cfg.target_label,
                seed=cfg.seed + cid,
                trigger_fn=_client_trigger_fn(cfg, cid, malicious_ids),
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
    backdoor_test_set = TriggerAllDataset(test_set, target_label=cfg.target_label, trigger_fn=_eval_trigger_fn(cfg))
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
    adapt_fn_factory: Optional[Callable] = None,
):
    """adapt_fn_factory, if given, is called once with (train_set, client_indices,
    malicious_ids) after partitioning, and must return an
    adapt_fn(cid, params, global_params) -> params hook applied to every
    malicious client's freshly-trained update before it's handed to
    aggregate_fn -- e.g. src.attacks.adaptive's defense-aware attacker, which
    reshapes the update to pass FLTrust's cosine-similarity trust check."""
    aggregate_kwargs = aggregate_kwargs or {}
    os.makedirs(cfg.results_dir, exist_ok=True)
    device = get_device()

    train_set, _ = load_cifar10(cfg.data_root)

    if cfg.root_size > 0:
        _root_idx, remaining_idx = reserve_root_set(len(train_set), cfg.root_size, cfg.seed)
        remaining_labels = np.array(train_set.targets)[remaining_idx]
        sub_indices = dirichlet_partition(remaining_labels, cfg.num_clients, cfg.dirichlet_alpha, seed=cfg.seed)
        client_indices = [remaining_idx[idxs] for idxs in sub_indices]
    else:
        client_indices = dirichlet_partition(train_set.targets, cfg.num_clients, cfg.dirichlet_alpha, seed=cfg.seed)

    malicious_ids = build_malicious_ids(cfg)

    shared_model = CNNCifar()
    clients = build_clients(cfg, client_indices, malicious_ids, shared_model, device)
    adapt_fn = adapt_fn_factory(train_set, client_indices, malicious_ids) if adapt_fn_factory else None

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
            if adapt_fn is not None and metrics["is_malicious"]:
                params = adapt_fn(cid=int(cid), params=params, global_params=global_params)
            results.append((params, num_examples))
            n_malicious_sampled += int(metrics["is_malicious"])

        global_params = aggregate_fn(results, global_params=global_params, round_num=rnd, **aggregate_kwargs)
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
