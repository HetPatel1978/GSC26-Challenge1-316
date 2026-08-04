"""Aggregation strategies. FedAvg/Krum are thin wrappers around Flower's own
implementations. FLTrust is implemented here directly (Flower doesn't ship
one): every aggregate_fn shares the call signature
    aggregate_fn(results, global_params, round_num, **kwargs) -> new_global_params
so `src/fl/experiment.py`'s round loop can drive any of them identically.
`results` is a list of (NDArrays, num_examples) tuples, exactly what
NumPyClient.fit() returns per client.
"""
from typing import Callable, List, Tuple

import numpy as np
import torch
from flwr.common import NDArrays
from flwr.server.strategy.aggregate import aggregate, aggregate_krum
from torch.utils.data import DataLoader

from src.fl.client import get_parameters, set_parameters, train


def fedavg_aggregate(results: List[Tuple[NDArrays, int]], **kwargs) -> NDArrays:
    return aggregate(results)


def krum_aggregate(results: List[Tuple[NDArrays, int]], num_malicious: int, to_keep: int = 0, **kwargs) -> NDArrays:
    """to_keep=0 -> single-Krum (pick the one update closest to the honest majority).
    to_keep>0 -> Multi-Krum (average the `to_keep` most representative updates)."""
    return aggregate_krum(results, num_malicious, to_keep)


def _flatten(ndarrays: NDArrays) -> np.ndarray:
    return np.concatenate([w.reshape(-1) for w in ndarrays])


def make_fltrust_aggregate(
    root_loader: DataLoader,
    model_ctor: Callable[[], torch.nn.Module],
    device: torch.device,
    local_epochs: int = 1,
    lr: float = 0.01,
) -> Callable:
    """FLTrust (Cao et al. 2021). Each round the server trains a fresh copy of
    the current global model on its own small trusted root dataset to get a
    reference update direction g0. Each client update gi is scored by
    ReLU-clipped cosine similarity to g0 (negative-similarity / adversarial
    directions get zero weight), then rescaled to g0's norm (so a malicious
    client can't just inflate its update's magnitude to dominate the
    average), and finally combined as a trust-weighted average."""

    def aggregate_fn(results: List[Tuple[NDArrays, int]], global_params: NDArrays, round_num: int, **kwargs) -> NDArrays:
        server_model = model_ctor()
        set_parameters(server_model, global_params)
        train(server_model, root_loader, epochs=local_epochs, lr=lr, device=device)
        server_delta = [sp - gp for sp, gp in zip(get_parameters(server_model), global_params)]
        g0_flat = _flatten(server_delta)
        g0_norm = float(np.linalg.norm(g0_flat)) + 1e-12

        trust_scores = []
        rescaled_deltas = []
        for client_params, _ in results:
            delta_i = [cp - gp for cp, gp in zip(client_params, global_params)]
            di_flat = _flatten(delta_i)
            di_norm = float(np.linalg.norm(di_flat)) + 1e-12
            cos_sim = float(np.dot(g0_flat, di_flat) / (g0_norm * di_norm))
            trust_scores.append(max(0.0, cos_sim))
            scale = g0_norm / di_norm
            rescaled_deltas.append([d * scale for d in delta_i])

        total_trust = sum(trust_scores)
        if total_trust <= 1e-12:
            # Nobody this round looked like the server's own trusted-data direction.
            return global_params

        aggregated_delta = [np.zeros_like(p) for p in global_params]
        for ts, delta in zip(trust_scores, rescaled_deltas):
            weight = ts / total_trust
            aggregated_delta = [ad + weight * d for ad, d in zip(aggregated_delta, delta)]

        return [gp + ad for gp, ad in zip(global_params, aggregated_delta)]

    return aggregate_fn
