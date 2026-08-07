"""Aggregation strategies. FedAvg/Krum are thin wrappers around Flower's own
implementations. FLTrust is implemented here directly (Flower doesn't ship
one): every aggregate_fn shares the call signature
    aggregate_fn(results, global_params, round_num, **kwargs) -> new_global_params
so `src/fl/experiment.py`'s round loop can drive any of them identically.
`results` is a list of (NDArrays, num_examples) tuples, exactly what
NumPyClient.fit() returns per client.
"""
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from flwr.common import NDArrays
from flwr.server.strategy.aggregate import aggregate, aggregate_krum
from sklearn.cluster import HDBSCAN
from torch.utils.data import DataLoader

from src.fl.client import get_parameters, set_parameters, train


def fedavg_aggregate(results: List[Tuple[NDArrays, int]], **kwargs) -> NDArrays:
    return aggregate(results)


def krum_aggregate(results: List[Tuple[NDArrays, int]], num_malicious: int, to_keep: int = 0, **kwargs) -> NDArrays:
    """to_keep=0 -> single-Krum (pick the one update closest to the honest majority).
    to_keep>0 -> Multi-Krum (average the `to_keep` most representative updates)."""
    return aggregate_krum(results, num_malicious, to_keep)


def multi_krum_aggregate(
    results: List[Tuple[NDArrays, int]], num_malicious: int, to_keep: Optional[int] = None, **kwargs
) -> NDArrays:
    """Multi-Krum: identical scoring to single-Krum (each update ranked by summed
    distance to its closest neighbors) but *averages* the `to_keep` most
    representative updates instead of keeping only one. Single-Krum's
    keep-exactly-one rule is what causes its accuracy collapse under non-IID
    data -- every round it throws away n-1 clients' worth of honest signal
    along with the attackers. Averaging the presumed-honest majority instead
    of picking a single "most central" client recovers most of that lost
    statistical efficiency while still excluding the most anomalous updates.
    Defaults to_keep to n - num_malicious (keep everyone not assumed Byzantine)."""
    n = len(results)
    if to_keep is None:
        to_keep = max(1, n - num_malicious)
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


def flame_aggregate(
    results: List[Tuple[NDArrays, int]],
    global_params: NDArrays,
    round_num: int,
    noise_lambda: float = 0.001,
    min_cluster_size: Optional[int] = None,
    seed: int = 0,
    **kwargs,
) -> NDArrays:
    """FLAME (Nguyen et al., 2022). Unlike FLTrust this needs no server-side
    trusted root dataset -- it separates honest from malicious updates purely
    from their geometry relative to each other, in three stages:

    1. Dynamic clustering: HDBSCAN over pairwise cosine *distance* between
       client update directions, with min_cluster_size fixed at n//2 + 1 so a
       cluster can only form from a majority of this round's clients (matching
       the standard Byzantine assumption of <50% malicious). The largest
       resulting cluster is kept; unclustered ("noise") points and any
       minority cluster are dropped as Byzantine.
    2. Adaptive clipping: each kept update's norm is clipped down to St, the
       *median* norm among kept updates -- bounds how much even an
       in-cluster outlier can contribute, without needing a fixed threshold.
    3. Adaptive noise: Gaussian noise scaled by noise_lambda * St is added to
       the aggregated model. This is the differential-privacy-style step that
       does the final cleanup -- it degrades any residual backdoor signal
       that survived clustering + clipping (e.g. a malicious cluster that
       narrowly beat the honest one on size), at the cost of a small amount
       of main-task accuracy.
    """
    deltas = [[cp - gp for cp, gp in zip(client_params, global_params)] for client_params, _ in results]
    flat = np.stack([_flatten(d) for d in deltas])
    n = flat.shape[0]

    norms = np.linalg.norm(flat, axis=1) + 1e-12
    cos_sim = (flat @ flat.T) / np.outer(norms, norms)
    cos_dist = np.clip(1.0 - cos_sim, 0.0, None)
    np.fill_diagonal(cos_dist, 0.0)

    mcs = min_cluster_size if min_cluster_size is not None else (n // 2 + 1)
    mcs = max(2, min(mcs, n))
    labels = HDBSCAN(min_cluster_size=mcs, metric="precomputed", copy=False).fit_predict(cos_dist.astype(np.float64))

    if (labels != -1).any():
        vals, counts = np.unique(labels[labels != -1], return_counts=True)
        largest = vals[np.argmax(counts)]
        keep_idx = np.where(labels == largest)[0]
    else:
        # Nothing clustered (e.g. too few clients this round) -- fail open
        # rather than discarding the whole round's work.
        keep_idx = np.arange(n)

    kept_norms = norms[keep_idx]
    S_t = float(np.median(kept_norms))

    clipped_deltas = []
    for i in keep_idx:
        scale = min(1.0, S_t / norms[i])
        clipped_deltas.append([d * scale for d in deltas[i]])

    avg_delta = [np.mean(np.stack(layer), axis=0) for layer in zip(*clipped_deltas)]

    rng = np.random.default_rng(seed * 1_000_003 + round_num)
    sigma = noise_lambda * S_t
    noisy_delta = [ad + rng.normal(0.0, sigma, size=ad.shape).astype(ad.dtype) for ad in avg_delta]

    return [gp + nd for gp, nd in zip(global_params, noisy_delta)]
