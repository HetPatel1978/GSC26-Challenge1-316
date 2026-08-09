"""Adaptive Trust Aggregation (ATA) -- this repo's own combined defense, not
a reproduction of a single paper. Motivated directly by two results already
in this repo: FLTrust's whole-update cosine check alone lets plain BadNets
through at 0.807 ASR (see badnets_fltrust.json / the adaptive-attacker
writeup in the README), and FLAME's peer-clustering never fires under this
repo's non-IID setting (see flame_aggregate). ATA stacks three mechanisms
into one aggregation function, each addressing a specific gap the other two
leave open:

1. FLTrust-style cosine trust scoring: a server-trained root-set reference
   direction g0, ReLU-clipped cosine similarity per client -> a trust
   weight, combined into a trust-weighted average delta. Same idea as
   make_fltrust_aggregate -- kept because a privileged reference direction
   is what let FLTrust survive non-IID skew where FLAME's peer-consensus
   approach failed.
2. Adaptive clipping: each trusted client's delta is norm-clipped to the
   *median* norm among clients that received nonzero trust before being
   folded into the weighted average -- bounds how much even a "trusted"
   outlier can contribute, independent of g0's own norm (FLTrust rescales
   every accepted update to g0's exact norm; ATA instead follows FLAME's
   median-of-accepted-updates approach, scoped to the trust-filtered set
   rather than a cluster).
3. RLR-style per-coordinate sign correction on the AGGREGATE: FLTrust's
   cosine score is a single whole-update number, which is exactly what let
   a partially-poisoned update (or this repo's own adaptive attacker) pass
   -- a malicious delta can have positive cosine similarity to g0 *on
   average* while a meaningful block of coordinates pushes the opposite
   way. This stage computes, per parameter coordinate, the trust-weighted
   fraction of clients whose sign agrees with the trust-weighted aggregate's
   own sign there; where that agreement is below `robust_threshold`, the
   aggregate is flipped at that coordinate before being applied. Adapted
   from Robust Learning Rate (Ozdayi et al., 2021), which flips the
   server's effective learning rate per-coordinate on weak cross-client
   sign agreement.

   An earlier version of this stage compared each *individual* client's
   per-coordinate sign against g0 directly and flipped disagreeing
   coordinates before aggregation. Diagnosing why that version made
   training diverge (loss increasing round-over-round in a smoke test)
   found the real problem: g0 is trained on only 500 root-set samples, so
   its per-coordinate sign is itself noisy across ~320K parameters --
   honest clients disagreed with it on 37-49% of coordinates even when
   restricted to g0's top 0.1% most-confident-by-magnitude coordinates,
   statistically indistinguishable from malicious clients' 30-61%. Flipping
   ~40% of every client's parameters, honest or not, destroys the update.
   Trust-weighted agreement *across the sampled clients* at each round
   (~10 of them) is a far more stable statistic than one small reference
   model's raw sign, and operating once on the aggregate (not on every
   client's raw contribution pre-aggregation) means a coordinate only gets
   corrected when the trusted mass actually disagrees with what's about to
   be applied, not whenever it disagrees with a noisy single reference.

FLAME's HDBSCAN clustering step is deliberately NOT included. This repo
already showed (flame_aggregate, and the README's FLAME writeup) that its
cosine-distance clustering never finds the required majority cluster under
this repo's Dirichlet(alpha=0.5) non-IID setting -- pairwise cosine
distances between ANY two client updates sit near-orthogonal (~0.85-1.05)
regardless of honest/malicious status, so every round falls back to
"keep everyone," and the clustering step contributes nothing. Stacking a
non-functional component into ATA would misrepresent what the defense
actually does; leaving it out is the honest choice given what this repo
has already measured, not an oversight.
"""
from typing import Callable, List, Tuple

import numpy as np
import torch
from flwr.common import NDArrays
from torch.utils.data import DataLoader

from src.fl.client import get_parameters, set_parameters, train


def _flatten(ndarrays: NDArrays) -> np.ndarray:
    return np.concatenate([w.reshape(-1) for w in ndarrays])


def _unflatten_like(flat: np.ndarray, like: NDArrays) -> NDArrays:
    out = []
    i = 0
    for w in like:
        n = w.size
        out.append(flat[i : i + n].reshape(w.shape).astype(w.dtype))
        i += n
    return out


def cosine_trust_scores(g0_flat: np.ndarray, flat_deltas: List[np.ndarray]) -> List[float]:
    """Stage 1: FLTrust-style ReLU-clipped whole-update cosine similarity of
    each client delta to the reference direction g0. Always in [0, 1] --
    negative similarity is clipped to zero (full exclusion), not merely
    down-weighted."""
    g0_norm = float(np.linalg.norm(g0_flat)) + 1e-12
    scores = []
    for df in flat_deltas:
        di_norm = float(np.linalg.norm(df)) + 1e-12
        cos_sim = float(np.dot(g0_flat, df) / (g0_norm * di_norm))
        scores.append(max(0.0, cos_sim))
    return scores


def adaptive_clip(flat_deltas: List[np.ndarray], trust_scores: List[float]) -> List[np.ndarray]:
    """Stage 2: clip each delta's norm down to the median norm among clients
    with nonzero trust. Every returned delta's norm is <= that median by
    construction (untrusted deltas get clipped too, even though their zero
    trust weight means they won't affect the aggregate)."""
    norms = [float(np.linalg.norm(df)) + 1e-12 for df in flat_deltas]
    trusted_norms = [norms[i] for i, ts in enumerate(trust_scores) if ts > 0.0]
    if not trusted_norms:
        return [np.array(df, copy=True) for df in flat_deltas]
    S_t = float(np.median(trusted_norms))
    return [df * min(1.0, S_t / n) for df, n in zip(flat_deltas, norms)]


def sign_correct_aggregate(
    agg_flat: np.ndarray,
    clipped_deltas: List[np.ndarray],
    trust_scores: List[float],
    robust_threshold: float,
) -> np.ndarray:
    """Stage 3: RLR-style per-coordinate sign correction of the trust-weighted
    aggregate. For each coordinate, the trust-weighted fraction of clients
    whose sign agrees with the aggregate's own sign there; where that
    agreement is below robust_threshold, the aggregate is flipped at that
    coordinate. Every output value's sign (np.sign of the result) is in
    {-1, 0, 1} by construction, and the result is finite wherever the input
    was -- flipping only negates, it never divides or otherwise risks
    introducing inf/nan."""
    total_trust = sum(trust_scores)
    agg_sign = np.sign(agg_flat)
    agree_weight = np.zeros_like(agg_flat)
    for ts, delta in zip(trust_scores, clipped_deltas):
        agree_weight += (ts / total_trust) * (np.sign(delta) == agg_sign)
    flip_mask = (agg_sign != 0) & (agree_weight < robust_threshold)
    return np.where(flip_mask, -agg_flat, agg_flat)


def make_ata_aggregate(
    root_loader: DataLoader,
    model_ctor: Callable[[], torch.nn.Module],
    device: torch.device,
    local_epochs: int = 1,
    lr: float = 0.01,
    robust_threshold: float = 0.7,
) -> Callable:
    def aggregate_fn(results: List[Tuple[NDArrays, int]], global_params: NDArrays, round_num: int, **kwargs) -> NDArrays:
        # Stage 0: server-side reference direction (same as FLTrust).
        server_model = model_ctor()
        set_parameters(server_model, global_params)
        train(server_model, root_loader, epochs=local_epochs, lr=lr, device=device)
        server_delta = [sp - gp for sp, gp in zip(get_parameters(server_model), global_params)]
        g0_flat = _flatten(server_delta)

        client_deltas = [[cp - gp for cp, gp in zip(cparams, global_params)] for cparams, _ in results]
        flat_deltas = [_flatten(d) for d in client_deltas]

        trust_scores = cosine_trust_scores(g0_flat, flat_deltas)
        total_trust = sum(trust_scores)
        if total_trust <= 1e-12:
            # Nobody this round looked like the server's own trusted-data direction.
            return global_params

        clipped_deltas = adaptive_clip(flat_deltas, trust_scores)

        agg_flat = np.zeros_like(g0_flat)
        for ts, delta in zip(trust_scores, clipped_deltas):
            agg_flat += (ts / total_trust) * delta

        agg_flat = sign_correct_aggregate(agg_flat, clipped_deltas, trust_scores, robust_threshold)

        agg_delta = _unflatten_like(agg_flat, global_params)
        return [gp + ad for gp, ad in zip(global_params, agg_delta)]

    return aggregate_fn
