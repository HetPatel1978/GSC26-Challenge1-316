"""Defense-aware adaptive attacker ("constrain-and-scale"): a malicious client
that knows the server is running a cosine-similarity trust defense (FLTrust)
and reshapes its poisoned update to pass that check, instead of submitting
its raw backdoor gradient the way BadNets/DBA's clients do.

Threat model: the attacker doesn't have access to the server's private root
set (and therefore not to g0 itself), but assumes -- optimistically for the
defender's sake, which is the point of a stress test -- that it can
approximate "the direction FLTrust's trust check rewards" using a reference
model trained on its own local *clean* data for the same number of epochs
the server trains on its root set. This is exactly analogous to g0: both are
"a plausible honest gradient from a small clean sample," just from different
data. The attacker then keeps the component of its malicious delta that is
orthogonal to this reference direction (the part that actually carries the
backdoor signal -- cosine similarity is blind to it) and adds back just
enough of the reference direction to hit a target cosine similarity, i.e.
the minimal distortion needed to look "trusted."
"""
from typing import Callable, Dict

import numpy as np
import torch
from flwr.common import NDArrays
from torch.utils.data import DataLoader

from src.fl.client import get_parameters, set_parameters, train
from src.fl.data import IndexedSubset


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


def constrain_to_cosine_similarity(delta_mal: NDArrays, delta_ref: NDArrays, target_cos_sim: float) -> NDArrays:
    """Return a delta whose cosine similarity to delta_ref equals target_cos_sim,
    built from delta_mal's component orthogonal to delta_ref (kept intact --
    this is the part a cosine check can't see) plus the minimal amount of the
    reference direction needed to reach the target angle. Closed form: for
    a = delta_mal, b_hat = unit(delta_ref), decompose a = a_par + a_perp along
    b_hat, then submit a_perp + k*b_hat where
        k = (t / sqrt(1 - t^2)) * ||a_perp||
    is the unique k >= 0 solving cos(a_perp + k*b_hat, b_hat) = t."""
    a = _flatten(delta_mal)
    b = _flatten(delta_ref)
    b_norm = float(np.linalg.norm(b)) + 1e-12
    b_hat = b / b_norm

    a_par_scalar = float(np.dot(a, b_hat))
    a_perp = a - a_par_scalar * b_hat
    a_perp_norm = float(np.linalg.norm(a_perp))

    t = float(np.clip(target_cos_sim, -0.999, 0.999))
    k = 0.0 if t <= 0 else (t / np.sqrt(1 - t**2)) * a_perp_norm

    submit_flat = a_perp + k * b_hat
    return _unflatten_like(submit_flat, delta_mal)


def make_adaptive_fltrust_adapt_fn_factory(
    model_ctor: Callable[[], torch.nn.Module],
    device: torch.device,
    local_epochs: int,
    lr: float,
    batch_size: int,
    target_cos_sim: float = 0.9,
) -> Callable:
    """Builds the adapt_fn_factory expected by src.fl.experiment.run_experiment.
    For each malicious client, trains a fresh reference model on that client's
    own *unpoisoned* local data (the same indices the client was assigned
    before BadNetsPoisonedDataset wrapped them) to get delta_ref, then
    constrains the client's already-poisoned delta toward it."""

    def factory(train_set, client_indices, malicious_ids) -> Callable:
        clean_loaders: Dict[int, DataLoader] = {
            cid: DataLoader(IndexedSubset(train_set, client_indices[cid]), batch_size=batch_size, shuffle=True)
            for cid in malicious_ids
        }

        def adapt_fn(cid: int, params: NDArrays, global_params: NDArrays) -> NDArrays:
            loader = clean_loaders.get(cid)
            if loader is None:
                return params

            ref_model = model_ctor()
            set_parameters(ref_model, global_params)
            train(ref_model, loader, epochs=local_epochs, lr=lr, device=device)
            delta_ref = [rp - gp for rp, gp in zip(get_parameters(ref_model), global_params)]

            delta_mal = [p - gp for p, gp in zip(params, global_params)]
            delta_submit = constrain_to_cosine_similarity(delta_mal, delta_ref, target_cos_sim)
            return [gp + d for gp, d in zip(global_params, delta_submit)]

        return adapt_fn

    return factory
