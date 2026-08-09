"""Tests for src/defenses/ata.py (Adaptive Trust Aggregation).

test_output_shape_matches_input, test_cosine_trust_scores_in_unit_interval,
test_adaptive_clip_never_exceeds_median_trusted_norm, and
test_sign_correction_produces_valid_signs exercise the three stages in
isolation on small synthetic data (no CIFAR-10, no real model -- fast and
hermetic). test_ata_reduces_asr_vs_fedavg_smoke is the one integration test
that needs the real attack/eval pipeline, so it needs CIFAR-10 (downloads on
first run if not already cached in ./data/, same as every scripts/run_*.py
in this repo) and is correspondingly slower (~1-2 min for two 3-round runs
at 10 clients).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.fl.client import get_parameters
from src.defenses.ata import (
    adaptive_clip,
    cosine_trust_scores,
    make_ata_aggregate,
    sign_correct_aggregate,
)


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 8))


def _tiny_loader(n: int = 16, batch_size: int = 4) -> DataLoader:
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(n, 8, generator=rng)
    y = torch.randint(0, 8, (n,), generator=rng)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def test_output_shape_matches_input():
    """ATA's aggregate_fn must return a parameter list with the same shapes
    and dtypes as global_params -- anything else would be silently
    incompatible with the next round's set_parameters() call."""
    model = _tiny_model()
    global_params = get_parameters(model)
    loader = _tiny_loader()
    device = torch.device("cpu")
    aggregate_fn = make_ata_aggregate(loader, _tiny_model, device, local_epochs=1, lr=0.01)

    rng = np.random.default_rng(0)
    results = []
    for _ in range(5):
        client_params = [p + rng.normal(0, 0.01, size=p.shape).astype(p.dtype) for p in global_params]
        results.append((client_params, 10))

    new_params = aggregate_fn(results, global_params=global_params, round_num=1)

    assert len(new_params) == len(global_params)
    for new_p, orig_p in zip(new_params, global_params):
        assert new_p.shape == orig_p.shape
        assert new_p.dtype == orig_p.dtype


def test_cosine_trust_scores_in_unit_interval():
    """cosine_trust_scores is a ReLU-clipped cosine similarity -- every score
    must land in [0, 1], and a delta pointing exactly opposite g0 must score
    exactly 0 (fully excluded, not just down-weighted)."""
    rng = np.random.default_rng(1)
    g0 = rng.normal(size=1000)
    deltas = [rng.normal(size=1000) for _ in range(6)]
    deltas.append(-g0)  # exact opposite direction

    scores = cosine_trust_scores(g0, deltas)

    assert len(scores) == len(deltas)
    for s in scores:
        assert 0.0 <= s <= 1.0
    assert scores[-1] == 0.0


def test_adaptive_clip_never_exceeds_median_trusted_norm():
    """adaptive_clip must never let a returned delta's norm exceed the
    median norm among trusted (nonzero trust score) clients, including for
    the untrusted deltas mixed in here -- they get clipped too even though
    their zero weight means it's moot for the aggregate."""
    rng = np.random.default_rng(2)
    deltas = [rng.normal(size=500) * scale for scale in [0.1, 0.5, 1.0, 5.0, 10.0]]
    trust_scores = [1.0, 1.0, 1.0, 0.0, 1.0]  # one untrusted, mixed in on purpose

    clipped = adaptive_clip(deltas, trust_scores)

    trusted_norms = [float(np.linalg.norm(d)) for d, ts in zip(deltas, trust_scores) if ts > 0.0]
    median_norm = float(np.median(trusted_norms))
    for c in clipped:
        assert float(np.linalg.norm(c)) <= median_norm + 1e-6


def test_sign_correction_produces_valid_signs():
    """sign_correct_aggregate only ever negates coordinates (never divides
    or rescales), so its output must stay finite wherever the input was,
    and np.sign of the result must land in {-1, 0, 1} for every coordinate
    -- a NaN/inf leak anywhere in the pipeline would violate this."""
    rng = np.random.default_rng(3)
    agg = rng.normal(size=200)
    clipped_deltas = [rng.normal(size=200) for _ in range(6)]
    trust_scores = [1.0] * 6

    corrected = sign_correct_aggregate(agg, clipped_deltas, trust_scores, robust_threshold=0.7)

    assert np.all(np.isfinite(corrected))
    signs = np.sign(corrected)
    assert np.all(np.isin(signs, [-1.0, 0.0, 1.0]))


def test_ata_reduces_asr_vs_fedavg_smoke():
    """Integration smoke test: over just 3 rounds with a fixed seed, ATA's
    final-round ASR against BadNets must be lower than undefended FedAvg's.
    Needs the real attack/eval pipeline (CIFAR-10, real CNN) since ASR is
    only meaningful against the real backdoor trigger dataset.

    Uses the same num_clients/seed/root_size as scripts/run_ata_badnets.py
    and scripts/run_multiseed.py (just truncated to 3 rounds) rather than a
    smaller ad hoc config: this is the scale/seed combination already shown,
    across every run at this scale, to separate the two by round 3 with a
    wide margin (e.g. ata_badnets.json: round 3 asr=0.011 vs the seed-42 arm
    of the FedAvg multi-seed check: round 3 asr=0.309). Note this is *not*
    bit-for-bit reproduction of those exact numbers -- this repo's training
    is not pinned to deterministic cuDNN kernels, so even identical
    seed+config runs diverge by round 2 in floating point (confirmed
    directly: see the README's note on GPU non-determinism). What carries
    over is the reliable *separation*, not the exact values. An earlier
    version of this test using num_clients=10/seed=123/3 rounds failed
    intermittently because at that smaller scale the separation itself
    isn't reliable, which is a different and worse problem than
    non-determinism in the exact numbers."""
    from src.fl.data import IndexedSubset, load_cifar10, reserve_root_set
    from src.fl.experiment import ExperimentConfig, get_device, run_experiment
    from src.fl.models import CNNCifar
    from src.defenses.aggregation import fedavg_aggregate

    common = dict(
        num_clients=20,
        dirichlet_alpha=0.5,
        fraction_fit=0.5,
        num_rounds=3,
        local_epochs=2,
        attack_type="badnets",
        fraction_malicious=0.2,
        poison_rate=0.5,
        target_label=0,
        seed=42,
        results_dir="./results/test_smoke_metrics",
    )

    cfg_fedavg = ExperimentConfig(name="test_smoke_fedavg", **common)
    fedavg_history = run_experiment(cfg_fedavg, fedavg_aggregate)

    cfg_ata = ExperimentConfig(name="test_smoke_ata", root_size=500, **common)
    train_set, _ = load_cifar10(cfg_ata.data_root)
    root_idx, _ = reserve_root_set(len(train_set), cfg_ata.root_size, cfg_ata.seed)
    root_loader = DataLoader(IndexedSubset(train_set, root_idx), batch_size=cfg_ata.batch_size, shuffle=True)
    device = get_device()
    aggregate_fn = make_ata_aggregate(root_loader, CNNCifar, device, local_epochs=cfg_ata.local_epochs, lr=cfg_ata.lr)
    ata_history = run_experiment(cfg_ata, aggregate_fn)

    assert ata_history[-1]["asr"] < fedavg_history[-1]["asr"]
