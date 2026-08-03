"""Thin wrappers around Flower's own aggregation implementations so every
strategy in this project shares one call signature: aggregate_fn(results, **kwargs).
`results` is a list of (NDArrays, num_examples) tuples, exactly what
NumPyClient.fit() returns per client.
"""
from typing import List, Tuple

from flwr.common import NDArrays
from flwr.server.strategy.aggregate import aggregate, aggregate_krum


def fedavg_aggregate(results: List[Tuple[NDArrays, int]], **kwargs) -> NDArrays:
    return aggregate(results)


def krum_aggregate(results: List[Tuple[NDArrays, int]], num_malicious: int, to_keep: int = 0, **kwargs) -> NDArrays:
    """to_keep=0 -> single-Krum (pick the one update closest to the honest majority).
    to_keep>0 -> Multi-Krum (average the `to_keep` most representative updates)."""
    return aggregate_krum(results, num_malicious, to_keep)
