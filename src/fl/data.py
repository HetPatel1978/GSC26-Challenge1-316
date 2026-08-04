"""CIFAR-10 loading and Dirichlet non-IID partitioning."""
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

_CACHE: dict = {}


def get_transforms(train: bool):
    if train:
        return T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


def load_cifar10(data_root: str = "./data"):
    """Load CIFAR-10 train/test sets, cached per-process to avoid re-reading
    from disk on every Ray-actor client_fn call."""
    key = data_root
    if key not in _CACHE:
        train_set = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=get_transforms(True)
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=get_transforms(False)
        )
        _CACHE[key] = (train_set, test_set)
    return _CACHE[key]


def dirichlet_partition(
    labels, num_clients: int, alpha: float, seed: int = 42, min_size_per_client: int = 10
):
    """Partition sample indices across clients using a symmetric Dirichlet(alpha)
    distribution per class (label-skew non-IID). Lower alpha -> more skewed."""
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    rng = np.random.default_rng(seed)

    min_size = 0
    client_indices = [[] for _ in range(num_clients)]
    while min_size < min_size_per_client:
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            cut_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, cut_points)
            for i, split in enumerate(splits):
                client_indices[i].extend(split.tolist())
        min_size = min(len(idxs) for idxs in client_indices)

    return [np.array(idxs, dtype=np.int64) for idxs in client_indices]


class IndexedSubset(Dataset):
    """Like torch.utils.data.Subset but picklable/lightweight for Ray transport."""

    def __init__(self, base_dataset: Dataset, indices: np.ndarray):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base_dataset[int(self.indices[i])]


def reserve_root_set(n_total: int, root_size: int, seed: int):
    """Split indices [0, n_total) into a small IID root set (e.g. for FLTrust's
    server-side trusted dataset) and the remaining pool, which callers then
    partition across clients -- keeping the root set strictly disjoint from
    every client's data."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    root_idx = perm[:root_size]
    remaining_idx = perm[root_size:]
    return root_idx, remaining_idx


def partition_class_distribution(labels, client_indices) -> np.ndarray:
    """Return a (num_clients, num_classes) count matrix, useful for sanity-checking
    and for plotting the non-IID skew."""
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    dist = np.zeros((len(client_indices), num_classes), dtype=int)
    for i, idxs in enumerate(client_indices):
        vals, counts = np.unique(labels[idxs], return_counts=True)
        dist[i, vals] = counts
    return dist
