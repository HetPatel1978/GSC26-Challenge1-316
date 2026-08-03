"""BadNets-style backdoor attack: a fixed pixel-pattern trigger patched onto
images, paired with a label flip to a fixed target class."""
import numpy as np
import torch
from torch.utils.data import Dataset


def square_trigger(img: torch.Tensor, size: int = 4, value: float = 2.5, position: str = "br") -> torch.Tensor:
    """Stamp a solid square patch onto a (already-normalized) CHW image tensor.
    `value` is set above the normalized max so the patch is a bright, unambiguous
    trigger regardless of the dataset's normalization stats."""
    img = img.clone()
    _, h, w = img.shape
    if position == "br":
        img[:, h - size : h, w - size : w] = value
    elif position == "tl":
        img[:, 0:size, 0:size] = value
    else:
        raise ValueError(f"unknown position {position}")
    return img


class BadNetsPoisonedDataset(Dataset):
    """Wraps a client's local dataset. A fixed fraction of samples get the
    trigger patched in and their label flipped to `target_label`."""

    def __init__(self, base_dataset: Dataset, poison_rate: float, target_label: int, seed: int = 0, trigger_fn=square_trigger):
        self.base_dataset = base_dataset
        self.target_label = target_label
        self.trigger_fn = trigger_fn

        n = len(base_dataset)
        rng = np.random.default_rng(seed)
        n_poison = int(n * poison_rate)
        poison_idx = rng.choice(n, size=n_poison, replace=False)
        self.poison_mask = np.zeros(n, dtype=bool)
        self.poison_mask[poison_idx] = True

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, i):
        img, label = self.base_dataset[i]
        if self.poison_mask[i]:
            img = self.trigger_fn(img)
            label = self.target_label
        return img, label


class TriggerAllDataset(Dataset):
    """Applies the trigger to every sample, for measuring Attack Success Rate:
    the fraction of triggered non-target-class test images classified as
    `target_label` by the global model."""

    def __init__(self, base_dataset: Dataset, target_label: int, trigger_fn=square_trigger):
        targets = np.asarray(getattr(base_dataset, "targets", None))
        if targets is None:
            targets = np.array([base_dataset[i][1] for i in range(len(base_dataset))])
        self.indices = np.where(targets != target_label)[0]
        self.base_dataset = base_dataset
        self.target_label = target_label
        self.trigger_fn = trigger_fn

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = int(self.indices[i])
        img, true_label = self.base_dataset[real_idx]
        img = self.trigger_fn(img)
        return img, self.target_label  # label here = attack's desired (mis)classification
