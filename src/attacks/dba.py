"""DBA (Distributed Backdoor Attack): decomposes a single BadNets-style trigger
into non-overlapping local patterns, one per malicious client, so no single
client's poisoned data ever contains the full trigger -- only the aggregated
global model does. Reuses BadNetsPoisonedDataset / TriggerAllDataset from
badnets.py unchanged; this module only supplies the local-vs-combined
trigger_fn, which is the entire difference from BadNets.
"""
import torch

DBA_NUM_PARTS = 4  # 2x2 decomposition of the trigger region


def make_local_trigger_fn(part_idx: int, num_parts: int = DBA_NUM_PARTS, block_size: int = 2, value: float = 2.5, position: str = "br"):
    """Stamps just one quadrant of a (block_size*2) x (block_size*2) trigger
    region -- the piece assigned to malicious client rank `part_idx`. With the
    defaults this is a single 2x2 sub-block inside the same 4x4 corner region
    BadNets uses, so the two attacks' *full* triggers are directly comparable
    in footprint; DBA just splits the stamping across clients."""
    if num_parts != 4:
        raise ValueError("only a 2x2 (num_parts=4) decomposition is implemented")

    def fn(img: torch.Tensor) -> torch.Tensor:
        img = img.clone()
        _, h, w = img.shape
        region = block_size * 2
        row_off = (part_idx // 2) * block_size
        col_off = (part_idx % 2) * block_size
        if position == "br":
            r0, c0 = h - region + row_off, w - region + col_off
        elif position == "tl":
            r0, c0 = row_off, col_off
        else:
            raise ValueError(f"unknown position {position}")
        img[:, r0 : r0 + block_size, c0 : c0 + block_size] = value
        return img

    return fn


def make_combined_trigger_fn(num_parts: int = DBA_NUM_PARTS, block_size: int = 2, value: float = 2.5, position: str = "br"):
    """The union of all local parts -- the full trigger. Used to build the ASR
    evaluation set, since only the *aggregated* global model has ever seen the
    complete pattern; no individual client's poisoned data has."""
    part_fns = [make_local_trigger_fn(i, num_parts, block_size, value, position) for i in range(num_parts)]

    def fn(img: torch.Tensor) -> torch.Tensor:
        for part_fn in part_fns:
            img = part_fn(img)
        return img

    return fn
