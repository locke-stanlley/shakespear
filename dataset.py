"""
dataset.py - Phase 3: Dataset Splitting & Structuring
Splits the tokenized binary into train/val/test, exposes PyTorch
Datasets and DataLoaders with context-window chunking.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import (
    TOKEN_IDS_PATH, SPLIT_DIR, BLOCK_SIZE,
    TRAIN_RATIO, VAL_RATIO, BATCH_SIZE, SEED
)


def split_data():
    """Split the token ID binary file into train/val/test memmaps."""
    print("=" * 60)
    print("[PHASE 3] Dataset Splitting")
    print("=" * 60)

    ids = np.memmap(str(TOKEN_IDS_PATH), dtype=np.uint16, mode="r")
    n = len(ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    for name, arr in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        path = SPLIT_DIR / f"{name}.bin"
        mm = np.memmap(str(path), dtype=np.uint16, mode="w+", shape=arr.shape)
        mm[:] = arr[:]
        mm.flush()
        print(f"[+] {name}: {len(arr)} tokens -> {path}")


class ShakespeareDataset(Dataset):
    """
    Streams chunks of BLOCK_SIZE from a memmap file.
    x = tokens[0:T], y = tokens[1:T+1] (next-token prediction).
    """

    def __init__(self, split: str, block_size: int = BLOCK_SIZE):
        assert split in ("train", "val", "test")
        self.path = SPLIT_DIR / f"{split}.bin"
        self.data = np.memmap(str(self.path), dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        # Number of non-overlapping chunks
        return max(1, len(self.data) // self.block_size - 1)

    def __getitem__(self, idx: int):
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def get_dataloader(split: str, batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    ds = ShakespeareDataset(split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
    split_data()
    # Sanity check
    dl = get_dataloader("train", batch_size=4)
    x, y = next(iter(dl))
    print(f"Batch shape: x={tuple(x.shape)}, y={tuple(y.shape)}")