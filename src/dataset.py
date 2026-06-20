from __future__ import annotations

import torch
from torch.utils.data import Dataset


class CharSequenceDataset(Dataset):
    """Next-token dataset where y is x shifted one character to the left."""

    def __init__(self, token_ids: list[int], block_size: int):
        if block_size < 1:
            raise ValueError("block_size must be positive.")
        if len(token_ids) <= block_size:
            raise ValueError("Need more token ids than block_size.")
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def split_ids(token_ids: list[int], train_ratio: float = 0.9) -> tuple[list[int], list[int]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    split = int(len(token_ids) * train_ratio)
    return token_ids[:split], token_ids[split:]
