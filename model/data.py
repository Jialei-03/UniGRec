"""Sequential recommendation data: (history item IDs, target item ID) pairs."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def pad_or_truncate(sequence, max_len, pad=0):
    """Keep the latest ``max_len`` items and left-pad with ``pad``."""
    sequence = list(sequence)[-max_len:]
    return [pad] * (max_len - len(sequence)) + sequence


class SequenceDataset(Dataset):
    """Rows of a ``{train,valid,test}.parquet`` file with columns ``history`` and ``target``.

    ``mode='train'`` expands every nonempty prefix of ``history + [target]`` into a
    training instance; ``mode='eval'`` uses each row once (history -> target).
    """

    def __init__(self, path, mode, max_len=20):
        if mode not in ("train", "eval"):
            raise ValueError("mode must be train or eval")
        df = pd.read_parquet(path)
        histories, targets = [], []
        for history, target in zip(df["history"], df["target"]):
            sequence = [int(x) for x in history] + [int(target)]
            if mode == "train":
                for i in range(1, len(sequence)):
                    histories.append(pad_or_truncate(sequence[:i], max_len))
                    targets.append(sequence[i])
            else:
                histories.append(pad_or_truncate(sequence[:-1], max_len))
                targets.append(sequence[-1])
        self.history = torch.as_tensor(np.asarray(histories, dtype=np.int64).reshape(-1, max_len))
        self.target = torch.as_tensor(np.asarray(targets, dtype=np.int64))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.history[index], self.target[index]


def make_loader(dataset, batch_size, shuffle, num_workers=0, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
                      generator=generator)


class ItemTable:
    """Content embeddings indexed by item ID (row 0 is a zero padding vector)."""

    def __init__(self, embeddings, device):
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.catalog = embeddings.to(device)                       # (N, d), row i-1 = item i
        self.padded = torch.cat([torch.zeros_like(self.catalog[:1]), self.catalog])
        self.num_items = embeddings.size(0)
        self.dim = embeddings.size(1)

    def batch(self, history_ids, target_ids, device):
        history_ids = history_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        mask = history_ids.ne(0)
        return dict(history_ids=history_ids, history_embs=self.padded[history_ids],
                    attention_mask=mask, target_ids=target_ids, target_embs=self.padded[target_ids])
