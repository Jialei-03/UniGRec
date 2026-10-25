import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


def load_item_embeddings(data_path):
    """Read an item embedding parquet (columns ``ItemID``, ``embedding``).

    Rows are returned in ItemID order; ItemIDs must be exactly 1..N so that row
    ``i - 1`` holds item ``i``.
    """
    df = pd.read_parquet(data_path)
    if "ItemID" in df.columns:
        df = df.sort_values("ItemID").reset_index(drop=True)
        ids = df["ItemID"].to_numpy(dtype=np.int64)
        if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
            raise ValueError(f"{data_path}: ItemID must be exactly 1..N")
    embeddings = np.stack(df["embedding"].values, axis=0).astype(np.float32)
    return embeddings


class EmbDataset(data.Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.embeddings = load_item_embeddings(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        return torch.from_numpy(self.embeddings[index])

    def __len__(self):
        return len(self.embeddings)
