import pandas as pd
import torch
import torch.utils.data as data
import numpy as np


class EmbDataset(data.Dataset):

    def __init__(self, data_path, return_item_id: bool = False):

        self.data_path = data_path
        self.return_item_id = return_item_id
        df = pd.read_parquet(data_path)
        self.embeddings = df['embedding'].values
        self.embeddings = np.stack(self.embeddings, axis=0)
        self.dim = self.embeddings.shape[-1]

        if 'ItemID' in df.columns:
            self.item_ids = df['ItemID'].to_numpy(dtype=np.int64)
        else:
            self.item_ids = np.arange(1, len(self.embeddings) + 1, dtype=np.int64)

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)

        if self.return_item_id:
            return tensor_emb, int(self.item_ids[index])
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
