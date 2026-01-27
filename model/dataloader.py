import torch
from torch.utils.data import DataLoader

class GenRecDataLoader(DataLoader):
    """
    GenRecDataLoader for Generative Recommendation tasks.
    
    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): Number of subprocesses to use for data loading.
        device (str): Device used when prefetching batches to GPU.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4,
                 device='cuda:0', pin_memory=True,
                 pin_memory_device=None, prefetch_factor=4, prefetch_to_gpu=False):
        self.device = torch.device(device)
        self.prefetch_to_gpu = prefetch_to_gpu
        collate_fn = self.collate_fn
        
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        
        if pin_memory_device is None and pin_memory and self.device.type.startswith('cuda'):
            pin_memory_device = self.device.type + (f":{self.device.index}" if self.device.index is not None else "")
        if pin_memory_device is not None:
            loader_kwargs['pin_memory_device'] = pin_memory_device
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = prefetch_factor
        
        super(GenRecDataLoader, self).__init__(**loader_kwargs)
    
    def __iter__(self):
        """
        Override __iter__ to move batches in the main process (avoid multiprocess + CUDA conflicts).
        """
        for batch in super().__iter__():
            yield self._move_batch_to_device_if_enabled(batch)
    
    def collate_fn(self, batch):
        """
        Create batched data. Supports real-time soft mode.
        
        Args:
            batch (list): List of samples from the dataset.
        
        Returns:
            dict: Batched data with padded sequences.
                  Real-time soft: {'history_embs', 'target_embs', 'attention_mask', 'target_item_id'}
        """
        history_embs = torch.stack([item['history_embs'] for item in batch])
        target_embs = torch.stack([item['target_emb'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])

        result = {
            'history_embs': history_embs,
            'target_embs': target_embs,
            'attention_mask': attention_mask,
        }
        if 'target_item_id' in batch[0]:
            result['target_item_id'] = [item['target_item_id'] for item in batch]
        return result

    def _move_batch_to_device_if_enabled(self, batch):
        if not self.prefetch_to_gpu or batch is None:
            return batch
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, list) and value and torch.is_tensor(value[0]):
                batch[key] = [tensor.to(self.device, non_blocking=True) for tensor in value]
        return batch
