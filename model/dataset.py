import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import os
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rqvae'))
from models.rqvae import RQVAE
from datasets import EmbDataset

def process_data(file_path, mode, max_len, PAD_TOKEN=0):
    """
    Process parquet data based on mode ('train' or 'evaluation').

    Args:
        file_path (str): Path to the parquet file.
        mode (str): Mode of operation ('train' or 'evaluation').
        max_len (int): Maximum length for padding or truncation.

    Returns:
        list: Processed data.
    """
    # Load parquet data
    data = pd.read_parquet(file_path)

    # Combine "history" and "target" columns into a single sequence
    # Important: Ensure 'history' is a list and 'target' is appended correctly
    data['sequence'] = data['history'].apply(lambda x: list(x)) + data['target'].apply(lambda x: [x])

    if mode == 'train':
        # Sliding window processing
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            for i in range(1, len(sequence)):
                processed_data.append({
                    'history': sequence[:i],
                    'target': sequence[i],
                })
    elif mode == 'evaluation':
        # Use the last item as target and the rest as history
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            processed_data.append({
                'history': sequence[:-1],
                'target': sequence[-1],
            })
    else:
        raise ValueError("Mode must be 'train' or 'evaluation'.")

    # Apply padding or truncation
    for item in processed_data:
        item['history'] = pad_or_truncate(item['history'], max_len)

    return processed_data

def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    """
    Pad or truncate a sequence to a specified maximum length.

    Args:
        sequence (list): Input sequence.
        max_len (int): Maximum length for the sequence.

    Returns:
        list: Padded or truncated sequence.
    """
    if len(sequence) > max_len:
        # Truncate sequence
        return sequence[-max_len:]
    else:
        # Left pad sequence with PAD_TOKEN
        return [PAD_TOKEN] * (max_len - len(sequence)) + sequence

# Global cache for RQVAE model to avoid multiple loading
_rqvae_cache = {}

def load_rqvae_model(rqvae_model_path, item_emb_path, device='cuda:0', codebook_size=256):
    """
    Load trained RQVAE model and create item-to-code converter (with caching)
    :param rqvae_model_path: Path to trained RQVAE model
    :param item_emb_path: Path to item embedding data
    :param device: Device to load model on
    :param codebook_size: Size of each codebook (default 256)
    :return: RQVAE model, item embeddings, codebook_size
    """
    cache_key = (rqvae_model_path, item_emb_path, device)
    
    # Check if already loaded
    if cache_key in _rqvae_cache:
        print(f"Using cached RQVAE model and embeddings")
        return _rqvae_cache[cache_key]
    
    # Load model checkpoint
    print(f"Loading RQVAE model from: {rqvae_model_path}")
    if not os.path.exists(rqvae_model_path):
        raise FileNotFoundError(f"RQVAE model file not found: {rqvae_model_path}")
        
    ckpt = torch.load(rqvae_model_path, map_location=torch.device('cpu'), weights_only=False)
    model_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    # Load item embeddings
    print(f"Loading item embeddings from: {item_emb_path}")
    if not os.path.exists(item_emb_path):
        raise FileNotFoundError(f"Item embedding file not found: {item_emb_path}")
        
    emb_dataset = EmbDataset(item_emb_path)
    
    # Create RQVAE model
    model = RQVAE(
        in_dim=emb_dataset.dim,
        num_emb_list=getattr(model_args, 'num_emb_list', [256, 256, 256]),
        e_dim=getattr(model_args, 'e_dim', 32),
        layers=getattr(model_args, 'layers', [512, 256, 128, 64]),
        dropout_prob=getattr(model_args, 'dropout_prob', 0.0),
        bn=getattr(model_args, 'bn', False),
        norm_type=getattr(model_args, 'norm_type', 'none'),
        loss_type=getattr(model_args, 'loss_type', 'mse'),
        quant_loss_weight=getattr(model_args, 'quant_loss_weight', 1.0),
        beta=getattr(model_args, 'beta', 0.25),
        kmeans_init=getattr(model_args, 'kmeans_init', True),
        kmeans_iters=getattr(model_args, 'kmeans_iters', 100),
        sk_epsilons=getattr(model_args, 'sk_epsilons', [0.0, 0.0, 0.0]),
        sk_iters=getattr(model_args, 'sk_iters', 50),
        # VQ params (read directly from checkpoint args to match training exactly)
        vq_type=getattr(model_args, 'vq_type', 'soft'),
        distance=getattr(model_args, 'distance', 'l2'),
        temperature=getattr(model_args, 'temperature', 0.001),
    )
    
    # Load model state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Restore training temperature (consistent with load_rqvae_checkpoint)
    # This ensures the RQVAE used during dataset init matches the RQVAE state used in training
    saved_temperature = ckpt.get('current_temperature', None)
    if saved_temperature is not None:
        if hasattr(model, 'rq') and hasattr(model.rq, 'set_temperature'):
            model.rq.set_temperature(saved_temperature)
    else:
        # If temperature is not saved, try to compute final temperature from args
        if model_args is not None:
            initial_temp = getattr(model_args, 'temperature', None)
            final_temp = getattr(model_args, 'temperature_final', initial_temp)
            target_temp = final_temp
            if hasattr(model, 'rq') and hasattr(model.rq, 'set_temperature'):
                model.rq.set_temperature(target_temp)

    
    print(f"RQVAE model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Cache the result
    result = (model, emb_dataset, codebook_size)
    _rqvae_cache[cache_key] = result
    
    return result

class GenRecDataset(Dataset):
    def __init__(self, dataset_path, rqvae_model_path, item_emb_path, mode, max_len, PAD_TOKEN=0, device='cuda:0', use_code_offset=False, real_time_soft=True, max_collision=100):
        """
        Initialize the GenRecDataset with real-time RQVAE code conversion.
        Args:
            dataset_path (str): Path to the dataset file.
            rqvae_model_path (str): Path to trained RQVAE model.
            item_emb_path (str): Path to item embedding data.
            mode (str): Mode of operation ('train' or 'evaluation').
            max_len (int): Maximum length for padding or truncation.
            PAD_TOKEN (int, optional): Token used for padding. Defaults to 0.
            device (str): Device to load RQVAE model on.
            use_code_offset (bool): Whether to apply offset to codes from different layers. Defaults to False.
            real_time_soft (bool): Whether to use real-time soft quantization.
        """
        self.dataset_path = dataset_path
        self.rqvae_model_path = rqvae_model_path
        self.item_emb_path = item_emb_path
        self.mode = mode
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.device = device
        self.use_code_offset = use_code_offset
        self.real_time_soft = real_time_soft
        
        # Load RQVAE model and item embeddings
        self.rqvae_model, self.emb_dataset, self.codebook_size = load_rqvae_model(
            rqvae_model_path, item_emb_path, device
        )
        self.all_item_embs = torch.from_numpy(self.emb_dataset.embeddings).float()
        self.pad_emb = torch.zeros(self.all_item_embs.shape[1])
        
        # Process the dataset (do not convert to codes; keep original item_ids)
        self.data = self._prepare_data()
    
    def _prepare_data(self):
        """
        Process the dataset (do not convert to codes; keep original item_ids).
        Returns:
            list: Processed data with original item_ids.
        """
        # Process the data using the process_data function
        processed_data = process_data(
            self.dataset_path, self.mode, self.max_len, self.PAD_TOKEN
        )
        
        # Keep original item_ids without code conversion
        # Code conversion will be performed uniformly in the training loop using the training-time RQVAE model
        for item in processed_data:
            item['history_item_ids'] = item['history'].copy()  # keep original item_ids
            item['target_item_id'] = item['target']  # keep original target_id
            # history and target remain as original item_ids (do not convert to codes)
        
        return processed_data
    
    def __getitem__(self, index):
        """
        Get a single data item by index.
        Args:
            index (int): Index of the data item.
        Returns:
            dict: Contains 'history_embs', 'target_emb', 'attention_mask', 'target_item_id'
        """
        item = self.data[index]
        
        history_item_ids = item['history_item_ids']  # List[int]，原始item_ids
        target_item_id = item['target_item_id']      # int，原始target_id
        
        # 获取item embeddings
        ids = torch.tensor(history_item_ids, dtype=torch.long)
        mask = (ids != self.PAD_TOKEN).long()
        index = torch.clamp(ids - 1, min=0)
        history_embs = self.all_item_embs[index]
        history_embs = history_embs * mask.unsqueeze(-1)
        attention_mask = mask
        if target_item_id != self.PAD_TOKEN:
            target_emb = self.all_item_embs[target_item_id - 1]
        else:
            target_emb = self.pad_emb.clone()
        
        return {
            'history_embs': history_embs,  # (seq_len, emb_dim)
            'target_emb': target_emb,      # (emb_dim,)
            'target_item_id': target_item_id,  # int, original target item ID (for evaluation)
            'attention_mask': attention_mask,  # (seq_len,)
        }
    
    def __len__(self):
        """
        Get the total number of data.
        Returns:
            int: Total number of data.
        """
        return len(self.data)
