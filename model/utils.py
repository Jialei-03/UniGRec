import random
import numpy as np
import torch
from collections import defaultdict


def set_seed(seed):
    """Set all random seeds to ensure experiment reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensure CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def calculate_codebook_usage(rqvae, item_embeddings, device):
    """
    Compute the codebook utilization for each layer (see analyze_codebook_usage.py).
    
    Args:
        rqvae: RQVAE model
        item_embeddings: All item embeddings (N, emb_dim)
        device: Device
    
    Returns:
        usage_rates: List[float] x 3 layers, usage per layer = (# used codes) / (total codebook size)
        usage_counts: List[int] x 3 layers, number of actually used codes per layer
    """
    rqvae.eval()
    with torch.no_grad():
        embs = item_embeddings.to(device)
        # Get indices: tensor of shape (N, num_layers)
        indices = rqvae.get_indices(embs, use_sk=False)
        indices_np = indices.cpu().numpy()
        
        usage_rates = []
        usage_counts = []
        
        # Iterate over layers
        for layer_idx in range(indices.shape[1]):
            # Codes of this layer
            layer_codes = indices_np[:, layer_idx]
            # Codebook size
            codebook_size = rqvae.rq.vq_layers[layer_idx].n_e
            # Number of unique codes actually used
            unique_codes = len(np.unique(layer_codes))
            # Utilization
            utilization = unique_codes / codebook_size
            
            usage_rates.append(utilization)
            usage_counts.append(unique_codes)
    
    return usage_rates, usage_counts


def calculate_collision_rate(rqvae, item_embeddings, device):
    """
    Compute the overall collision rate based on the first 3 code layers (excluding collision index).
    See analyze_codebook_usage.py for reference.
    
    Note: we only compute collision rate on the first 3 layers. The 4th layer is a collision index;
    once included, collisions are resolved.
    
    Args:
        rqvae: RQVAE model
        item_embeddings: All item embeddings (N, emb_dim)
        device: Device
    
    Returns:
        collision_rate: float, collision rate = (total_count - unique_count) / total_count
        unique_count: int, number of unique code combinations (first 3 layers)
        total_count: int, total number of items
    """
    rqvae.eval()
    with torch.no_grad():
        embs = item_embeddings.to(device)
        # Get indices: tensor of shape (N, num_layers)
        indices = rqvae.get_indices(embs, use_sk=False)
        indices_np = indices.cpu().numpy()
        
        n_items = len(item_embeddings)
        
        # Use only the first 3 layers to build code-combination strings (e.g. "12-45-78")
        # Note: exclude the 4th layer (collision index), since including it removes collisions
        all_codes = []
        for idx_seq in indices_np:
            # Take first 3 layers only
            code_str = "-".join([str(int(c)) for c in idx_seq[:3]])
            all_codes.append(code_str)
        
        # Number of unique code combinations (first 3 layers)
        unique_codes = len(set(all_codes))
        
        # collision rate = (total - unique) / total
        collision_rate = (n_items - unique_codes) / n_items if n_items > 0 else 0.0
    
    return collision_rate, unique_codes, n_items


def calculate_soft_entropy(rqvae, item_embeddings, tau, device):
    """
    Compute the average entropy of soft distributions (measuring distribution "sharpness").
    
    Args:
        rqvae: RQVAE model
        item_embeddings: All item embeddings (N, emb_dim)
        tau: Softmax temperature
        device: Device
    
    Returns:
        avg_entropies: List[float] x 3 layers, average entropy per layer
        Lower entropy -> sharper distribution (closer to one-hot)
        Uniform entropy: log(256) ~= 5.55
    """
    rqvae.eval()
    with torch.no_grad():
        embs = item_embeddings.to(device)
        soft_list, _ = rqvae.encode_soft(embs, tau=tau)
        
        avg_entropies = []
        for layer_idx in range(len(soft_list)):
            soft_indices = soft_list[layer_idx]  # (N, 256)
            
            # Entropy: H = -Σ p*log(p)
            entropy = -torch.sum(
                soft_indices * torch.log(soft_indices + 1e-10), 
                dim=1
            ).mean()
            
            avg_entropies.append(entropy.item())
    
    return avg_entropies
    


def calculate_pos_index(preds, labels, beam_size, code_to_item, target_item_ids=None,
                        use_code_offset=False, codebook_size=256):
    """
    Compute position index (4-token code version).
    
    Args:
        preds: (B, beam_size, 4) - generated 4-token embedding IDs
        labels: (B, 4) - target 4-token embedding IDs
        beam_size: number of beams
        code_to_item: dict - {4-token code_tuple: item_id}
            Format: (raw_code0, raw_code1, raw_code2, collision_idx)
        target_item_ids: (B,) - target item IDs (optional, for verification)
        use_code_offset: whether to use code-offset encoding
        codebook_size: codebook size
    
    Returns:
        pos_index: (B, beam_size) boolean tensor; pos_index[i, j] = True indicates the j-th beam is correct
    """
    B = preds.shape[0]
    # Return a 2D boolean tensor (B, beam_size), not a 1D position index
    pos_index = torch.zeros((B, beam_size), dtype=torch.bool)
    
    # Start embedding id for collision indices
    # Offset mode: 769 (3*256+1)
    # Non-offset mode: 257 (256+1)
    # Note: derived from vocab_size; in non-offset mode vocab_size=358, so collision indices start at 257
    if use_code_offset:
        collision_embedding_start = 3 * codebook_size + 1  # 769
    else:
        collision_embedding_start = codebook_size + 1  # 257
    
    for i in range(B):
        # Convert label embedding IDs to raw codes
        label_emb_ids = labels[i].cpu().numpy().astype(int)  # should be (4,)
        
        # Validate length
        if len(label_emb_ids) < 4:
            # Not enough tokens; skip this sample (all beams remain False)
            continue
        
        if use_code_offset:
            # Offset mode: reverse mapping
            raw_code0 = label_emb_ids[0] - 1  # 1-256 -> 0-255
            raw_code1 = label_emb_ids[1] - 257  # 257-512 -> 0-255
            raw_code2 = label_emb_ids[2] - 513  # 513-768 -> 0-255
        else:
            # Non-offset mode: reverse mapping
            raw_code0 = label_emb_ids[0] - 1  # 1-256 -> 0-255
            raw_code1 = label_emb_ids[1] - 1  # 1-256 -> 0-255
            raw_code2 = label_emb_ids[2] - 1  # 1-256 -> 0-255
        
        # Collision index: embedding ID - collision_embedding_start
        collision_idx = label_emb_ids[3] - collision_embedding_start  # 769-869 -> 0-100
        
        target_code = (raw_code0, raw_code1, raw_code2, collision_idx)
        target_item_id = code_to_item.get(target_code, None)
        
        if target_item_id is None:
            # If missing, fall back to target_item_ids (if provided)
            if target_item_ids is not None:
                target_item_id = target_item_ids[i]
            else:
                # Not found; all beams remain False
                continue
        
        # Search within beams
        for j in range(beam_size):
            # Convert predicted embedding IDs to raw codes
            pred_emb_ids_raw = preds[i, j].cpu().numpy()  # may be a scalar or an array
            
            # Ensure pred_emb_ids is a 1D array
            if pred_emb_ids_raw.ndim == 0:
                # Scalar -> 1D array (should not happen, but handle defensively)
                pred_emb_ids = np.array([pred_emb_ids_raw.item()])
            elif pred_emb_ids_raw.ndim > 1:
                pred_emb_ids = pred_emb_ids_raw.flatten().astype(int)
            else:
                pred_emb_ids = pred_emb_ids_raw.astype(int)
            
            # Validate length: skip if sequence is too short
            if len(pred_emb_ids) < 4:
                # Generated sequence too short; skip this beam
                continue
            
            # Safe indexing
            try:
                if use_code_offset:
                    pred_raw_code0 = int(pred_emb_ids[0]) - 1
                    pred_raw_code1 = int(pred_emb_ids[1]) - 257
                    pred_raw_code2 = int(pred_emb_ids[2]) - 513
                else:
                    pred_raw_code0 = int(pred_emb_ids[0]) - 1
                    pred_raw_code1 = int(pred_emb_ids[1]) - 1
                    pred_raw_code2 = int(pred_emb_ids[2]) - 1
                
                pred_collision_idx = int(pred_emb_ids[3]) - collision_embedding_start
            except (IndexError, ValueError) as e:
                # If indexing still fails, print debug info and skip
                print(f"  Warning: pred_emb_ids access failed, shape={pred_emb_ids.shape}, len={len(pred_emb_ids)}, error={e}. Skipping this beam")
                continue
            
            pred_code = (pred_raw_code0, pred_raw_code1, pred_raw_code2, pred_collision_idx)
            pred_item_id = code_to_item.get(pred_code, None)
            
            if pred_item_id == target_item_id:
                pos_index[i, j] = True
                break  # stop at the first match (beam search is already score-sorted)
    
    return pos_index


def recall_at_k(pos_index, k):
    """Compute Recall@k (same as the original main.py)."""
    return pos_index[:, :k].sum(dim=1).float()


def ndcg_at_k(pos_index, k):
    """Compute NDCG@k (same as the original main.py)."""
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).float()

def compute_code_change_rate(old_codes, new_codes):
    """
    Compute code change rate.
    
    Args:
        old_codes: (N, 3) old hard codes
        new_codes: (N, 3) new hard codes
    
    Returns:
        change_rate: fraction of changed samples
    """
    changed = (old_codes != new_codes).any(dim=-1).sum().item()
    total = old_codes.shape[0]
    return changed / total

def generate_labels_from_mapping(
    target_item_ids, item_to_code, use_code_offset=False, 
    codebook_size=256, device='cuda:0'
):
    """
    Query the 4-token code for target items from a mapping table (no real-time computation),
    and convert the 4-token code to embedding IDs.
    
    Args:
        target_item_ids: (B,) - target item IDs (1-indexed)
        item_to_code: dict - {item_id: 4-token code_tuple}
            Format: (raw_code0, raw_code1, raw_code2, collision_idx)
            raw_code is the raw RQVAE code (0-255)
        use_code_offset: whether to use code-offset encoding
        codebook_size: codebook size
        device: device
    
    Returns:
        decoder_input_ids: (B, 5) = [0, emb_id0, emb_id1, emb_id2, emb_id_collision]
        labels: (B, 5) = [emb_id0, emb_id1, emb_id2, emb_id_collision, 0] (right-shifted, ending with 0)
    """
    B = len(target_item_ids)
    decoder_input_ids_list = []
    labels_list = []
    
    # Start embedding id for collision indices
    # Offset mode: 769 (3*256+1)
    # Non-offset mode: 257 (256+1)
    # Note: derived from vocab_size; in non-offset mode vocab_size=358, so collision indices start at 257
    if use_code_offset:
        collision_embedding_start = 3 * codebook_size + 1  # 769
    else:
        collision_embedding_start = codebook_size + 1  # 257
    
    for item_id in target_item_ids:
        if item_id not in item_to_code:
            raise ValueError(f"Item {item_id} is not in the mapping table")
        
        code_tuple = item_to_code[item_id]  # (raw_code0, raw_code1, raw_code2, collision_idx)
        raw_code0, raw_code1, raw_code2, collision_idx = code_tuple
        
        # Convert raw codes to embedding IDs
        if use_code_offset:
            # Offset mode: each layer uses a different embedding range
            emb_id0 = raw_code0 + 1  # 1-256
            emb_id1 = raw_code1 + 257  # 257-512
            emb_id2 = raw_code2 + 513  # 513-768
        else:
            # Non-offset mode: all layers share embedding[1:257]
            emb_id0 = raw_code0 + 1  # 1-256
            emb_id1 = raw_code1 + 1  # 1-256
            emb_id2 = raw_code2 + 1  # 1-256
        
        # Collision index embedding id = collision_embedding_start + collision_idx
        emb_id_collision = collision_embedding_start + collision_idx  # 769-869
        
        # Build embedding-id sequence
        embedding_ids = torch.tensor(
            [emb_id0, emb_id1, emb_id2, emb_id_collision],
            dtype=torch.long,
            device=device
        )  # (4,)
        
        # Build decoder_input_ids (including start token)
        start_token = torch.zeros(1, dtype=torch.long, device=device)
        decoder_input_ids = torch.cat([start_token, embedding_ids]).unsqueeze(0)  # (1, 5)
        
        # Build labels (right-shifted, ending with 0)
        labels = torch.cat([embedding_ids, torch.zeros(1, dtype=torch.long, device=device)]).unsqueeze(0)  # (1, 5)
        
        decoder_input_ids_list.append(decoder_input_ids)
        labels_list.append(labels)
    
    decoder_input_ids = torch.cat(decoder_input_ids_list, dim=0)  # (B, 5)
    labels = torch.cat(labels_list, dim=0)  # (B, 5)
    
    return decoder_input_ids, labels


def build_code_with_collision_index(rqvae, item_embeddings, use_code_offset=True, 
                                     codebook_size=256, device='cuda:0', max_collision=100):
    """
    Build 4-token codes (first 3 tokens are RQVAE codes + 4th token is collision index).
    
    Note: this function stores **raw codes** (0-255), not embedding IDs.
    When used in the decoder, they must be converted to embedding IDs.
    
    Args:
        rqvae: RQVAE model
        item_embeddings: (N, emb_dim) all item embeddings
        use_code_offset: whether to use code-offset encoding (used later when converting to embedding IDs)
        codebook_size: codebook size
        device: device
        max_collision: maximum collision count (for validation)
    
    Returns:
        all_item_codes: (N+1, 4) - 4-token code per item (row 0 is PAD token [-1,-1,-1,-1])
            Format: [raw_code0, raw_code1, raw_code2, collision_idx]
            raw_code is the raw RQVAE code (0-255)
        code_to_item: dict - {4-token code_tuple: item_id} unique mapping
        item_to_code: dict - {item_id: 4-token code_tuple} reverse mapping
        collision_stats: dict - collision statistics
    """
    
    # Ensure RQVAE is in eval mode to obtain consistent results
    rqvae.eval()
    
    all_item_codes = [[-1, -1, -1, -1]]  # PAD token
    code_to_item = {}
    item_to_code = {}
    
    # 1. Get the first 3 codes for all items (raw codes, 0-255)
    with torch.no_grad():
        batch_size = 4096  # keep consistent with dataset.py
        n_items = len(item_embeddings)
        all_prefix_codes = []
        
        for i in range(0, n_items, batch_size):
            end_idx = min(i + batch_size, n_items)
            batch_embs = item_embeddings[i:end_idx].to(device)
            indices = rqvae.get_indices(batch_embs, use_sk=False)  # (batch_size, 3)
            # indices are raw codes (0-255); no +1 or offset is needed
            
            all_prefix_codes.append(indices.cpu().numpy())
        
        all_prefix_codes = np.vstack(all_prefix_codes)  # (N, 3), raw codes (0-255)
    
    # 2. Compute collision indices (4th token)
    # Use raw codes (0-255) as prefixes, without offsets
    prefix_to_items = defaultdict(list)
    max_conflict = 0
    
    for item_id in range(1, n_items + 1):  # item_id starts from 1
        prefix = tuple(all_prefix_codes[item_id - 1].astype(int).tolist())
        prefix_to_items[prefix].append(item_id)
        max_conflict = max(max_conflict, len(prefix_to_items[prefix]))
    
    # Validate maximum collision count
    if max_conflict > max_collision:
        raise ValueError(
            f'Max collision count {max_conflict} > {max_collision}. '
            f'Please increase max_collision or codebook_size.'
        )
    
    # 3. Build 4-token codes (raw codes + collision index)
    for prefix, items in prefix_to_items.items():
        for collision_idx, item_id in enumerate(items):
            four_digit_code = list(prefix) + [collision_idx]  # [raw_code0, raw_code1, raw_code2, collision_idx]
            all_item_codes.append(four_digit_code)
            
            code_tuple = tuple(four_digit_code)
            code_to_item[code_tuple] = item_id
            item_to_code[item_id] = code_tuple
    
    all_item_codes = np.array(all_item_codes, dtype=np.int64)  # (N+1, 4)
    
    # 4. Statistics
    collision_stats = {
        'total_items': n_items,
        'unique_prefixes': len(prefix_to_items),
        'max_collision': max_conflict,
        'collision_rate': (n_items - len(prefix_to_items)) / n_items if n_items > 0 else 0.0,
    }
    
    return all_item_codes, code_to_item, item_to_code, collision_stats

