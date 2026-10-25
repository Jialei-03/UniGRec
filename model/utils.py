"""Identifier maps, labels and ranking metrics."""
import random

import numpy as np
import torch


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def token_offsets(codebook_size, device=None):
    """Vocabulary offsets of the four identifier positions.

    Token 0 is PAD/BOS; level-l codes occupy ``[l*K+1, (l+1)*K]`` and the
    collision token starts at ``3K+1``.
    """
    k = codebook_size
    return torch.tensor([1, k + 1, 2 * k + 1, 3 * k + 1], dtype=torch.long, device=device)


@torch.no_grad()
def build_code_map(rqvae, item_embeddings, max_collision, codebook_size, batch_size=4096):
    """Exact full-catalog item -> identifier map of the current tokenizer.

    The first three tokens are the hard RQ codes (argmax of every soft level);
    items sharing the same three codes receive a collision token assigned by
    ascending item ID (0, 1, 2, ...). Returns an ``(N + 1, 4)`` long tensor
    (row 0 is padding, filled with -1) and the collision statistics.
    """
    was_training = rqvae.training
    rqvae.eval()
    try:
        prefixes = torch.cat([rqvae.get_indices(item_embeddings[s:s + batch_size])
                              for s in range(0, len(item_embeddings), batch_size)]).long()
    finally:
        rqvae.train(was_training)
    n_items = prefixes.size(0)
    k = codebook_size
    keys = (prefixes[:, 0] * k + prefixes[:, 1]) * k + prefixes[:, 2]
    order = torch.sort(keys, stable=True).indices          # stable: item-ID order inside a prefix
    sorted_keys = keys[order]
    positions = torch.arange(n_items, device=keys.device)
    new_group = torch.ones_like(sorted_keys, dtype=torch.bool)
    new_group[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_start = torch.cummax(torch.where(new_group, positions, torch.zeros_like(positions)), dim=0).values
    collision = torch.empty_like(keys)
    collision[order] = positions - group_start
    max_index = int(collision.max().item()) if n_items else 0
    if max_index > max_collision:
        raise ValueError(f"collision index {max_index} > max_collision={max_collision}; increase --max_collision")
    codes = torch.cat([prefixes, collision.unsqueeze(1)], dim=1)
    codes = torch.cat([codes.new_full((1, 4), -1), codes], dim=0)
    unique_prefixes = int(new_group.sum().item())
    stats = dict(collision_rate=(n_items - unique_prefixes) / max(n_items, 1),
                 unique_prefixes=unique_prefixes, max_collision_index=max_index)
    return codes, stats


def make_decoder_io(target_item_ids, item_codes, codebook_size):
    """Teacher-forcing inputs ``[0, t1, t2, t3]`` and labels ``[t1, t2, t3, t4]``."""
    tokens = item_codes.index_select(0, target_item_ids) + token_offsets(codebook_size, item_codes.device)
    start = torch.zeros_like(tokens[:, :1])
    return torch.cat([start, tokens[:, :3]], dim=1), tokens


class CodeLookup:
    """Vectorized map from generated tokens back to item IDs (0 = no item)."""

    def __init__(self, item_codes, codebook_size, max_collision):
        self.k = codebook_size
        self.c = max_collision + 1
        self.offsets = token_offsets(codebook_size, item_codes.device)
        keys = self._keys(item_codes[1:])
        self.sorted_keys, perm = torch.sort(keys)
        self.items = perm + 1

    def _keys(self, raw):
        return ((raw[..., 0] * self.k + raw[..., 1]) * self.k + raw[..., 2]) * self.c + raw[..., 3]

    def __call__(self, tokens):
        keys = self._keys(tokens - self.offsets.to(tokens.device))
        sorted_keys = self.sorted_keys.to(tokens.device)
        pos = torch.searchsorted(sorted_keys, keys).clamp(max=len(sorted_keys) - 1)
        found = sorted_keys[pos] == keys
        return torch.where(found, self.items.to(tokens.device)[pos], torch.zeros_like(keys))


def hit_ranks(scores_or_none, candidate_items, target_items):
    """1-based rank of the target among candidates (0 if absent).

    With ``scores_or_none=None`` the candidates are already in ranking order.
    """
    hits = candidate_items == target_items.unsqueeze(1)
    if scores_or_none is not None:
        order = torch.sort(scores_or_none, dim=1, descending=True, stable=True).indices
        hits = torch.gather(hits, 1, order)
    found = hits.any(dim=1)
    return torch.where(found, hits.float().argmax(dim=1) + 1, torch.zeros_like(target_items))


def metrics_from_ranks(ranks, topk_list=(5, 10, 20)):
    ranks = torch.as_tensor(ranks).float()
    out = {}
    for k in topk_list:
        hit = (ranks > 0) & (ranks <= k)
        out[f"recall@{k}"] = hit.float().mean().item()
        out[f"ndcg@{k}"] = torch.where(hit, 1.0 / torch.log2(ranks.clamp(min=1) + 1),
                                       torch.zeros_like(ranks)).mean().item()
    return out
