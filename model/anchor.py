"""Dual Collaborative Anchoring.

Collaborative item anchors are computed from the fixed content embedding x_i by
an anchor network phi (two-layer MLP without biases); a momentum copy produces
slowly moving anchors used as alignment targets:

    u_i = norm(phi(norm(x_i)))            (online anchor)
    a_i = norm(phi_bar(norm(x_i)))        (momentum anchor, phi_bar <- m phi_bar + (1-m) phi)

The anchors become collaborative through L_Seq, a full-catalog next-item
objective of a one-layer causal Transformer over the online anchors. The
recommender is aligned with the momentum anchors on the input side (L_Hist,
encoder states of historical items) and on the output side (L_Tgt, the BOS
decoder state). The anchor network is trained by L_Seq only.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLinear(nn.Module):
    """Square projection without bias, initialized to the identity."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(int(dim)))

    def forward(self, x):
        return F.linear(x, self.weight)


class CollaborativeAnchor(nn.Module):

    def __init__(self, catalog_embeddings, dim=128, hidden_dim=128, temperature=0.07, momentum=0.99,
                 sequence_max_len=20, sequence_num_heads=4, init_seed=17):
        super().__init__()
        catalog_embeddings = torch.as_tensor(catalog_embeddings, dtype=torch.float32)
        self.num_items, self.text_dim = catalog_embeddings.shape
        self.dim = int(dim)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.sequence_max_len = int(sequence_max_len)
        # kappa: scales unit-norm anchors to the magnitude of the token embeddings.
        self.scale = 0.02 * math.sqrt(self.dim)
        # The content table is an input, not a learned parameter.
        self.register_buffer("catalog", catalog_embeddings.contiguous(), persistent=False)

        # Auxiliary modules are initialized from local seeds so that they do not
        # change the global random stream of the recommender.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(init_seed))
            self.phi = nn.Sequential(
                nn.Linear(self.text_dim, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, self.dim, bias=False),
            )
        self.phi_momentum = copy.deepcopy(self.phi)
        self.phi_momentum.requires_grad_(False)

        self.query_projection = IdentityLinear(self.dim)       # W_q (output side)
        self.history_projection = IdentityLinear(self.dim)     # W_h (input side)
        self.sequence_projection = IdentityLinear(self.dim)    # W_z (sequence state)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(init_seed) + 1)
            self.position_embeddings = nn.Embedding(self.sequence_max_len, self.dim)
            self.sequence_encoder = nn.TransformerEncoderLayer(
                d_model=self.dim, nhead=sequence_num_heads, dim_feedforward=4 * self.dim, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True)

    # ------------------------------------------------------------------ anchors
    def anchors(self, content, momentum=False):
        """Anchors of content embeddings, scaled to norm kappa (u_i or a_i up to scale)."""
        network = self.phi_momentum if momentum else self.phi
        return F.normalize(network(F.normalize(content, dim=-1)), dim=-1) * self.scale

    @torch.no_grad()
    def update_momentum(self):
        for target, online in zip(self.phi_momentum.parameters(), self.phi.parameters()):
            target.mul_(self.momentum).add_(online, alpha=1.0 - self.momentum)

    # --------------------------------------------------------- sequence (L_Seq)
    def sequence_state(self, history_embs, attention_mask):
        """z: normalized output of the causal Transformer at the last history position."""
        mask = attention_mask.bool()
        history_anchors = self.anchors(history_embs) * mask.unsqueeze(-1)
        # Move the (left-padded) valid items to the front.
        compact_positions = mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        compacted = torch.zeros_like(history_anchors)
        compacted.scatter_add_(1, compact_positions.unsqueeze(-1).expand_as(history_anchors), history_anchors)
        lengths = mask.sum(dim=1)
        length = compacted.size(1)
        if length > self.sequence_max_len:
            raise ValueError(f"history length {length} exceeds sequence_max_len={self.sequence_max_len}")
        positions = torch.arange(length, device=compacted.device)
        compact_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        inputs = (compacted + self.position_embeddings(positions).unsqueeze(0)) * compact_mask.unsqueeze(-1)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=compacted.device), diagonal=1)
        outputs = self.sequence_encoder(inputs, src_mask=causal, src_key_padding_mask=~compact_mask)
        state = outputs[torch.arange(outputs.size(0), device=outputs.device), lengths - 1]
        return F.normalize(self.sequence_projection(state), dim=-1)

    def sequence_loss(self, history_embs, attention_mask, target_ids):
        """L_Seq: full-catalog InfoNCE with the online anchors as classifier weights."""
        z = self.sequence_state(history_embs, attention_mask)
        catalog = F.normalize(self.anchors(self.catalog), dim=-1)
        logits = z @ catalog.t() / self.temperature
        return F.cross_entropy(logits, target_ids - 1)

    # ------------------------------------------------------- output side (L_Tgt)
    def query(self, bos_states):
        return F.normalize(self.query_projection(bos_states), dim=-1)

    def target_loss(self, bos_states, target_embs, target_ids):
        """L_Tgt: symmetric in-batch InfoNCE between q and the momentum anchor of the target.

        Users sharing a target item are merged (first occurrence kept), so a
        target is never used as a negative for a user with the same target.
        """
        _, inverse = target_ids.unique(sorted=True, return_inverse=True)
        positions = torch.arange(target_ids.numel(), device=target_ids.device)
        first = torch.full((int(inverse.max()) + 1,), target_ids.numel(), dtype=torch.long, device=target_ids.device)
        first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
        if first.numel() < 2:
            return bos_states.sum() * 0.0
        q = self.query(bos_states.index_select(0, first))
        a = F.normalize(self.anchors(target_embs.index_select(0, first), momentum=True), dim=-1).detach()
        logits = q @ a.t() / self.temperature
        labels = torch.arange(first.numel(), device=q.device)
        return F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)

    # -------------------------------------------------------- input side (L_Hist)
    def history_loss(self, encoder_layer_states, history_embs, attention_mask, tokens_per_item=3):
        """L_Hist: cosine alignment of every encoder layer's item states with the momentum anchors."""
        valid = attention_mask.bool()
        batch, length = valid.shape
        targets = F.normalize(self.anchors(history_embs[valid], momentum=True), dim=-1).detach()
        losses = []
        for states in encoder_layer_states:
            items = states.reshape(batch, length, tokens_per_item, -1).mean(dim=2)
            g = F.normalize(self.history_projection(items[valid]), dim=-1)
            losses.append(1.0 - (g * targets).sum(dim=-1).mean())
        return torch.stack(losses).mean()
