"""UniGRec: tokenizer (soft RQ-VAE) + T5 recommender + Dual Collaborative Anchoring."""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rqvae.models.rqvae import RQVAE  # noqa: E402

from anchor import CollaborativeAnchor  # noqa: E402
from t5 import SoftT5  # noqa: E402


def load_tokenizer_checkpoint(path):
    """Read a stage-1 checkpoint; returns (rqvae_config, state_dict, temperature tau*)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    if "current_temperature" not in ckpt:
        raise ValueError(f"{path} has no 'current_temperature'; re-run stage 1")
    config = dict(
        num_emb_list=list(a.num_emb_list), e_dim=a.e_dim, layers=list(a.layers),
        dropout_prob=a.dropout_prob, bn=a.bn, norm_type=getattr(a, "norm_type", "none"),
        loss_type=a.loss_type, quant_loss_weight=a.quant_loss_weight, beta=a.beta,
        kmeans_init=False, kmeans_iters=0, sk_epsilons=list(a.sk_epsilons), sk_iters=a.sk_iters,
        vq_type=str(a.vq_type).lower(), distance=a.distance, temperature=float(ckpt["current_temperature"]),
        diversity_scale=getattr(a, "diversity_scale", 0.0),
    )
    return config, ckpt["state_dict"], float(ckpt["current_temperature"])


class UniGRec(nn.Module):

    def __init__(self, rqvae_config, t5_config, catalog_embeddings, use_anchor=True,
                 anchor_hidden_dim=128, anchor_temperature=0.07, anchor_momentum=0.99,
                 sequence_max_len=20, sequence_num_heads=4, anchor_init_seed=17,
                 hard_identifier_input=False):
        super().__init__()
        rqvae_config = dict(rqvae_config, in_dim=catalog_embeddings.shape[1])
        if len(set(rqvae_config["num_emb_list"])) != 1:
            raise ValueError("all tokenizer levels must have the same codebook size")
        self.codebook_size = int(rqvae_config["num_emb_list"][0])
        self.num_levels = len(rqvae_config["num_emb_list"])
        self.hard_identifier_input = bool(hard_identifier_input)

        self.rqvae = RQVAE(**rqvae_config)
        self.t5 = SoftT5(t5_config, codebook_size=self.codebook_size, num_levels=self.num_levels)
        self.anchor = None
        if use_anchor:
            self.anchor = CollaborativeAnchor(
                catalog_embeddings, dim=t5_config.d_model, hidden_dim=anchor_hidden_dim,
                temperature=anchor_temperature, momentum=anchor_momentum,
                sequence_max_len=sequence_max_len, sequence_num_heads=sequence_num_heads,
                init_seed=anchor_init_seed)

    def load_tokenizer(self, state_dict, temperature):
        self.rqvae.load_state_dict(state_dict)
        self.rqvae.rq.set_temperature(temperature)

    # ------------------------------------------------------------ identifiers
    def soft_identifiers(self, item_embs, tau):
        """Per-level codeword distributions p_l and the soft quantized latent r~."""
        if self.hard_identifier_input:  # ablation: one-hot hard residual codes (no gradient)
            with torch.no_grad():
                codes = self.rqvae.get_indices(item_embs)
            probs = [F.one_hot(codes[:, l], self.codebook_size).to(item_embs.dtype) for l in range(self.num_levels)]
            return probs, None
        return self.rqvae.encode_soft(item_embs, tau=tau)

    def history_inputs(self, history_embs, attention_mask, tau):
        batch, length, dim = history_embs.shape
        flat = history_embs.reshape(-1, dim)
        probs, quantized = self.soft_identifiers(flat, tau)
        soft = torch.stack([p.reshape(batch, length, -1) for p in probs], dim=2)
        soft = soft.reshape(batch, length * self.num_levels, -1)
        token_mask = attention_mask.long().repeat_interleave(self.num_levels, dim=1)
        return soft, token_mask, quantized

    # -------------------------------------------------------------- training
    def forward(self, history_embs, attention_mask, target_embs, target_ids, decoder_input_ids, labels,
                tau, alpha=0.0, w_seq=0.0, w_hist=0.0, w_tgt=0.0):
        """Returns the total loss and a dict of its (unweighted) terms.

        L = L_Rec + alpha L_Recon + w_seq L_Seq + w_hist L_Hist + w_tgt L_Tgt
        """
        soft, token_mask, hist_quantized = self.history_inputs(history_embs, attention_mask, tau)
        out = self.t5(soft, token_mask, decoder_input_ids, labels)
        losses = {"rec": out.loss}
        total = out.loss

        if alpha > 0:
            # Soft reconstruction of the history (all positions) and target items.
            flat = history_embs.reshape(-1, history_embs.size(-1))
            if hist_quantized is None:
                _, hist_quantized = self.rqvae.encode_soft(flat, tau=tau)
            _, target_quantized = self.rqvae.encode_soft(target_embs, tau=tau)
            recon = (F.mse_loss(self.rqvae.decoder(hist_quantized), flat)
                     + F.mse_loss(self.rqvae.decoder(target_quantized), target_embs)) / 2
            losses["recon"] = recon
            total = total + alpha * recon

        if self.anchor is not None:
            if w_seq > 0:
                losses["seq"] = self.anchor.sequence_loss(history_embs, attention_mask, target_ids)
                total = total + w_seq * losses["seq"]
            if w_tgt > 0:
                bos = out.decoder_hidden_states[-1][:, 0, :]
                losses["tgt"] = self.anchor.target_loss(bos, target_embs, target_ids)
                total = total + w_tgt * losses["tgt"]
            if w_hist > 0:
                # One state per encoder block (index 0 of HF hidden states is the embedding output).
                layer_states = out.encoder_hidden_states[1:]
                losses["hist"] = self.anchor.history_loss(layer_states, history_embs, attention_mask,
                                                          tokens_per_item=self.num_levels)
                total = total + w_hist * losses["hist"]
        return total, losses

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, history_embs, attention_mask, tau, num_beams, max_collision):
        soft, token_mask, _ = self.history_inputs(history_embs, attention_mask, tau)
        return self.t5.generate(soft, token_mask, num_beams=num_beams, max_collision=max_collision)
