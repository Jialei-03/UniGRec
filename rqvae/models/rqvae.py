import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 norm_type="none",
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=True,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 vq_type='soft',   # 'standard' | 'soft' (SoftVectorQuantizer)
                 distance='l2',       # 'l2', 'dot', 'cos'
                 temperature=0.01,     # Temperature parameter (soft only)
                 diversity_scale=1e-4, # Diversity loss weight (soft only, reduces collision rate)
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.norm_type = (norm_type or "none")
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        
        self.vq_type = vq_type
        self.distance = distance
        self.temperature = temperature

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn, norm_type=self.norm_type)

        self.rq = ResidualVectorQuantizer(
            n_e_list=num_emb_list, 
            e_dim=e_dim,
            sk_epsilons=self.sk_epsilons,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_iters=self.sk_iters,
            vq_type=self.vq_type,
            distance=self.distance,
            temperature=self.temperature,
            diversity_scale=diversity_scale
        )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob, bn=self.bn, norm_type=self.norm_type)

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices
    
    def encode_soft(self, x, tau=1.0, hard=False):
        """
        Soft encoding (for end-to-end training)
        
        Args:
            x: Input embeddings (B, in_dim)
            tau: Temperature (default 1.0)
            hard: Whether to harden
        
        Returns:
            all_soft_indices: List[(B, n_e)], length=num_layers
            x_q: Quantized vectors (B, e_dim)
        """
        z = self.encoder(x)  # (B, e_dim)
        x_q, all_soft_indices, _ = self.rq.forward_soft(z, tau=tau, hard=hard)
        
        return all_soft_indices, x_q
    
    def forward_soft(self, x, tau=1.0, hard=False):
        """
        Complete soft forward pass (for joint training)
        
        Args:
            x: Input embeddings (B, in_dim)
            tau: Temperature
            hard: Whether to harden
        
        Returns:
            out: Reconstructed output (B, in_dim)
            rq_loss: VQ loss
            all_soft_indices: List[(B, n_e)]
        """
        z = self.encoder(x)
        x_q, all_soft_indices, rq_loss = self.rq.forward_soft(z, tau=tau, hard=hard)
        
        out = self.decoder(x_q)
        return out, rq_loss, all_soft_indices

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon