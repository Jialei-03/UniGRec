import torch
import torch.nn as nn

from .vq import VectorQuantizer, SoftVectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta=0.25,
                 kmeans_init=True, kmeans_iters=100, sk_iters=100,
                 vq_type='standard', distance='l2', temperature=0.1,
                 diversity_scale=0.0):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_type = vq_type.lower()
        self.distance = distance
        self.temperature = temperature

        self.last_diversity_loss = None
        self.last_diversity_loss_scaled = None

        # Create quantizers based on VQ type
        self.vq_layers = nn.ModuleList()
        for n_e, sk_epsilon in zip(n_e_list, sk_epsilons):
            if self.vq_type == 'standard':
                quantizer = VectorQuantizer(
                    n_e=n_e, e_dim=e_dim, beta=self.beta,
                    kmeans_init=self.kmeans_init, kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon, sk_iters=sk_iters
                )
            elif self.vq_type == 'soft':
                quantizer = SoftVectorQuantizer(
                    n_e=n_e, e_dim=e_dim, beta=self.beta,
                    kmeans_init=self.kmeans_init, kmeans_iters=self.kmeans_iters,
                    distance=self.distance, temperature=self.temperature,
                    diversity_scale=diversity_scale
                )
            else:
                raise ValueError(f"Unknown VQ type: {self.vq_type}")
            
            self.vq_layers.append(quantizer)

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def set_temperature(self, temperature: float):
        """Update temperature for all quantizers that support it."""
        self.temperature = temperature
        for quantizer in self.vq_layers:
            if hasattr(quantizer, "set_temperature"):
                quantizer.set_temperature(temperature)
            elif hasattr(quantizer, "tau"):
                quantizer.tau = temperature

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []
        all_diversity_losses = []
        all_diversity_losses_scaled = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

            div_loss = getattr(quantizer, 'last_diversity_loss', None)
            div_loss_scaled = getattr(quantizer, 'last_diversity_loss_scaled', None)
            if div_loss is not None:
                all_diversity_losses.append(div_loss)
            if div_loss_scaled is not None:
                all_diversity_losses_scaled.append(div_loss_scaled)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        if all_diversity_losses:
            self.last_diversity_loss = torch.stack(all_diversity_losses).mean()
        else:
            self.last_diversity_loss = None

        if all_diversity_losses_scaled:
            self.last_diversity_loss_scaled = torch.stack(all_diversity_losses_scaled).mean()
        else:
            self.last_diversity_loss_scaled = None

        return x_q, mean_losses, all_indices
    
    def forward_soft(self, x, tau=1.0, hard=False):
        """
        Residual soft quantization (for end-to-end training)
        
        Args:
            x: Input vectors (B, e_dim)
            tau: Softmax temperature
            hard: Whether to harden
        
        Returns:
            x_q: Quantized vectors (B, e_dim)
            all_soft_indices: Soft probability distributions for all layers List[(B, n_e)], length=num_layers
            mean_losses: Average VQ loss
        """
        all_losses = []
        all_soft_indices = []
        
        x_q = 0
        residual = x
        
        for quantizer in self.vq_layers:
            # Soft quantization for each layer
            x_res, soft_idx, loss = quantizer.forward_soft(residual, tau=tau, hard=hard)
            
            residual = residual - x_res
            x_q = x_q + x_res
            
            all_losses.append(loss)
            all_soft_indices.append(soft_idx)  # (B, n_e)
        
        mean_losses = torch.stack(all_losses).mean()
        
        return x_q, all_soft_indices, mean_losses