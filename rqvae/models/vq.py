import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):
    """Standard Vector Quantizer - takes only the nearest codeword + uses STE"""

    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=False, kmeans_iters=10,
                 sk_epsilon=0.003, sk_iters=100):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

        if not self.kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True, conflict=False):
        
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, self.embedding.weight.t())
        
        # Choose quantization strategy
        if self.sk_epsilon > 0 and (use_sk and (self.training or conflict)):
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # Compute loss for embedding
        codebook_loss = F.mse_loss(x_q, x.detach())
        commitment_loss = F.mse_loss(x_q.detach(), x)
        loss = codebook_loss + self.beta * commitment_loss

        # Preserve gradients (STE)
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices
    

    def forward_soft(self, x, tau=1.0, hard=False):
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, self.embedding.weight.t())

        logits = -d / tau
        soft_indices = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

        x_q = torch.matmul(soft_indices, self.embedding.weight)
        x_q = x_q.view(x.shape)

        codebook_loss = F.mse_loss(x_q, x.detach())
        commitment_loss = F.mse_loss(x_q.detach(), x)
        loss = codebook_loss + self.beta * commitment_loss

        return x_q, soft_indices, loss


    def get_maxk_indices(self, x, maxk=1, used=False):
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, self.embedding.weight.t())
        
        # Choose quantization strategy
        if self.sk_epsilon > 0 and (self.training or used):
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.topk(Q, maxk, dim=-1).indices
        else:
            indices = torch.topk(d, maxk, dim=-1, largest=False).indices

        indices = indices.view(x.shape[:-1])
        
        return indices, None

class SoftVectorQuantizer(nn.Module):
    """
    SoftVectorQuantizer - Uses deterministic Softmax and diversity loss

    Design:
    1. Uses standard VQ distance calculation (L2/dot/cos)
    2. Soft sampling based on distance using deterministic Softmax
    3. Uses only diversity loss (encourages uniform codeword usage)
    4. Supports temperature adjustment

    Parameters:
    - distance: Distance type ('l2', 'dot', 'cos')
    - temperature: Softmax temperature parameter
    - diversity_scale: Diversity loss weight (reduces collision rate)
    """

    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=False, kmeans_iters=10,
                 distance='l2', temperature=1.0, diversity_scale=0.0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.dist = distance.lower()  # 'l2', 'dot', 'cos'
        self.tau = temperature  # Softmax temperature
        self.diversity_scale = diversity_scale  # Diversity loss weight (encourages uniform codeword usage)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        self.last_diversity_loss = None
        self.last_diversity_loss_scaled = None

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

        if not self.kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False

    def init_emb(self, x):
        """Initialize embedding using k-means"""
        if self.initted:
            return
        
        print(f"Initializing embedding with k-means (k={self.n_e}, iters={self.kmeans_iters})")
        
        # Flatten input
        x_flat = x.view(-1, self.e_dim)
        
        # Initialize using k-means
        centroids = kmeans(x_flat, self.n_e, self.kmeans_iters)
        
        # Update embedding weights
        self.embedding.weight.data.copy_(centroids)
        
        self.initted = True
        print("Embedding initialized successfully")

    def set_temperature(self, temperature: float):
        self.tau = temperature

    def _compute_distance(self, x, codebook):
        """Compute distance between input and codebook"""
        if self.dist == 'l2':
            # L2 distance: ||x - e||^2 = ||x||^2 + ||e||^2 - 2*x*e
            x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
            e_norm_sq = torch.sum(codebook**2, dim=-1)
            xe = torch.matmul(x, codebook.t())
            dist = x_norm_sq + e_norm_sq - 2 * xe
        elif self.dist == 'dot':
            # Dot product distance (negative dot product, since we want to minimize distance)
            dist = -torch.matmul(x, codebook.t())
        elif self.dist == 'cos':
            # Cosine distance
            x_norm = F.normalize(x, p=2, dim=-1)
            e_norm = F.normalize(codebook, p=2, dim=-1)
            dist = -torch.matmul(x_norm, e_norm.t())
        else:
            raise ValueError(f"Unknown distance type: {self.dist}")
        
        return dist

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def forward(self, x, use_sk=True, conflict=False):
        """
        Forward pass - SoftVectorQuantizer (uses deterministic Softmax, no Gumbel noise)

        Process:
        1. Compute distance between input and codebook
        2. Convert distance to logits (negative distance)
        3. Use deterministic Softmax for soft weighting (consistent in training and inference)
        4. Compute diversity loss (encourages uniform codeword usage)

        - Removed Gumbel noise, uses deterministic Softmax
        - Training: Softmax weighted sum (differentiable)
        - Inference: Also uses Softmax (maintains consistency)
        - Uses only diversity loss, no VQ loss or KL loss
        """
        original_shape = x.shape
        latent = x.view(-1, self.e_dim)  # [batch, e_dim]

        if self.kmeans_init and not self.initted and self.training:
            self.init_emb(latent)

        # 1. Compute distance
        dist = self._compute_distance(latent, self.embedding.weight)  # [batch, n_e]
        
        # 2. Convert distance to logits (smaller distance, larger logits)
        logits = -dist / self.tau  # Temperature scaling
        
        # 3. Deterministic Softmax (no Gumbel noise)
        soft_one_hot = F.softmax(logits, dim=1)  # [batch, n_e]
        
        # 4. Get quantized vectors (soft weighted)
        x_q = torch.matmul(soft_one_hot, self.embedding.weight)  # [batch, e_dim]
        
        # 5. Get hard indices (for recording and analysis)
        indices = soft_one_hot.argmax(dim=1)  # Index based on Softmax probability
        
        # 6. Compute loss (only diversity loss)
        total_loss = 0.0
        
        # Diversity loss (encourages more uniform codeword usage within batch, reduces collision rate)
        self.last_diversity_loss = None
        self.last_diversity_loss_scaled = None
        if self.diversity_scale > 0:
            # Compute average distribution within batch
            q_bar = soft_one_hot.mean(dim=0)  # (n_e,)
            # Encourage uniform distribution: minimize negative entropy (equivalent to maximizing entropy)
            # diversity_loss = -entropy(q_bar) = sum(q_bar * log(q_bar + eps))
            eps = 1e-10
            diversity_loss = torch.sum(q_bar * torch.log(q_bar + eps))
            total_loss += self.diversity_scale * diversity_loss

            self.last_diversity_loss = diversity_loss.detach()
            self.last_diversity_loss_scaled = (self.diversity_scale * diversity_loss).detach()

        # Restore original shape
        x_q = x_q.view(original_shape)
        indices = indices.view(original_shape[:-1])

        return x_q, total_loss, indices
    
    def forward_soft(self, x, tau=None, hard=False):
        """
        SoftVectorQuantizer soft quantization forward pass (for end-to-end training and Phase 1 real-time soft quantization)

        Args:
            x: Input vectors (B, e_dim)
            tau: Softmax temperature (if None, uses self.tau)
            hard: Ignore this parameter (maintains interface consistency)

        Returns:
            x_q: Quantized vectors (B, e_dim)
            soft_indices: Soft probability distributions (B, n_e)
            loss: Placeholder loss (kept for interface consistency)

        - Uses deterministic Softmax (no Gumbel noise)
        - Training and inference are completely consistent
        - Same input always produces the same soft distribution
        """
        original_shape = x.shape
        latent = x.view(-1, self.e_dim)  # [batch, e_dim]
        
        # K-means initialization (if needed)
        if self.kmeans_init and not self.initted and self.training:
            self.init_emb(latent)
        
        # Use specified temperature or default temperature
        temperature = tau if tau is not None else self.tau
        
        # 1. Compute distance
        dist = self._compute_distance(latent, self.embedding.weight)  # [batch, n_e]
        
        # 2. Convert distance to logits (smaller distance, larger logits)
        logits = -dist / temperature
        
        # 3. Deterministic Softmax (no Gumbel noise)
        soft_one_hot = F.softmax(logits, dim=1)  # [batch, n_e]
        
        # 4. Soft quantization: probability-weighted sum
        x_q = torch.matmul(soft_one_hot, self.embedding.weight)  # [batch, e_dim]

        loss = torch.tensor(0.0, device=x_q.device)
        
        # Restore shape
        x_q = x_q.view(original_shape)
        soft_indices = soft_one_hot  # (batch, n_e)

        return x_q, soft_indices, loss
