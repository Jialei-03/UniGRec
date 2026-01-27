import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rqvae'))

from t5 import SoftT5
from models.rqvae import RQVAE


def build_adaptive_adapter_layers(input_dim, output_dim, min_hidden_dim=64, max_layers=3):
    """
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        min_hidden_dim: Minimum hidden dimension (default 64)
        max_layers: Maximum number of layers (default 3, excluding the input layer)
    
    Returns:
        layers: List[int], layer dimension list, e.g. [128, 256, 512, 768]
    """
    if input_dim == output_dim:
        return [input_dim, output_dim]
    is_expanding = output_dim > input_dim
    ratio = max(input_dim, output_dim) / min(input_dim, output_dim)
    
    if ratio < 2.0:
        return [input_dim, output_dim]
    
    layers = [input_dim]
    current_dim = input_dim
    
    if is_expanding:
        num_steps = 0
        temp_dim = input_dim
        while temp_dim < output_dim and num_steps < max_layers - 1:
            temp_dim *= 2
            num_steps += 1
        for _ in range(num_steps - 1):
            current_dim *= 2
            current_dim = max(current_dim, min_hidden_dim)
            layers.append(current_dim)
        layers.append(output_dim)
    else:
        num_steps = 0
        temp_dim = input_dim
        while temp_dim > output_dim and num_steps < max_layers - 1:
            temp_dim //= 2
            num_steps += 1
        for _ in range(num_steps - 1):
            current_dim //= 2
            current_dim = max(current_dim, min_hidden_dim)
            layers.append(current_dim)
        layers.append(output_dim)
    return layers


class MLPLayers(nn.Module):
    def __init__(self, layers, dropout=0.0, activation="relu", bn=False):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))

            if self.use_bn and idx != (len(self.layers)-2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            activation_func = self._get_activation(activation, output_size)
            if activation_func is not None and idx != (len(self.layers)-2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def _get_activation(self, activation_name, emb_dim=None):
        if activation_name is None:
            return None
        elif isinstance(activation_name, str):
            if activation_name.lower() == "sigmoid":
                return nn.Sigmoid()
            elif activation_name.lower() == "tanh":
                return nn.Tanh()
            elif activation_name.lower() == "relu":
                return nn.ReLU()
            elif activation_name.lower() == "leakyrelu":
                return nn.LeakyReLU()
            elif activation_name.lower() == "none":
                return None
        return None

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def compute_discrete_contrastive_loss_kl(x_logits, y_logits):
    code_num = x_logits.size(-1)
    # Convert to log-probabilities
    x_log_prob = F.log_softmax(x_logits.view(-1, code_num), dim=-1)
    y_log_prob = F.log_softmax(y_logits.view(-1, code_num), dim=-1)
    
    # Compute KL divergence
    loss = F.kl_div(x_log_prob, y_log_prob, reduction='batchmean', log_target=True)
    return loss


def compute_contrastive_loss(query_embeds, semantic_embeds, temperature=0.07, sim="cos", gathered=False):
    """
    Contrastive learning loss (InfoNCE, in embedding space).
    
    Args:
        query_embeds: (B, emb_dim) - query embeddings
        semantic_embeds: (B, emb_dim) - semantic embeddings
        temperature: Temperature
        sim: Similarity type ("cos" means cosine)
        gathered: Whether to gather across processes (for distributed training; keep False here)
    
    Returns:
        co_loss: InfoNCE loss
    """
    if sim == "cos":
        query_embeds = F.normalize(query_embeds, dim=-1)
        semantic_embeds = F.normalize(semantic_embeds, dim=-1)

    effective_bsz = query_embeds.size(0)
    labels = torch.arange(effective_bsz, dtype=torch.long, device=query_embeds.device)
    similarities = torch.matmul(query_embeds, semantic_embeds.transpose(0, 1)) / temperature

    co_loss = F.cross_entropy(similarities, labels)
    return co_loss




class UniGRec(nn.Module):
    """
    End-to-end TIGER model.
    
    Components:
    - RQVAE: trainable residual-quantization VAE
    - T5: SoftT5 (soft encoder input, hard autoregressive decoder)
    
    Training flow:
    1. item embedding -> RQVAE.encode_soft() -> soft distributions
    2. soft distributions -> expand to seq_len * 3 tokens -> T5 encoder
    """
    
    def __init__(self, rqvae_config, t5_config, freeze_rqvae=False, 
                 use_code_offset=False, codebook_size=256):
        """
        Args:
            rqvae_config: RQVAE configuration dictionary
            t5_config: T5Config object
            freeze_rqvae: Whether to freeze RQVAE (typically True during finetuning)
            use_code_offset: Whether to use code offsets
            codebook_size: Codebook size per layer (default 256)
            num_users: Number of users (deprecated)
            user_emb_dim: User embedding dimension (deprecated)
            use_user_embedding: Whether to use user embedding (deprecated)
        """
        super().__init__()

        self.sasrec_dim = None
        self.sasrec_to_e_adapter = None
        self.dec_to_sasrec_adapter = None
        
        # RQVAE
        self.rqvae = RQVAE(**rqvae_config)
        
        # T5 (hybrid mode: soft encoder, hard decoder)
        self.t5 = SoftT5(
            t5_config, 
            use_code_offset=use_code_offset, 
            codebook_size=codebook_size,
        )
        
        # Save config
        self.use_code_offset = use_code_offset
        self.codebook_size = codebook_size
        
        # Encoder adapter: T5 encoder output -> RQVAE quantization space
        # Use adaptive-depth MLP (automatically select number of layers based on dimension mismatch)
        # Gradually transition across dimensions using adaptive layers
        enc_adapter_layers = build_adaptive_adapter_layers(
            input_dim=t5_config.d_model,
            output_dim=rqvae_config['e_dim'],
            min_hidden_dim=32,
            max_layers=3
        )
        self.enc_adapter = MLPLayers(
            layers=enc_adapter_layers,
            dropout=0.0,
            activation="relu",
            bn=False
        )
        print(f"  Encoder adapter: {enc_adapter_layers} ({len(enc_adapter_layers)-1} layers)")

        self._t5_d_model = int(t5_config.d_model)
        self._rqvae_e_dim = int(rqvae_config['e_dim'])
        
        # Whether to freeze RQVAE
        if freeze_rqvae:
            for param in self.rqvae.parameters():
                param.requires_grad = False
            self.rqvae.eval()  # set to eval mode when frozen
        else:
            # Keep eval mode to stabilize BN/Dropout behavior; params remain trainable (requires_grad=True)
            self.rqvae.eval()

    def init_sasrec_teacher(self, sasrec_embedding_table: torch.Tensor):
        """Initialize SASRec teacher embedding table and build adaptive MLP adapters.

        sasrec_embedding_table must be shaped (num_items, sasrec_dim), aligned to ItemID 1..N.
        """
        if not isinstance(sasrec_embedding_table, torch.Tensor):
            raise TypeError("sasrec_embedding_table must be a torch.Tensor")
        if sasrec_embedding_table.ndim != 2:
            raise ValueError(f"sasrec_embedding_table must be 2D, got shape={tuple(sasrec_embedding_table.shape)}")

        sasrec_embedding_table = sasrec_embedding_table.float().contiguous()
        self.register_buffer('sasrec_embedding_table', sasrec_embedding_table)
        self.sasrec_dim = int(sasrec_embedding_table.size(1))

        sasrec_to_e_layers = build_adaptive_adapter_layers(
            input_dim=self.sasrec_dim,
            output_dim=self._rqvae_e_dim,
            min_hidden_dim=32,
            max_layers=4,
        )
        self.sasrec_to_e_adapter = MLPLayers(
            layers=sasrec_to_e_layers,
            dropout=0.0,
            activation="relu",
            bn=False,
        )
        print(f"  SASRec->E adapter: {sasrec_to_e_layers} ({len(sasrec_to_e_layers)-1} layers)")

        dec_to_sasrec_layers = build_adaptive_adapter_layers(
            input_dim=self._t5_d_model,
            output_dim=self.sasrec_dim,
            min_hidden_dim=32,
            max_layers=4,
        )
        self.dec_to_sasrec_adapter = MLPLayers(
            layers=dec_to_sasrec_layers,
            dropout=0.0,
            activation="relu",
            bn=False,
        )
        print(f"  Dec->SASRec adapter: {dec_to_sasrec_layers} ({len(dec_to_sasrec_layers)-1} layers)")
    
    def _get_code_logits(self, latents):
        """
        Get code logits (distance matrix) from latents.
        
        Args:
            latents: (B, e_dim) - latents in quantization space (encoder output)
        
        Returns:
            code_logits: (B, code_len, code_num) - logits (distance matrices) for all layers
        """
        all_logits = []
        residual = latents
        
        # Iterate over all quantizer layers
        for quantizer in self.rqvae.rq.vq_layers:
            # Compute distance matrix
            latent_flat = residual.view(-1, quantizer.e_dim)
            
            # Compute distances based on distance type
            if quantizer.dist == 'l2':
                # L2 distance: d = ||x||^2 + ||e||^2 - 2*x*e^T
                d = torch.sum(latent_flat**2, dim=1, keepdim=True) + \
                    torch.sum(quantizer.embedding.weight**2, dim=1, keepdim=True).t() - \
                    2 * torch.matmul(latent_flat, quantizer.embedding.weight.t())
            elif quantizer.dist == 'dot':
                # Dot product: d = -x*e^T
                d = -torch.matmul(latent_flat, quantizer.embedding.weight.t())
            elif quantizer.dist == 'cos':
                # Cosine: d = -cos(x, e)
                d = -torch.matmul(
                    F.normalize(latent_flat, dim=-1),
                    F.normalize(quantizer.embedding.weight, dim=-1).t()
                )
            else:
                raise ValueError(f"Unknown distance type: {quantizer.dist}")
            
            # d is logits (distance matrix)
            all_logits.append(d)
            
            # Update residual (for next layer)
            # Get quantized vector
            indices = d.argmin(dim=-1)  # (B,)
            x_q = quantizer.embedding(indices)  # (B, e_dim)
            residual = residual - x_q.view(residual.shape)
        
        # Stack logits from all layers: (B, code_len, code_num)
        code_logits = torch.stack(all_logits, dim=1)
        return code_logits
    
    def forward(self, history_embs, target_embs, decoder_input_ids, 
                attention_mask, labels=None, tau=0.5, 
                alpha=1.0, beta=1.0, gamma=0.0, delta=0.0, target_item_ids=None,
                diversity_weight=0.0):
        """
        End-to-end forward pass.
        
        Args:
            history_embs: (B, seq_len, emb_dim) history item embeddings
            target_embs: (B, emb_dim) target item embedding
            decoder_input_ids: (B, 4) decoder input hard codes
                e.g. [0, code0+1, code1+1, code2+1]
            attention_mask: (B, seq_len) attention mask
                1=valid item, 0=padding
            labels: (B, 4) target hard codes (for recommendation loss)
                e.g. [code0+1, code1+1, code2+1, 0]
            tau: Gumbel-Softmax temperature
            alpha: reconstruction loss weight
            beta: recommendation loss weight
            gamma: CDT loss weight (kl_loss)
            delta: CDR loss weight (dec_cl_loss)
        
        Returns:
            total_loss: total loss
            rec_loss: recommendation loss
            recon_loss: reconstruction loss
            kl_loss: CDT loss (kl_loss)
            dec_cl_loss: CDR loss (dec_cl_loss)
            diversity_loss: diversity loss (for monitoring)
        """
        B, seq_len, emb_dim = history_embs.shape
        device = history_embs.device
        
        def _compute_diversity_loss_from_soft_list(soft_list):
            total = torch.tensor(0.0, device=device)
            eps = 1e-10
            for soft_indices in soft_list:
                q_bar = soft_indices.mean(dim=0)  # (codebook_size,)
                total = total + torch.sum(q_bar * torch.log(q_bar + eps))
            return total / max(1, len(soft_list))
        
        # 1. RQVAE soft-quantize history sequence
        # (B, seq_len, emb_dim) -> (B*seq_len, emb_dim)
        history_embs_flat = history_embs.reshape(-1, emb_dim)
        
        # Soft quantization
        history_soft_list, history_quantized = self.rqvae.encode_soft(history_embs_flat, tau=tau)
        # history_soft_list: List[3 x (B*seq_len, 256)]
        # history_quantized: (B*seq_len, e_dim)
        history_diversity_loss = _compute_diversity_loss_from_soft_list(history_soft_list)
        
        # 2. Expand to (B, seq_len * 3, 256)
        # Expand 3 layers per item into 3 tokens, interleaved
        # e.g. item0 [c0_0, c1_0, c2_0], item1 [c0_1, c1_1, c2_1], ...
        layers_reshaped = [s.reshape(B, seq_len, 256) for s in history_soft_list]
        history_soft_expanded = torch.stack(layers_reshaped, dim=2).reshape(B, seq_len * 3, 256)
        
        # 3. Expand attention_mask
        # Original: (B, seq_len), each position is one item
        # Expanded: (B, seq_len * 3), all 3 layers share the same mask per item
        attention_mask_expanded = attention_mask.unsqueeze(-1).repeat(1, 1, 3).reshape(B, seq_len * 3)
        
        # 4. T5 recommendation loss (need encoder/decoder hidden states for distillation losses)
        rec_loss, logits, encoder_hidden_states, decoder_hidden_states = self.t5(
                encoder_soft_indices=history_soft_expanded,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask_expanded,
                labels=labels,
                return_encoder_hidden_states=True,
                return_decoder_hidden_states=True,
            )
        
        # 5. RQVAE reconstruction loss
        # Reconstruct history
        history_recon = self.rqvae.decoder(history_quantized)  # (B*seq_len, emb_dim)
        history_recon_loss = nn.functional.mse_loss(history_recon, history_embs_flat)
        
        # Reconstruct target
        target_soft_list, target_quantized = self.rqvae.encode_soft(target_embs, tau=tau)
        target_diversity_loss = _compute_diversity_loss_from_soft_list(target_soft_list)
        target_recon = self.rqvae.decoder(target_quantized)  # (B, emb_dim)
        target_recon_loss = nn.functional.mse_loss(target_recon, target_embs)
        
        recon_loss = (history_recon_loss + target_recon_loss) / 2
        
        # Aggregate diversity loss (history + target)
        # History loss is averaged (since it contains B*seq_len samples)
        total_diversity_loss = (history_diversity_loss * seq_len + target_diversity_loss) / (seq_len + 1)
        
        # 6. Two distillation losses: kl_loss and dec_cl_loss
        kl_loss = torch.tensor(0.0, device=device)
        dec_cl_loss = torch.tensor(0.0, device=device)

        if gamma > 0 or delta > 0:
            if target_item_ids is None:
                raise ValueError("SASRec teacher loss requires target_item_ids")
            if not hasattr(self, 'sasrec_embedding_table') or self.sasrec_dim is None:
                raise ValueError("SASRec teacher is not initialized. Call model.init_sasrec_teacher(...) before training.")
            if self.sasrec_to_e_adapter is None or self.dec_to_sasrec_adapter is None:
                raise ValueError("SASRec teacher adapters are not initialized. Call model.init_sasrec_teacher(...) before training.")

            # Deduplicate: compute loss only for unique items
            target_ids_np = target_item_ids.detach().cpu().numpy() if isinstance(target_item_ids, torch.Tensor) else np.asarray(target_item_ids)
            _, unq_index_np = np.unique(target_ids_np, return_index=True)
            unq_index = torch.tensor(unq_index_np, dtype=torch.long, device=device)

            # seq -> code logits
            seq_latents = encoder_hidden_states.clone()
            seq_latents[~attention_mask_expanded] = 0
            seq_last_latents = torch.sum(seq_latents, dim=1) / attention_mask_expanded.sum(dim=1, keepdim=True)
            seq_project_latents = self.enc_adapter(seq_last_latents)  # (B, e_dim)
            seq_code_logits = self._get_code_logits(seq_project_latents)  # (B, code_len, code_num)

            # teacher sasrec -> code logits
            sasrec_embeds = self.sasrec_embedding_table[target_item_ids.long() - 1]  # (B, sasrec_dim)
            sasrec_e = self.sasrec_to_e_adapter(sasrec_embeds)  # (B, e_dim)
            sasrec_code_logits = self._get_code_logits(sasrec_e)  # (B, code_len, code_num)

            if gamma > 0:
                seq_code_logits_unq = seq_code_logits[unq_index]
                sasrec_code_logits_unq = sasrec_code_logits[unq_index]
                kl_loss = compute_discrete_contrastive_loss_kl(
                    seq_code_logits_unq, sasrec_code_logits_unq
                ) + compute_discrete_contrastive_loss_kl(
                    sasrec_code_logits_unq, seq_code_logits_unq
                )

            if delta > 0:
                dec_latents = decoder_hidden_states[:, 0, :]  # (B, d_model)
                dec_sasrec = self.dec_to_sasrec_adapter(dec_latents)  # (B, sasrec_dim)
                dec_sasrec_unq = dec_sasrec[unq_index]
                sasrec_embeds_unq = sasrec_embeds[unq_index]
                dec_cl_loss = compute_contrastive_loss(
                    dec_sasrec_unq, sasrec_embeds_unq,
                    sim="cos", gathered=False
                ) + compute_contrastive_loss(
                    sasrec_embeds_unq, dec_sasrec_unq,
                    sim="cos", gathered=False
                )
        
        # 7. Combine losses
        total_loss = alpha * recon_loss + beta * rec_loss + gamma * kl_loss + delta * dec_cl_loss + diversity_weight * total_diversity_loss

        # Return 6 values
        return total_loss, rec_loss, recon_loss, kl_loss, dec_cl_loss, total_diversity_loss
    
    def load_rqvae_checkpoint(self, ckpt_path):
        """
        Load a pretrained RQVAE.
        
        Args:
            ckpt_path: Path to RQVAE checkpoint
        """
        print(f"Loading RQVAE from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.rqvae.load_state_dict(ckpt['state_dict'])
        
        # Restore training temperature (if saved)
        saved_temperature = ckpt.get('current_temperature', None)
        if saved_temperature is not None:
            print(f"  Restored training temperature: {saved_temperature:.6f}")
            if hasattr(self.rqvae, 'rq') and hasattr(self.rqvae.rq, 'set_temperature'):
                self.rqvae.rq.set_temperature(saved_temperature)
        else:
            # If temperature is not saved, try to compute final temperature from args
            rqvae_args = ckpt.get('args', None)
            if rqvae_args is not None:
                initial_temp = getattr(rqvae_args, 'temperature', None)
                final_temp = getattr(rqvae_args, 'temperature_final', initial_temp)
                schedule = getattr(rqvae_args, 'temperature_schedule', 'none')
                
                if schedule != 'none' and final_temp is not None and initial_temp is not None:
                    # Assume to use final temperature (temperature at the end of training)
                    target_temp = final_temp
                    print(f"  Using final temperature (computed from args): {target_temp:.6f}")
                    if hasattr(self.rqvae, 'rq') and hasattr(self.rqvae.rq, 'set_temperature'):
                        self.rqvae.rq.set_temperature(target_temp)
                else:
                    initial_temp_val = getattr(rqvae_args, 'temperature', 0.1)
                    print(f"  Using initial temperature: {initial_temp_val:.6f}")
        
        print("RQVAE loaded!")
    
    def load_t5_checkpoint(self, ckpt_path):
        """
        Load T5 checkpoint.
         
        Args:
            ckpt_path: Path to T5 checkpoint
        """
        print(f"Loading T5 from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.t5.load_state_dict(ckpt['model_state_dict'])
        print("T5 loaded!")
    
    def generate(self, history_embs, attention_mask, max_length=5, num_beams=10, tau=0.5,
                 use_code_range_constraint=True, max_collision=100):
        """
        Generate recommendation codes.
        
        Args:
            history_embs: (B, seq_len, emb_dim) history item embeddings
            attention_mask: (B, seq_len) attention mask
            max_length: max generation length (including start token)
            num_beams: number of beams
            tau: Gumbel-Softmax temperature
            use_code_range_constraint: whether to use code range constraint (default True)
            max_collision: max collision index (default 100)
        
        Returns:
            generated: (B * num_beams, max_length) generated token ids
        """
        B, seq_len, emb_dim = history_embs.shape
        device = history_embs.device
        
        # 1. RQVAE soft-quantize history sequence
        history_embs_flat = history_embs.reshape(-1, emb_dim)
        
        with torch.no_grad():
            history_soft_list, _ = self.rqvae.encode_soft(history_embs_flat, tau=tau)
        
        # 2. Expand to (B, seq_len * 3, 256)
        layers_reshaped = [s.reshape(B, seq_len, 256) for s in history_soft_list]
        history_soft_expanded = torch.stack(layers_reshaped, dim=2).reshape(B, seq_len * 3, 256)
        
        # 3. Expand attention_mask
        attention_mask_expanded = attention_mask.unsqueeze(-1).repeat(1, 1, 3).reshape(B, seq_len * 3)
        
        # 4. Call T5.generate
        generated = self.t5.generate(
            encoder_soft_indices=history_soft_expanded,
            attention_mask=attention_mask_expanded,
            max_length=max_length,
            num_beams=num_beams,
            use_code_range_constraint=use_code_range_constraint,
            max_collision=max_collision,
        )
        
        return generated