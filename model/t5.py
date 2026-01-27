"""
Soft T5 Model: HybridSoftT5
 - Encoder: soft distribution input (expected embeddings)
 - Decoder: hard autoregressive decoding (standard T5)
 - Embedding: shared (no offset)
 """

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
from transformers.generation.logits_process import LogitsProcessor


class CodeRangeLogitsProcessor(LogitsProcessor):
    """
    Constrain the valid code range for each generated position.

    Based on the generation position and use_code_offset, constrain the valid logits range.

    Note: This constrains decoder token ids (embedding/vocab ids), not raw RQVAE codes.
    - raw code: 0-255
    - token id: raw code + 1 (0 is reserved for start/pad/eos)

    In T5.generate, the first decoder token (start token) is usually provided by decoder_start_token_id.
    Therefore, this processor typically starts taking effect at current_length==1 (constraining the first generated code).

    Valid ranges by position (depending on use_code_offset):
    - position 0: start token (0)
    - position 1: layer-1 code
    - position 2: layer-2 code
    - position 3: layer-3 code
    - position 4: collision index
    """
    
    def __init__(self, use_code_offset=True, codebook_size=256, max_collision=100, vocab_size=None):
        """
        Args:
            use_code_offset: Whether to use code offset encoding
            codebook_size: Codebook size (default 256)
            max_collision: Maximum collision index (default 100, total 101: 0-100)
        """
        self.use_code_offset = use_code_offset
        self.codebook_size = codebook_size
        self.max_collision = max_collision
        self.vocab_size = vocab_size
        
        # Define allowed ranges for each position
        # position 0: start token (0)
        # position 1: layer-1 code
        # position 2: layer-2 code
        # position 3: layer-3 code
        # position 4: collision index
        
        if use_code_offset:
            # Offset mode: each layer uses a different embedding range
            self.allowed_ranges = [
                [0],  # position 0: start token
                list(range(1, codebook_size + 1)),  # position 1: 1-256
                list(range(codebook_size + 1, 2 * codebook_size + 1)),  # position 2: 257-512
                list(range(2 * codebook_size + 1, 3 * codebook_size + 1)),  # position 3: 513-768
                list(range(3 * codebook_size + 1, 3 * codebook_size + 1 + max_collision + 1)),  # position 4: 769-869
            ]
        else:
            # Non-offset mode: all layers share embedding[1:257]
            # Collision indices start from 257 (vocab_size=358, collision range is 257-357)
            collision_start = codebook_size + 1  # 257
            self.allowed_ranges = [
                [0],  # position 0: start token
                list(range(1, codebook_size + 1)),  # position 1: 1-256
                list(range(1, codebook_size + 1)),  # position 2: 1-256
                list(range(1, codebook_size + 1)),  # position 3: 1-256
                list(range(collision_start, collision_start + max_collision + 1)),  # position 4: 257-357
            ]
        
        # For fast lookup, build a mask for each position
        self.allowed_masks = []
        # If vocab_size is provided, use the larger value to avoid masking out all tokens due to a short mask
        max_vocab_size = max(max(r) for r in self.allowed_ranges) + 1
        if self.vocab_size is not None:
            max_vocab_size = max(max_vocab_size, self.vocab_size)
        for allowed_range in self.allowed_ranges:
            mask = torch.zeros(max_vocab_size, dtype=torch.bool)
            mask[allowed_range] = True
            self.allowed_masks.append(mask)
    
    def __call__(self, input_ids, scores):
        """
        Constrain logits range at each generation step.
        
        Args:
            input_ids: (batch_size * num_beams, current_length) tokens generated so far
            scores: (batch_size * num_beams, vocab_size) logits at current step
        
        Returns:
            scores: Updated logits (disallowed positions set to -inf)
        """
        batch_size = scores.shape[0]
        current_length = input_ids.shape[1]
        
        # Determine current generation position
        # In T5 generation:
        # - current_length=0: generation not started (start token is usually already present)
        # - current_length=1: generate layer-1 code, use allowed_masks[1]
        # - current_length=2: generate layer-2 code, use allowed_masks[2]
        # - current_length=3: generate layer-3 code, use allowed_masks[3]
        # - current_length=4: generate collision index, use allowed_masks[4]
        
        if current_length == 0:
            # Generation not started; return original scores (start token is usually already present)
            return scores
        
        # Position index to generate:
        # - allowed_masks[0] is for start token (usually unused)
        # - from current_length==1, generate layer-1 code using allowed_masks[1]
        position_idx = current_length
        
        # Clamp position index to a valid range
        if position_idx >= len(self.allowed_masks):
            # If out of range, allow all tokens (backward compatibility)
            return scores
        
        # Get mask for current position
        allowed_mask = self.allowed_masks[position_idx].to(scores.device)
        
        # Expand mask to batch dimension
        vocab_size = scores.shape[1]
        if vocab_size > allowed_mask.shape[0]:
            # Extend mask if vocab_size > mask length
            extended_mask = torch.zeros(vocab_size, dtype=torch.bool, device=scores.device)
            extended_mask[:allowed_mask.shape[0]] = allowed_mask
            allowed_mask = extended_mask
        elif vocab_size < allowed_mask.shape[0]:
            # Truncate mask if vocab_size < mask length
            allowed_mask = allowed_mask[:vocab_size]
        
        # Set logits of disallowed positions to -inf
        scores = scores.masked_fill(~allowed_mask.unsqueeze(0), float('-inf'))
        
        return scores


class SoftEmbeddingLayer(nn.Module):
    """
    Soft embedding layer: convert RQVAE soft distributions to expected embeddings.
    
    Supports two modes:
    1. Non-offset (use_code_offset=False):
       - 3 RQ layers share the same embedding table (vocab_size=257)
       - 0: PAD (no soft sampling; directly use embedding[0])
       - 1-256: code 0-255 (soft-weighted sum, embedding[code+1])
    
    2. Offset (use_code_offset=True):
       - Each layer uses a different embedding range (vocab_size=769)
       - 0: PAD
       - 1-256: layer 0 code 0-255
       - 257-512: layer 1 code 0-255
       - 513-768: layer 2 code 0-255
       - Layer-wise weighted sum: for layer i, weight embedding[i*256+1:(i+1)*256+1] by the layer probabilities
    """
    
    def __init__(self, vocab_size, d_model, pad_token_id=0, use_code_offset=False, codebook_size=256):
        """
        Args:
            vocab_size: Vocabulary size
                - Non-offset: 257 (1 PAD + 256 codes)
                - Offset: 769 (1 PAD + 256*3 codes)
            d_model: T5 model dimension (e.g., 128)
            pad_token_id: PAD token ID (default 0)
            use_code_offset: Whether to use code offset
            codebook_size: Codebook size per layer (default 256)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.use_code_offset = use_code_offset
        self.codebook_size = codebook_size
        
        # Shared embedding table (shared with T5 decoder)
        # Note: the embedding is initialized externally and passed in
        self.shared_embedding = None  # to be set later from T5
        
    def set_shared_embedding(self, embedding):
        """Set shared embedding (from T5.shared)."""
        self.shared_embedding = embedding
    
    def forward(self, soft_indices, attention_mask=None):
        """
        Args:
            soft_indices: (B, seq_len * 3, 256), expanded soft probability distributions
                        Every 3 positions correspond to 3 code layers of one item
                        Positions 0,3,6,... correspond to layer 0
                        Positions 1,4,7,... correspond to layer 1
                        Positions 2,5,8,... correspond to layer 2
            attention_mask: (B, seq_len * 3), 1 indicates valid position, 0 indicates padding
        
        Returns:
            embeddings: (B, seq_len * 3, d_model)
        """
        # soft_indices: (B, seq_len * 3, 256)
        B, total_seq_len, code_dim = soft_indices.shape
        
        if self.use_code_offset:
            # Offset mode: layer-wise weighted sum
            # Every 3 positions correspond to 3 code layers of one item and need to be handled separately
            seq_len = total_seq_len // 3
            expected_emb = torch.zeros(B, total_seq_len, self.d_model, 
                                      device=soft_indices.device, dtype=soft_indices.dtype)
            
            for layer_idx in range(3):
                # Get all positions for layer layer_idx
                layer_positions = torch.arange(layer_idx, total_seq_len, 3, device=soft_indices.device)
                layer_soft = soft_indices[:, layer_positions, :]  # (B, seq_len, 256)
                
                # Layer layer_idx uses embedding range: [layer_idx*256+1, (layer_idx+1)*256+1]
                start_idx = layer_idx * self.codebook_size + 1
                end_idx = (layer_idx + 1) * self.codebook_size + 1
                layer_embeddings = self.shared_embedding.weight[start_idx:end_idx]  # (256, d_model)
                
                # Expected embedding: E[e] = Σ p_i * embedding[layer_idx*256+1+code_i]
                # (B, seq_len, 256) @ (256, d_model) = (B, seq_len, d_model)
                layer_expected_emb = torch.matmul(layer_soft, layer_embeddings)
                
                # Write back to the original positions
                expected_emb[:, layer_positions, :] = layer_expected_emb
        else:
            # Non-offset mode: all layers share embedding[1:257]
            code_embeddings = self.shared_embedding.weight[1:257]  # (256, d_model)
            
            # Expected embedding: E[e] = Σ p_i * embedding[code_i+1]
            # (B, seq_len * 3, 256) @ (256, d_model) = (B, seq_len * 3, d_model)
            expected_emb = torch.matmul(soft_indices, code_embeddings)
        
        # Handle padding: replace positions with mask=0 by PAD embedding
        if attention_mask is not None:
            pad_emb = self.shared_embedding.weight[self.pad_token_id]  # (d_model,)
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, seq_len * 3, 1)
            expected_emb = expected_emb * mask_expanded + pad_emb * (1 - mask_expanded)
        
        return expected_emb


class SoftT5(nn.Module):
    """
    Hybrid-mode T5 (soft encoder, hard decoder, shared embedding).
    
    Design notes:
    - Encoder takes soft distributions (expected embeddings)
    - Decoder is standard autoregressive decoding (discrete tokens)
    - Encoder and Decoder share the embedding table

    Supports two modes:
    1. Non-offset (use_code_offset=False):
       - vocab: 0=PAD, 1-256=code 0-255 (shared across layers)
       - vocab_size=257

    2. Offset (use_code_offset=True):
       - vocab: 0=PAD, 1-256=layer0, 257-512=layer1, 513-768=layer2
       - vocab_size=769
    """
    
    def __init__(self, t5_config, use_code_offset=True, codebook_size=256):
        """
        Args:
            t5_config: T5Config
                - Non-offset: vocab_size should be 257 (0=PAD, 1-256=codes)
                - Offset: vocab_size should be 769 (0=PAD, 1-768=codes)
            use_code_offset: Whether to use code offset
            codebook_size: Codebook size per layer (default 256)
            num_users: Deprecated, kept for compatibility. Hashing Trick is used in practice, with a fixed 2000 tokens.
            user_emb_dim: User embedding dimension (default 128)
            use_user_embedding: Whether to use user embedding (default False, must be explicitly enabled)
        
        Note:
            Following the TIGER paper, use the Hashing Trick to map user IDs into 2000 fixed tokens.
            Original user IDs are mapped by hash(user_id) % 2000 into the range 0-1999.
        """
        super().__init__()
        
        # T5 model (standard initialization)
        self.t5 = T5ForConditionalGeneration(t5_config)
        
        # Soft embedding layer (for encoder)
        self.soft_embedding = SoftEmbeddingLayer(
            vocab_size=t5_config.vocab_size,
            d_model=t5_config.d_model,
            pad_token_id=t5_config.pad_token_id if hasattr(t5_config, 'pad_token_id') else 0,
            use_code_offset=use_code_offset,
            codebook_size=codebook_size,
        )
        
        # Set shared embedding (T5.shared is shared by encoder and decoder)
        self.soft_embedding.set_shared_embedding(self.t5.shared)
        
        # Save config
        self.config = t5_config
        self.use_code_offset = use_code_offset
        self.codebook_size = codebook_size
    
    def forward(self, 
                encoder_soft_indices,  # (B, seq_len * 3, 256)
                decoder_input_ids,     # (B, target_len), discrete tokens
                attention_mask,        # (B, seq_len * 3)
                labels=None,           # (B, target_len), discrete tokens
                return_encoder_hidden_states=False,
                return_decoder_hidden_states=False):
        """
        Hybrid-mode forward pass.
        
        Args:
            encoder_soft_indices: (B, seq_len * 3, 256), expanded soft distributions.
                Every 3 positions correspond to 3 code layers of one item.
            decoder_input_ids: (B, target_len), discrete target tokens (teacher forcing)
                Non-offset: [0, 13, 46, 79] = [start, code0+1, code1+1, code2+1]
                Offset: [0, 13, 257+46, 513+79] = [start, code0+1, code1+257, code2+513]
            attention_mask: (B, seq_len * 3), 1=valid, 0=padding
            labels: (B, target_len), target tokens (for loss)
                Non-offset: [13, 46, 79, 0] = [code0+1, code1+1, code2+1, EOS]
                Offset: [13, 257+46, 513+79, 0] = [code0+1, code1+257, code2+513, EOS]
            return_encoder_hidden_states: Whether to return encoder hidden states
            return_decoder_hidden_states: Whether to return decoder hidden states
        Returns:
            loss: cross-entropy loss
            logits: (B, target_len, vocab_size)
                - Non-offset: vocab_size=257
                - Offset: vocab_size=769
            encoder_hidden_states (optional): (B, 1+seq_len * 3, d_model) if a user token was inserted
            decoder_hidden_states (optional): (B, target_len, d_model)
        """
        # 1. Encoder: soft distribution -> expected embeddings
        encoder_embeds = self.soft_embedding(
            encoder_soft_indices, 
            attention_mask=attention_mask
        )  # (B, seq_len * 3, d_model)
        
        # Ensure encoder_embeds and attention_mask shapes match
        if encoder_embeds.size(1) != attention_mask.size(1):
            raise ValueError(
                f"encoder_embeds and attention_mask sequence lengths do not match: "
                f"encoder_embeds={encoder_embeds.shape}, attention_mask={attention_mask.shape}"
            )
        
        # Check NaN/Inf (only during training to avoid performance overhead)
        if self.training:
            if torch.isnan(encoder_embeds).any() or torch.isinf(encoder_embeds).any():
                raise ValueError("encoder_embeds contains NaN or Inf")
        
        # 2. T5 forward (decoder uses standard input_ids)
        output_hidden_states = return_encoder_hidden_states or return_decoder_hidden_states

        outputs = self.t5(
            inputs_embeds=encoder_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )
        loss = outputs.loss
        logits = outputs.logits
        
        # 3. Assemble return values
        ret = [loss, logits]
        
        if return_encoder_hidden_states:
            # encoder_hidden_states is the output of the last encoder layer
            encoder_hidden_states = outputs.encoder_last_hidden_state
            ret.append(encoder_hidden_states)
        
        if return_decoder_hidden_states:
            # decoder_hidden_states is the output of the last decoder layer
            decoder_hidden_states = outputs.decoder_hidden_states[-1]  # (B, target_len, d_model)
            ret.append(decoder_hidden_states)
        
        return tuple(ret)
    
    def generate(self, encoder_soft_indices, attention_mask, 
                 max_length=5, num_beams=20, use_code_range_constraint=True, 
                 max_collision=100, **kwargs):
        """
        Autoregressive generation (inference).
        
        Args:
            encoder_soft_indices: (B, seq_len * 3, 256), expanded soft distributions.
                Every 3 positions correspond to 3 code layers of one item.
            attention_mask: (B, seq_len * 3)
            max_length: Generated length (1 start + 4 codes = 5)
            num_beams: beam size
            use_code_range_constraint: Whether to constrain code ranges (default True)
            max_collision: Max collision index (default 100)
            **kwargs: Other arguments passed to T5.generate
        Returns:
            generated_ids: (B*num_beams, max_length)
                Non-offset: [[0, 13, 46, 79, 769], ...] = [[start, code0+1, code1+1, code2+1, collision_idx+769], ...]
                Offset: [[0, 13, 257+46, 513+79, 769], ...] = [[start, code0+1, code1+257, code2+513, collision_idx+769], ...]
        """
        # Encoder embeddings (soft)
        encoder_embeds = self.soft_embedding(
            encoder_soft_indices,
            attention_mask=attention_mask
        )  # (B, seq_len * 3, d_model)
        
        # Create logits processor (if range constraint is enabled)
        logits_processor = None
        if use_code_range_constraint:
            logits_processor = CodeRangeLogitsProcessor(
                use_code_offset=self.use_code_offset,
                codebook_size=self.codebook_size,
                max_collision=max_collision,
                vocab_size=self.t5.config.vocab_size,  # use actual vocab_size to avoid short masks
            )
        
        # Build logits_processor list
        logits_processor_list = []
        if logits_processor is not None:
            logits_processor_list.append(logits_processor)
        
        # Merge with logits_processor passed via kwargs
        if 'logits_processor' in kwargs:
            existing_processors = kwargs.pop('logits_processor')
            if isinstance(existing_processors, list):
                logits_processor_list.extend(existing_processors)
            else:
                logits_processor_list.append(existing_processors)
        
        # T5 autoregressive generation (standard)
        # Note: set min_length=max_length and eos_token_id=None to force full-length sequences
        pad_token_id = self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else 0
        generated = self.t5.generate(
            inputs_embeds=encoder_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=max_length,  # ensure full sequence (including start token)
            num_beams=num_beams,
            num_return_sequences=num_beams,
            pad_token_id=pad_token_id,
            eos_token_id=None,  # disable EOS token to force full sequence
            logits_processor=logits_processor_list if logits_processor_list else None,
            **kwargs
        )
        
        return generated  # (B*num_beams, max_length)
    
    def save_pretrained(self, save_path):
        """Save the model."""
        self.t5.save_pretrained(save_path)
    
    def load_pretrained(self, load_path):
        """Load the model."""
        self.t5 = T5ForConditionalGeneration.from_pretrained(load_path)
        # Re-set shared embedding
        self.soft_embedding.set_shared_embedding(self.t5.shared)
        # Re-set use_code_offset and codebook_size (if saved)
        if hasattr(self.soft_embedding, 'use_code_offset'):
            self.use_code_offset = self.soft_embedding.use_code_offset
        if hasattr(self.soft_embedding, 'codebook_size'):
            self.codebook_size = self.soft_embedding.codebook_size

