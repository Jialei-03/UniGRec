"""T5 recommender that reads soft identifiers and generates discrete identifiers."""
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessor


class CodeRangeLogitsProcessor(LogitsProcessor):
    """Constrained decoding: step l may only emit tokens of identifier position l."""

    def __init__(self, codebook_size, max_collision, vocab_size, num_levels=3):
        k = codebook_size
        ranges = [(0, 1)] + [(l * k + 1, (l + 1) * k + 1) for l in range(num_levels)]
        ranges.append((num_levels * k + 1, num_levels * k + 2 + max_collision))
        masks = torch.zeros(len(ranges), vocab_size, dtype=torch.bool)
        for position, (start, end) in enumerate(ranges):
            masks[position, start:end] = True
        self.masks = masks

    def __call__(self, input_ids, scores):
        position = input_ids.shape[1]
        if position == 0 or position >= self.masks.size(0):
            return scores
        mask = self.masks[position].to(scores.device)
        return scores.masked_fill(~mask.unsqueeze(0), float("-inf"))


class SoftT5(nn.Module):
    """T5 encoder-decoder with a probability-weighted (soft) encoder input.

    The l-th token of a historical item is embedded as ``o_l = p_l^T O_l`` where
    ``O_l`` are the embedding rows of the level-l code tokens (shared with the
    decoder). Padding positions use the PAD embedding.
    """

    def __init__(self, config, codebook_size=256, num_levels=3):
        super().__init__()
        self.t5 = T5ForConditionalGeneration(config)
        self.config = config
        self.codebook_size = codebook_size
        self.num_levels = num_levels

    def embed_soft(self, soft_inputs, attention_mask):
        """soft_inputs: (B, L * levels, K) -> (B, L * levels, D)."""
        batch, total, k = soft_inputs.shape
        weight = self.t5.shared.weight
        per_level = soft_inputs.reshape(batch, total // self.num_levels, self.num_levels, k)
        embeds = torch.stack(
            [per_level[:, :, l] @ weight[l * k + 1:(l + 1) * k + 1] for l in range(self.num_levels)], dim=2
        ).reshape(batch, total, -1)
        mask = attention_mask.unsqueeze(-1).to(embeds.dtype)
        return embeds * mask + weight[self.config.pad_token_id] * (1 - mask)

    def forward(self, soft_inputs, attention_mask, decoder_input_ids, labels=None):
        """Returns the Hugging Face output with all encoder/decoder hidden states."""
        embeds = self.embed_soft(soft_inputs, attention_mask)
        return self.t5(inputs_embeds=embeds, attention_mask=attention_mask.long(),
                       decoder_input_ids=decoder_input_ids, labels=labels, output_hidden_states=True)

    def encode(self, soft_inputs, attention_mask):
        embeds = self.embed_soft(soft_inputs, attention_mask)
        return self.t5.encoder(inputs_embeds=embeds, attention_mask=attention_mask.long()).last_hidden_state

    @torch.no_grad()
    def generate(self, soft_inputs, attention_mask, num_beams, max_collision):
        """Constrained beam search; returns (B * num_beams, 1 + levels + 1) token IDs in beam order."""
        embeds = self.embed_soft(soft_inputs, attention_mask)
        length = self.num_levels + 2
        processor = CodeRangeLogitsProcessor(self.codebook_size, max_collision, self.config.vocab_size,
                                             self.num_levels)
        return self.t5.generate(inputs_embeds=embeds, attention_mask=attention_mask.long(),
                                max_length=length, min_length=length, num_beams=num_beams,
                                num_return_sequences=num_beams, do_sample=False,
                                pad_token_id=self.config.pad_token_id, eos_token_id=None,
                                logits_processor=[processor])
