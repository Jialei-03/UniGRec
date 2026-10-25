"""Beam candidates and their AVR features (inference only).

For every user and each of the beam candidates i (in beam order):
  lp[l]  log p(c_{i,l} | c_{i,<l}, X) for the L+1 identifier tokens (teacher-forced pass)
  tgt    <q, a_i>   output-side query vs. momentum anchor of the candidate item
  seq    <z, u_i>   sequence state vs. online anchor of the candidate item
Candidates whose identifier does not denote an item get item 0 and zero anchor scores.
"""
import numpy as np
import torch
import torch.nn.functional as F

from utils import CodeLookup


@torch.no_grad()
def extract(model, loader, table, item_codes, tau, beam_size, max_collision, device):
    model.eval()
    anchor = model.anchor
    if anchor is None:
        raise ValueError("AVR needs a model trained with collaborative anchoring")
    lookup = CodeLookup(item_codes.to(device), model.codebook_size, max_collision)
    zero = torch.zeros(1, anchor.dim, device=device)
    tgt_keys = torch.cat([zero, F.normalize(anchor.anchors(anchor.catalog, momentum=True), dim=-1)])
    seq_keys = torch.cat([zero, F.normalize(anchor.anchors(anchor.catalog), dim=-1)])

    rows = {k: [] for k in ("cand", "lp", "tgt", "seq", "is_target")}
    for history_ids, target_ids in loader:
        batch = table.batch(history_ids, target_ids, device)
        b = target_ids.size(0)
        soft, token_mask, _ = model.history_inputs(batch["history_embs"], batch["attention_mask"], tau)
        states = model.t5.encode(soft, token_mask)
        generated = model.t5.generate(soft, token_mask, num_beams=beam_size, max_collision=max_collision)
        tokens = generated[:, 1:]                                    # (B * beam, L + 1), beam order
        decoder_input = torch.cat([torch.zeros_like(tokens[:, :1]), tokens[:, :-1]], dim=1)
        out = model.t5.t5(encoder_outputs=(states.repeat_interleave(beam_size, 0),),
                          attention_mask=token_mask.repeat_interleave(beam_size, 0),
                          decoder_input_ids=decoder_input)
        lp = torch.log_softmax(out.logits.float(), -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        items = lookup(tokens.reshape(b, beam_size, -1))

        bos = model.t5.t5(encoder_outputs=(states,), attention_mask=token_mask,
                          decoder_input_ids=torch.zeros(b, 1, dtype=torch.long, device=device),
                          output_hidden_states=True).decoder_hidden_states[-1][:, 0, :]
        q = anchor.query(bos)
        z = anchor.sequence_state(batch["history_embs"], batch["attention_mask"])
        rows["tgt"].append((q[:, None, :] * tgt_keys[items]).sum(-1).cpu())
        rows["seq"].append((z[:, None, :] * seq_keys[items]).sum(-1).cpu())
        rows["cand"].append(items.cpu())
        rows["lp"].append(lp.reshape(b, beam_size, -1).cpu())
        rows["is_target"].append((items == batch["target_ids"][:, None]).cpu())
    data = {k: torch.cat(v).numpy() for k, v in rows.items()}
    # Baseline ranking = beam order.
    data["gen_order"] = np.broadcast_to(-np.arange(beam_size, dtype=np.float32), data["cand"].shape).copy()
    return data
