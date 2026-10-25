"""Full-ranking evaluation with constrained beam search."""
import torch
from transformers import T5Config

from unigrec import UniGRec
from utils import CodeLookup, hit_ranks, metrics_from_ranks


@torch.no_grad()
def evaluate(model, loader, table, item_codes, tau, beam_size, max_collision, device,
             topk_list=(5, 10, 20)):
    """Recall/NDCG of the beam order. ``item_codes`` is the (N+1, 4) identifier map."""
    model.eval()
    lookup = CodeLookup(item_codes.to(device), model.codebook_size, max_collision)
    all_ranks = []
    for history_ids, target_ids in loader:
        batch = table.batch(history_ids, target_ids, device)
        generated = model.generate(batch["history_embs"], batch["attention_mask"], tau,
                                   num_beams=beam_size, max_collision=max_collision)
        tokens = generated[:, 1:].reshape(target_ids.size(0), beam_size, -1)
        items = lookup(tokens)
        all_ranks.append(hit_ranks(None, items, batch["target_ids"]).cpu())
    return metrics_from_ranks(torch.cat(all_ranks), topk_list)


def build_model_from_checkpoint(ckpt, catalog_embeddings):
    """Rebuild a UniGRec model from a checkpoint written by train.py."""
    cfg = ckpt["model_config"]
    model = UniGRec(cfg["rqvae_config"], T5Config(**cfg["t5_config"]), catalog_embeddings, **cfg["kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.rqvae.rq.set_temperature(ckpt["tau"])
    return model
