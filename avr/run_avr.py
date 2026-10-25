"""Fit Anchor-Verified Re-ranking on validation and report test.

Usage:
  python avr/run_avr.py --checkpoint ckpt/best_finetune_model.pth --data_dir dataset/Beauty \
      --item_emb_path dataset/Beauty/item_emb_td.parquet --output_dir ckpt/avr
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(1, ROOT)

from data import ItemTable, SequenceDataset, make_loader  # noqa: E402
from evaluation import build_model_from_checkpoint  # noqa: E402
from rqvae.datasets import load_item_embeddings  # noqa: E402
from utils import set_seed  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402
import protocol  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Anchor-Verified Re-ranking")
    p.add_argument("--checkpoint", required=True, help="best_finetune_model.pth written by model/train.py")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--item_emb_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--beam_size", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(0)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    embeddings = torch.from_numpy(load_item_embeddings(args.item_emb_path))
    table = ItemTable(embeddings, device)
    model = build_model_from_checkpoint(ckpt, embeddings).to(device)
    max_len = ckpt["args"]["max_len"]
    item_codes = ckpt["item_codes"]

    data = {}
    for split in ("valid", "test"):
        loader = make_loader(SequenceDataset(os.path.join(args.data_dir, f"{split}.parquet"), "eval", max_len),
                             args.batch_size, False)
        data[split] = features.extract(model, loader, table, item_codes, ckpt["tau"], args.beam_size,
                                       ckpt["max_collision"], device)
        np.savez_compressed(os.path.join(args.output_dir, f"{split}.avr_features.npz"), **data[split])
        print(f"{split}: {data[split]['cand'].shape[0]} users")

    tau_a = ckpt["model_config"]["kwargs"]["anchor_temperature"]
    protocol.TAU_A = tau_a
    dv, dt = data["valid"], data["test"]
    chosen, cv = protocol.select_on_validation(dv)
    params = protocol.fit(dv, protocol.feature_sets()[chosen])
    result = dict(chosen_feature_set=chosen, cv_heldout_ndcg10_gain=cv, weights=params)
    for split, d in (("valid", dv), ("test", dt)):
        result[f"{split}_beam_order"] = protocol.metrics(d["gen_order"], d["is_target"])
        result[f"{split}_avr"] = protocol.metrics(protocol.score(d, params, tau_a), d["is_target"])
    with open(os.path.join(args.output_dir, "avr_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
