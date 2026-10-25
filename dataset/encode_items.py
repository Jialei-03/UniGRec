"""Encode item titles and descriptions with sentence-T5 (768-d).

Reads <data_dir>/item_text.jsonl (written by preprocess_amazon.py) and writes
<data_dir>/item_emb_td.parquet with columns ItemID (1..N, sorted) and embedding.

Usage: python dataset/encode_items.py --data_dir dataset/Beauty
Requires: pip install sentence-transformers
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def item_text(row):
    # Same template as used for the reported results; missing fields print as None.
    return f"'title':{row.get('title')}\n 'description':{row.get('description')}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model", default="sentence-transformers/sentence-t5-base",
                   help="Hugging Face name or local path of the sentence encoder")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default="item_emb_td.parquet")
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer

    with open(os.path.join(args.data_dir, "item_text.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    rows.sort(key=lambda r: r["ItemID"])
    ids = np.array([r["ItemID"] for r in rows], dtype=np.int64)
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise ValueError("ItemID must be exactly 1..N")

    model = SentenceTransformer(args.model, device=args.device)
    embeddings = model.encode([item_text(r) for r in rows], batch_size=args.batch_size,
                              show_progress_bar=True, convert_to_numpy=True)
    out = os.path.join(args.data_dir, args.output)
    pd.DataFrame({"ItemID": ids, "embedding": [e.astype(np.float32).tolist() for e in embeddings]}).to_parquet(
        out, index=False)
    print(f"wrote {out}: {embeddings.shape}")


if __name__ == "__main__":
    main()
