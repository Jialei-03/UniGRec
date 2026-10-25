"""Generate a tiny synthetic dataset in the UniGRec format (for the smoke test)."""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_items", type=int, default=400)
    p.add_argument("--num_users", type=int, default=150)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Clustered random content embeddings; users mostly stay within a few clusters.
    centers = rng.normal(size=(20, args.dim))
    cluster = rng.integers(0, 20, size=args.num_items)
    emb = centers[cluster] + 0.5 * rng.normal(size=(args.num_items, args.dim))
    pd.DataFrame({"ItemID": np.arange(1, args.num_items + 1),
                  "embedding": [e.astype(np.float32).tolist() for e in emb]}).to_parquet(
        os.path.join(args.out_dir, "item_emb_td.parquet"), index=False)

    sequences = []
    for _ in range(args.num_users):
        length = int(rng.integers(5, 25))
        home = rng.integers(0, 20, size=2)
        pool = np.flatnonzero(np.isin(cluster, home)) + 1
        sequences.append(rng.choice(pool, size=length).tolist())

    def frame(cut):
        rows = []
        for user, seq in enumerate(sequences, 1):
            seq = seq[:len(seq) - cut]
            rows.append({"user": user, "history": seq[:-1], "target": seq[-1]})
        return pd.DataFrame(rows)

    frame(2).to_parquet(os.path.join(args.out_dir, "train.parquet"), index=False)
    frame(1).to_parquet(os.path.join(args.out_dir, "valid.parquet"), index=False)
    frame(0).to_parquet(os.path.join(args.out_dir, "test.parquet"), index=False)
    print(f"toy data written to {args.out_dir}")


if __name__ == "__main__":
    main()
