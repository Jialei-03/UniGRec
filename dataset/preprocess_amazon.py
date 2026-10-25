"""Build UniGRec sequential data from the Amazon Reviews (2014) 5-core files.

Input (download from https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html):
  <raw_dir>/reviews_<DATASET>_5.json.gz   5-core reviews, e.g. reviews_Beauty_5.json.gz
  <raw_dir>/meta_<DATASET>.json.gz        item metadata,  e.g. meta_Beauty.json.gz

Output (<out_dir>, default ./dataset/<DATASET>):
  train.parquet / valid.parquet / test.parquet   columns: user, history (list[int]), target (int)
  item_text.jsonl                                 ItemID, asin, title, description (for encode_items.py)
  user_mapping.json / item_mapping.json           raw ID -> integer ID (1-based)

Each user's interactions are sorted by time and split leave-one-out: the last item is the
test target, the second-to-last the validation target, and the training row holds the rest
(the trainer expands every prefix of it). Histories are truncated to the latest 20 items
inside the trainer (--max_len).

Usage: python dataset/preprocess_amazon.py --dataset Beauty --raw_dir raw/
"""
import argparse
import ast
import gzip
import json
import os

import pandas as pd


def read_lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)  # the 2014 metadata files are Python literals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Amazon category, e.g. Beauty or Pet_Supplies")
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--min_len", type=int, default=3, help="minimum interactions per user")
    args = p.parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # ---- interactions: IDs are assigned in order of first appearance (1-based).
    user_map, item_map, sequences = {}, {}, {}
    for review in read_lines(os.path.join(args.raw_dir, f"reviews_{args.dataset}_5.json.gz")):
        user = user_map.setdefault(review["reviewerID"], len(user_map) + 1)
        item = item_map.setdefault(review["asin"], len(item_map) + 1)
        sequences.setdefault(user, []).append((item, review["unixReviewTime"]))
    for user in sequences:
        sequences[user].sort(key=lambda x: x[1])  # stable: ties keep file order
        sequences[user] = [item for item, _ in sequences[user]]
    sequences = {u: s for u, s in sequences.items() if len(s) >= args.min_len}
    print(f"users={len(sequences)} items={len(item_map)} "
          f"interactions={sum(len(s) for s in sequences.values())}")

    def frame(cut):
        rows = []
        for user, seq in sequences.items():
            seq = seq[:len(seq) - cut]
            rows.append({"user": user, "history": seq[:-1], "target": seq[-1]})
        return pd.DataFrame(rows)

    frame(2).to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    frame(1).to_parquet(os.path.join(out_dir, "valid.parquet"), index=False)
    frame(0).to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
    with open(os.path.join(out_dir, "user_mapping.json"), "w") as f:
        json.dump(user_map, f)
    with open(os.path.join(out_dir, "item_mapping.json"), "w") as f:
        json.dump(item_map, f)

    # ---- item text (title + description) for the content embeddings.
    info = {}
    for meta in read_lines(os.path.join(args.raw_dir, f"meta_{args.dataset}.json.gz")):
        asin = meta.get("asin")
        if asin in item_map:
            info[item_map[asin]] = dict(title=meta.get("title") or None,
                                        description=meta.get("description") or None)
    missing = len(item_map) - len(info)
    if missing:
        print(f"warning: {missing} items have no metadata; their text fields are empty")
    with open(os.path.join(out_dir, "item_text.jsonl"), "w") as f:
        for asin, item in sorted(item_map.items(), key=lambda x: x[1]):
            row = info.get(item, dict(title=None, description=None))
            f.write(json.dumps(dict(ItemID=item, asin=asin, **row)) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
