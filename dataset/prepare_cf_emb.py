#!/usr/bin/env python3

import argparse
import contextlib
import io
import logging
import os
import sys
from pathlib import Path
import random
import warnings

import numpy as np
import torch


def _ensure_recbole_importable(recbole_path: str | Path | None) -> None:
    if recbole_path is None:
        return
    p = Path(recbole_path)
    if not p.exists():
        raise FileNotFoundError(f"recbole_path does not exist: {p}")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_recbole_item_token2id(recbole_path: Path, recbole_dataset: str):
    _ensure_recbole_importable(recbole_path)
    try:
        from recbole.config import Config
        from recbole.data import create_dataset
    except Exception as e:
        raise ImportError(
            f"Failed to import RecBole from {recbole_path}. Please ensure RecBole and its dependencies are installed."
        ) from e

    model_cfg = recbole_path / "recbole" / "properties" / "model" / "SASRec.yaml"
    dataset_cfg = recbole_path / "recbole" / "properties" / "dataset" / f"{recbole_dataset}.yaml"
    if not model_cfg.exists() or not dataset_cfg.exists():
        raise FileNotFoundError(
            f"RecBole config files not found: {model_cfg} / {dataset_cfg}. Please pass valid --recbole_path and --recbole_dataset."
        )

    config = Config(
        model="SASRec",
        dataset=recbole_dataset,
        config_file_list=[str(model_cfg), str(dataset_cfg)],
        config_dict={"data_path": str(recbole_path / "dataset")},
    )
    dataset = create_dataset(config)
    if not hasattr(dataset, "field2token_id") or "item_id" not in dataset.field2token_id:
        raise ValueError('RecBole dataset does not provide field2token_id["item_id"].')
    return dataset.field2token_id["item_id"]


@contextlib.contextmanager
def _suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield


def _extract_item_embedding_tensor(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["state_dict"]
    elif isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj
    else:
        raise ValueError("Unsupported checkpoint object type")

    for k in ["item_embedding.weight", "model.item_embedding.weight"]:
        if k in state_dict and torch.is_tensor(state_dict[k]) and state_dict[k].dim() == 2:
            return k, state_dict[k]

    for k, v in state_dict.items():
        if not torch.is_tensor(v) or v.dim() != 2:
            continue
        lk = k.lower()
        if "item" in lk and "emb" in lk and k.endswith("weight"):
            return k, v

    raise ValueError("Cannot find item embedding weight in checkpoint state_dict")


def _load_unigrec_inv_item_mapping(mapping_path: Path) -> dict[int, int] | None:
    if not mapping_path.exists():
        return None
    obj = np.load(mapping_path, allow_pickle=True).item()
    # mapping: raw -> UniGRec item id
    inv = {int(v): int(k) for k, v in obj.items()}
    return inv


def _read_item_ids_from_parquet(item_emb_parquet: Path, item_id_col: str):
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("pandas is required to read parquet.") from e

    try:
        df = pd.read_parquet(item_emb_parquet, columns=[item_id_col])
    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet: {item_emb_parquet}. Ensure pyarrow/fastparquet is installed in this environment."
        ) from e

    item_ids = df[item_id_col].to_numpy()
    uniq = np.unique(item_ids)
    uniq = uniq.astype(np.int64)
    uniq.sort()
    return uniq


def _write_cf_parquet(output_path: Path, item_ids: np.ndarray, emb_list: list[np.ndarray], item_id_col: str, emb_col: str):
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("pandas is required to write parquet.") from e

    df = pd.DataFrame({item_id_col: item_ids.astype(np.int64), emb_col: emb_list})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to write parquet: {output_path}. Ensure pyarrow/fastparquet is installed in this environment."
        ) from e


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sasrec_checkpoint", type=str, required=True)
    p.add_argument("--recbole_path", type=str, default="/NAS/lijialei/RecBole-master")
    p.add_argument("--recbole_dataset", type=str, required=True)
    p.add_argument("--unigrec_item_emb_parquet", type=str, required=True)
    p.add_argument("--unigrec_item_mapping_npy", type=str, default=None)
    p.add_argument("--output_parquet", type=str, default=None)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--item_id_col", type=str, default="ItemID")
    p.add_argument("--output_emb_col", type=str, default="cf_embedding")
    p.add_argument("--verbose", action="store_true", help="Show RecBole/pandas warnings and verbose logs")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.verbose:
        # Keep our own report clean; still raise on errors.
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        # RecBole uses logging; silence it.
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger("recbole").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        # Avoid tokenizers-related noise if it appears in this environment.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ckpt_path = Path(args.sasrec_checkpoint)
    recbole_path = Path(args.recbole_path)
    unigrec_item_emb_parquet = Path(args.unigrec_item_emb_parquet)

    print("=" * 60)
    print("[SASRec->UniGRec] Export CF embeddings")
    print(f"- sasrec_checkpoint: {ckpt_path}")
    print(f"- recbole_path: {recbole_path}")
    print(f"- recbole_dataset: {args.recbole_dataset}")
    print(f"- unigrec_item_emb_parquet: {unigrec_item_emb_parquet}")

    if args.output_parquet is None:
        out_path = unigrec_item_emb_parquet.parent / f"cf_emb_sasrec{args.hidden_dim}.parquet"
    else:
        out_path = Path(args.output_parquet)

    print(f"- output_parquet: {out_path}")

    unigrec_inv_item = None
    if args.unigrec_item_mapping_npy:
        unigrec_inv_item = _load_unigrec_inv_item_mapping(Path(args.unigrec_item_mapping_npy))

    item_ids = _read_item_ids_from_parquet(unigrec_item_emb_parquet, args.item_id_col)
    if len(item_ids) == 0:
        raise ValueError("No item ids found.")

    if int(item_ids[0]) != 1:
        raise ValueError(f"Expected item ids to start from 1, but got min={int(item_ids[0])}")

    max_item_id = int(item_ids[-1])

    print(f"[1/3] Loaded UniGRec ItemIDs: n={len(item_ids)}, range={int(item_ids[0])}..{int(item_ids[-1])}")

    with _suppress_output(enabled=not args.verbose):
        token2id = _load_recbole_item_token2id(recbole_path, args.recbole_dataset)
    if "[PAD]" not in token2id or int(token2id["[PAD]"]) != 0:
        raise ValueError("Unexpected RecBole token2id: missing [PAD] or [PAD] id is not 0.")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    emb_key, item_emb = _extract_item_embedding_tensor(checkpoint)

    print(f"[2/3] Loaded RecBole token2id: size={len(token2id)}")
    print(f"[2/3] Loaded SASRec item embedding: key={emb_key}, shape=({int(item_emb.shape[0])}, {int(item_emb.shape[1])})")

    actual_dim = int(item_emb.shape[1])
    if args.hidden_dim is not None and int(args.hidden_dim) != actual_dim:
        raise ValueError(f"hidden_dim mismatch: expected {args.hidden_dim} but ckpt dim={actual_dim}")

    if len(token2id) != int(item_emb.shape[0]):
        raise ValueError(
            f"Vocab mismatch: ckpt rows={int(item_emb.shape[0])} but token2id size={len(token2id)}"
        )

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    emb_list: list[np.ndarray] = []
    iterable = item_ids.tolist()
    if tqdm is not None:
        iterable = tqdm(iterable, desc="Exporting cf embeddings", unit="item")

    for item_id in iterable:
        tok = str(int(item_id))
        if tok not in token2id and unigrec_inv_item is not None and int(item_id) in unigrec_inv_item:
            tok = str(int(unigrec_inv_item[int(item_id)]))
        if tok not in token2id:
            raise KeyError(f"RecBole token2id missing item token for item_id={int(item_id)} (tried '{tok}')")
        rid = int(token2id[tok])
        vec = item_emb[rid].detach().cpu().numpy().astype(np.float32, copy=False)
        emb_list.append(vec)

    if len(emb_list) != len(item_ids):
        raise RuntimeError("Internal error: embedding count mismatch.")

    print(f"[3/3] Writing CF parquet: {out_path}")

    _write_cf_parquet(out_path, item_ids.astype(np.int64), emb_list, args.item_id_col, args.output_emb_col)

    # Post-write validation (quick sanity checks)
    try:
        import pandas as pd
        df_out = pd.read_parquet(out_path, columns=[args.item_id_col, args.output_emb_col])
        out_n = int(df_out.shape[0])
        out_min = int(df_out[args.item_id_col].min())
        out_max = int(df_out[args.item_id_col].max())
        v0 = df_out[args.output_emb_col].iloc[0]
        out_dim = len(v0) if hasattr(v0, '__len__') else None
    except Exception:
        df_out = None
        out_n = len(item_ids)
        out_min = int(item_ids[0])
        out_max = int(item_ids[-1])
        out_dim = actual_dim

    sample_ids = [int(item_ids[0]), int(item_ids[-1])]
    if len(item_ids) > 2:
        sample_ids.append(int(random.choice(item_ids.tolist())))

    # Compute norms from in-memory vectors (no extra IO)
    id_to_idx = {int(i): idx for idx, i in enumerate(item_ids.tolist())}
    sample_norms = {}
    for sid in sample_ids:
        idx = id_to_idx.get(sid)
        if idx is None:
            continue
        sample_norms[sid] = float(np.linalg.norm(emb_list[idx]))

    size_mb = out_path.stat().st_size / 1024 / 1024 if out_path.exists() else -1

    print("=" * 60)
    print("Exported SASRec CF embeddings (UniGRec parquet)")
    print(f"- checkpoint: {ckpt_path}")
    print(f"- ckpt emb key: {emb_key}")
    print(f"- recbole_dataset: {args.recbole_dataset}")
    print(f"- unigrec_item_emb_parquet: {unigrec_item_emb_parquet}")
    print(f"- output_parquet: {out_path} ({size_mb:.2f} MB)")
    print(f"- output schema: {args.item_id_col} (int64), {args.output_emb_col} (ndarray)")
    print(f"- items: {out_n} (expected {len(item_ids)})")
    print(f"- ItemID range: {out_min}..{out_max} (expected 1..{max_item_id})")
    print(f"- embedding dim: {out_dim} (expected {actual_dim})")
    print(f"- sample L2 norms (sanity): {sample_norms}")


if __name__ == "__main__":
    main()
