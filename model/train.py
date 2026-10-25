"""UniGRec stage 2 (end-to-end joint training) and stage 3 (recommender fine-tuning).

Stage 2  L_Joint = L_Rec + alpha L_Recon + lambda_S L_Seq
                   + 1[epoch > E_w] (lambda_H L_Hist + lambda_T L_Tgt)
         The tokenizer is frozen during the first E_w epochs (unified warm-up) and the
         item -> identifier map is recomputed from the current tokenizer before every
         optimization step. The checkpoint with the best validation NDCG@10 is kept.
Stage 3  Tokenizer and identifiers frozen, fresh AdamW, L_Rec only, linear decay,
         selected on validation NDCG@10. The selected model is evaluated on test.
"""
import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import T5Config
from transformers.optimization import get_scheduler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data import ItemTable, SequenceDataset, make_loader  # noqa: E402
from rqvae.datasets import load_item_embeddings  # noqa: E402
from evaluation import evaluate  # noqa: E402
from unigrec import UniGRec, load_tokenizer_checkpoint  # noqa: E402
from utils import build_code_map, make_decoder_io, set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="UniGRec joint training + fine-tuning")
    # data
    p.add_argument("--data_dir", required=True, help="directory with train/valid/test.parquet")
    p.add_argument("--item_emb_path", required=True, help="item content embedding parquet")
    p.add_argument("--rqvae_path", required=True, help="stage-1 tokenizer checkpoint (best_loss_model.pth)")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--max_len", type=int, default=20, help="history length (latest items)")
    # recommender
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_decoder_layers", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_kv", type=int, default=64)
    p.add_argument("--dropout_rate", type=float, default=0.1)
    p.add_argument("--max_collision", type=int, default=100, help="largest collision token (101 values)")
    # anchoring
    p.add_argument("--lambda_seq", type=float, default=0.05, help="weight of L_Seq")
    p.add_argument("--lambda_hist", type=float, default=1.5, help="weight of L_Hist")
    p.add_argument("--lambda_tgt", type=float, default=0.05, help="weight of L_Tgt")
    p.add_argument("--anchor_temperature", type=float, default=0.07, help="tau_a")
    p.add_argument("--anchor_momentum", type=float, default=0.99, help="m")
    p.add_argument("--anchor_hidden_dim", type=int, default=128)
    p.add_argument("--anchor_lr", type=float, default=1e-3)
    p.add_argument("--anchor_seed", type=int, default=17, help="local initialization seed of the anchor modules")
    # joint training
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--t5_lr", type=float, default=5e-3)
    p.add_argument("--rqvae_lr", type=float, default=2e-7)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.1, help="linear LR warm-up, then constant")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.5, help="weight of L_Recon")
    p.add_argument("--warm_epochs", type=int, default=20,
                   help="E_w: tokenizer frozen and alignment losses off for the first E_w epochs")
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--selection_start_epoch", type=int, default=11,
                   help="first (1-based) joint epoch eligible for checkpoint selection")
    # fine-tuning
    p.add_argument("--finetune_lr", type=float, default=2e-4)
    p.add_argument("--finetune_epochs", type=int, default=100)
    p.add_argument("--finetune_patience", type=int, default=10)
    p.add_argument("--skip_finetune", action="store_true", help="evaluate the joint checkpoint directly")
    # ablations
    p.add_argument("--freeze_tokenizer", action="store_true", help="w/o joint training: tokenizer never updated")
    p.add_argument("--hard_identifier_input", action="store_true",
                   help="encoder reads one-hot hard identifiers (use with --freeze_tokenizer --alpha 0)")
    # evaluation / misc
    p.add_argument("--beam_size", type=int, default=30)
    p.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--map_batch_size", type=int, default=4096, help="catalog batch size for map refresh")
    p.add_argument("--use_swanlab", action="store_true", help="log to SwanLab (off by default)")
    p.add_argument("--swanlab_project", default="UniGRec")
    p.add_argument("--swanlab_mode", default="cloud", choices=["cloud", "local", "offline", "disabled"])
    p.add_argument("--exp_name", default=None)
    return p.parse_args()


class Logger:
    def __init__(self, args):
        self.run = None
        if args.use_swanlab:
            import swanlab
            self.run = swanlab.init(project=args.swanlab_project, mode=args.swanlab_mode, config=vars(args),
                                    experiment_name=args.exp_name or os.path.basename(os.path.normpath(args.save_dir)))

    def log(self, values, step):
        if self.run is not None:
            self.run.log(values, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()


def fmt(metrics):
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    logger = Logger(args)

    # ---------------------------------------------------------------- data
    embeddings = torch.from_numpy(load_item_embeddings(args.item_emb_path))
    table = ItemTable(embeddings, device)
    train_set = SequenceDataset(os.path.join(args.data_dir, "train.parquet"), "train", args.max_len)
    valid_set = SequenceDataset(os.path.join(args.data_dir, "valid.parquet"), "eval", args.max_len)
    test_set = SequenceDataset(os.path.join(args.data_dir, "test.parquet"), "eval", args.max_len)
    print(f"items={table.num_items} dim={table.dim} train={len(train_set)} valid={len(valid_set)} test={len(test_set)}")
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    valid_loader = make_loader(valid_set, args.eval_batch_size, False, args.num_workers)
    test_loader = make_loader(test_set, args.eval_batch_size, False, args.num_workers)

    # --------------------------------------------------------------- model
    rqvae_config, rqvae_state, tau = load_tokenizer_checkpoint(args.rqvae_path)
    codebook_size = rqvae_config["num_emb_list"][0]
    num_levels = len(rqvae_config["num_emb_list"])
    if num_levels != 3:
        raise ValueError("this implementation uses L=3 identifier levels")
    print(f"Tokenizer temperature tau* = {tau:.6f} (kept fixed)")
    t5_config = T5Config(
        vocab_size=num_levels * codebook_size + 1 + args.max_collision + 1,
        d_model=args.d_model, num_layers=args.num_layers, num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff, num_heads=args.num_heads, d_kv=args.d_kv, dropout_rate=args.dropout_rate,
        pad_token_id=0, eos_token_id=0, decoder_start_token_id=0)
    use_anchor = max(args.lambda_seq, args.lambda_hist, args.lambda_tgt) > 0
    kwargs = dict(use_anchor=use_anchor, anchor_hidden_dim=args.anchor_hidden_dim,
                  anchor_temperature=args.anchor_temperature, anchor_momentum=args.anchor_momentum,
                  sequence_max_len=args.max_len, anchor_init_seed=args.anchor_seed,
                  hard_identifier_input=args.hard_identifier_input)
    model = UniGRec(rqvae_config, t5_config, embeddings, **kwargs)
    model.load_tokenizer(rqvae_state, tau)
    model.to(device)
    model_config = dict(rqvae_config=dict(rqvae_config, in_dim=table.dim), t5_config=t5_config.to_dict(),
                        kwargs=kwargs)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def checkpoint(path, item_codes, **extra):
        torch.save(dict(model_state_dict=model.state_dict(), item_codes=item_codes.cpu(), tau=tau,
                        model_config=model_config, max_collision=args.max_collision, args=vars(args), **extra), path)

    def refresh_map():
        codes, stats = build_code_map(model.rqvae, table.catalog, args.max_collision, codebook_size,
                                      args.map_batch_size)
        return codes, stats

    def run_eval(loader, item_codes):
        return evaluate(model, loader, table, item_codes, tau, args.beam_size, args.max_collision, device,
                        args.topk)

    # ================================================== stage 2: joint training
    joint_path = os.path.join(args.save_dir, "best_model.pth")
    rqvae_params = [p for p in model.rqvae.parameters() if p.requires_grad]
    groups = [dict(params=rqvae_params, lr=args.rqvae_lr, weight_decay=args.weight_decay),
              dict(params=list(model.t5.parameters()), lr=args.t5_lr, weight_decay=args.weight_decay)]
    if model.anchor is not None:
        groups.append(dict(params=[p for p in model.anchor.parameters() if p.requires_grad],
                           lr=args.anchor_lr, weight_decay=args.weight_decay))
    optimizer = torch.optim.AdamW(groups)
    total_steps = args.num_epochs * len(train_loader)
    scheduler = get_scheduler("constant_with_warmup", optimizer=optimizer,
                              num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps)

    item_codes, stats = refresh_map()
    print(f"Initial map: collision rate {stats['collision_rate']:.4f}, max collision index {stats['max_collision_index']}")
    best, patience, step = -float("inf"), 0, 0
    for epoch in range(args.num_epochs):
        model.train()
        warm = epoch < args.warm_epochs
        model.rqvae.requires_grad_(not (warm or args.freeze_tokenizer))
        w_align = 0.0 if warm else 1.0
        sums, t0 = {}, time.time()
        for history_ids, target_ids in tqdm(train_loader, desc=f"Joint {epoch + 1}/{args.num_epochs}", leave=False):
            # The identifier map always reflects the current tokenizer.
            item_codes, _ = refresh_map()
            batch = table.batch(history_ids, target_ids, device)
            decoder_input_ids, labels = make_decoder_io(batch["target_ids"], item_codes, codebook_size)
            loss, terms = model(batch["history_embs"], batch["attention_mask"], batch["target_embs"],
                                batch["target_ids"], decoder_input_ids, labels, tau, alpha=args.alpha,
                                w_seq=args.lambda_seq, w_hist=w_align * args.lambda_hist,
                                w_tgt=w_align * args.lambda_tgt)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if model.anchor is not None:
                model.anchor.update_momentum()
            scheduler.step()
            for k, v in dict(terms, total=loss).items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            logger.log({f"joint/train/{k}": float(v.detach()) for k, v in dict(terms, total=loss).items()}, step)
            step += 1
        train_log = {k: v / len(train_loader) for k, v in sums.items()}

        item_codes, stats = refresh_map()
        valid = run_eval(valid_loader, item_codes)
        print(f"[Joint {epoch + 1}] {time.time() - t0:.1f}s  loss: {fmt(train_log)}  "
              f"collision={stats['collision_rate']:.4f}\n  valid: {fmt(valid)}")
        logger.log({**{f"joint/val/{k}": v for k, v in valid.items()},
                    "joint/map/collision_rate": stats["collision_rate"]}, step)

        if epoch + 1 >= args.selection_start_epoch:
            if valid["ndcg@10"] > best:
                best, patience = valid["ndcg@10"], 0
                checkpoint(joint_path, item_codes, epoch=epoch + 1, valid=valid)
                print(f"  saved best joint checkpoint (valid NDCG@10={best:.4f})")
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"Early stopping joint training after epoch {epoch + 1}")
                    break
    if not os.path.exists(joint_path):  # fewer epochs than selection_start_epoch
        checkpoint(joint_path, item_codes, epoch=args.num_epochs, valid=valid)

    # ================================================= stage 3: fine-tuning
    joint = torch.load(joint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(joint["model_state_dict"])
    item_codes = joint["item_codes"].to(device)          # identifiers frozen
    final_path = joint_path
    results = dict(joint_epoch=joint["epoch"], joint_valid=joint["valid"])
    if not args.skip_finetune:
        model.rqvae.requires_grad_(False)
        if model.anchor is not None:
            model.anchor.requires_grad_(False)
        model.t5.requires_grad_(True)
        t5_params = list(model.t5.parameters())
        optimizer = torch.optim.AdamW([dict(params=t5_params, lr=args.finetune_lr, weight_decay=args.weight_decay)])
        ft_steps = args.finetune_epochs * len(train_loader)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=ft_steps)
        # Stage boundary: fresh data order and random state.
        train_loader = make_loader(train_set, args.batch_size, True, args.num_workers,
                                   generator=torch.Generator().manual_seed(args.seed))
        set_seed(args.seed)
        ft_path = os.path.join(args.save_dir, "best_finetune_model.pth")
        best, patience = -float("inf"), 0
        for epoch in range(args.finetune_epochs):
            model.train()
            total, t0 = 0.0, time.time()
            for history_ids, target_ids in tqdm(train_loader, desc=f"Finetune {epoch + 1}/{args.finetune_epochs}",
                                                leave=False):
                batch = table.batch(history_ids, target_ids, device)
                decoder_input_ids, labels = make_decoder_io(batch["target_ids"], item_codes, codebook_size)
                loss, _ = model(batch["history_embs"], batch["attention_mask"], batch["target_embs"],
                                batch["target_ids"], decoder_input_ids, labels, tau)
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(t5_params, args.grad_clip)
                optimizer.step()
                scheduler.step()
                total += float(loss.detach())
                logger.log({"finetune/train/rec": float(loss.detach())}, step)
                step += 1
            valid = run_eval(valid_loader, item_codes)
            print(f"[Finetune {epoch + 1}] {time.time() - t0:.1f}s  rec={total / len(train_loader):.4f}\n"
                  f"  valid: {fmt(valid)}")
            logger.log({f"finetune/val/{k}": v for k, v in valid.items()}, step)
            if valid["ndcg@10"] > best:
                best, patience = valid["ndcg@10"], 0
                checkpoint(ft_path, item_codes, epoch=epoch + 1, valid=valid)
                print(f"  saved best fine-tuned checkpoint (valid NDCG@10={best:.4f})")
            else:
                patience += 1
                if patience >= args.finetune_patience:
                    print(f"Early stopping fine-tuning after epoch {epoch + 1}")
                    break
        final_path = ft_path
        ft = torch.load(ft_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ft["model_state_dict"])
        results.update(finetune_epoch=ft["epoch"], finetune_valid=ft["valid"])

    # ====================================================== test evaluation
    test = run_eval(test_loader, item_codes)
    results.update(checkpoint=os.path.basename(final_path), test=test)
    print(f"Test (beam order, checkpoint {os.path.basename(final_path)}): {fmt(test)}")
    logger.log({f"test/{k}": v for k, v in test.items()}, step)
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.finish()


if __name__ == "__main__":
    main()
