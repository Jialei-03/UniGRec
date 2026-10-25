"""Stage 1: tokenizer pretraining.

RQ-VAE with soft identifiers (``--vq_type soft``), Codeword Uniformity
Regularization (``--diversity_scale``, batch-mean entropy of the assignment
distribution of every level) and Annealed Inference Alignment (temperature
decayed linearly from ``--temperature`` to ``--temperature_final``).

The checkpoint with the lowest training loss (checked every ``--eval_step``
epochs) is saved as ``<ckpt_dir>/best_loss_model.pth`` together with its
temperature, which joint training keeps fixed.
"""
import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import Trainer


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")


def parse_args():
    parser = argparse.ArgumentParser(description="UniGRec stage 1: RQ-VAE tokenizer pretraining")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=3000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=50, help="evaluate / checkpoint every N epochs")
    parser.add_argument("--learner", type=str, default="AdamW", help="optimizer")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="linear | constant")
    parser.add_argument("--warmup_epochs", type=int, default=50, help="warmup epochs")
    parser.add_argument("--data_path", type=str, required=True, help="item embedding parquet (columns: ItemID, embedding)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=str2bool, default=False, help="use batch norm (deprecated, see --norm_type)")
    parser.add_argument("--norm_type", type=str, default="none", choices=["none", "batchnorm", "layernorm"],
                        help="normalization inside MLP blocks")
    parser.add_argument("--loss_type", type=str, default="mse", help="reconstruction loss: mse | l1")
    parser.add_argument("--kmeans_init", type=str2bool, default=True, help="initialize codebooks with k-means")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max k-means iterations")
    parser.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0], help="sinkhorn epsilons (standard VQ only)")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iterations (standard VQ only)")

    # Quantizer: soft = UniGRec soft identifiers; standard = hard RQ-VAE (TIGER-style, used by an ablation).
    parser.add_argument("--vq_type", type=str, default="soft", choices=["standard", "soft"])
    parser.add_argument("--distance", type=str, default="l2", help="distance: l2 | dot | cos")
    parser.add_argument("--temperature", type=float, default=0.01, help="initial temperature (tau_max)")
    parser.add_argument("--temperature_schedule", type=str, default="linear", choices=["none", "linear", "cosine"],
                        help="temperature annealing schedule ('none' = constant temperature ablation)")
    parser.add_argument("--temperature_final", type=float, default=0.001, help="final temperature (tau_min)")
    parser.add_argument("--temperature_anneal_start_epoch", type=int, default=0)
    parser.add_argument("--temperature_anneal_end_epoch", type=int, default=3000)
    parser.add_argument("--diversity_scale", type=float, default=1e-4,
                        help="weight of Codeword Uniformity Regularization (0 disables it)")

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[256, 256, 256], help="codebook size of every level")
    parser.add_argument("--e_dim", type=int, default=32, help="latent / codeword dimension")
    parser.add_argument("--quant_loss_weight", type=float, default=1.0, help="weight of the quantizer loss term")
    parser.add_argument("--beta", type=float, default=0.25, help="commitment weight (standard VQ only)")
    parser.add_argument("--layers", type=int, nargs="+", default=[512, 256, 128, 64], help="encoder hidden sizes")

    parser.add_argument("--ckpt_dir", type=str, required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=2025, help="random seed")
    parser.add_argument("--drop_last", action="store_true", help="drop the last incomplete batch")

    parser.add_argument("--use_swanlab", action="store_true", help="log to SwanLab (off by default)")
    parser.add_argument("--swanlab_project", type=str, default="UniGRec")
    parser.add_argument("--swanlab_mode", type=str, default="cloud", choices=["cloud", "local", "offline", "disabled"])
    parser.add_argument("--experiment_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.temperature_schedule == "none":
        args.temperature_final = args.temperature

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.basicConfig(level=logging.INFO)

    logger = None
    if args.use_swanlab:
        import swanlab
        logger = swanlab.init(project=args.swanlab_project,
                              experiment_name=args.experiment_name or os.path.basename(os.path.normpath(args.ckpt_dir)),
                              config=vars(args), mode=args.swanlab_mode)

    data = EmbDataset(args.data_path)
    print(f"Dataset loaded: {len(data)} items, dimension: {data.dim}")
    print(args)

    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  norm_type=args.norm_type,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  vq_type=args.vq_type,
                  distance=args.distance,
                  temperature=args.temperature,
                  diversity_scale=args.diversity_scale)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    data_loader = DataLoader(data, num_workers=args.num_workers, batch_size=args.batch_size,
                             shuffle=True, pin_memory=torch.cuda.is_available(), drop_last=args.drop_last)

    trainer = Trainer(args, model, len(data_loader), logger=logger)
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("=" * 60)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Best collision rate: {best_collision_rate:.4f}")
    print(f"Tokenizer checkpoint: {os.path.join(args.ckpt_dir, 'best_loss_model.pth')}")
    print("=" * 60)
    if logger is not None:
        logger.log({"final/best_loss": best_loss, "final/best_collision_rate": best_collision_rate})
        logger.finish()


if __name__ == "__main__":
    main()
