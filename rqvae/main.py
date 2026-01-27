import argparse
import random
import torch
import numpy as np
from time import time
import logging
import swanlab
import os
from datetime import datetime

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import  Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="RQVAE Training with SwanLab")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str, default="../dataset/Beauty/item_emb_td.parquet", help="Input data path.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not (deprecated if --norm_type is set)")
    parser.add_argument("--norm_type", type=str, default="none", choices=["none", "batchnorm", "layernorm"],
                        help="normalization type inside MLP blocks")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    
    # VQ parameters (standard: VectorQuantizer | soft: SoftVectorQuantizer)
    parser.add_argument("--vq_type", type=str, default="standard", choices=["standard", "soft"], help="VQ type: standard, soft (SoftVectorQuantizer)")
    parser.add_argument("--distance", type=str, default="l2", help="Distance type: l2, dot, cos")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument("--temperature_schedule", type=str, default="none",
                        choices=["none", "linear", "cosine"], help="Temperature annealing schedule type")
    parser.add_argument("--temperature_final", type=float, default=None,
                        help="Target temperature after annealing")
    parser.add_argument("--temperature_anneal_start_epoch", type=int, default=0,
                        help="Epoch to start temperature annealing")
    parser.add_argument("--temperature_anneal_end_epoch", type=int, default=0,
                        help="Epoch to end temperature annealing")
    # SoftVectorQuantizer specific parameters (standard VQ does not use)
    parser.add_argument("--diversity_scale", type=float, default=0.0, help="Diversity loss scale (soft only, reduce collision rate)")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")
    parser.add_argument("--gpu_ids", type=str, default="7", help="Comma-separated GPU IDs to use (e.g., '0' or '0,1,2')")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512,256,128,64], help='hidden sizes of every layer')
    parser.add_argument('--save_limit', type=int, default=5, help='save limit for ckpt')
    
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt/Beauty", help="please specify output directory for model")
    parser.add_argument("--experiment_name", type=str, default="RQVAE-Baseline", help="experiment name for SwanLab")
    parser.add_argument("--seed", type=int, default=2025, help="random seed")

    parser.add_argument("--drop_last", action='store_true',
                        help="Drop the last incomplete batch in training DataLoader")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.temperature_final is None:
        args.temperature_final = args.temperature
    
    # Set visible GPUs
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"[GPU] Setting CUDA_VISIBLE_DEVICES to: {args.gpu_ids}")
    
    # Use seed from arguments
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize SwanLab
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.experiment_name}_{timestamp}"
    
    swanlab.init(
        project="UniGRec-RQVAE",
        name=run_name,
        config=vars(args),
        tags=["UniGRec", "RQVAE"]
    )
    
    print("=================================================")
    print(f"Starting RQVAE Training: {run_name}")
    print(f"SwanLab Project: UniGRec-RQVAE")
    print("=================================================")
    print(args)
    print("=================================================")

    logging.basicConfig(level=logging.INFO)
    
    # Reduce debug logs from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("swanlab").setLevel(logging.INFO)

    # Build dataset
    data = EmbDataset(args.data_path, return_item_id=False)
    print(f"Dataset loaded: {len(data)} items, dimension: {data.dim}")
    
    # Display GPU information
    if torch.cuda.is_available():
        print(f"[GPU] CUDA is available!")
        print(f"[GPU] Number of GPUs: {torch.cuda.device_count()}")
        print(f"[GPU] Current GPU: {torch.cuda.current_device()}")
        print(f"[GPU] GPU Name: {torch.cuda.get_device_name()}")
        print(f"[GPU] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(f"[GPU] Visible GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print(f"[GPU] CUDA is not available, using CPU")
    
    print(f"[GPU] Using device: {args.device}")
    
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
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Log model information to SwanLab
    swanlab.log({
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
        "model/size_mb": total_params * 4 / 1024 / 1024,
        "dataset/size": len(data),
        "dataset/dimension": data.dim
    })
    
    print(model)
    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=args.drop_last
    )
    
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("=" * 60)
    print("Training Completed!")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Best Collision Rate: {best_collision_rate:.4f}")
    print("=" * 60)
    
    # Log final results
    swanlab.log({
        "final/best_loss": best_loss,
        "final/best_collision_rate": best_collision_rate
    })
    
    swanlab.finish()

