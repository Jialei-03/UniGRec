"""
End-to-end training from scratch.

RQVAE: load pretrained weights (frozen or trainable)
T5: randomly initialized and trained from scratch
Training mode: joint training or alternating training
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
import sys
import numpy as np
import pandas as pd
import time


sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rqvae'))

from unigrec import UniGRec
from dataset import GenRecDataset
from dataloader import GenRecDataLoader
from datasets import EmbDataset
from transformers import T5Config
from transformers.optimization import get_scheduler as get_transformers_scheduler
from utils import (
    set_seed,
    calculate_codebook_usage,
    calculate_collision_rate,
    calculate_soft_entropy,
    calculate_pos_index,
    recall_at_k,
    ndcg_at_k,
    compute_code_change_rate,
    generate_labels_from_mapping,
    build_code_with_collision_index
)

import swanlab

def evaluate(model, eval_loader, code_to_item, device, 
                      topk_list=[5, 10, 20], beam_size=20, tau=0.5, 
                      compute_loss=False, alpha=1.0, beta=1.0, gamma=0.0, delta=0.0,
                      diversity_weight=0.0,
                      item_embeddings=None, use_code_offset=True, codebook_size=256,
                      rebuild_mapping=True, max_collision=100, item_to_code=None,
                      use_code_range_constraint=True):
    """
    Evaluate the model on the validation set, computing Recall and NDCG.
    Optionally compute validation loss (4-token code version).
    
    Args:
        model: UniGRec model
        eval_loader: validation data loader
        code_to_item: {4-token code_tuple: item_id} unique mapping (rebuilt if rebuild_mapping=True)
        device: device
        topk_list: list of k values
        beam_size: beam search size
        tau: Gumbel-Softmax temperature
        compute_loss: whether to compute validation loss (for early stopping)
        alpha: reconstruction loss weight
        beta: recommendation loss weight
        gamma: KL loss weight (kl_loss)
        delta: decoder contrastive loss weight (dec_cl_loss)
        diversity_weight: additional diversity-loss weight
        item_embeddings: (N, emb_dim) all item embeddings (for rebuilding the mapping)
        use_code_offset: whether to use code-offset encoding
        codebook_size: codebook size
        rebuild_mapping: whether to rebuild mapping using current RQVAE (default True)
        max_collision: maximum collision index
        item_to_code: {item_id: 4-token code_tuple} reverse mapping (required if rebuild_mapping=False)
    """
    model.eval()
    
    # Rebuild mapping with the current RQVAE (if item_embeddings is provided)
    if rebuild_mapping and item_embeddings is not None:
        print("  [Eval] Rebuilding code mapping using current RQVAE (with collision index)...")
        all_item_codes, code_to_item_new, item_to_code_new, collision_stats = \
            build_code_with_collision_index(
                model.rqvae, item_embeddings, use_code_offset, codebook_size, device, max_collision
            )
        code_to_item = code_to_item_new
        if item_to_code is None:
            item_to_code = item_to_code_new
        print(f"  Rebuilt mapping: {len(code_to_item)} unique codes")
        print(f"  Max collisions: {collision_stats['max_collision']}")
        print(f"  Collision rate: {collision_stats['collision_rate']:.4f}")
    else:
        # Use provided mapping
        if item_to_code is None:
            raise ValueError("When rebuild_mapping=False, item_to_code must be provided")
    
    recalls = {f'Recall@{k}': [] for k in topk_list}
    ndcgs = {f'NDCG@{k}': [] for k in topk_list}
    
    # Used to compute validation loss
    total_loss = 0.0
    total_rec_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_dec_cl_loss = 0.0
    total_diversity_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            history_embs = batch['history_embs'].to(device, non_blocking=True)  # (B, seq_len, emb_dim)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)  # (B, seq_len)
            target_item_ids = batch.get('target_item_id', None)  # (B,) - raw target item IDs (if available)
            target_item_ids_np = None
            target_item_ids_tensor = None
            if target_item_ids is not None:
                target_item_ids_np = np.array(target_item_ids)
                target_item_ids_tensor = torch.tensor(target_item_ids, dtype=torch.long, device=device)
            
            # Query target codes from mapping (4-token codes)
            target_embs = batch['target_embs'].to(device, non_blocking=True)  # (B, emb_dim)
            if target_item_ids is None:
                raise ValueError("Batch must contain target_item_id")

            # Lookup via mapping (4-token codes)
            decoder_input_ids, labels = generate_labels_from_mapping(
                target_item_ids_np, item_to_code, use_code_offset, codebook_size, device
            )
            # decoder_input_ids: (B, 5) = [0, emb_id0, emb_id1, emb_id2, emb_id_collision]
            # labels: (B, 5) = [emb_id0, emb_id1, emb_id2, emb_id_collision, 0]
                
            # Optionally compute validation loss
            if compute_loss:
                # Forward pass for loss
                decoder_input_ids_for_loss = decoder_input_ids[:, :-1].contiguous()  # (B, 4) - [0, emb_id0, emb_id1, emb_id2]
                decoder_labels_for_loss = labels[:, :-1].contiguous()  # (B, 4) - [emb_id0, emb_id1, emb_id2, emb_id_collision]
                
                # UniGRec.forward() returns 6 values
                # Prepare target_item_ids (for deduplication)
                target_item_ids_for_loss = None
                if target_item_ids_tensor is not None:
                    target_item_ids_for_loss = target_item_ids_tensor
                
                raw_total_loss, rec_loss, recon_loss, kl_loss, dec_cl_loss, diversity_loss = model(
                    history_embs=history_embs,
                    target_embs=target_embs,
                    decoder_input_ids=decoder_input_ids_for_loss,
                    attention_mask=attention_mask,
                    labels=decoder_labels_for_loss,
                    tau=tau,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    delta=delta,
                    target_item_ids=target_item_ids_for_loss,  # for deduplication
                    diversity_weight=diversity_weight,
                )
                batch_total_loss = raw_total_loss
                
                # Accumulate losses
                total_loss += batch_total_loss.item()
                total_rec_loss += rec_loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_dec_cl_loss += dec_cl_loss.item()
                total_diversity_loss += diversity_loss.item()
                num_batches += 1
            
            # Generation (for Recall/NDCG)
            generated = model.generate(
                history_embs=history_embs,
                attention_mask=attention_mask,
                max_length=5,  # 4-token code + start token = 5
                num_beams=beam_size,
                tau=tau,
                use_code_range_constraint=use_code_range_constraint,  # enable range constraint
                max_collision=max_collision,
            )  # (B*beam_size, 5)
            
            # Parse codes
            preds = generated[:, 1:]  # remove start token, (B*beam_size, 4)
            
            # Validate generated sequence length
            if preds.shape[1] < 4:
                print(f"  Warning: generated sequence is too short. Expected 4 tokens, got {preds.shape[1]}")
                # Pad if too short (use PAD token 0)
                if preds.shape[1] == 0:
                    preds = torch.zeros(preds.shape[0], 4, dtype=preds.dtype, device=preds.device)
                else:
                    padding = torch.zeros(preds.shape[0], 4 - preds.shape[1], dtype=preds.dtype, device=preds.device)
                    preds = torch.cat([preds, padding], dim=1)
            
            # Ensure preds has the correct shape
            if preds.shape[1] != 4:
                print(f"  Warning: preds.shape[1]={preds.shape[1]}, expected 4. Padding/truncating.")
                if preds.shape[1] < 4:
                    padding = torch.zeros(preds.shape[0], 4 - preds.shape[1], dtype=preds.dtype, device=preds.device)
                    preds = torch.cat([preds, padding], dim=1)
                else:
                    preds = preds[:, :4]  # truncate to 4 tokens
            
            preds = preds.reshape(labels.shape[0], beam_size, -1)  # (B, beam_size, 4)
            
            # Final check: ensure shape after reshape is correct
            if preds.shape[2] != 4:
                print(f"  Warning: after reshape preds.shape[2]={preds.shape[2]}, expected 4")
                # Force reshape if still incorrect
                preds = preds.reshape(labels.shape[0] * beam_size, -1)
                if preds.shape[1] < 4:
                    padding = torch.zeros(preds.shape[0], 4 - preds.shape[1], dtype=preds.dtype, device=preds.device)
                    preds = torch.cat([preds, padding], dim=1)
                preds = preds.reshape(labels.shape[0], beam_size, 4)
            
            # Compute position index (4-token codes)
            # labels[:, :-1] removes the trailing 0 -> (B, 4)
            pos_index = calculate_pos_index(
                preds, labels[:, :-1], beam_size, code_to_item, 
                target_item_ids=target_item_ids_np,
                use_code_offset=use_code_offset,
                codebook_size=codebook_size
            )
            
            # Compute recall and ndcg for each k
            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls[f'Recall@{k}'].append(recall)
                ndcgs[f'NDCG@{k}'].append(ndcg)
    
    # Compute averages
    avg_recalls = {k: sum(v) / len(v) if len(v) > 0 else 0.0 for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) if len(v) > 0 else 0.0 for k, v in ndcgs.items()}
    
    # Compute average losses
    avg_valid_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_valid_rec_loss = total_rec_loss / num_batches if num_batches > 0 else 0.0
    avg_valid_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0
    avg_valid_kl_loss = total_kl_loss / num_batches if num_batches > 0 else 0.0
    avg_valid_dec_cl_loss = total_dec_cl_loss / num_batches if num_batches > 0 else 0.0
    avg_valid_diversity_loss = total_diversity_loss / num_batches if num_batches > 0 else 0.0
    
    if compute_loss:
        return avg_recalls, avg_ndcgs, avg_valid_loss, avg_valid_rec_loss, avg_valid_recon_loss, avg_valid_kl_loss, avg_valid_dec_cl_loss, avg_valid_diversity_loss
    else:
        return avg_recalls, avg_ndcgs


def train(args):
    """Main entry for end-to-end training from scratch."""
    
    # Check finetune-only mode
    finetune_only = getattr(args, 'finetune_only', False)
    
    if finetune_only:
        print("=" * 70)
        print("Finetune-Only Mode")
        print("=" * 70)
        print("Skip joint training and start finetuning from a joint-training checkpoint")
        print("=" * 70)
        
        # Validate required args
        if not args.joint_checkpoint_path:
            raise ValueError("Finetune-only mode requires --joint_checkpoint_path (points to joint best_model.pth)")
        if not os.path.exists(args.joint_checkpoint_path):
            raise FileNotFoundError(f"Joint-training checkpoint not found: {args.joint_checkpoint_path}")
        
        # Force-enable finetune
        args.enable_finetune = True
    else:
        print("=" * 70)
        print("End-to-End from Scratch")
        print("=" * 70)
        print("RQVAE: pretrained weights (trainable)")
        print("T5: randomly initialized (trained from scratch)")
        print("=" * 70)
    
    # Set random seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize SwanLab
    run = None
    if args.use_swanlab:
        # Build experiment name
        if args.exp_name:
            exp_name = args.exp_name
        else:
            # Auto-generate experiment name with key hyperparameters
            exp_name = f"E2E-t5_{args.t5_lr}-rq_{args.rqvae_lr}-a{args.alpha}b{args.beta}"
            if args.gamma > 0:
                exp_name += f"-g{args.gamma}"
            if args.delta > 0:
                exp_name += f"-d{args.delta}"
        run = swanlab.init(
            project="UniGRec-E2E",
            experiment_name=exp_name,
            config=vars(args),
        )
        print(f"\nExperiment name: {exp_name}")
    
    print("\n[1/7] Loading item embeddings...")
    emb_dataset = EmbDataset(args.item_emb_path)
    item_embeddings = torch.from_numpy(emb_dataset.embeddings).float()
    print(f"Loaded {len(item_embeddings)} items, dim={item_embeddings.shape[1]}")
    
    print("\n[2/7] Loading pretrained RQVAE...")
    rqvae_ckpt = torch.load(args.rqvae_model_path, map_location='cpu', weights_only=False)
    rqvae_args = rqvae_ckpt['args']
    
    # Select temperature based on rqvae_temperature_mode
    temperature_mode = getattr(args, 'rqvae_temperature_mode', 'current')
    saved_temperature = rqvae_ckpt.get('current_temperature', None)
    final_temperature = getattr(rqvae_args, 'temperature_final', None)
    
    if temperature_mode == 'current':
        # current: prefer saved current_temperature; otherwise use final/initial
        if saved_temperature is not None:
            effective_temperature = saved_temperature
            print(f"  [current] Using saved current_temperature: {effective_temperature:.6f}")
        else:
            # If not saved, use final temperature
            effective_temperature = final_temperature if final_temperature is not None else 0.001
            print(f"  [current] No saved current_temperature. Using initial/final temperature: {effective_temperature:.6f}")
    elif temperature_mode == 'final':
        # final: use final temperature (ignore current_temperature)
        effective_temperature = final_temperature if final_temperature is not None else 0.001
        print(f"  [final] Using final temperature: {effective_temperature:.6f}")
    else:
        raise ValueError(f"Unknown temperature_mode: {temperature_mode}")

    # Force tau used by encode_soft during E2E training to match the checkpoint temperature.
    # Note: the CLI argument --tau will be ignored.
    if saved_temperature is None:
        raise ValueError(
            "RQVAE checkpoint does not contain 'current_temperature'. "
            "Cannot force tau to match checkpoint temperature."
        )
    args.tau = float(saved_temperature)
    
    rqvae_config = {
        'in_dim': emb_dataset.dim,
        'num_emb_list': rqvae_args.num_emb_list,
        'e_dim': rqvae_args.e_dim,
        'layers': rqvae_args.layers,
        'dropout_prob': rqvae_args.dropout_prob,
        'bn': rqvae_args.bn,
        'loss_type': rqvae_args.loss_type,
        'quant_loss_weight': rqvae_args.quant_loss_weight,
        'beta': rqvae_args.beta,
        'kmeans_init': False,
        'kmeans_iters': 0,
        'sk_epsilons': rqvae_args.sk_epsilons,
        'sk_iters': rqvae_args.sk_iters,
        'vq_type': rqvae_args.vq_type,
        'distance': rqvae_args.distance,
        'temperature': effective_temperature,
        'diversity_scale': getattr(rqvae_args, 'diversity_scale', 0.0001),
    }
    
    print("\n[3/7] Building T5 config...")
    # Set vocab_size dynamically based on use_code_offset
    # Default is offset mode (True)
    use_code_offset = getattr(args, 'use_code_offset', True)
    max_collision = getattr(args, 'max_collision', 100)  # default 100, total 101 collision indices (0-100)
    codebook_size = 256

    collision_vocab = max_collision + 1
    if use_code_offset:
        vocab_size = (3 * codebook_size + 1) + collision_vocab
    else:
        vocab_size = (codebook_size + 1) + collision_vocab
    
    # # Compute max sequence length
    # # Original: max_len * 3 (3 tokens per item for 3 layers)
    # max_seq_len = args.max_len * 3
    
    # # T5 relative position config
    # # relative_attention_max_distance: maximum relative distance (default 128)
    # # relative_attention_num_buckets: number of buckets (default 32)
    # # When sequence length increases, these may need to be increased to avoid index overflow
    # # We set max_distance to 2x sequence length (bidirectional) and increase buckets accordingly
    # relative_attention_max_distance = max(128, max_seq_len * 2)  # at least 128, or 2x sequence length
    # relative_attention_num_buckets = max(32, relative_attention_max_distance // 4)  # buckets ~= max_distance/4
    
    t5_config = T5Config(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        d_kv=args.d_kv,
        dropout_rate=args.dropout_rate,
        pad_token_id=0,
        eos_token_id=0,
        decoder_start_token_id=0,
        # Keep hooks for longer sequences (e.g. when inserting user token)
        # relative_attention_max_distance=relative_attention_max_distance,
        # relative_attention_num_buckets=relative_attention_num_buckets,
    )

    print(f"T5 vocab_size: {vocab_size} (use_code_offset={use_code_offset}, max_collision={max_collision})")
    # print(f"  Max sequence length: {max_seq_len} (max_len={args.max_len} * 3)")
    # print(f"  Relative position encoding: max_distance={relative_attention_max_distance}, num_buckets={relative_attention_num_buckets}")
    
    print("\n[4/7] Loading datasets...")
    # Load datasets early to obtain dataset configuration
    use_code_offset_for_dataset = getattr(args, 'use_code_offset', True)  # default True
    max_collision = getattr(args, 'max_collision', 100)
    
    train_dataset = GenRecDataset(
        dataset_path=args.train_data,
        rqvae_model_path=args.rqvae_model_path,
        item_emb_path=args.item_emb_path,
        mode='train',
        max_len=args.max_len,
        device=args.device,
        real_time_soft=True,
        use_code_offset=use_code_offset_for_dataset,
        max_collision=max_collision,
    )

    valid_dataset = GenRecDataset(
        dataset_path=args.valid_data,
        rqvae_model_path=args.rqvae_model_path,
        item_emb_path=args.item_emb_path,
        mode='evaluation',
        max_len=args.max_len,
        device=args.device,
        real_time_soft=True,
        use_code_offset=use_code_offset_for_dataset,
        max_collision=max_collision,
    )

    test_dataset = GenRecDataset(
        dataset_path=args.test_data,
        rqvae_model_path=args.rqvae_model_path,
        item_emb_path=args.item_emb_path,
        mode='evaluation',
        max_len=args.max_len,
        device=args.device,
        real_time_soft=True,
        use_code_offset=use_code_offset_for_dataset,
        max_collision=max_collision,
    )
    
    # Fetch effective use_code_offset and codebook_size from dataset
    use_code_offset = train_dataset.use_code_offset
    codebook_size = train_dataset.codebook_size
    
    print(f"use_code_offset: {use_code_offset}")
    print(f"codebook_size: {codebook_size}")
    print(f"max_collision: {max_collision}")


    print("\n[5/7] Building end-to-end model...")
    model = UniGRec(
        rqvae_config,
        t5_config,
        freeze_rqvae=False,
        use_code_offset=use_code_offset,
        codebook_size=codebook_size,
    )

    # Optional SASRec teacher: load embeddings and initialize teacher adapters
    if getattr(args, 'use_sasrec_teacher', False):
        if not getattr(args, 'sasrec_emb_path', None):
            raise ValueError("--use_sasrec_teacher requires --sasrec_emb_path")
        sasrec_df = pd.read_parquet(args.sasrec_emb_path)
        if 'ItemID' not in sasrec_df.columns:
            raise ValueError(f"sasrec_emb parquet must contain column: ItemID. Available columns: {list(sasrec_df.columns)}")
        emb_candidates = ['sasrec_embedding', 'embedding', 'cf_embedding']
        emb_col = next((c for c in emb_candidates if c in sasrec_df.columns), None)
        if emb_col is None:
            raise ValueError(
                f"sasrec_emb parquet must contain an embedding column in {emb_candidates}. Available columns: {list(sasrec_df.columns)}"
            )
        sasrec_item_ids = sasrec_df['ItemID'].to_numpy(dtype=np.int64)
        sasrec_mat = np.stack(sasrec_df[emb_col].values, axis=0).astype(np.float32)
        if sasrec_mat.shape[0] != len(item_embeddings):
            raise ValueError(f"sasrec_embedding rows ({sasrec_mat.shape[0]}) != num_items ({len(item_embeddings)})")
        order = np.argsort(sasrec_item_ids)
        sasrec_item_ids = sasrec_item_ids[order]
        sasrec_mat = sasrec_mat[order]
        expected_ids = np.arange(1, len(item_embeddings) + 1, dtype=np.int64)
        if not np.array_equal(sasrec_item_ids, expected_ids):
            raise ValueError(
                f"sasrec_emb ItemID must be exactly 1..{len(item_embeddings)} with no duplicates and aligned to rows"
            )
        model.init_sasrec_teacher(torch.from_numpy(sasrec_mat))
        print(f"  SASRec teacher enabled: sasrec_emb_path={args.sasrec_emb_path}, dim={sasrec_mat.shape[1]}")
    
    # Load pretrained RQVAE
    model.load_rqvae_checkpoint(args.rqvae_model_path)
    
    # T5 is initialized from scratch (no pretrained weights)
    print("T5 initialized from scratch (random weights)")
    print("RQVAE pretrained weights loaded; will be jointly trained with T5")
    # Print SoftVectorQuantizer params (if vq_type=soft)
    if rqvae_config.get('vq_type') == 'soft':
        print("SoftVectorQuantizer config:")
        print(f"diversity_scale: {rqvae_config.get('diversity_scale', 0.0)}")
        if rqvae_config.get('diversity_scale', 0.0) > 0:
            print("Diversity loss enabled")
    
    # Add key RQVAE config to SwanLab
    if args.use_swanlab:
        rqvae_config_for_swanlab = {
            'rqvae_vq_type': rqvae_config.get('vq_type', 'standard'),
            'rqvae_diversity_scale': rqvae_config.get('diversity_scale', 0.0),
            'rqvae_distance': rqvae_config.get('distance', 'l2'),
            'rqvae_temperature': rqvae_config.get('temperature', 0.001),
        }
        swanlab.config.update(rqvae_config_for_swanlab)
        print(f"  Added RQVAE config to SwanLab (including diversity_scale={rqvae_config.get('diversity_scale', 0.0)})")
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("  Model built successfully")
    print(f"    Total params: {total_params:,}")
    print(f"    Trainable params: {trainable_params:,}")
    
    print("\n[5/7] Configuring optimizer (grouped learning rates)...")
    rqvae_params = list(model.rqvae.parameters())
    t5_params = list(model.t5.parameters())
    
    optimizer = optim.AdamW([
        {'params': rqvae_params, 'lr': args.rqvae_lr, 'weight_decay': args.weight_decay},
        {'params': t5_params, 'lr': args.t5_lr, 'weight_decay': args.weight_decay},
    ])
    
    print(f"  RQVAE LR: {args.rqvae_lr:.2e}")
    print(f"  T5 LR: {args.t5_lr:.2e}")
    print(f"  Weight Decay: {args.weight_decay:.2e}")
    
    # LR scheduler (only cosine_with_warmup / constant_with_warmup / none)
    scheduler = None
    
    print("\n[6/7] Building data loaders...")
    
    train_loader = GenRecDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        device=args.device,
        pin_memory=args.pin_memory,
        pin_memory_device=args.pin_memory_device,
        prefetch_factor=args.prefetch_factor,
        prefetch_to_gpu=args.prefetch_to_gpu,
    )
    
    print(f"  Training data: {len(train_dataset)} samples")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps per epoch: {len(train_loader)}")
    
    scheduler_name_to_transformers = {
        'cosine_with_warmup': 'cosine',
        'constant_with_warmup': 'constant_with_warmup',
    }
    if args.scheduler_type in scheduler_name_to_transformers:
        train_steps_per_epoch = len(train_loader)
        total_train_steps = args.num_epochs * train_steps_per_epoch
        warmup_ratio = args.warmup_ratio
        num_warmup_steps = int(total_train_steps * warmup_ratio)
        scheduler = get_transformers_scheduler(
            name=scheduler_name_to_transformers[args.scheduler_type],
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_train_steps,
        )
        scheduler_name_map = {
            'cosine_with_warmup': 'Cosine with Warmup',
            'constant_with_warmup': 'Constant with Warmup',
        }
        print(f"  Using {scheduler_name_map.get(args.scheduler_type, args.scheduler_type)} LR scheduler (transformers): "
              f"total_train_steps={total_train_steps}, num_warmup_steps={num_warmup_steps} "
              f"(≈ 前 {warmup_ratio*100:.1f}% 的 step)")
    else:
        scheduler = None
        print(f"No LR scheduler (scheduler_type={args.scheduler_type})")
    
    valid_loader = GenRecDataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.eval_num_workers,
        device=args.device,
        pin_memory=args.pin_memory,
        pin_memory_device=args.pin_memory_device,
        prefetch_factor=args.prefetch_factor,
        prefetch_to_gpu=args.prefetch_to_gpu,
    )
    
    print(f"Validation data: {len(valid_dataset)} samples")
    
    test_loader = GenRecDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.eval_num_workers,
        device=args.device,
        pin_memory=args.pin_memory,
        pin_memory_device=args.pin_memory_device,
        prefetch_factor=args.prefetch_factor,
        prefetch_to_gpu=args.prefetch_to_gpu,
    )
    
    print(f"Test data: {len(test_dataset)} samples")
    
    # Build code_to_item mapping (for evaluation; will be updated during training)
    # Initialize from training dataset mapping (4-token codes)
    if hasattr(train_dataset, 'item_to_code') and train_dataset.item_to_code is not None:
        # Build reverse mapping from item_to_code
        code_to_item = {v: k for k, v in train_dataset.item_to_code.items()}
        print(f"Code mapping: {len(code_to_item)} unique codes (loaded from dataset, 4-token codes)")
    else:
        code_to_item = {}
        print("Code mapping will be built during the training loop")
    
    print("\n[7/7] Recording initial codes (for monitoring change rate)...")
    initial_codes = []
    with torch.no_grad():
        N = len(item_embeddings)
        bs = max(1, args.init_code_batch_size)
        for i in tqdm(range(0, N, bs), desc="Initial encoding"):
            batch = item_embeddings[i:i+bs].to(device)
            codes = model.rqvae.get_indices(batch, use_sk=False)
            initial_codes.append(codes.cpu())
    initial_codes = torch.cat(initial_codes, dim=0)
    print(f"  Recorded initial codes for {len(initial_codes)} items")
    
    # =========================================================================
    # Pre-training evaluation: verify loaded model correctness
    # =========================================================================
    print("\n" + "=" * 70)
    print("Initial evaluation before training (verify model loading)")
    print("=" * 70)
    
    print("\n[Initial Eval] Rebuilding code mapping using current RQVAE (with collision index)...")
    max_collision = getattr(args, 'max_collision', 100)
    all_item_codes, eval_code_to_item, eval_item_to_code, collision_stats = \
        build_code_with_collision_index(
            model.rqvae,
            item_embeddings,
            use_code_offset=train_dataset.use_code_offset,
            codebook_size=train_dataset.codebook_size,
            device=device,
            max_collision=max_collision,
        )
    # Update 4-token mapping in training dataset
    train_dataset.item_to_code = eval_item_to_code
    train_dataset.all_item_codes = torch.tensor(all_item_codes).to(device)
    print(f"Rebuilt mapping: {len(eval_code_to_item)} unique codes")
    print(f"Max collisions: {collision_stats['max_collision']}")
    print(f"Collision rate: {collision_stats['collision_rate']:.4f}")

    warm_epoch = getattr(args, 'warm_epoch', 10)
    effective_gamma_init = float(args.gamma) if 0 >= int(warm_epoch) else 0.0
    effective_delta_init = float(args.delta) if 0 >= int(warm_epoch) else 0.0

    print("\nEvaluating initial model on validation set...")
    init_valid_recalls, init_valid_ndcgs, init_valid_loss, init_valid_rec_loss, init_valid_recon_loss, init_valid_kl_loss, init_valid_dec_cl_loss, init_valid_diversity_loss = evaluate(
        model, valid_loader, eval_code_to_item, device, 
        topk_list=args.topk_list, beam_size=args.beam_size, tau=args.tau,
        compute_loss=True, alpha=args.alpha, beta=args.beta, gamma=effective_gamma_init, delta=effective_delta_init,
        diversity_weight=args.diversity_weight,
        item_embeddings=item_embeddings,
        use_code_offset=train_dataset.use_code_offset,
        codebook_size=train_dataset.codebook_size,
        rebuild_mapping=False,
        max_collision=max_collision,
        item_to_code=eval_item_to_code
    )
    print("Initial validation performance:")
    print(f"  Total Loss: {init_valid_loss:.4f}")
    print(f"  Rec Loss: {init_valid_rec_loss:.4f}")
    print(f"  Recon Loss: {init_valid_recon_loss:.4f}")
    print(f"  Diversity Loss: {init_valid_diversity_loss:.6f}")
    if effective_gamma_init > 0:
        print(f"  CDT Loss: {init_valid_kl_loss:.4f}")
    if effective_delta_init > 0:
        print(f"  CDR Loss: {init_valid_dec_cl_loss:.4f}")
    for k in args.topk_list:
        print(f"  Recall@{k}: {init_valid_recalls[f'Recall@{k}']:.4f}  NDCG@{k}: {init_valid_ndcgs[f'NDCG@{k}']:.4f}")

    print("\nEvaluating initial model on test set...")
    init_test_recalls, init_test_ndcgs, init_test_loss, init_test_rec_loss, init_test_recon_loss, init_test_kl_loss, init_test_dec_cl_loss, init_test_diversity_loss = evaluate(
        model, test_loader, eval_code_to_item, device, 
        topk_list=args.topk_list, beam_size=args.beam_size, tau=args.tau,
        compute_loss=True, alpha=args.alpha, beta=args.beta, gamma=effective_gamma_init, delta=effective_delta_init,
        diversity_weight=args.diversity_weight,
        item_embeddings=item_embeddings,
        use_code_offset=train_dataset.use_code_offset,
        codebook_size=train_dataset.codebook_size,
        rebuild_mapping=False,
        max_collision=max_collision,
        item_to_code=eval_item_to_code
    )
    print("Initial test performance:")
    print(f"  Total Loss: {init_test_loss:.4f}")
    print(f"  Rec Loss: {init_test_rec_loss:.4f}")
    print(f"  Recon Loss: {init_test_recon_loss:.4f}")
    print(f"  Diversity Loss: {init_test_diversity_loss:.6f}")
    if effective_gamma_init > 0:
        print(f"  CDT Loss: {init_test_kl_loss:.4f}")
    if effective_delta_init > 0:
        print(f"  CDR Loss: {init_test_dec_cl_loss:.4f}")
    for k in args.topk_list:
        print(f"  Recall@{k}: {init_test_recalls[f'Recall@{k}']:.4f}  NDCG@{k}: {init_test_ndcgs[f'NDCG@{k}']:.4f}")

    # Log initial performance to SwanLab
    if run:
        init_metrics = {
            'epoch': 0,
            'val/total_loss': init_valid_loss,
            'val/rec_loss': init_valid_rec_loss,
            'val/recon_loss': init_valid_recon_loss,
            'val/diversity_loss': init_valid_diversity_loss,
        }
        for k in args.topk_list:
            init_metrics[f'val/recall@{k}'] = init_valid_recalls[f'Recall@{k}']
            init_metrics[f'val/ndcg@{k}'] = init_valid_ndcgs[f'NDCG@{k}']
        run.log(init_metrics, step=0)
    
    print("\n" + "=" * 70)
    if finetune_only:
        print("Skipping joint training and entering finetune stage")
        print("=" * 70)
        # Set variables for finetune stage
        global_step = 0
        best_model_path = args.joint_checkpoint_path  # use the specified joint-training checkpoint
        
        # Load joint-training checkpoint
        print(f"\n[Finetune-Only] Loading joint-training checkpoint: {best_model_path}")
        joint_ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(joint_ckpt['model_state_dict'])
        model.to(device)
        print(f"Loaded joint-training model (epoch {joint_ckpt.get('epoch', 'unknown')})")
        
        # Rebuild mapping (using the current RQVAE state)
        print("\n[Finetune-Only] Rebuilding code mapping...")
        max_collision = getattr(args, 'max_collision', 100)
        all_item_codes, code_to_item, item_to_code, collision_stats = \
            build_code_with_collision_index(
                model.rqvae,
                item_embeddings,
                use_code_offset=train_dataset.use_code_offset,
                codebook_size=train_dataset.codebook_size,
                device=device,
                max_collision=max_collision,
            )
        train_dataset.item_to_code = item_to_code
        train_dataset.all_item_codes = torch.tensor(all_item_codes).to(device)
        print(f"Rebuilt mapping: {len(code_to_item)} unique codes")
        print(f"Collision rate: {collision_stats['collision_rate']:.4f}")
    else:
        print(f"Initial evaluation complete. Start training (total {args.num_epochs} epochs)")
        print("=" * 70)
    
    # ===========================
    # Joint training stage (runs only when not in finetune_only mode)
    # ===========================
    if not finetune_only:
        # Joint training initialization
        global_step = 0
        best_metric = float('inf') if args.early_stop_metric == 'valid_loss' else -float('inf')
        best_model_path = None
        patience_counter = 0
        patience = args.patience  # early-stop patience (valid_loss)
        
        stop_training = False
        epoch = 0
        while epoch < args.num_epochs and not stop_training:
            model.train()

            # Keep distillation loss weights unchanged (even if RQVAE is frozen, distillation still helps T5)
            # Two distillation loss weights (support warm_epoch)
            # Do not enable distillation losses before warm_epoch; enable after warm_epoch
            warm_epoch = getattr(args, 'warm_epoch', 10)
            if epoch < warm_epoch:
                # Do not use these two losses for the first warm_epoch epochs
                effective_gamma = 0.0  # kl_loss weight
                effective_delta = 0.0  # dec_cl_loss weight
                if epoch == 0:
                    print(f"Warm-up: for the first {warm_epoch} epochs, CDT Loss and CDR Loss are disabled")
            else:
                effective_gamma = args.gamma  # kl_loss weight
                effective_delta = args.delta  # dec_cl_loss weight
                if epoch == warm_epoch:
                    print(f"Warm-up finished: enable CDT Loss (gamma={effective_gamma}) and CDR Loss (delta={effective_delta}) from epoch {warm_epoch+1}")
            # ======================================
            
            print(f"\n[Epoch {epoch+1}] Joint training")

            total_loss = 0
            total_rec_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            total_dec_cl_loss = 0
            total_diversity_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # Get batch data
                history_embs = batch['history_embs'].to(device, non_blocking=True)  # (B, seq_len, emb_dim)
                target_embs = batch['target_embs'].to(device, non_blocking=True)    # (B, emb_dim)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)  # (B, seq_len)
                
                # Lookup target codes from mapping (4-token codes)
                target_item_ids = batch.get('target_item_id', None)  # (B,) - raw target item IDs
                if target_item_ids is None:
                    raise ValueError("Batch must contain target_item_id")

                # Lookup via mapping (4-token codes)
                if not hasattr(train_dataset, 'item_to_code') or train_dataset.item_to_code is None:
                    raise ValueError("Training dataset must provide item_to_code mapping (4-token codes)")

                # Ensure target_item_ids is a numpy array
                if isinstance(target_item_ids, torch.Tensor):
                    target_item_ids_np = target_item_ids.cpu().numpy()
                else:
                    target_item_ids_np = np.array(target_item_ids)

                decoder_input_ids, labels = generate_labels_from_mapping(
                    target_item_ids_np, train_dataset.item_to_code,
                    use_code_offset=train_dataset.use_code_offset,
                    codebook_size=train_dataset.codebook_size,
                    device=device
                )
                # decoder_input_ids: (B, 5) = [0, emb_id0, emb_id1, emb_id2, emb_id_collision]
                # labels: (B, 5) = [emb_id0, emb_id1, emb_id2, emb_id_collision, 0]
                
                # Forward
                decoder_input_ids_for_loss = decoder_input_ids[:, :-1].contiguous()  # (B, 4) - [0, emb_id0, emb_id1, emb_id2]
                labels_for_loss = labels[:, :-1].contiguous()  # (B, 4) - [emb_id0, emb_id1, emb_id2, emb_id_collision]
                
                # Prepare target_item_ids
                target_item_ids_for_loss = None
                if target_item_ids is not None:
                    if isinstance(target_item_ids, torch.Tensor):
                        target_item_ids_for_loss = target_item_ids
                    else:
                        target_item_ids_for_loss = torch.tensor(target_item_ids, dtype=torch.long, device=device)
                    
                total_loss_val, rec_loss_val, recon_loss_val, kl_loss_val, dec_cl_loss_val, diversity_loss_val = model(
                    history_embs=history_embs,
                    target_embs=target_embs,
                    decoder_input_ids=decoder_input_ids_for_loss,
                    attention_mask=attention_mask,
                    labels=labels_for_loss,
                    tau=args.tau,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=effective_gamma,  # kl_loss weight
                    delta=effective_delta,  # dec_cl_loss weight
                    target_item_ids=target_item_ids_for_loss,
                    diversity_weight=args.diversity_weight,
                )
              
                # Backward
                optimizer.zero_grad()
                total_loss_val.backward()
                
                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                # For transformers schedulers (cosine/linear/polynomial, etc.), step the scheduler each update
                if scheduler is not None:
                    scheduler.step()

                # Logging
                total_loss += total_loss_val.item()
                total_rec_loss += rec_loss_val.item()
                total_recon_loss += recon_loss_val.item()
                total_kl_loss += kl_loss_val.item()
                total_dec_cl_loss += dec_cl_loss_val.item()
                total_diversity_loss += diversity_loss_val.item()
                
                postfix_dict = {
                    'total': f'{total_loss_val.item():.4f}',
                    'rec': f'{rec_loss_val.item():.4f}',
                    'recon': f'{recon_loss_val.item():.4f}',
                    'diversity': f'{diversity_loss_val.item():.4f}'
                }
                if effective_gamma > 0:
                    postfix_dict['cdt'] = f'{kl_loss_val.item():.4f}'
                if effective_delta > 0:
                    postfix_dict['cdr'] = f'{dec_cl_loss_val.item():.4f}'
                pbar.set_postfix(postfix_dict)
                
                # SwanLab logging (per step)
                if run:
                    log_dict = {
                        'train/total_loss': total_loss_val.item(),
                        'train/rec_loss': rec_loss_val.item(),
                        'train/recon_loss': recon_loss_val.item(),
                        'train/diversity_loss': diversity_loss_val.item(),
                        'epoch': epoch + 1,
                    }
                    if effective_gamma > 0:
                        log_dict['train/cdt_loss'] = kl_loss_val.item()
                    if effective_delta > 0:
                        log_dict['train/cdr_loss'] = dec_cl_loss_val.item()
                    run.log(log_dict, step=global_step)
                
                global_step += 1
            
            # Epoch summary
            avg_total_loss = total_loss / len(train_loader)
            avg_rec_loss = total_rec_loss / len(train_loader)
            avg_recon_loss = total_recon_loss / len(train_loader)
            avg_kl_loss = total_kl_loss / len(train_loader)
            avg_dec_cl_loss = total_dec_cl_loss / len(train_loader)
            avg_diversity_loss = total_diversity_loss / len(train_loader)
            
            print(f"\n" + "=" * 70)
            print(f"Epoch {epoch+1}/{args.num_epochs} finished")
            print(f"  Total Loss: {avg_total_loss:.4f}")
            print(f"  Rec Loss: {avg_rec_loss:.4f}")
            print(f"  Recon Loss: {avg_recon_loss:.4f}")
            print(f"  Diversity Loss: {avg_diversity_loss:.6f}")
            if effective_gamma > 0:
                print(f"  CDT Loss: {avg_kl_loss:.4f}")
            if effective_delta > 0:
                print(f"  CDR Loss: {avg_dec_cl_loss:.4f}")
            
            # After each epoch (before validation), rebuild mapping using the latest RQVAE (4-token codes)
            # Note: mapping is built over the full dataset and shared by validation and test
            print("\n  [Eval] Rebuilding code mapping using current RQVAE (with collision index, after epoch)...")
            max_collision = getattr(args, 'max_collision', 100)
            all_item_codes, code_to_item, item_to_code, collision_stats = \
                build_code_with_collision_index(
                    model.rqvae,
                    item_embeddings,
                    use_code_offset=train_dataset.use_code_offset,
                    codebook_size=train_dataset.codebook_size,
                    device=device,
                    max_collision=max_collision,
                )
            # Update mapping in training dataset
            train_dataset.item_to_code = item_to_code
            train_dataset.all_item_codes = torch.tensor(all_item_codes).to(device)
            print(f"Rebuilt mapping: {len(code_to_item)} unique codes (shared by validation and test)")
            print(f"  Max collisions: {collision_stats['max_collision']}")
            print(f"  Collision rate: {collision_stats['collision_rate']:.4f}")
            
            # Evaluate on validation set (compute loss for early stopping)
            print("\n  Evaluating on validation set...")
            valid_recalls, valid_ndcgs, valid_loss, valid_rec_loss, valid_recon_loss, valid_kl_loss, valid_dec_cl_loss, valid_diversity_loss = evaluate(
                model, valid_loader, code_to_item, device,
                topk_list=args.topk_list, beam_size=args.beam_size, tau=args.tau,
                compute_loss=True, alpha=args.alpha, beta=args.beta, gamma=effective_gamma, delta=effective_delta,
                diversity_weight=args.diversity_weight,
                item_embeddings=item_embeddings,
                use_code_offset=train_dataset.use_code_offset,
                codebook_size=train_dataset.codebook_size,
                rebuild_mapping=False,  # use the rebuilt mapping above (shared by validation and test)
                max_collision=getattr(args, 'max_collision', 100),
                item_to_code=train_dataset.item_to_code
            )
            print("  Validation results:")
            print(f"    Total Loss: {valid_loss:.4f}")
            print(f"    Rec Loss: {valid_rec_loss:.4f}")
            print(f"    Recon Loss: {valid_recon_loss:.4f}")
            print(f"    Diversity Loss: {valid_diversity_loss:.6f}")
            if effective_gamma > 0:
                print(f"    CDT Loss: {valid_kl_loss:.4f}")
            if effective_delta > 0:
                print(f"    CDR Loss: {valid_dec_cl_loss:.4f}")
            for k in args.topk_list:
                print(f"    Recall@{k}: {valid_recalls[f'Recall@{k}']:.4f}  NDCG@{k}: {valid_ndcgs[f'NDCG@{k}']:.4f}")
    
            # ===== 早停与保存：可配置指标（valid_loss / ndcg@10） =====
            if args.early_stop_metric == 'valid_loss':
                current_metric = valid_loss
                better = current_metric < best_metric
                metric_desc = f"Valid Loss: {current_metric:.4f}"
            else:
                current_metric = valid_ndcgs['NDCG@10']
                better = current_metric > best_metric
                metric_desc = f"NDCG@10: {current_metric:.4f}"

            if better:
                prev_best = best_metric
                best_metric = current_metric
                patience_counter = 0

                prev_best_str = f"{prev_best:.4f}" if prev_best not in (float('inf'), -float('inf')) else str(prev_best)
                print(f"\n  [Early Stop] Metric improved ({args.early_stop_metric}): {prev_best_str} -> {best_metric:.4f}")
                print(f"              Current {metric_desc}")

                # Save best model immediately
                save_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_total_loss,
                    'valid_loss': valid_loss,
                    'valid_ndcg@10': valid_ndcgs['NDCG@10'],
                    'valid_ndcg@5': valid_ndcgs['NDCG@5'],
                    'valid_recall@5': valid_recalls['Recall@5'],
                    'valid_recall@10': valid_recalls['Recall@10'],
                    'best_metric': best_metric,
                    'early_stop_metric': args.early_stop_metric,
                    'rqvae_lr': args.rqvae_lr,
                    't5_lr': args.t5_lr,
                }, save_path)
                best_model_path = save_path
                print(f"\n  !!! Saved best model! ({args.early_stop_metric}={best_metric:.4f})")
                print(f"     Valid: Loss={valid_loss:.4f}, NDCG@10={valid_ndcgs['NDCG@10']:.4f}, Recall@10={valid_recalls['Recall@10']:.4f}")
                print(f"     -> {save_path}")

                if args.use_swanlab and run is not None:
                    run.log({
                        'best/metric': best_metric,
                        'best/valid_loss': valid_loss,
                        'best/valid_ndcg@5': valid_ndcgs['NDCG@5'],
                        'best/valid_ndcg@10': valid_ndcgs['NDCG@10'],
                        'best/valid_recall@5': valid_recalls['Recall@5'],
                        'best/valid_recall@10': valid_recalls['Recall@10'],
                        'best/epoch_saved': epoch + 1,
                    }, step=global_step)
            else:
                patience_counter += 1
                print(f"\n  [Early Stop] No improvement ({args.early_stop_metric}): current={current_metric:.4f}, best={best_metric:.4f}")
                print(f"              Patience {patience_counter}/{patience}")
                
                # Log evaluation to SwanLab
                if run:
                    metrics_log = {
                        'epoch': epoch + 1,
                        'train/avg_total_loss': avg_total_loss,
                        'train/avg_rec_loss': avg_rec_loss,
                        'train/avg_recon_loss': avg_recon_loss,
                        'train/avg_diversity_loss': avg_diversity_loss,
                        'train/rqvae_lr': optimizer.param_groups[0]['lr'],
                        'train/effective_gamma': effective_gamma,
                        'train/effective_delta': effective_delta,
                        # Validation losses/metrics
                        'val/total_loss': valid_loss,
                        'val/rec_loss': valid_rec_loss,
                        'val/recon_loss': valid_recon_loss,
                        'val/diversity_loss': valid_diversity_loss,
                    }
                    if effective_gamma > 0:
                        metrics_log['train/avg_cdt_loss'] = avg_kl_loss
                        metrics_log['val/cdt_loss'] = valid_kl_loss
                    if effective_delta > 0:
                        metrics_log['train/avg_cdr_loss'] = avg_dec_cl_loss
                        metrics_log['val/cdr_loss'] = valid_dec_cl_loss
                    # Validation performance (always log)
                    for k in args.topk_list:
                        metrics_log[f'val/recall@{k}'] = valid_recalls[f'Recall@{k}']
                        metrics_log[f'val/ndcg@{k}'] = valid_ndcgs[f'NDCG@{k}']
                    run.log(metrics_log, step=global_step)
            
                # Compute code change rate
                if (epoch + 1) % args.check_code_interval == 0:
                    print("  Checking code change rate...")
                    current_codes = []
                    with torch.no_grad():
                        N = len(item_embeddings)
                        bs = max(1, args.init_code_batch_size)
                        for i in tqdm(range(0, N, bs), desc="  Current encoding", leave=False):
                            batch = item_embeddings[i:i+bs].to(device)
                            codes = model.rqvae.get_indices(batch, use_sk=False)
                            current_codes.append(codes.cpu())
                    current_codes = torch.cat(current_codes, dim=0)
                    
                    change_rate = compute_code_change_rate(initial_codes, current_codes)
                    print(f"  Code change rate: {change_rate:.4%}")
                    
                    if run:
                        run.log({'train/code_change_rate': change_rate}, step=global_step)
            
                    # Codebook quality monitoring (each epoch, SwanLab only)
                    if run:
                        usage_rates, usage_counts = calculate_codebook_usage(model.rqvae, item_embeddings, device)
                        collision_rate, unique_codes, total_codes = calculate_collision_rate(model.rqvae, item_embeddings, device)
                        avg_entropies = calculate_soft_entropy(model.rqvae, item_embeddings, args.tau, device)

                        codebook_metrics = {}
                        for layer_idx in range(len(usage_rates)):
                            codebook_metrics[f'codebook/usage_rate_layer{layer_idx}'] = usage_rates[layer_idx]
                        codebook_metrics['codebook/collision_rate'] = collision_rate
                        codebook_metrics['codebook/unique_codes'] = unique_codes
                        for layer_idx in range(len(avg_entropies)):
                            codebook_metrics[f'codebook/entropy_layer{layer_idx}'] = avg_entropies[layer_idx]
                        run.log(codebook_metrics, step=global_step)

                # Early stopping based on valid_loss
                if patience_counter >= patience:
                    print(f"\nEarly stopping! Validation loss did not decrease for {patience} consecutive epochs")
                    stop_training = True
                    
                    # Periodically save checkpoints
                    if (epoch + 1) % args.save_interval == 0:
                        save_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_total_loss,
                        }, save_path)
                        print(f"  Saved checkpoint to {save_path}")
            
            # Log learning rate to SwanLab (each epoch)
            if run:
                current_t5_lr = optimizer.param_groups[1]['lr']  # T5 LR
                current_rqvae_lr = optimizer.param_groups[0]['lr']  # RQVAE LR
                run.log({
                    'train/t5_lr': current_t5_lr,
                    'train/rqvae_lr': current_rqvae_lr,
                }, step=global_step)
            
            epoch += 1
        
    # Final save (only when not finetune_only)
    if not finetune_only:
        final_path = os.path.join(args.save_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'rqvae_config': rqvae_config,
            't5_config': t5_config.to_dict(),
        }, final_path)
        print(f"\nEnd-to-end training completed. Final model saved to {final_path}")
    
    # After training, evaluate the best model on test set and log to SwanLab
    if best_model_path is not None and os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        ckpt_epoch = int(ckpt.get('epoch', 0))
        effective_gamma_test = float(args.gamma) if ckpt_epoch >= int(getattr(args, 'warm_epoch', 10)) else 0.0
        effective_delta_test = float(args.delta) if ckpt_epoch >= int(getattr(args, 'warm_epoch', 10)) else 0.0
        # Rebuild mapping (4-token codes)
        all_item_codes, code_to_item, item_to_code, collision_stats = build_code_with_collision_index(
            model.rqvae,
            item_embeddings,
            use_code_offset=train_dataset.use_code_offset,
            codebook_size=train_dataset.codebook_size,
            device=device,
            max_collision=getattr(args, 'max_collision', 100),
        )

        test_recalls, test_ndcgs, test_loss, test_rec_loss, test_recon_loss, test_kl_loss, test_dec_cl_loss, test_diversity_loss = evaluate(
            model, test_loader, code_to_item, device,
            topk_list=args.topk_list, beam_size=args.beam_size, tau=args.tau,
            compute_loss=True, alpha=args.alpha, beta=args.beta, gamma=effective_gamma_test, delta=effective_delta_test,
            diversity_weight=args.diversity_weight,
            item_embeddings=item_embeddings,
            use_code_offset=train_dataset.use_code_offset,
            codebook_size=train_dataset.codebook_size,
            rebuild_mapping=False,  # use rebuilt mapping above
            max_collision=getattr(args, 'max_collision', 100),
            item_to_code=item_to_code  # use rebuilt item_to_code (aligned with validation)
        )
        print("\nBest model performance on test set:")
        print(f"  Total Loss: {test_loss:.4f}")
        print(f"  Rec Loss: {test_rec_loss:.4f}")
        print(f"  Recon Loss: {test_recon_loss:.4f}")
        print(f"  Diversity Loss: {test_diversity_loss:.6f}")
        if effective_gamma_test > 0:
            print(f"  CDT Loss: {test_kl_loss:.4f}")
        if effective_delta_test > 0:
            print(f"  CDR Loss: {test_dec_cl_loss:.4f}")
        for k in args.topk_list:
            print(f"  Recall@{k}: {test_recalls[f'Recall@{k}']:.4f}  NDCG@{k}: {test_ndcgs[f'NDCG@{k}']:.4f}")

        if run:
            best_test_log = {
                'best/test_total_loss': test_loss,
                'best/test_rec_loss': test_rec_loss,
                'best/test_recon_loss': test_recon_loss,
                'best/test_diversity_loss': test_diversity_loss,
                'best/epoch_saved': ckpt.get('epoch', 0) + 1,
            }
            if effective_gamma_test > 0:
                best_test_log['best/test_cdt_loss'] = test_kl_loss
            if effective_delta_test > 0:
                best_test_log['best/test_cdr_loss'] = test_dec_cl_loss
            for k in args.topk_list:
                best_test_log[f'best/test_recall@{k}'] = test_recalls[f'Recall@{k}']
                best_test_log[f'best/test_ndcg@{k}'] = test_ndcgs[f'NDCG@{k}']
            run.log(best_test_log, step=global_step)
    else:
        print("\nBest-model checkpoint not found; skipping test-set evaluation")
    
    # =========================================================================
    # Finetune stage: train T5 only, freeze RQVAE
    # =========================================================================
    if args.enable_finetune:
        print("\n" + "=" * 70)
        print("Starting finetune stage (train T5 only, freeze RQVAE)")
        print("=" * 70)
        
        # Load best model
        # In finetune-only mode, the model has already been loaded
        if not finetune_only:
            if best_model_path is not None and os.path.exists(best_model_path):
                print(f"\n[Finetune] Loading best model: {best_model_path}")
                ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                model.to(device)
            else:
                print("\nBest-model checkpoint not found; finetuning with current model state")
        else:
            print("\n[Finetune] Finetune-only mode: using the already loaded joint-training model")
        
        # Freeze RQVAE
        for param in model.rqvae.parameters():
            param.requires_grad = False
        # Train T5 parameters only
        for param in model.t5.parameters():
            param.requires_grad = True
        print("  RQVAE frozen; training T5 only")
        
        # Create a new optimizer
        t5_params = list(model.t5.parameters())
        finetune_lr = getattr(args, 'finetune_lr', 5e-4)
        # Use AdamW
        finetune_optimizer = optim.AdamW(
            t5_params, 
            lr=finetune_lr, 
            weight_decay=args.weight_decay
        )
        print(f"  Finetune optimizer: T5 LR={finetune_lr:.2e}, Weight Decay={args.weight_decay:.2e}")
        
        # Compute training steps
        finetune_epochs = getattr(args, 'finetune_epochs', 100)
        train_steps_per_epoch = len(train_loader)
        total_train_steps = finetune_epochs * train_steps_per_epoch
        # Finetune warmup ratio (decoupled from joint training)
        finetune_warmup_ratio = getattr(args, 'finetune_warmup_ratio', 0.0)
        num_warmup_steps = int(total_train_steps * finetune_warmup_ratio)
        
        # Finetune scheduler (only cosine_with_warmup / constant_with_warmup / none; and legacy constant)
        finetune_scheduler_type = getattr(args, 'finetune_scheduler_type', 'cosine_with_warmup')
        finetune_scheduler = None
        if finetune_scheduler_type == 'constant':
            finetune_scheduler = get_transformers_scheduler(
                name='constant',
                optimizer=finetune_optimizer,
                num_warmup_steps=num_warmup_steps,
            )
            print(f"  Finetune scheduler: Constant (transformers), num_warmup_steps={num_warmup_steps}")
        elif finetune_scheduler_type in ['cosine_with_warmup', 'constant_with_warmup']:
            finetune_scheduler_name_to_transformers = {
                'cosine_with_warmup': 'cosine',
                'constant_with_warmup': 'constant_with_warmup',
            }
            finetune_scheduler = get_transformers_scheduler(
                name=finetune_scheduler_name_to_transformers[finetune_scheduler_type],
                optimizer=finetune_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_train_steps,
            )
            scheduler_name_map = {
                'cosine_with_warmup': 'Cosine with Warmup',
                'constant_with_warmup': 'Constant with Warmup',
            }
            print(f"  Finetune scheduler: {scheduler_name_map.get(finetune_scheduler_type, finetune_scheduler_type)} (transformers), "
                  f"num_training_steps={total_train_steps}, num_warmup_steps={num_warmup_steps} "
                  f"(≈ 前 {finetune_warmup_ratio*100:.1f}% 的 step)")
        else:
            finetune_scheduler = None
            print("  Finetune scheduler: None")
        
        # In finetune-only mode, mapping has already been rebuilt
        if finetune_only:
            # Use rebuilt mapping
            finetune_code_to_item = code_to_item
            finetune_item_to_code = item_to_code
            print("\n[Finetune] Finetune-only mode: using rebuilt code mapping")
            print(f"  Mapping: {len(finetune_code_to_item)} unique codes")
        else:
            print("\n[Finetune] Rebuilding code mapping...")
            all_item_codes, finetune_code_to_item, finetune_item_to_code, finetune_collision_stats = \
                build_code_with_collision_index(
                    model.rqvae,
                    item_embeddings,
                    use_code_offset=train_dataset.use_code_offset,
                    codebook_size=train_dataset.codebook_size,
                    device=device,
                    max_collision=getattr(args, 'max_collision', 100),
                )
            # Update mapping in training dataset
            train_dataset.item_to_code = finetune_item_to_code
            train_dataset.all_item_codes = torch.tensor(all_item_codes).to(device)
            print(f"  Rebuilt mapping: {len(finetune_code_to_item)} unique codes")
            print(f"  Collision rate: {finetune_collision_stats['collision_rate']:.4f}")
        
        # Finetune early-stop settings
        finetune_patience = getattr(args, 'finetune_patience', 10)
        finetune_eval_step = 1  # evaluate every epoch
        finetune_best_metric = float('inf') if args.finetune_early_stop_metric == 'valid_loss' else -float('inf')
        finetune_patience_counter = 0
        finetune_best_model_path = None
        finetune_global_step = global_step  # continue using global step counter
        finetune_stop = False

        effective_gamma_finetune = 0.0
        effective_delta_finetune = 0.0
        
        print(f"\n[Finetune] Start training (epochs={finetune_epochs}, patience={finetune_patience}, metric=valid_loss, eval every epoch)...")
        
        for finetune_epoch in range(finetune_epochs):
            if finetune_stop:
                break
            model.train()

            # No warmup epochs during finetuning
            # warm_epoch = 0
            # Do not use kl_loss or dec_cl_loss during finetuning
            # effective_gamma_finetune = float(args.gamma) if finetune_epoch >= int(warm_epoch) else 0.0
            # effective_delta_finetune = float(args.delta) if finetune_epoch >= int(warm_epoch) else 0.0
            effective_gamma_finetune = 0.0
            effective_delta_finetune = 0.0
            
            training_start_time = time.time()
            total_loss = 0
            total_rec_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            total_dec_cl_loss = 0
            total_diversity_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Finetune Epoch {finetune_epoch + 1}/{finetune_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # Get batch data
                history_embs = batch['history_embs'].to(device, non_blocking=True)
                target_embs = batch['target_embs'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                
                # Lookup target codes from mapping
                target_item_ids = batch.get('target_item_id', None)
                if target_item_ids is None:
                    raise ValueError("Batch must contain target_item_id")

                if isinstance(target_item_ids, torch.Tensor):
                    target_item_ids_np = target_item_ids.cpu().numpy()
                elif isinstance(target_item_ids, list):
                    target_item_ids_np = np.array(target_item_ids)
                else:
                    target_item_ids_np = np.array(target_item_ids)

                decoder_input_ids, labels = generate_labels_from_mapping(
                    target_item_ids_np, train_dataset.item_to_code,
                    use_code_offset=train_dataset.use_code_offset,
                    codebook_size=train_dataset.codebook_size,
                    device=device
                )
                
                # Forward
                decoder_input_ids_for_loss = decoder_input_ids[:, :-1].contiguous()
                labels_for_loss = labels[:, :-1].contiguous()
                
                # Prepare target_item_ids
                target_item_ids_for_loss = None
                if target_item_ids is not None:
                    if isinstance(target_item_ids, torch.Tensor):
                        target_item_ids_for_loss = target_item_ids
                    else:
                        target_item_ids_for_loss = torch.tensor(target_item_ids, dtype=torch.long, device=device)
                
                total_loss_val, rec_loss_val, recon_loss_val, kl_loss_val, dec_cl_loss_val, diversity_loss_val = model(
                    history_embs=history_embs,
                    target_embs=target_embs,
                    decoder_input_ids=decoder_input_ids_for_loss,
                    attention_mask=attention_mask,
                    labels=labels_for_loss,
                    tau=args.tau,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=effective_gamma_finetune,
                    delta=effective_delta_finetune,
                    target_item_ids=target_item_ids_for_loss,
                    diversity_weight=0.0,
                )
                
                # Backward
                finetune_optimizer.zero_grad()
                total_loss_val.backward()
                
                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.t5.parameters(), args.grad_clip)
                
                finetune_optimizer.step()
                # Update LR scheduler (per step)
                if finetune_scheduler is not None:
                    finetune_scheduler.step()
                
                # Logging
                total_loss += total_loss_val.item()
                total_rec_loss += rec_loss_val.item()
                total_recon_loss += recon_loss_val.item()
                total_kl_loss += kl_loss_val.item()
                total_dec_cl_loss += dec_cl_loss_val.item()
                total_diversity_loss += diversity_loss_val.item()
                
                postfix_dict = {
                    'total': f'{total_loss_val.item():.4f}',
                    'rec': f'{rec_loss_val.item():.4f}',
                    'recon': f'{recon_loss_val.item():.4f}',
                }
                if effective_gamma_finetune > 0:
                    postfix_dict['cdt'] = f'{kl_loss_val.item():.4f}'
                if effective_delta_finetune > 0:
                    postfix_dict['cdr'] = f'{dec_cl_loss_val.item():.4f}'
                pbar.set_postfix(postfix_dict)
                
                # SwanLab logging (per step, consistent with joint training)
                if run:
                    log_dict = {
                        'finetune_train/total_loss': total_loss_val.item(),
                        'finetune_train/rec_loss': rec_loss_val.item(),
                        'finetune_train/recon_loss': recon_loss_val.item(),
                        'finetune_train/diversity_loss': diversity_loss_val.item(),
                        'finetune_train/epoch': finetune_epoch + 1,
                    }
                    if effective_gamma_finetune > 0:
                        log_dict['finetune_train/cdt_loss'] = kl_loss_val.item()
                    if effective_delta_finetune > 0:
                        log_dict['finetune_train/cdr_loss'] = dec_cl_loss_val.item()
                    run.log(log_dict, step=finetune_global_step)
                
                finetune_global_step += 1
            
            training_end_time = time.time()
            avg_total_loss = total_loss / len(train_loader)
            avg_rec_loss = total_rec_loss / len(train_loader)
            avg_recon_loss = total_recon_loss / len(train_loader)
            avg_kl_loss = total_kl_loss / len(train_loader)
            avg_dec_cl_loss = total_dec_cl_loss / len(train_loader)
            avg_diversity_loss = total_diversity_loss / len(train_loader)
            
            # Get current LR (if scheduler is None, read from optimizer)
            if finetune_scheduler is not None:
                current_lr = finetune_scheduler.get_last_lr()[0] if isinstance(finetune_scheduler.get_last_lr(), list) else finetune_scheduler.get_last_lr()
            else:
                current_lr = finetune_optimizer.param_groups[0]['lr']
            print(f"\n[Finetune Epoch {finetune_epoch+1}/{finetune_epochs}]")
            print(f"  Total Loss: {avg_total_loss:.4f} (train time: {training_end_time - training_start_time:.2f}s)")
            print(f"  Rec Loss: {avg_rec_loss:.4f}")
            print(f"  Recon Loss: {avg_recon_loss:.4f}")
            print(f"  Diversity Loss: {avg_diversity_loss:.6f}")
            if effective_gamma_finetune > 0:
                print(f"  CDT Loss: {avg_kl_loss:.4f}")
            if effective_delta_finetune > 0:
                print(f"  CDR Loss: {avg_dec_cl_loss:.4f}")
            print(f"  T5 LR: {current_lr:.2e}")
            
            # SwanLab: log training losses
            if run:
                metrics_log = {
                    'finetune_train/epoch': finetune_epoch + 1,
                    'finetune_train/avg_total_loss': avg_total_loss,
                    'finetune_train/avg_rec_loss': avg_rec_loss,
                    'finetune_train/avg_recon_loss': avg_recon_loss,
                    'finetune_train/avg_diversity_loss': avg_diversity_loss,
                    'finetune_train/t5_lr': current_lr,
                }
                if effective_gamma_finetune > 0:
                    metrics_log['finetune_train/avg_cdt_loss'] = avg_kl_loss
                if effective_delta_finetune > 0:
                    metrics_log['finetune_train/avg_cdr_loss'] = avg_dec_cl_loss
                run.log(metrics_log, step=finetune_global_step)
            
            # Evaluation
            if (finetune_epoch + 1) % finetune_eval_step == 0:
                print("\n  Evaluating on validation set...")
                valid_recalls, valid_ndcgs, valid_loss, valid_rec_loss, valid_recon_loss, valid_kl_loss, valid_dec_cl_loss, valid_diversity_loss = evaluate(
                    model, valid_loader, finetune_code_to_item, device,
                    topk_list=args.topk_list, beam_size=args.beam_size, tau=args.tau,
                    compute_loss=True, alpha=0.0, beta=args.beta, gamma=effective_gamma_finetune, delta=effective_delta_finetune,
                    diversity_weight=0.0,
                    item_embeddings=item_embeddings,
                    use_code_offset=train_dataset.use_code_offset,
                    codebook_size=train_dataset.codebook_size,
                    rebuild_mapping=False,
                    max_collision=getattr(args, 'max_collision', 100),
                    item_to_code=finetune_item_to_code
                )
                print(f"  [Epoch {finetune_epoch+1}] Val Results:")
                print(f"    Total Loss: {valid_loss:.4f}")
                print(f"    Rec Loss: {valid_rec_loss:.4f}")
                print(f"    Recon Loss: {valid_recon_loss:.4f}")
                print(f"    Diversity Loss: {valid_diversity_loss:.6f}")
                if effective_gamma_finetune > 0:
                    print(f"    CDT Loss: {valid_kl_loss:.4f}")
                if effective_delta_finetune > 0:
                    print(f"    CDR Loss: {valid_dec_cl_loss:.4f}")
                for k in args.topk_list:
                    print(f"    Recall@{k}: {valid_recalls[f'Recall@{k}']:.4f}  NDCG@{k}: {valid_ndcgs[f'NDCG@{k}']:.4f}")
                
                # SwanLab logging: validation metrics (consistent with joint training)
                if run:
                    finetune_val_log = {
                        'finetune_val/total_loss': valid_loss,
                        'finetune_val/rec_loss': valid_rec_loss,
                        'finetune_val/recon_loss': valid_recon_loss,
                        'finetune_val/diversity_loss': valid_diversity_loss,
                    }
                    if effective_gamma_finetune > 0:
                        finetune_val_log['finetune_val/cdt_loss'] = valid_kl_loss
                    if effective_delta_finetune > 0:
                        finetune_val_log['finetune_val/cdr_loss'] = valid_dec_cl_loss
                    for k in args.topk_list:
                        finetune_val_log[f'finetune_val/recall@{k}'] = valid_recalls[f'Recall@{k}']
                        finetune_val_log[f'finetune_val/ndcg@{k}'] = valid_ndcgs[f'NDCG@{k}']
                    run.log(finetune_val_log, step=finetune_global_step)
                
                # Early-stop decision (valid_loss / ndcg@10)
                current_ndcg10 = valid_ndcgs['NDCG@10']
                
                if args.finetune_early_stop_metric == 'valid_loss':
                    current_metric = valid_loss
                    better = current_metric < finetune_best_metric
                    metric_desc = f"Valid Loss: {current_metric:.4f}"
                else:
                    current_metric = current_ndcg10
                    better = current_metric > finetune_best_metric
                    metric_desc = f"NDCG@10: {current_metric:.4f}"
                
                if better:
                    prev_best = finetune_best_metric
                    finetune_best_metric = current_metric
                    finetune_patience_counter = 0
                    prev_best_str = "inf" if prev_best == float('inf') else f"{prev_best:.4f}"
                    print(f"\n  [Finetune Early Stop] Metric improved ({args.finetune_early_stop_metric}): {prev_best_str} -> {finetune_best_metric:.4f}. Reset patience.")
                    
                    # Save best model
                    finetune_best_model_path = os.path.join(args.save_dir, "best_finetune_model.pth")
                    torch.save({
                        'epoch': finetune_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': finetune_optimizer.state_dict(),
                        'train_loss': avg_total_loss,
                        'valid_loss': valid_loss,
                        'valid_ndcg@10': current_ndcg10,
                        'valid_ndcg@5': valid_ndcgs['NDCG@5'],
                        'valid_recall@5': valid_recalls['Recall@5'],
                        'valid_recall@10': valid_recalls['Recall@10'],
                        'best_metric': finetune_best_metric,
                        'finetune_early_stop_metric': args.finetune_early_stop_metric,
                        'finetune_lr': finetune_lr,
                    }, finetune_best_model_path)
                    print(f"   Saved best finetune model! ({args.finetune_early_stop_metric}={finetune_best_metric:.4f}, NDCG@10={current_ndcg10:.4f})")
                    print(f"     Valid: Loss={valid_loss:.4f}, NDCG@10={valid_ndcgs['NDCG@10']:.4f}, Recall@10={valid_recalls['Recall@10']:.4f}")
                    print(f"     -> {finetune_best_model_path}")
                    
                    # Log best model metrics to SwanLab (consistent with joint training)
                    if run:
                        run.log({
                            'finetune_best/metric': finetune_best_metric,
                            'finetune_best/valid_loss': valid_loss,
                            'finetune_best/valid_ndcg@5': valid_ndcgs['NDCG@5'],
                            'finetune_best/valid_ndcg@10': current_ndcg10,
                            'finetune_best/valid_recall@5': valid_recalls['Recall@5'],
                            'finetune_best/valid_recall@10': valid_recalls['Recall@10'],
                            'finetune_best/epoch_saved': finetune_epoch + 1,
                        }, step=finetune_global_step)
                else:
                    finetune_patience_counter += 1
                    print(f"\n  [Finetune Early Stop] No improvement ({args.finetune_early_stop_metric}). Patience {finetune_patience_counter}/{finetune_patience}")
                
                # Early stopping
                if finetune_patience_counter >= finetune_patience:
                    finetune_stop = True
                    print(f"\nFinetune early stopping! No validation improvement for {finetune_patience} epochs")
        
        # After finetuning, evaluate the best model on the test set
        if finetune_best_model_path is not None and os.path.exists(finetune_best_model_path):
            print("\n[Finetune] Evaluating best model on test set...")
            ckpt = torch.load(finetune_best_model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device)
            
            test_recalls, test_ndcgs, test_loss, test_rec_loss, test_recon_loss, test_kl_loss, test_dec_cl_loss, test_diversity_loss = evaluate(
                model, test_loader, finetune_code_to_item, device,
                topk_list=args.topk_list, beam_size=args.beam_size, tau=args.tau,
                compute_loss=True, alpha=0.0, beta=args.beta, gamma=effective_gamma_finetune, delta=effective_delta_finetune,
                diversity_weight=0.0,
                item_embeddings=item_embeddings,
                use_code_offset=train_dataset.use_code_offset,
                codebook_size=train_dataset.codebook_size,
                rebuild_mapping=False,
                max_collision=getattr(args, 'max_collision', 100),
                item_to_code=finetune_item_to_code
            )
            print("\nBest finetune model performance on test set:")
            print(f"  Total Loss: {test_loss:.4f}")
            print(f"  Rec Loss: {test_rec_loss:.4f}")
            print(f"  Recon Loss: {test_recon_loss:.4f}")
            print(f"  Diversity Loss: {test_diversity_loss:.6f}")
            if effective_gamma_finetune > 0:
                print(f"  CDT Loss: {test_kl_loss:.4f}")
            if effective_delta_finetune > 0:
                print(f"  CDR Loss: {test_dec_cl_loss:.4f}")
            for k in args.topk_list:
                print(f"  Recall@{k}: {test_recalls[f'Recall@{k}']:.4f}  NDCG@{k}: {test_ndcgs[f'NDCG@{k}']:.4f}")

            if run:
                finetune_best_test_log = {
                    'finetune_best/test_total_loss': test_loss,
                    'finetune_best/test_rec_loss': test_rec_loss,
                    'finetune_best/test_recon_loss': test_recon_loss,
                    'finetune_best/test_diversity_loss': test_diversity_loss,
                }
                if effective_gamma_finetune > 0:
                    finetune_best_test_log['finetune_best/test_cdt_loss'] = test_kl_loss
                if effective_delta_finetune > 0:
                    finetune_best_test_log['finetune_best/test_cdr_loss'] = test_dec_cl_loss
                for k in args.topk_list:
                    finetune_best_test_log[f'finetune_best/test_recall@{k}'] = test_recalls[f'Recall@{k}']
                    finetune_best_test_log[f'finetune_best/test_ndcg@{k}'] = test_ndcgs[f'NDCG@{k}']
                run.log(finetune_best_test_log, step=finetune_global_step)
        else:
            print("\nBest finetune-model checkpoint not found; skipping test-set evaluation")
        
        print("\n" + "=" * 70)
        print("Finetune stage completed!")
        print("=" * 70)
    
    if run:
        run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end joint training')
    
    # Data paths
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--valid_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--rqvae_model_path', type=str, required=True)
    parser.add_argument('--item_emb_path', type=str, required=True)

    # Optional SASRec teacher (used by kl_loss/dec_cl_loss)
    parser.add_argument('--use_sasrec_teacher', action='store_true', default=False,
                       help='Enable SASRec teacher (requires sasrec_emb_path; columns: ItemID, sasrec_embedding)')
    parser.add_argument('--sasrec_emb_path', type=str, default=None,
                       help='Path to SASRec embedding parquet (columns: ItemID, sasrec_embedding)')
    # T5 initialization
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (for SwanLab; auto-generated by default)')
    
    # Model config
    parser.add_argument('--vocab_size', type=int, default=257,
                       help='T5 vocabulary size (auto-adjusted based on use_code_offset)')
    parser.add_argument('--use_code_offset', action='store_true',default=True,
                       help='Whether to use code offsets (each layer uses a different embedding range; vocab_size=769)')
    parser.add_argument('--max_collision', type=int, default=100,
                       help='Max collision index (0-100, total 101; used for code deduplication; default 100)')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--d_kv', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    
    # Training config
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--train_num_workers', type=int, default=8,
                        help='Number of workers for training DataLoader')
    parser.add_argument('--eval_num_workers', type=int, default=4,
                        help='Number of workers for validation/test DataLoader')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='Enable pinned memory for DataLoader')
    parser.add_argument('--pin_memory_device', type=str, default=None,
                        help='Target device for pinned memory (defaults to training device)')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='Prefetch factor per worker (only for num_workers>0)')
    parser.add_argument('--prefetch_to_gpu', action='store_true', default=True,
                        help='Move DataLoader output tensors to GPU before returning batches')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--t5_lr', type=float, default=1e-4)
    parser.add_argument('--rqvae_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Optimizer L2 weight decay (applies to both T5 and RQVAE params)')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--beam_size', type=int, default=30,
                       help='Beam size for validation/test generation')
    parser.add_argument('--topk_list', type=int, nargs='+', default=[5, 10, 20],
                       help='Top-K list for evaluation, e.g. --topk_list 5 10 20')
    
    # LR warmup / scheduler config
    parser.add_argument('--warmup_ratio', type=float, default=0.0,
                       help='Warmup ratio (fraction of total steps) during joint training (0 disables warmup)')
    parser.add_argument('--scheduler_type', type=str, default='cosine_with_warmup',
                       choices=['cosine_with_warmup', 'constant_with_warmup', 'none'],
                       help='LR scheduler type: cosine_with_warmup / constant_with_warmup / none')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Reconstruction loss weight')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Recommendation loss weight')
    parser.add_argument('--gamma', type=float, default=0.0,
                       help='CDT loss weight (tokenizer distillation; kl_loss in implementation; 0.0 disables)')
    parser.add_argument('--delta', type=float, default=0.0,
                       help='CDR loss weight (recommender distillation; dec_cl_loss in implementation; 0.0 disables)')
    parser.add_argument('--warm_epoch', type=int, default=10,
                       help='Warmup epochs for CDT/CDR (disable these losses for first N epochs; default 10)')
    parser.add_argument('--diversity_weight', type=float, default=0.0,
                       help='Additional diversity-loss weight added to total loss')
    
    # Monitoring
    parser.add_argument('--check_code_interval', type=int, default=5,
                       help='Check code change rate every N epochs')
    parser.add_argument('--max_code_change_rate', type=float, default=0.05,
                       help='Maximum allowed code change rate (reduce RQVAE LR if exceeded)')
    
    # Saving
    parser.add_argument('--save_dir', type=str, default='./ckpt/e2e')
    parser.add_argument('--save_interval', type=int, default=10)
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10,
                       help='Early-stop patience (stop after N epochs with no validation improvement)')
    parser.add_argument('--early_stop_metric', type=str, default='valid_loss',
                       choices=['valid_loss', 'ndcg@10'],
                       help='Metric for early stopping/saving best model in joint training: valid_loss or ndcg@10')
    parser.add_argument('--use_swanlab', action='store_true',
                       help='Log experiments to SwanLab')
    parser.add_argument('--init_code_batch_size', type=int, default=4096,
                       help='GPU batch size for initial codes and code-change-rate computation')
    parser.add_argument('--rqvae_temperature_mode', type=str, default='current',
                       choices=['current', 'initial'],
                       help='RQVAE temperature mode: current=prefer current_temperature (fallback to initial); initial=use initial temperature directly')
    
    # Finetune stage args
    parser.add_argument('--enable_finetune', action='store_true',
                       help='Enable finetune stage (train T5 only, freeze RQVAE)')
    parser.add_argument('--finetune_only', action='store_true',
                       help='Finetune-only mode: skip joint training and finetune from a specified joint checkpoint')
    parser.add_argument('--joint_checkpoint_path', type=str, default=None,
                       help='Path to joint-training checkpoint (required for finetune-only mode; points to best_model.pth)')
    parser.add_argument('--finetune_lr', type=float, default=5e-4,
                       help='T5 learning rate for finetune stage (default 5e-4)')
    parser.add_argument('--finetune_epochs', type=int, default=100,
                       help='Maximum epochs for finetune stage (default 100)')
    parser.add_argument('--finetune_patience', type=int, default=10,
                       help='Early-stop patience for finetune stage')
    parser.add_argument('--finetune_early_stop_metric', type=str, default='valid_loss',
                       choices=['valid_loss', 'ndcg@10'],
                       help='Metric for early stopping/saving best model in finetune: valid_loss or ndcg@10')
    parser.add_argument('--finetune_warmup_ratio', type=float, default=0.0,
                       help='Warmup ratio for finetune cosine scheduler (fraction of total steps)')
    parser.add_argument('--finetune_scheduler_type', type=str, default='cosine_with_warmup',
                       choices=['cosine_with_warmup', 'constant_with_warmup', 'constant', 'none'],
                       help='Finetune LR scheduler type: cosine_with_warmup / constant_with_warmup / constant / none')
    
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    train(args)
