import csv
import logging
import math
import os
from time import time

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from utils import ensure_dir, set_color


class Trainer(object):
    """Stage-1 trainer: soft RQ-VAE with temperature annealing and codeword uniformity."""

    def __init__(self, args, model, data_num, logger=None):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()
        self.run = logger

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = torch.device(args.device)

        self.ckpt_dir = args.ckpt_dir
        ensure_dir(self.ckpt_dir)
        self.metrics_csv_path = os.path.join(self.ckpt_dir, "metrics.csv")
        self._metrics_fieldnames = ["epoch", "phase", "temperature", "learning_rate", "train_loss",
                                    "train_recon_loss", "train_cu_loss", "eval_collision_rate", "epoch_time"]
        with open(self.metrics_csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._metrics_fieldnames).writeheader()

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)
        self.initial_temperature = args.temperature
        self.final_temperature = args.temperature_final

    def _log(self, values, step):
        if self.run is not None:
            self.run.log(values, step=step)

    def _append_metrics_row(self, row):
        with open(self.metrics_csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._metrics_fieldnames).writerow(
                {k: row.get(k) for k in self._metrics_fieldnames})

    def _build_optimizer(self):
        params = self.model.parameters()
        name = self.learner.lower()
        if name == "adamw":
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if name == "adam":
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if name == "sgd":
            return optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(f"unsupported optimizer: {self.learner}")

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            return get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps,
                                                   num_training_steps=self.max_steps)
        return get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps)

    def _update_temperature(self, epoch_idx):
        """Annealed Inference Alignment: tau(t) = tau_max - t / T * (tau_max - tau_min)."""
        schedule = self.args.temperature_schedule
        init_temp, final_temp = self.initial_temperature, self.final_temperature
        if schedule == "none":
            target = init_temp
        else:
            start = max(0, self.args.temperature_anneal_start_epoch)
            end = max(start, self.args.temperature_anneal_end_epoch)
            if end <= start:
                progress = 1.0 if epoch_idx >= end else 0.0
            else:
                progress = min(max((epoch_idx - start) / (end - start), 0.0), 1.0)
            if schedule == "linear":
                target = init_temp + (final_temp - init_temp) * progress
            else:  # cosine
                target = final_temp + (init_temp - final_temp) * 0.5 * (1 + math.cos(math.pi * progress))
        self.model.rq.set_temperature(target)
        return target

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()
        total_loss, total_recon_loss, total_cu, cu_steps = 0.0, 0.0, 0.0, 0
        for data in tqdm(train_data, total=len(train_data), ncols=100,
                         desc=set_color(f"Train {epoch_idx}", "pink")):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, _ = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            if torch.isnan(loss):
                raise ValueError("Training loss is nan")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            cu = self.model.rq.last_diversity_loss
            if cu is not None:
                total_cu += float(cu.item())
                cu_steps += 1
        return total_loss, total_recon_loss, total_cu / max(1, cu_steps)

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()
        codes, num_sample = set(), 0
        for data in valid_data:
            num_sample += len(data)
            indices = self.model.get_indices(data.to(self.device))
            for index in indices.view(-1, indices.shape[-1]).cpu().numpy():
                codes.add("-".join(str(int(c)) for c in index))
        return (num_sample - len(codes)) / num_sample

    def _save_checkpoint(self, epoch, ckpt_file, current_temperature):
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "current_temperature": current_temperature,
        }
        path = os.path.join(self.ckpt_dir, ckpt_file)
        torch.save(state, path, pickle_protocol=4)
        self.logger.info(set_color("Saving", "blue") + f": {path}")
        return path

    def fit(self, data):
        print(f"Training RQ-VAE for {self.epochs} epochs; checkpoints in {self.ckpt_dir}")
        for epoch_idx in range(self.epochs):
            temperature = self._update_temperature(epoch_idx)
            t0 = time()
            train_loss, train_recon_loss, train_cu = self._train_epoch(data, epoch_idx)
            epoch_time = time() - t0
            lr = self.scheduler.get_last_lr()[0]
            self.logger.info(f"epoch {epoch_idx} [time {epoch_time:.2f}s, loss {train_loss:.4f}, "
                             f"recon {train_recon_loss:.4f}, tau {temperature:.6f}]")
            self._log({"train/loss": train_loss, "train/recon_loss": train_recon_loss, "train/cu_loss": train_cu,
                       "train/learning_rate": lr, "train/temperature": temperature}, epoch_idx)
            self._append_metrics_row(dict(epoch=epoch_idx, phase="train", temperature=temperature,
                                          learning_rate=lr, train_loss=train_loss,
                                          train_recon_loss=train_recon_loss, train_cu_loss=train_cu,
                                          epoch_time=epoch_time))

            if (epoch_idx + 1) % self.eval_step == 0:
                collision_rate = self._valid_epoch(data)
                self._log({"eval/collision_rate": collision_rate}, epoch_idx)
                self._append_metrics_row(dict(epoch=epoch_idx, phase="eval", temperature=temperature,
                                              eval_collision_rate=collision_rate))
                self.logger.info(f"epoch {epoch_idx} collision rate {collision_rate:.6f}")
                # The tokenizer used for joint training is the checkpoint with the lowest loss.
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch_idx, self.best_loss_ckpt, temperature)
                    print(f"New best loss: {self.best_loss:.6f} (tau={temperature:.6f})")
                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    self._save_checkpoint(epoch_idx, self.best_collision_ckpt, temperature)
        return self.best_loss, self.best_collision_rate
