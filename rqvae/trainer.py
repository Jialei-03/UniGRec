import logging
import math

import numpy as np
import torch
from time import time
import csv
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import swanlab

from utils import ensure_dir,set_color,get_local_time,delete_file
import os

import heapq
class Trainer(object):

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.metrics_csv_path = os.path.join(self.ckpt_dir, "metrics.csv")
        self._metrics_fieldnames = [
            "epoch",
            "phase",
            "temperature",
            "learning_rate",
            "train_loss",
            "train_recon_loss",
            "train_diversity_loss",
            "train_diversity_loss_scaled",
            "eval_collision_rate",
            "epoch_time",
            "eval_time",
        ]
        with open(self.metrics_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._metrics_fieldnames)
            writer.writeheader()

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)
        self.initial_temperature = getattr(args, "temperature", None)
        self.final_temperature = getattr(args, "temperature_final", self.initial_temperature)

    def _append_metrics_row(self, row: dict):
        safe_row = {k: row.get(k, None) for k in self._metrics_fieldnames}
        with open(self.metrics_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._metrics_fieldnames)
            writer.writerow(safe_row)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _update_temperature(self, epoch_idx: int):
        schedule = getattr(self.args, "temperature_schedule", "none")
        if self.initial_temperature is None or schedule == "none":
            target_temperature = self.initial_temperature
        else:
            start = max(0, getattr(self.args, "temperature_anneal_start_epoch", 0))
            end = max(start, getattr(self.args, "temperature_anneal_end_epoch", start))
            init_temp = self.initial_temperature
            final_temp = self.final_temperature if self.final_temperature is not None else init_temp

            if end <= start:
                progress = 1.0 if epoch_idx >= end else 0.0
            else:
                if epoch_idx < start:
                    progress = 0.0
                elif epoch_idx >= end:
                    progress = 1.0
                else:
                    progress = (epoch_idx - start) / (end - start)

            if schedule == "linear":
                target_temperature = init_temp + (final_temp - init_temp) * progress
            elif schedule == "cosine":
                target_temperature = final_temp + (init_temp - final_temp) * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                target_temperature = init_temp

        if hasattr(self.model, "rq") and target_temperature is not None:
            self.model.rq.set_temperature(target_temperature)
        return target_temperature


    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_diversity_loss = 0.0
        total_diversity_loss_scaled = 0.0
        diversity_steps = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            out, rq_loss, indices = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            # print(self.scheduler.get_last_lr())
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()

            div_loss = getattr(getattr(self.model, 'rq', None), 'last_diversity_loss', None)
            div_loss_scaled = getattr(getattr(self.model, 'rq', None), 'last_diversity_loss_scaled', None)
            if div_loss is not None:
                total_diversity_loss += float(div_loss.item())
                diversity_steps += 1
            if div_loss_scaled is not None:
                total_diversity_loss_scaled += float(div_loss_scaled.item())

        avg_div_loss = total_diversity_loss / max(1, diversity_steps)
        avg_div_loss_scaled = total_diversity_loss_scaled / max(1, diversity_steps)
        return total_loss, total_recon_loss, avg_div_loss, avg_div_loss_scaled

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )

        indices_set = set()
        num_sample = 0
        for batch_idx, data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set)))/num_sample

        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None, current_temperature=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        # Save current temperature (for analysis scripts)
        if current_temperature is not None:
            state["current_temperature"] = current_temperature
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

        return ckpt_path
    
    def _save_latest_checkpoint(self, is_best_collision=False, is_best_loss=False, current_temperature=None):
        """Save latest checkpoint version for automated workflows"""
        # Create latest directory
        latest_dir = os.path.join(os.path.dirname(self.ckpt_dir), "latest")
        os.makedirs(latest_dir, exist_ok=True)
        
        state = {
            "args": self.args,
            "epoch": -1,  # Mark as latest
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        # Save current temperature (for analysis scripts)
        if current_temperature is not None:
            state["current_temperature"] = current_temperature
        
        # Save latest version
        latest_path = os.path.join(latest_dir, "latest_model.pth")
        torch.save(state, latest_path, pickle_protocol=4)
        
        # If it's the best collision model, also save as best_collision_model.pth
        if is_best_collision:
            best_collision_path = os.path.join(latest_dir, "best_collision_model.pth")
            torch.save(state, best_collision_path, pickle_protocol=4)
            print(f"!!!Latest best collision model saved: {best_collision_path}")
        
        # If it's the best loss model, also save as best_loss_model.pth
        if is_best_loss:
            best_loss_path = os.path.join(latest_dir, "best_loss_model.pth")
            torch.save(state, best_loss_path, pickle_protocol=4)
            print(f"!!!Latest best loss model saved: {best_loss_path}")
        
        print(f"!!!Latest model saved: {latest_path}")
        return latest_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"


    def fit(self, data):

        cur_eval_step = 0
        print(f"Starting RQVAE training for {self.epochs} epochs...")
        print(f"Checkpoints will be saved to: {self.ckpt_dir}")

        for epoch_idx in range(self.epochs):
            # train
            current_temperature = self._update_temperature(epoch_idx)
            training_start_time = time()
            train_loss, train_recon_loss, train_div_loss, train_div_loss_scaled = self._train_epoch(data, epoch_idx)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            self.logger.info(train_loss_output)
            
            # Log training metrics to SwanLab
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.lr
            swanlab.log({
                "train/loss": train_loss,
                "train/recon_loss": train_recon_loss,
                "train/diversity_loss": train_div_loss,
                "train/diversity_loss_scaled": train_div_loss_scaled,
                "train/learning_rate": current_lr,
                "train/temperature": current_temperature if current_temperature is not None else 0.0,
                "train/epoch": epoch_idx,
                "train/epoch_time": training_end_time - training_start_time
            }, step=epoch_idx)

            self._append_metrics_row({
                "epoch": epoch_idx,
                "phase": "train",
                "temperature": current_temperature if current_temperature is not None else 0.0,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "train_recon_loss": train_recon_loss,
                "train_diversity_loss": train_div_loss,
                "train_diversity_loss_scaled": train_div_loss_scaled,
                "epoch_time": training_end_time - training_start_time,
            })

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                eval_time = time() - valid_start_time

                # Log validation metrics to SwanLab
                swanlab.log({
                    "eval/collision_rate": collision_rate,
                    "eval/best_loss": self.best_loss,
                    "eval/best_collision_rate": self.best_collision_rate,
                    "eval/eval_time": eval_time
                }, step=epoch_idx)

                self._append_metrics_row({
                    "epoch": epoch_idx,
                    "phase": "eval",
                    "temperature": current_temperature if current_temperature is not None else 0.0,
                    "eval_collision_rate": collision_rate,
                    "eval_time": eval_time,
                })

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt,
                                          current_temperature=current_temperature)
                    # Also save latest version
                    self._save_latest_checkpoint(is_best_loss=True, current_temperature=current_temperature)
                    print(f"New best loss: {self.best_loss:.6f}")
                    
                    # Log best model update
                    swanlab.log({
                        "best/loss_updated": True,
                        "best/loss": self.best_loss
                    }, step=epoch_idx)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          current_temperature=current_temperature,
                                          ckpt_file=self.best_collision_ckpt)
                    # Also save latest version
                    self._save_latest_checkpoint(is_best_collision=True, current_temperature=current_temperature)
                    print(f"New best collision rate: {self.best_collision_rate:.4f}")
                    
                    # Log best collision rate update
                    swanlab.log({
                        "best/collision_rate_updated": True,
                        "best/collision_rate": self.best_collision_rate
                    }, step=epoch_idx)
                else:
                    cur_eval_step += 1

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                                  current_temperature=current_temperature)
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])

        return self.best_loss, self.best_collision_rate




