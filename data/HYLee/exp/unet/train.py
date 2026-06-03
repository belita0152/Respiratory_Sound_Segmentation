# -*- coding:utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parents[3]  # /home/coder/workspace
DATA_DIR = WORKSPACE_ROOT / "data" / "hylee" / "data"

sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(DATA_DIR))

from data.hylee.data.utils import label_folder, data_folder
from data.hylee.data.data_loader import LungSoundDataset
from data.hylee.utils import metric
from data.hylee.utils.loss import FocalLoss
from data.hylee.utils.utils import save_best_model
from model import UNet1D

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch_x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return batch_x.to(torch.float32, non_blocking=True).to(device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_base_path", default=data_folder)
    parser.add_argument("--label_base_path", default=label_folder)
    # parser.add_argument("--save_path", type=str, default=str(CURRENT_DIR / "runs"))
    parser.add_argument("--model_name", type=str, default="unet1d_signal")

    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--gamma", default=2.0, type=float)

    parser.add_argument("--num_classes", default=5, type=int)
    parser.add_argument("--ignore_index", default=-1, type=int)
    parser.add_argument("--target_sr", default=16000, type=int)
    parser.add_argument("--window_sec", default=8.0, type=float)
    parser.add_argument("--step_sec", default=4.0, type=float)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--down_sampling", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument("--max_train_batches", default=None, type=int)
    parser.add_argument("--max_eval_batches", default=None, type=int)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # self.save_path = os.path.join(args.save_path, args.model_name)
        # os.makedirs(self.save_path, exist_ok=True)

        # Data Loader
        (self.train_loader, self.eval_loader), (self.channel_num, self.class_num) = self._build_dataloaders()

        # Model
        self.model_cfg = {
            "in_channels": self.channel_num,
            "out_channels": self.class_num,
            "stem_channels": 32,
            "stage_channels": (32, 64, 128, 128),
            "stage_blocks": (2, 2, 2, 1),
            "stem_kernel": 31,
            "block_kernel": 15,
        }

        self.model: nn.Module = UNet1D(**self.model_cfg).to(self.device)

        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.criterion = FocalLoss(
            gamma=args.gamma,
            alpha=[1.0] * self.class_num,
            ignore_index=args.ignore_index,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

    def _make_input(self, data: torch.Tensor) -> torch.Tensor:
        return to_device(data, self.device)

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_count = 0

        for batch_idx, (data, target) in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad(set_to_none=True)

            x = self._make_input(data)  # signal: [B, 1, T]
            y = target.long().to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            if self.args.grad_clip is not None and self.args.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = x.size(0)
            total_loss += float(loss.detach()) * batch_size
            total_count += batch_size

            if self.args.max_train_batches is not None and batch_idx >= self.args.max_train_batches:
                break

        self.scheduler.step()
        return total_loss / max(total_count, 1)

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int) -> Dict:
        self.model.eval()
        preds_all, reals_all = [], []
        total_loss = 0.0
        total_count = 0

        for batch_idx, (data, target) in enumerate(self.eval_loader, start=1):
            x = self._make_input(data)
            y = target.long().to(self.device, non_blocking=True)

            logits = self.model(x)
            loss = self.criterion(logits, y)
            preds = logits.argmax(dim=1)

            preds_all.append(preds.detach().cpu())
            reals_all.append(y.detach().cpu())

            batch_size = x.size(0)
            total_loss += float(loss.detach()) * batch_size
            total_count += batch_size

            if self.args.max_eval_batches is not None and batch_idx >= self.args.max_eval_batches:
                break

        y_pred = torch.cat(preds_all, dim=0).reshape(-1).numpy()
        y_true = torch.cat(reals_all, dim=0).reshape(-1).numpy()

        result = metric.calculate_segmentation_metrics(
            y_pred,
            y_true,
            num_classes=self.class_num,
            ignore_index=self.args.ignore_index,
        )
        result["eval_loss"] = total_loss / max(total_count, 1)

        accuracy, iou_macro, dice_macro = result["accuracy"], result["iou_macro"], result["dice_macro"]
        print(
            f"[Epoch]: {epoch:03d} => "
            f"[Loss] : {result['eval_loss']:.4f} "
            f"[Accuracy] : {accuracy * 100:.2f} "
            f"[IoU Macro] : {iou_macro * 100:.2f} "
            f"[Dice Macro] : {dice_macro * 100:.2f}"
        )
        return result

    def _build_dataloaders(self) -> Tuple[Tuple[DataLoader, DataLoader], Tuple[int, int]]:
        train_dataset = LungSoundDataset(
            self.args.wav_base_path,
            self.args.label_base_path,
            train=True,
            train_ratio=self.args.train_ratio,
            down_sampling=self.args.down_sampling,
            target_sr=self.args.target_sr,
            window_sec=self.args.window_sec,
            step_sec=self.args.step_sec,
            input_type="signal",
        )
        eval_dataset = LungSoundDataset(
            self.args.wav_base_path,
            self.args.label_base_path,
            train=False,
            train_ratio=self.args.train_ratio,
            down_sampling=self.args.down_sampling,
            target_sr=self.args.target_sr,
            window_sec=self.args.window_sec,
            step_sec=self.args.step_sec,
            input_type="signal",
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        return (train_loader, eval_loader), (1, self.args.num_classes)

    def run(self):
        results = []
        best_iou = -1.0

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            result = self.eval_one_epoch(epoch)
            result["epoch"] = epoch
            result["train_loss"] = train_loss
            results.append(result)

            if result["iou_macro"] > best_iou:
                best_iou = result["iou_macro"]
                # save_best_model(
                #     model=self.model,
                #     optimizer=self.optimizer,
                #     epoch=epoch,
                #     iou=best_iou,
                #     save_dir=self.save_path,
                #     model_name=self.args.model_name,
                # )

        # file_path = os.path.join(self.save_path, f"{self.args.model_name}.json")
        # with open(file_path, "w", encoding="utf-8") as f:
        #     json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    Trainer(args).run()
