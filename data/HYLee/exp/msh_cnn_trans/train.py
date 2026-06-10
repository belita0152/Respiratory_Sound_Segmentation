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
from torch.utils.data import DataLoader, WeightedRandomSampler

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parents[3]  # /home/coder/workspace
DATA_DIR = WORKSPACE_ROOT / "data" / "hylee" / "data"

sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(DATA_DIR))

from data.hylee.data.utils import label_folder, data_folder
from data.hylee.data.transform import WindowTransform
from data.hylee.data.data_loader import LungSoundDataset
from data.hylee.utils import metric
from data.hylee.utils.loss import FocalLoss, CEDiceLoss, FocalDiceLoss
# from data.hylee.utils.utils import save_best_model
from model import HybridCNNTransformer

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
    parser.add_argument("--model_name", type=str, default="msh_cnn_trans_mel")

    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--gamma", default=2.0, type=float)

    parser.add_argument("--ignore_index", default=-1, type=int)
    parser.add_argument("--target_sr", default=16000, type=int)
    parser.add_argument("--window_sec", default=8.0, type=float)
    parser.add_argument("--step_sec", default=4.0, type=float)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--down_sampling", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--loss_weight", default=0.5, type=float)
    parser.add_argument("--sampler_scale", default=5.0, type=float)
    parser.add_argument("--sampler_weight", default=5.0, type=float)
    parser.add_argument("--input_type", default='mel', type=str)  # signal, wavelet, mel

    parser.add_argument("--output_channels", default=3, type=int)
    parser.add_argument("--n_fft", default=512, type=int)
    parser.add_argument("--hop_length", default=128, type=int)
    parser.add_argument("--win_length", default=512, type=int)
    parser.add_argument("--n_mels", default=128, type=int)
    parser.add_argument("--f_min", default=50.0, type=float)
    parser.add_argument("--f_max", default=2000.0, type=float)
    parser.add_argument("--no_normalize", action="store_true")

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--max_train_batches", default=None, type=int)
    parser.add_argument("--max_eval_batches", default=None, type=int)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        (self.train_loader, self.eval_loader), (self.channel_num, self.class_num) = self._build_dataloaders(
            use_train_sampler=True)

        self.model_cfg = {
            "num_classes": self.class_num,
            "input_channels": self.channel_num,
            "pretrained_backbone": False,
            "transformer_layers": 4,
            "attention_heads": 8,
            "ffn_hidden_size": 2048,
            "dropout": 0.3,
        }

        self.model: nn.Module = HybridCNNTransformer(**self.model_cfg).to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 5:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.criterion = CEDiceLoss(weight=args.loss_weight, num_classes=args.num_classes,
                                    ignore_index=args.ignore_index)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

    def _make_input(self, data: torch.Tensor) -> torch.Tensor:
        return to_device(data, self.device)

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_count = 0

        for batch_idx, (data, target) in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad(set_to_none=True)

            x = self._make_input(data)
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
        per_class_str = " ".join(
            [
                f"C{cls}:I={result['per_class_iou'][cls] * 100:.1f},"
                f"D={result['per_class_dice'][cls] * 100:.1f},"
                f"P={result['per_class_precision'][cls] * 100:.1f},"
                f"R={result['per_class_recall'][cls] * 100:.1f}"
                for cls in range(self.class_num)
            ]
        )

        print(
            f"[Epoch]: {epoch:03d} => "
            f"[Loss] : {result['eval_loss']:.4f} "
            f"[Accuracy] : {accuracy * 100:.2f} "
            f"[IoU Macro] : {iou_macro * 100:.2f} "
            f"[Dice Macro] : {dice_macro * 100:.2f} "
            f"| PerClass {per_class_str}"
        )
        return result

    def _build_dataloaders(self, use_train_sampler=True) -> Tuple[Tuple[DataLoader, DataLoader], Tuple[int, int]]:
        input_transform = WindowTransform(
            kind=self.args.input_type,
            sample_rate=self.args.target_sr,
            output_channels=self.args.output_channels,
            n_fft=self.args.n_fft,
            hop_length=self.args.hop_length,
            win_length=self.args.win_length,
            n_mels=self.args.n_mels,
            f_min=self.args.f_min,
            f_max=self.args.f_max,
        )

        train_dataset = LungSoundDataset(
            self.args.wav_base_path,
            self.args.label_base_path,
            train=True,
            train_ratio=self.args.train_ratio,
            down_sampling=self.args.down_sampling,
            target_sr=self.args.target_sr,
            window_sec=self.args.window_sec,
            step_sec=self.args.step_sec,
            input_type=input_transform,
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
            input_type=input_transform,
        )

        train_counts, train_ratios = self._count_dataset_labels(
            train_dataset,
            num_classes=self.args.num_classes,
            ignore_index=self.args.ignore_index,
        )
        eval_counts, eval_ratios = self._count_dataset_labels(
            eval_dataset,
            num_classes=self.args.num_classes,
            ignore_index=self.args.ignore_index,
        )

        self._print_label_distribution("Train", train_counts, train_ratios)
        self._print_label_distribution("Eval", eval_counts, eval_ratios)

        train_sampler = None
        if use_train_sampler:
            train_sampler = self._build_rare_ratio_sampler(
                train_dataset,
                normal_class=0,
                ignore_index=self.args.ignore_index,
                scale=self.args.sampler_scale,
                max_weight=self.args.sampler_weight,
            )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
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

        sampled_counts, sampled_ratios = self._count_loader_labels(
            train_loader,
            num_classes=self.args.num_classes,
            ignore_index=self.args.ignore_index,
            max_batches=20,
        )
        self._print_label_distribution("Sampled Train Loader", sampled_counts, sampled_ratios)

        return (train_loader, eval_loader), (self.args.output_channels, self.args.num_classes)

    def _build_rare_ratio_sampler(
            self,
            train_dataset,
            normal_class: int = 0,
            ignore_index: int = -1,
            scale: float = 10.0,
            max_weight: float = 5.0,
    ):
        sample_weights = []

        for i in range(len(train_dataset)):
            _, y = train_dataset[i]

            if not torch.is_tensor(y):
                y = torch.as_tensor(y)

            y = y.long()
            valid = y != ignore_index
            rare = (y != normal_class) & valid

            valid_count = valid.sum().item()
            rare_count = rare.sum().item()
            rare_ratio = rare_count / max(valid_count, 1)

            weight = 1.0 + scale * rare_ratio
            weight = min(weight, max_weight)
            sample_weights.append(weight)

        sample_weights = torch.DoubleTensor(sample_weights)
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def _count_dataset_labels(
            self,
            dataset,
            num_classes: int,
            ignore_index: int = -1,
    ):
        counts = torch.zeros(num_classes, dtype=torch.long)

        for i in range(len(dataset)):
            _, y = dataset[i]

            if not torch.is_tensor(y):
                y = torch.as_tensor(y)

            y = y.long().reshape(-1)
            y = y[y != ignore_index]

            if y.numel() == 0:
                continue

            counts += torch.bincount(y, minlength=num_classes)[:num_classes]

        ratios = counts.float() / counts.sum().clamp_min(1)
        return counts, ratios

    def _print_label_distribution(self, name: str, counts: torch.Tensor, ratios: torch.Tensor):
        print(f"[{name} label distribution]")
        for cls in range(len(counts)):
            print(
                f"  C{cls}: count={counts[cls].item()} "
                f"ratio={ratios[cls].item() * 100:.4f}%"
            )

    def _count_loader_labels(
            self,
            loader,
            num_classes: int,
            ignore_index: int = -1,
            max_batches: int = 20,
    ):
        counts = torch.zeros(num_classes, dtype=torch.long)

        for batch_idx, (_, y) in enumerate(loader, start=1):
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)

            y = y.long().reshape(-1)
            y = y[y != ignore_index]

            if y.numel() > 0:
                counts += torch.bincount(y, minlength=num_classes)[:num_classes]

            if batch_idx >= max_batches:
                break

        ratios = counts.float() / counts.sum().clamp_min(1)
        return counts, ratios

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

        return results


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    Trainer(args).run()
