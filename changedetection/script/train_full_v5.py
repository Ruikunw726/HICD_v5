#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HICD V5 双分支训练脚本

路径策略: 脚本自动检测项目根目录，所有默认路径相对于项目根目录。
PYTHONPATH 需要包含项目根目录的父目录（即 HICD_v5/ 所在目录）。

用法 (Linux/WSL):
    # 假设项目在 /path/to/HICD_v5
    export PYTHONPATH="/path/to:$PYTHONPATH"
    python HICD_v5/changedetection/script/train_full_v5.py \
        --dataset 0617final --batch_size 4 --max_epochs 100 --use_amp

用法 (Windows):
    # 假设项目在 F:\mambacd\home\HICD_v5
    set PYTHONPATH=F:\mambacd\home;%PYTHONPATH%
    python HICD_v5\changedetection\script\train_full_v5.py ^
        --dataset 0617final --batch_size 4 --max_epochs 100 --use_amp
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path

# ── 自动检测项目根目录 (HICD_v5/) ──
# 脚本位于 HICD_v5/changedetection/script/train_full_v5.py
# 向上 3 级到 HICD_v5/
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # HICD_v5/

# 确保项目根目录的父目录在 sys.path (for HICD_v5.* imports)
_PARENT = str(_PROJECT_ROOT.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from HICD_v5.changedetection.configs.config import get_config
from HICD_v5.changedetection.datasets.dataset_v5 import ChangeDetectionDatasetV5
from HICD_v5.changedetection.models.HICD_v5 import HICD_v5
from HICD_v5.changedetection.models.HierarchicalInstanceLoss import HierarchicalInstanceLoss
from HICD_v5.changedetection.models.DualBranchLoss import DualBranchLoss
from HICD_v5.changedetection.models.class_mapping import (
    DatasetConfig, TARGET_NAMES, STATE_NAMES, NUM_TARGETS, NUM_STATES,
)
from HICD_v5.changedetection.script.metrics import InstanceMetrics, compute_model_stats

from osgeo import gdal
gdal.UseExceptions()


def win_to_wsl(path):
    """Windows 路径 → WSL 路径 (D:\\xxx → /mnt/d/xxx)"""
    if path and len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        rest = path[2:].replace('\\', '/')
        return f"/mnt/{drive}{rest}"
    return path


def resolve_path(path_str, project_root=None):
    """将路径字符串解析为绝对路径。
    - 如果已是绝对路径，直接返回
    - 否则相对于 project_root 解析
    """
    if project_root is None:
        project_root = _PROJECT_ROOT
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((project_root / p).resolve())


def collate_fn(batch):
    """自定义 collate：实例 GT 用 list，语义 GT 用 stack"""
    out = {
        'pre_img': torch.stack([b['pre_img'] for b in batch]),
        'post_img': torch.stack([b['post_img'] for b in batch]),
        'gt_target_map': torch.stack([b['gt_target_map'] for b in batch]),
        'gt_state_map': torch.stack([b['gt_state_map'] for b in batch]),
        'filename': [b['filename'] for b in batch],
        'gt_boxes_list': [b['gt_boxes'] for b in batch],
        'gt_target_list': [b['gt_target'] for b in batch],
        'gt_state_list': [b['gt_state'] for b in batch],
    }
    return out


# =====================================================================
# Trainer
# =====================================================================
class TrainerV5:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        print(f"Project root: {_PROJECT_ROOT}")

        # ── 解析路径: 相对路径 → 绝对路径 ──
        args.data_dir = resolve_path(args.data_dir)
        args.classes_csv = resolve_path(args.classes_csv) if args.classes_csv else None
        args.pretrained_weight_path = resolve_path(args.pretrained_weight_path) if args.pretrained_weight_path else None
        args.clip_weights_path = resolve_path(args.clip_weights_path) if args.clip_weights_path else None
        args.cfg = resolve_path(args.cfg) if args.cfg else None
        args.output_dir = resolve_path(args.output_dir)

        # WSL 兼容
        args.data_dir = win_to_wsl(args.data_dir)
        args.classes_csv = win_to_wsl(args.classes_csv) if args.classes_csv else None

        # ── Dataset Config ──
        dataset_config = None
        if args.dataset:
            dataset_config = DatasetConfig.load(args.dataset)
            print(f"[Dataset] Loaded config: {args.dataset}")
            dataset_config.print_summary()
        else:
            csv_name = Path(args.classes_csv).parent.name
            try:
                dataset_config = DatasetConfig.load(csv_name)
                print(f"[Dataset] Auto-detected: {csv_name}")
                dataset_config.print_summary()
            except FileNotFoundError:
                print(f"[Dataset] No config for '{csv_name}', using defaults")

        # ── Load datasets ──
        scenes = args.scenes.split(",")
        print("\nLoading datasets...")
        train_datasets = []
        val_datasets = []

        for scene in scenes:
            scene_dir = os.path.join(args.data_dir, scene.strip())
            json_path = os.path.join(scene_dir, "instances.json")
            if not os.path.exists(json_path):
                print(f"Warning: {json_path} not found, skipping {scene}")
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                instances = json.load(f)

            train_inst = {k: v for k, v in instances.items() if k.startswith("train/")}
            val_inst = {k: v for k, v in instances.items() if k.startswith("val/")}
            print(f"  {scene.strip()}: train={len(train_inst)}, val={len(val_inst)}")

            if train_inst:
                train_datasets.append(ChangeDetectionDatasetV5(
                    scene_dir, train_inst, dataset_config, args.crop_size, mode="train"))
            if val_inst:
                val_datasets.append(ChangeDetectionDatasetV5(
                    scene_dir, val_inst, dataset_config, args.crop_size, mode="val"))

        if not train_datasets:
            raise ValueError("No training data found!")

        self.train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        self.val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else (val_datasets[0] if val_datasets else None)

        print(f"\nDataset loaded: {len(self.train_dataset)} train, "
              f"{len(self.val_dataset) if self.val_dataset else 0} val")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            collate_fn=collate_fn, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_fn, pin_memory=True,
        ) if self.val_dataset else None

        # ── Model ──
        print("\nBuilding model...")
        cfg = get_config(args)
        cfg.defrost()
        vssm = cfg.MODEL.VSSM
        cfg_dict = {
            'norm_layer': vssm.NORM_LAYER,
            'ssm_act_layer': vssm.SSM_ACT_LAYER,
            'mlp_act_layer': vssm.MLP_ACT_LAYER,
            'ssm_d_state': vssm.SSM_D_STATE,
            'ssm_ratio': vssm.SSM_RATIO,
            'ssm_dt_rank': vssm.SSM_DT_RANK,
            'ssm_conv': vssm.SSM_CONV,
            'ssm_conv_bias': vssm.SSM_CONV_BIAS,
            'ssm_drop_rate': vssm.SSM_DROP_RATE,
            'ssm_init': vssm.SSM_INIT,
            'forward_type': vssm.SSM_FORWARDTYPE,
            'mlp_ratio': vssm.MLP_RATIO,
            'mlp_drop_rate': vssm.MLP_DROP_RATE,
            'gmlp': vssm.GMLP,
            'use_checkpoint': cfg.TRAIN.USE_CHECKPOINT,
            'drop_path_rate': cfg.MODEL.DROP_PATH_RATE,
            'patch_size': vssm.PATCH_SIZE,
            'in_chans': vssm.IN_CHANS,
            'embed_dim': vssm.EMBED_DIM,
            'depths': vssm.DEPTHS,
            'downsample': vssm.DOWNSAMPLE,
            'patchembed': vssm.PATCHEMBED,
            'patch_norm': vssm.PATCH_NORM,
        }

        self.model = HICD_v5(
            pretrained=args.pretrained_weight_path,
            num_queries_per_scale=args.num_queries,
            clip_weights_path=args.clip_weights_path,
            clip_mode=args.clip_mode,
            **cfg_dict,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        print(f"Model parameters: {total_params:.2f}M (trainable: {trainable:.2f}M)")

        # CLIP two-stage
        if args.clip_unfreeze_epoch == -1:
            print("[Two-Stage] CLIP always unfrozen")
            self.model.clip_text_encoder.unfreeze(n_layers=2)
        elif args.clip_unfreeze_epoch == 0:
            print("[Two-Stage] CLIP always frozen")
        else:
            print(f"[Two-Stage] CLIP frozen until epoch {args.clip_unfreeze_epoch}")

        # ── Loss ──
        self.criterion_instance = HierarchicalInstanceLoss(
            num_targets=NUM_TARGETS, num_states=NUM_STATES,
        ).to(self.device)
        self.criterion_semantic = DualBranchLoss(
            num_targets=NUM_TARGETS, num_states=NUM_STATES,
        ).to(self.device)

        # ── Optimizer ──
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # LR scheduler
        total_steps = len(self.train_loader) * args.max_epochs
        warmup_steps = len(self.train_loader) * 5
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = args.learning_rate

        self.scaler = torch.amp.GradScaler(enabled=args.use_amp)

        # ── Save dir ──
        self.save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_map = 0.0
        self.start_epoch = 0

        # Resume
        if args.resume:
            print(f"Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.start_epoch = ckpt.get('epoch', 0)
            self.best_map = ckpt.get('best_map', 0.0)

        # Metrics
        self.metrics = InstanceMetrics(
            num_targets=NUM_TARGETS, num_states=NUM_STATES,
            target_names=TARGET_NAMES, state_names=STATE_NAMES,
        )
        self.val_results = {}

    def _get_lr(self, step):
        if step < self.warmup_steps:
            return self.base_lr * step / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_inst = 0.0
        total_sem = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            pre_imgs = batch['pre_img'].to(self.device).float()
            post_imgs = batch['post_img'].to(self.device).float()

            gt_boxes_list = [b.to(self.device) for b in batch['gt_boxes_list']]
            gt_target_list = [b.to(self.device) for b in batch['gt_target_list']]
            gt_state_list = [b.to(self.device) for b in batch['gt_state_list']]
            gt_target_map = batch['gt_target_map'].to(self.device).long()
            gt_state_map = batch['gt_state_map'].to(self.device).long()

            # LR schedule
            step = epoch * len(self.train_loader) + batch_idx
            lr = self._get_lr(step)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            with torch.amp.autocast(device_type='cuda', enabled=self.args.use_amp):
                outputs = self.model(pre_imgs, post_imgs)
                loss_inst, _ = self.criterion_instance(
                    outputs['instance'], gt_boxes_list, gt_target_list, gt_state_list)
                loss_sem = self.criterion_semantic(
                    outputs['semantic'], gt_target_map, gt_state_map)
                loss = self.args.w_instance * loss_inst + self.args.w_semantic * loss_sem

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.args.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_inst += loss_inst.item()
            total_sem += loss_sem.item()
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'inst': f"{loss_inst.item():.4f}",
                'sem': f"{loss_sem.item():.4f}",
                'lr': f"{lr:.2e}"
            })

        elapsed = time.time() - t0
        sps = len(self.train_dataset) / max(elapsed, 1e-6)
        print(f"  Train loss: {total_loss/n_batches:.4f} "
              f"(inst={total_inst/n_batches:.4f}, sem={total_sem/n_batches:.4f}) "
              f"| {sps:.1f} samples/s")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            self.val_results = {}
            return 0.0

        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_gts = []
        t0 = time.time()

        for batch in tqdm(self.val_loader, desc="Val"):
            pre_imgs = batch['pre_img'].to(self.device).float()
            post_imgs = batch['post_img'].to(self.device).float()

            gt_boxes_list = [b.to(self.device) for b in batch['gt_boxes_list']]
            gt_target_list = [b.to(self.device) for b in batch['gt_target_list']]
            gt_state_list = [b.to(self.device) for b in batch['gt_state_list']]
            gt_target_map = batch['gt_target_map'].to(self.device).long()
            gt_state_map = batch['gt_state_map'].to(self.device).long()

            with torch.amp.autocast(device_type='cuda', enabled=self.args.use_amp):
                outputs = self.model(pre_imgs, post_imgs)
                loss_inst, _ = self.criterion_instance(
                    outputs['instance'], gt_boxes_list, gt_target_list, gt_state_list)
                loss_sem = self.criterion_semantic(
                    outputs['semantic'], gt_target_map, gt_state_map)
                loss = self.args.w_instance * loss_inst + self.args.w_semantic * loss_sem

            total_loss += loss.item()
            n_batches += 1

            # Collect predictions for metrics
            pred_target = outputs['instance']['pred_target'].argmax(-1)  # (B, N)
            pred_state = outputs['instance']['pred_state'].argmax(-1)
            for b in range(len(gt_target_list)):
                all_preds.append({
                    'pred_target': pred_target[b].cpu(),
                    'pred_state': pred_state[b].cpu(),
                    'pred_boxes': outputs['instance']['pred_boxes'][b].cpu(),
                })
                all_gts.append({
                    'gt_target': gt_target_list[b].cpu(),
                    'gt_state': gt_state_list[b].cpu(),
                    'gt_boxes': gt_boxes_list[b].cpu(),
                })

        elapsed = time.time() - t0
        sps = len(self.val_dataset) / max(elapsed, 1e-6) if self.val_dataset else 0
        avg_loss = total_loss / max(n_batches, 1)

        r = self.metrics.compute_all(all_preds, all_gts)
        r['infer_samples_per_sec'] = sps
        self.val_results = r

        return avg_loss

    def train(self):
        print(f"\n{'='*60}")
        print(f"Starting training: {self.args.max_epochs} epochs")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Learning rate: {self.args.learning_rate}")
        print(f"  AMP: {self.args.use_amp}")
        print(f"  Save dir: {self.save_dir}")
        print(f"{'='*60}\n")

        log_path = os.path.join(self.save_dir, "train_log.csv")
        log_header = "epoch,train_loss,val_loss,mAP@0.5,target_F1,state_F1,loss_inst,loss_sem\n"
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write(log_header)

        for epoch in range(self.start_epoch, self.args.max_epochs):
            # CLIP unfreeze
            if (self.args.clip_unfreeze_epoch >= 0 and
                    epoch == self.args.clip_unfreeze_epoch and
                    self.model.clip_text_encoder.is_frozen):
                print(f"\n[Two-Stage] Unfreezing CLIP at epoch {epoch+1}")
                self.model.clip_text_encoder.unfreeze(n_layers=2)
                clip_params = [p for p in self.model.clip_text_encoder.parameters() if p.requires_grad]
                self.optimizer.add_param_group({
                    "params": clip_params, "lr": self.base_lr * 0.1,
                    "weight_decay": self.args.weight_decay,
                })

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            r = self.val_results

            print(f"\nEpoch {epoch+1}/{self.args.max_epochs} — "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            print(self.metrics.format_results(r))

            with open(log_path, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},"
                        f"{r.get('mAP@0.5', 0):.6f},{r.get('target_macro_f1', 0):.6f},"
                        f"{r.get('state_macro_f1', 0):.6f},"
                        f"{r.get('loss_instance', 0):.6f},{r.get('loss_semantic', 0):.6f}\n")

            # Save best
            current_map = r.get('mAP@0.5', 0)
            if current_map > self.best_map:
                self.best_map = current_map
                save_path = os.path.join(self.save_dir, "best.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'best_map': self.best_map,
                }, save_path)
                print(f"  New best mAP: {self.best_map:.4f}")

            # Save latest
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'best_map': self.best_map,
            }, os.path.join(self.save_dir, "latest.pth"))

        print(f"\nTraining complete! Best mAP: {self.best_map:.4f}")
        print(f"  Best model: {os.path.join(self.save_dir, 'best.pth')}")


def main():
    parser = argparse.ArgumentParser(description="HICD V5 Training")
    # Data
    parser.add_argument("--data_dir", type=str, default="0617final")
    parser.add_argument("--scenes", type=str, default="Airports,Ports,Urban-Rural Areas")
    parser.add_argument("--classes_csv", type=str, default="0617final/classes.csv")
    parser.add_argument("--dataset", type=str, default=None)
    # Weights
    parser.add_argument("--pretrained_weight_path", type=str, default="weights/vssmtiny_dp01_ckpt_epoch_292.pth")
    parser.add_argument("--clip_weights_path", type=str, default="weights/open_clip_pytorch_model.bin")
    # Model
    parser.add_argument("--num_queries", type=int, default=17)
    parser.add_argument("--clip_mode", type=str, default="both", choices=["both", "target", "state", "none"])
    parser.add_argument("--clip_unfreeze_epoch", type=int, default=20)
    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # V5 dual-branch weights
    parser.add_argument("--w_instance", type=float, default=3.0, help="实例分支损失权重")
    parser.add_argument("--w_semantic", type=float, default=1.0, help="语义分支损失权重")
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--exp_name", type=str, default="v5_dual_branch")
    parser.add_argument("--resume", type=str, default=None)
    # Config
    parser.add_argument("--cfg", type=str,
                        default="changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)

    args = parser.parse_args()

    trainer = TrainerV5(args)
    trainer.train()


if __name__ == "__main__":
    main()
