"""
HICD V5 数据集：同时加载实例标注（bbox）和像素级标注（label TIF）。
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from osgeo import gdal

from HICD_v5.changedetection.datasets.imutils import normalize_img


def read_tif(path):
    """读取影像，返回 (H, W, C) float32"""
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    arr = ds.ReadAsArray()
    ds = None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] > 3:
            arr = arr[:3]
        arr = np.transpose(arr, (1, 2, 0))  # (C,H,W) → (H,W,C)
    return arr.astype(np.float32)


def read_label_tif(path):
    """读取像素级标签 TIF，返回 (H, W) int"""
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    arr = ds.ReadAsArray()
    ds = None
    if arr.ndim == 3:
        arr = arr[0]  # 取第一个通道
    return arr.astype(np.int64)


class ChangeDetectionDatasetV5(Dataset):
    """V5 双分支数据集。
    
    返回:
        pre_img:       (3, H, W) 前时相影像
        post_img:      (3, H, W) 后时相影像
        gt_boxes:      (N, 4) 实例分支 bbox [cx, cy, w, h] 归一化
        gt_target:     (N,) 实例分支目标类别
        gt_state:      (N,) 实例分支变化状态
        gt_target_map: (H, W) 语义分支目标类别像素图
        gt_state_map:  (H, W) 语义分支变化状态像素图
    """
    def __init__(self, dataset_path, instances_dict, dataset_config,
                 crop_size=512, mode="train"):
        self.dataset_path = dataset_path
        self.instances_dict = instances_dict
        self.dataset_config = dataset_config
        self.crop_size = crop_size
        self.mode = mode
        self.samples = []

        # 检测目录结构
        train_dir = os.path.join(dataset_path, "train", "image")
        self.flat_structure = not os.path.exists(os.path.join(train_dir, "pre"))

        # 构建 train_id → (target_idx, state_idx) 映射
        self.train_id_to_target = {}
        self.train_id_to_state = {}
        for tid, (t_idx, s_idx) in dataset_config.train_id_map.items():
            self.train_id_to_target[tid] = t_idx
            self.train_id_to_state[tid] = s_idx

        # 构建分支路由：哪些 target_idx 走 semantic 分支
        self.semantic_target_indices = set()
        for t_name, branch in dataset_config.branch_routing.items():
            if branch == 'semantic' and t_name in dataset_config.target_names:
                self.semantic_target_indices.add(dataset_config.target_names.index(t_name))

        # 收集样本
        for key in instances_dict:
            if key.startswith(f"{mode}/"):
                fname = instances_dict[key]["filename"]
                self.samples.append((mode, fname))

        # 也收集 split 内的 label 文件（用于语义分支）
        label_dir = os.path.join(dataset_path, mode, "label")
        if os.path.exists(label_dir):
            self.label_files = [f for f in os.listdir(label_dir) if f.endswith('.tif')]
        else:
            self.label_files = []

        print(f"  Dataset V5: {len(self.samples)} instance samples, "
              f"{len(self.label_files)} label files ({mode}, flat={self.flat_structure})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        split, fname = self.samples[index]
        stem = os.path.splitext(fname)[0]
        img_stem = stem.replace('_target', '')

        # 读取影像
        if self.flat_structure:
            pre_path = os.path.join(self.dataset_path, split, "image", img_stem + "_pre_war.tif")
            post_path = os.path.join(self.dataset_path, split, "image", img_stem + "_post_war.tif")
            label_path = os.path.join(self.dataset_path, split, "label", img_stem + "_target.tif")
        else:
            pre_path = os.path.join(self.dataset_path, split, "image", "pre", stem + ".tif")
            post_path = os.path.join(self.dataset_path, split, "image", "post", stem + ".tif")
            label_path = os.path.join(self.dataset_path, split, "label", img_stem + "_target.tif")

        pre_img = read_tif(pre_path)
        post_img = read_tif(post_path)

        # ── 实例分支 GT（从 instances.json）──
        key = f"{split}/{fname}"
        inst_data = self.instances_dict.get(key, {})
        instances = inst_data.get("instances", [])

        gt_boxes = torch.tensor([inst['bbox'] for inst in instances], dtype=torch.float32) if instances else torch.zeros(0, 4)
        gt_target = torch.tensor([inst['target_idx'] for inst in instances], dtype=torch.long) if instances else torch.zeros(0, dtype=torch.long)
        gt_state = torch.tensor([inst['state_idx'] for inst in instances], dtype=torch.long) if instances else torch.zeros(0, dtype=torch.long)

        # ── 语义分支 GT（从 label TIF）──
        if os.path.exists(label_path):
            label_arr = read_label_tif(label_path)
            # 映射 train_id → target_idx, state_idx
            gt_target_map = np.zeros_like(label_arr, dtype=np.int64)  # 0 = background
            gt_state_map = np.zeros_like(label_arr, dtype=np.int64)   # 0 = NoChange

            for tid in np.unique(label_arr):
                if tid == 0:
                    continue
                t_idx = self.train_id_to_target.get(int(tid), -1)
                s_idx = self.train_id_to_state.get(int(tid), -1)
                if t_idx >= 0:
                    # 只保留 semantic 分支负责的类别
                    if t_idx in self.semantic_target_indices:
                        gt_target_map[label_arr == tid] = t_idx + 1  # +1 因为 0 是背景
                    gt_state_map[label_arr == tid] = s_idx
        else:
            gt_target_map = np.zeros((self.crop_size, self.crop_size), dtype=np.int64)
            gt_state_map = np.zeros((self.crop_size, self.crop_size), dtype=np.int64)

        gt_target_map = torch.from_numpy(gt_target_map)
        gt_state_map = torch.from_numpy(gt_state_map)

        # ── 数据增强 ──
        if self.mode == "train":
            pre_img, post_img, gt_boxes, gt_target_map, gt_state_map = \
                self._random_augment(pre_img, post_img, gt_boxes, gt_target_map, gt_state_map)

        # 归一化
        pre_img = normalize_img(pre_img)
        post_img = normalize_img(post_img)

        # 转 CHW
        pre_img = torch.from_numpy(np.transpose(pre_img, (2, 0, 1)).astype(np.float32))
        post_img = torch.from_numpy(np.transpose(post_img, (2, 0, 1)).astype(np.float32))

        return {
            'pre_img': pre_img,
            'post_img': post_img,
            'gt_boxes': gt_boxes,
            'gt_target': gt_target,
            'gt_state': gt_state,
            'gt_target_map': gt_target_map,
            'gt_state_map': gt_state_map,
            'filename': key,
        }

    def _random_augment(self, pre_img, post_img, gt_boxes, gt_target_map, gt_state_map):
        """同步随机增强"""
        # 水平翻转
        if np.random.rand() > 0.5:
            pre_img = pre_img[:, ::-1, :].copy()
            post_img = post_img[:, ::-1, :].copy()
            if gt_boxes.numel() > 0:
                gt_boxes[:, 0] = 1.0 - gt_boxes[:, 0]
            gt_target_map = torch.flip(gt_target_map, dims=[1])
            gt_state_map = torch.flip(gt_state_map, dims=[1])

        # 垂直翻转
        if np.random.rand() > 0.5:
            pre_img = pre_img[::-1, :, :].copy()
            post_img = post_img[::-1, :, :].copy()
            if gt_boxes.numel() > 0:
                gt_boxes[:, 1] = 1.0 - gt_boxes[:, 1]
            gt_target_map = torch.flip(gt_target_map, dims=[0])
            gt_state_map = torch.flip(gt_state_map, dims=[0])

        return pre_img, post_img, gt_boxes, gt_target_map, gt_state_map
