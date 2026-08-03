# -*- coding: utf-8 -*-
"""
SECOND 数据集转换脚本 V5：6 类别 + 4 变化状态 + 双分支标签
============================================================
原始 SECOND 数据集:
  - 6 地物类别: Non-veg ground(1), Trees(2), Low veg(3), Water(4), Building(5), Playground(6)
  - 背景: 0 (white)
  - label1.png = T1 标签, label2.png = T2 标签

变化状态 (对比 label1 和 label2):
  - NoChange(0): T1 == T2 (同类)
  - Disappeared(1): T1 != 0, T2 == 0 (前有后无)
  - Appeared(2): T1 == 0, T2 != 0 (前无后有)
  - Transitioned(3): T1 != 0, T2 != 0, T1 != T2 (类别转变)

输出:
  - 像素标签 TIF: train_id = class_id * 4 + state_id (class_id 1-6)
  - instances.json: Building/Playground 的 bbox 实例标注
"""

import os
import json
import shutil
import random
import numpy as np
from PIL import Image
from scipy import ndimage

# ── RGB → 类别 ID 映射 ──
COLOR_TO_CLASS = {
    (255, 255, 255): 0,  # Background
    (128, 0, 0): 1,      # Non-vegetated ground
    (0, 128, 0): 2,      # Trees
    (0, 255, 0): 3,      # Low vegetation
    (0, 0, 255): 4,      # Water
    (128, 128, 128): 5,  # Building
    (255, 0, 0): 6,      # Playground
}

# ── 状态 ID ──
NOCHANGE = 0
DISAPPEARED = 1
APPEARED = 2
TRANSITIONED = 3

# ── 实例分支目标类别 (像素值) ──
INSTANCE_CLASSES = {5, 6}  # Building, Playground


def rgb_to_class(rgb_arr):
    """RGB 标签图 → 类别 ID 图 (H, W)"""
    h, w, _ = rgb_arr.shape
    class_map = np.zeros((h, w), dtype=np.uint8)
    for color, cls_id in COLOR_TO_CLASS.items():
        if cls_id == 0:
            continue  # skip background
        mask = np.all(rgb_arr == color, axis=-1)
        class_map[mask] = cls_id
    return class_map


def compute_change_label(t1_class, t2_class):
    """对比 T1 和 T2 类别图，生成变化标签 train_id 图。
    
    train_id = 0: background
    train_id = class_id * 4 + state_id (class_id 1-6, state_id 0-3)
    """
    h, w = t1_class.shape
    label = np.zeros((h, w), dtype=np.uint8)
    
    # 背景 mask
    bg_t1 = (t1_class == 0)
    bg_t2 = (t2_class == 0)
    
    # NoChange: 同类且非背景
    same = (~bg_t1) & (~bg_t2) & (t1_class == t2_class)
    label[same] = t1_class[same] * 4 + NOCHANGE
    
    # Disappeared: T1 有, T2 无
    disappeared = (~bg_t1) & bg_t2
    label[disappeared] = t1_class[disappeared] * 4 + DISAPPEARED
    
    # Appeared: T1 无, T2 有
    appeared = bg_t1 & (~bg_t2)
    label[appeared] = t2_class[appeared] * 4 + APPEARED
    
    # Transitioned: T1 有, T2 有, 但不同类
    transitioned = (~bg_t1) & (~bg_t2) & (t1_class != t2_class)
    label[transitioned] = t2_class[transitioned] * 4 + TRANSITIONED
    
    return label


def extract_instances(t1_class, t2_class, img_w, img_h, min_area=50):
    """提取 Building/Playground 的实例 bbox。
    
    对 T1 和 T2 中的实例分别提取，然后对比得到变化状态。
    """
    instances = []
    
    for cls_id in INSTANCE_CLASSES:
        # T1 中的实例
        t1_binary = (t1_class == cls_id).astype(np.uint8)
        t1_labeled, t1_n = ndimage.label(t1_binary)
        
        # T2 中的实例
        t2_binary = (t2_class == cls_id).astype(np.uint8)
        t2_labeled, t2_n = ndimage.label(t2_binary)
        
        # 收集 T1 实例
        t1_instances = {}
        for i in range(1, t1_n + 1):
            region = (t1_labeled == i)
            area = region.sum()
            if area < min_area:
                continue
            rows, cols = np.where(region)
            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            pixels = set(zip(rows.tolist(), cols.tolist()))
            t1_instances[i] = {
                'bbox': [float(cx), float(cy), float(bw), float(bh)],
                'pixels': pixels,
                'area': area,
            }
        
        # 收集 T2 实例
        t2_instances = {}
        for i in range(1, t2_n + 1):
            region = (t2_labeled == i)
            area = region.sum()
            if area < min_area:
                continue
            rows, cols = np.where(region)
            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            pixels = set(zip(rows.tolist(), cols.tolist()))
            t2_instances[i] = {
                'bbox': [float(cx), float(cy), float(bw), float(bh)],
                'pixels': pixels,
                'area': area,
            }
        
        # 匹配 T1 和 T2 实例 (像素重叠)
        matched_t1 = set()
        matched_t2 = set()
        for i, inst1 in t1_instances.items():
            best_j = None
            best_overlap = 0
            for j, inst2 in t2_instances.items():
                if j in matched_t2:
                    continue
                overlap = len(inst1['pixels'] & inst2['pixels'])
                ratio = overlap / min(len(inst1['pixels']), len(inst2['pixels']))
                if ratio > best_overlap:
                    best_overlap = ratio
                    best_j = j
            if best_j is not None and best_overlap >= 0.3:
                matched_t1.add(i)
                matched_t2.add(best_j)
                # Matched: NoChange (bbox 用 T2 的)
                inst = t2_instances[best_j]
                instances.append({
                    'bbox': inst['bbox'],
                    'target_idx': cls_id - 1,  # 0-indexed
                    'state_idx': NOCHANGE,
                    'area': inst['area'],
                })
        
        # 未匹配的 T1 实例 → Disappeared
        for i, inst in t1_instances.items():
            if i not in matched_t1:
                instances.append({
                    'bbox': inst['bbox'],
                    'target_idx': cls_id - 1,
                    'state_idx': DISAPPEARED,
                    'area': inst['area'],
                })
        
        # 未匹配的 T2 实例 → Appeared
        for j, inst in t2_instances.items():
            if j not in matched_t2:
                instances.append({
                    'bbox': inst['bbox'],
                    'target_idx': cls_id - 1,
                    'state_idx': APPEARED,
                    'area': inst['area'],
                })
    
    return instances


def process_split(src_dir, dst_dir, split_name, all_instances):
    """处理一个 split (train/val/test)"""
    im1_dir = os.path.join(src_dir, 'im1')
    im2_dir = os.path.join(src_dir, 'im2')
    lb1_dir = os.path.join(src_dir, 'label1')
    lb2_dir = os.path.join(src_dir, 'label2')
    
    out_img_dir = os.path.join(dst_dir, split_name, 'image')
    out_lbl_dir = os.path.join(dst_dir, split_name, 'label')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(im1_dir) if f.endswith('.png')])
    
    for fname in files:
        stem = fname.replace('.png', '')
        
        # 读取影像 → TIF
        img1 = np.array(Image.open(os.path.join(im1_dir, fname)))
        img2 = np.array(Image.open(os.path.join(im2_dir, fname)))
        
        # 保存为 TIF (H, W, C) → (C, H, W)
        from osgeo import gdal, gdal_array
        
        for arr, suffix in [(img1, '_pre_war'), (img2, '_post_war')]:
            out_path = os.path.join(out_img_dir, stem + suffix + '.tif')
            h, w, c = arr.shape
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(out_path, w, h, c, gdal.GDT_Byte)
            for ch in range(c):
                ds.GetRasterBand(ch + 1).WriteArray(arr[:, :, ch])
            ds = None
        
        # 读取标签
        lb1 = rgb_to_class(np.array(Image.open(os.path.join(lb1_dir, fname))))
        lb2 = rgb_to_class(np.array(Image.open(os.path.join(lb2_dir, fname))))
        
        # 生成变化标签
        change_label = compute_change_label(lb1, lb2)
        
        # 保存标签 TIF
        lbl_path = os.path.join(out_lbl_dir, stem + '_target.tif')
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(lbl_path, change_label.shape[1], change_label.shape[0], 1, gdal.GDT_Byte)
        ds.GetRasterBand(1).WriteArray(change_label)
        ds = None
        
        # 提取实例
        img_h, img_w = lb1.shape
        instances = extract_instances(lb1, lb2, img_w, img_h)
        
        key = f"{split_name}/{stem}_pre_war.tif"
        all_instances[key] = {
            'filename': f'{stem}_pre_war.tif',
            'instances': [
                {'bbox': inst['bbox'], 'target_idx': inst['target_idx'], 'state_idx': inst['state_idx']}
                for inst in instances
            ]
        }
    
    print(f"  {split_name}: {len(files)} samples processed")


def main():
    src_dir = '/autodl-fs/data/SECOND_train'
    dst_dir = '/autodl-fs/data/SECOND_hicd_v5'
    
    # 清理旧数据
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    
    all_instances = {}
    
    # 获取所有文件名并随机分 train/val/test (7:2:1)
    files = sorted([f for f in os.listdir(os.path.join(src_dir, 'im1')) if f.endswith('.png')])
    random.seed(42)
    random.shuffle(files)
    
    n = len(files)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    
    splits = {
        'train': files[:n_train],
        'val': files[n_train:n_train + n_val],
        'test': files[n_train + n_val:],
    }
    
    # 复制文件到对应 split
    for split_name, split_files in splits.items():
        im1_dir = os.path.join(src_dir, 'im1')
        im2_dir = os.path.join(src_dir, 'im2')
        lb1_dir = os.path.join(src_dir, 'label1')
        lb2_dir = os.path.join(src_dir, 'label2')
        
        out_img_dir = os.path.join(dst_dir, split_name, 'image')
        out_lbl_dir = os.path.join(dst_dir, split_name, 'label')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)
        
        for fname in split_files:
            stem = fname.replace('.png', '')
            
            # 读取影像
            img1 = np.array(Image.open(os.path.join(im1_dir, fname)))
            img2 = np.array(Image.open(os.path.join(im2_dir, fname)))
            
            from osgeo import gdal
            
            for arr, suffix in [(img1, '_pre_war'), (img2, '_post_war')]:
                out_path = os.path.join(out_img_dir, stem + suffix + '.tif')
                h, w, c = arr.shape
                driver = gdal.GetDriverByName('GTiff')
                ds = driver.Create(out_path, w, h, c, gdal.GDT_Byte)
                for ch in range(c):
                    ds.GetRasterBand(ch + 1).WriteArray(arr[:, :, ch])
                ds = None
            
            # 标签
            lb1 = rgb_to_class(np.array(Image.open(os.path.join(lb1_dir, fname))))
            lb2 = rgb_to_class(np.array(Image.open(os.path.join(lb2_dir, fname))))
            
            change_label = compute_change_label(lb1, lb2)
            
            lbl_path = os.path.join(out_lbl_dir, stem + '_target.tif')
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(lbl_path, change_label.shape[1], change_label.shape[0], 1, gdal.GDT_Byte)
            ds.GetRasterBand(1).WriteArray(change_label)
            ds = None
            
            # 实例
            img_h, img_w = lb1.shape
            instances = extract_instances(lb1, lb2, img_w, img_h)
            
            key = f"{split_name}/{stem}_pre_war.tif"
            all_instances[key] = {
                'filename': f'{stem}_pre_war.tif',
                'instances': [
                    {'bbox': inst['bbox'], 'target_idx': inst['target_idx'], 'state_idx': inst['state_idx']}
                    for inst in instances
                ]
            }
        
        print(f"  {split_name}: {len(split_files)} samples")
    
    # 保存 instances.json
    with open(os.path.join(dst_dir, 'instances.json'), 'w') as f:
        json.dump(all_instances, f, indent=2)
    
    # 统计
    total_inst = sum(len(v['instances']) for v in all_instances.values())
    state_counts = {}
    target_counts = {}
    for v in all_instances.values():
        for inst in v['instances']:
            s = inst['state_idx']
            t = inst['target_idx']
            state_counts[s] = state_counts.get(s, 0) + 1
            target_counts[t] = target_counts.get(t, 0) + 1
    
    print(f"\nTotal instances: {total_inst}")
    print(f"State distribution: {state_counts}")
    print(f"Target distribution: {target_counts}")
    print(f"\nSaved to {dst_dir}")


if __name__ == '__main__':
    main()

