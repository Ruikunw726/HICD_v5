# HICD V5 — Dual-Branch Change Detection

基于 Mamba (VSSM) + CLIP 文本引导的**双分支**遥感变化检测框架。

## 核心创新

在 V4 实例级检测的基础上，新增语义分割分支，解决大范围目标（跑道、停机坪）局部损坏无法精确定位的问题。

| 创新 | 说明 |
|------|------|
| 双分支解码器 | 实例检测（小目标）+ 语义分割（大/线性目标）并行 |
| Task-Specific Adapters | 轻量适配层（~100K 参数），避免梯度冲突 |
| DatasetConfig 分支路由 | YAML 配置指定每个类别走哪个分支 |
| Masked Dual-Branch Loss | 实例损失 + Masked CE + Masked Dice（只在语义目标区域计算，背景不回传梯度） |
| SD-SSM + SparseChangeGate | V4 继承：显式建模双时相差值 + 稀疏门控 |
| CLIP 文本引导 | 两阶段训练：冻结→解冻 |

## 分支路由

| 分支 | 负责类别 | 输出格式 |
|------|---------|---------|
| 实例检测 | Building, Aircraft, Tank, Vessel, Crater | bbox + target + state |
| 语义分割 | Runway, Taxiway, Apron, Highway, Farmland | 像素级 target_map + state_map |

## 快速开始

```bash
cd /mnt/f/mambacd/home
export PYTHONPATH="/mnt/f/mambacd/home:$PYTHONPATH"
source ~/miniconda/bin/activate && conda activate mamba

python HICD_v5/changedetection/script/train_full_v5.py \
    --dataset 0617final \
    --data_dir HICD/0617final \
    --batch_size 4 --grad_accum 4 \
    --learning_rate 3e-4 --max_epochs 100 \
    --w_instance 1.0 --w_semantic 1.0 \
    --clip_mode both --clip_unfreeze_epoch 20 \
    --use_amp --exp_name v5_dual_branch
```

## 项目结构

```
HICD_v5/
├── changedetection/
│   ├── configs/
│   │   ├── config.py
│   │   └── datasets/
│   │       ├── 0617final.yaml    # 含 branch_routing
│   │       └── xbd.yaml
│   ├── datasets/
│   │   ├── dataset_v5.py         # 双分支数据集
│   │   └── imutils.py
│   ├── models/
│   │   ├── HICD_v5.py            # 主模型（双分支）
│   │   ├── TaskAdapter.py        # 任务特定适配器
│   │   ├── SemanticSegmentationHead.py  # 语义分割头
│   │   ├── DualBranchLoss.py     # 双分支联合损失
│   │   ├── HierarchicalInstanceHead.py  # 实例检测头（沿用 V4）
│   │   ├── HierarchicalInstanceLoss.py  # 实例损失（沿用 V4）
│   │   ├── ChangeDecoder.py      # SD-SSM + SparseChangeGate
│   │   ├── CLIPTextEncoder.py
│   │   ├── CrossAttentionFusion.py
│   │   ├── Mamba_backbone.py
│   │   └── class_mapping.py      # 含 DatasetConfig + branch_routing
│   └── script/
│       ├── train_full_v5.py      # V5 训练脚本
│       └── metrics.py
└── README.md
```


## Masked Dual-Branch Loss

**问题**：建筑占 91% 像素，语义分割 loss 被"预测背景"主导，Runway/Taxiway 的梯度信号被稀释。

**解决**：只在 `gt_target_mask > 0` 的语义目标区域计算 CE + Dice，背景区域不回传梯度。

| 组件 | 改动前 | 改动后 |
|------|--------|--------|
| CE Loss | 整张图计算 | 只在语义目标像素计算 |
| Dice Loss | 整张图计算 | 只在语义目标像素计算 |
| 梯度回传 | 91% 背景 + 9% 目标 | 仅目标区域 |

**效果**：
- 模型被迫聚焦学习分割 Runway/Taxiway 等语义目标
- 分割 loss 不再被建筑背景淹没
- `sem_pixel_ratio` 指标记录每 batch 语义像素占比，方便调试

## 版本历史

| 版本 | 日期 | 主要改动 |
|------|------|----------|
| V1-V3 | 2026-07-28~30 | 见 [HICD](https://github.com/Ruikunw726/HICD) |
| V4 | 2026-07-30 | SD-SSM, Context-SSM, Pair-weighted Loss |
| V4.2 | 2026-07-31 | SparseChangeGate |
| V5 | 2026-08-03 | 双分支解码器：实例检测 + 语义分割，Task Adapters，DatasetConfig 路由，Dual-Branch Loss |
| V5.1 | 2026-08-03 | Masked Dual-Branch Loss：语义分支只在目标区域计算损失，背景像素不回传梯度，避免 91% 建筑主导分割 loss |

## 致谢

- [VMamba](https://github.com/MzeroMiko/VMamba) — VSSM 骨干网络
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — 文本编码器
- [HICD](https://github.com/Ruikunw726/HICD) — V1-V4 基础架构

