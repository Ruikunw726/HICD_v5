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


## 实际像素分布与损失设计

**实测像素占比（0617final Airports 前100张）：**

| 类别 | 像素占比 | 分支 |
|------|---------|------|
| Background | 56.98% | — |
| Taxiway | 20.81% | 语义 |
| Runway | 15.73% | 语义 |
| Apron | 3.98% | 语义 |
| Building | 2.00% | 实例 |
| Farmland | 0.36% | 语义 |
| Highway | 0.09% | 语义 |
| Crater | 0.03% | 实例 |
| Aircraft | 0.02% | 实例 |

**关键发现**：
- 语义分支目标占 41% 非背景像素，数据充足
- 实例分支目标仅占 2%，是真正的少数派
- 背景占 57%，需要排除

**损失设计**：
- **Masked Loss**：排除 57% 背景，只在语义目标区域计算 CE + Dice
- **Loss 权重**：`w_instance=3.0, w_semantic=1.0`，实例分支像素少需要更强梯度
- **语义类别权重**：inverse frequency 平衡 Taxiway(20.8%) vs Highway(0.09%)

## 版本历史

| 版本 | 日期 | 主要改动 |
|------|------|----------|
| V1-V3 | 2026-07-28~30 | 见 [HICD](https://github.com/Ruikunw726/HICD) |
| V4 | 2026-07-30 | SD-SSM, Context-SSM, Pair-weighted Loss |
| V4.2 | 2026-07-31 | SparseChangeGate |
| V5 | 2026-08-03 | 双分支解码器：实例检测 + 语义分割，Task Adapters，DatasetConfig 路由，Dual-Branch Loss |
| V5.1 | 2026-08-03 | Masked Dual-Branch Loss + 实测像素分布分析：排除 57% 背景，w_instance=3.0 补偿实例分支 2% 像素占比，语义类别 inverse frequency 权重 |

## 致谢

- [VMamba](https://github.com/MzeroMiko/VMamba) — VSSM 骨干网络
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — 文本编码器
- [HICD](https://github.com/Ruikunw726/HICD) — V1-V4 基础架构

