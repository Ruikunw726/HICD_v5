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
| 实例检测 | Building, Aircraft, Tank, Vessel, Crater, Playground | bbox + target + state |
| 语义分割 | Runway, Taxiway, Apron, Highway, Farmland, Non-veg ground, Trees, Low veg, Water | 像素级 target_map + state_map |

具体类别路由通过 YAML 配置文件按数据集指定，模型不写死类别数。

## 支持的数据集

| 数据集 | 类别数 | 状态数 | 实例分支 | 语义分支 |
|--------|--------|--------|----------|----------|
| 0617final | 10 | 6 | Building, Aircraft, Tank, Vessel, Crater | Runway, Taxiway, Apron, Highway, Farmland |
| SECOND | 6 | 4 | Building, Playground | Non-veg ground, Trees, Low veg, Water |
| xbd | - | - | 待配置 | 待配置 |

## 快速开始

```bash
cd /mnt/f/mambacd/home
export PYTHONPATH="/mnt/f/mambacd/home:$PYTHONPATH"
source ~/miniconda/bin/activate && conda activate mamba

# 训练 0617final
python HICD_v5/changedetection/script/train_full_v5.py \
    --dataset 0617final \
    --data_dir HICD/0617final \
    --batch_size 4 --grad_accum 4 \
    --learning_rate 3e-4 --max_epochs 100 \
    --w_instance 3.0 --w_semantic 1.0 \
    --clip_mode target --clip_unfreeze_epoch 20 \
    --use_amp --exp_name v5_0617final

# 训练 SECOND（需要先转换数据集）
python HICD_v5/changedetection/datasets/convert_second_v5.py  # 生成 SECOND_hicd_v5/
python HICD_v5/changedetection/script/train_full_v5.py \
    --dataset second \
    --data_dir SECOND_hicd_v5 \
    --scenes train \
    --batch_size 4 --grad_accum 4 \
    --learning_rate 3e-4 --max_epochs 100 \
    --w_instance 3.0 --w_semantic 1.0 \
    --clip_mode target --clip_unfreeze_epoch 20 \
    --use_amp --exp_name v5_second
```

## 项目结构

```
HICD_v5/
├── changedetection/
│   ├── configs/
│   │   ├── config.py
│   │   └── datasets/
│   │       ├── 0617final.yaml    # 10类, 6状态
│   │       ├── second.yaml       # 6类, 4状态
│   │       └── xbd.yaml
│   ├── datasets/
│   │   ├── dataset_v5.py         # 双分支数据集
│   │   ├── convert_second_v5.py  # SECOND 数据集转换
│   │   └── imutils.py
│   ├── models/
│   │   ├── HICD_v5.py            # 主模型（双分支）
│   │   ├── TaskAdapter.py        # 任务特定适配器
│   │   ├── SemanticSegmentationHead.py  # 语义分割头
│   │   ├── DualBranchLoss.py     # 双分支联合损失（Masked CE + Masked Dice）
│   │   ├── HierarchicalInstanceHead.py  # 实例检测头（沿用 V4）
│   │   ├── HierarchicalInstanceLoss.py  # 实例损失（沿用 V4）
│   │   ├── ChangeDecoder.py      # SD-SSM + SparseChangeGate
│   │   ├── CLIPTextEncoder.py
│   │   ├── CrossAttentionFusion.py
│   │   ├── Mamba_backbone.py
│   │   └── class_mapping.py      # DatasetConfig + 分支路由 + train_id_map
│   └── script/
│       ├── train_full_v5.py      # V5 训练脚本
│       └── metrics.py
└── README.md
```

## 实际像素分布与损失设计

**0617final Airports：**

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

**SECOND：**

| 类别 | 状态 | 分支 |
|------|------|------|
| Building | Disappeared/Appeared | 实例 (42,426 instances) |
| Playground | Disappeared/Appeared | 实例 (321 instances) |
| Non-veg ground | 变化 | 语义 |
| Trees | 变化 | 语义 |
| Low vegetation | 变化 | 语义 |
| Water | 变化 | 语义 |

**损失设计**：
- **Masked Loss**：排除背景，只在语义目标区域计算 CE + Dice
- **GT 下采样**：语义分支输出 H/4 分辨率，GT 自动下采样对齐
- **Loss 权重**：`w_instance=3.0, w_semantic=1.0`

## 版本历史

| 版本 | 日期 | 主要改动 |
|------|------|----------|
| V1-V3 | 2026-07-28~30 | 见 [HICD](https://github.com/Ruikunw726/HICD) |
| V4 | 2026-07-30 | SD-SSM, Context-SSM, Pair-weighted Loss |
| V4.2 | 2026-07-31 | SparseChangeGate |
| V5 | 2026-08-03 | 双分支解码器：实例检测 + 语义分割，Task Adapters，DatasetConfig 路由，Dual-Branch Loss |
| V5.1 | 2026-08-03 | Masked Dual-Branch Loss + 实测像素分布分析：排除背景，w_instance=3.0 补偿实例分支 |
| V5.2 | 2026-08-03 | SECOND 数据集支持：6 类地物 + 4 种变化状态（NoChange/Disappeared/Appeared/Transitioned），RGB 标签→train_id 编码，实例+语义双分支标签生成 |

## 踩坑记录

1. **导入路径**：V4 代码中残留 `from HICD.` / `from MambaCD.`，需全部改为 `from HICD_v5.`
2. **TARGET_VALID_STATES**：V5 class_mapping 移入 DatasetConfig，但 HierarchicalInstanceHead 仍直接导入 → 添加兼容性常量
3. **数据路径双重拼接**：`scene_dir` 已含 split 路径，dataset 不应再拼 `split`
4. **target_state_mask 维度**：默认 10×6，需根据实际 num_targets/num_states 动态构建
5. **YAML 编码**：PowerShell 的 `Set-Content` 会加 BOM 或改 GBK → 用 Python 写文件或在服务器端 `cat` 写入
6. **Loss 分辨率对齐**：语义头输出 H/4，GT 是原始分辨率 → 需在 loss 中下采样 GT
7. **Dice/CE mask 维度**：pred (B,C,H,W) vs mask (B,H,W) → 需 `mask.unsqueeze(1).expand_as(pred)` 或 `pred.permute(0,2,3,1)[mask]`

## 致谢

- [VMamba](https://github.com/MzeroMiko/VMamba) — VSSM 骨干网络
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — 文本编码器
- [HICD](https://github.com/Ruikunw726/HICD) — V1-V4 基础架构
- [SECOND](https://captain-whu.github.io/SCD/) — 语义变化检测数据集
