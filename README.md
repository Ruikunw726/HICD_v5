# HICD V5 �� Dual-Branch Change Detection

���� Mamba (VSSM) + CLIP �ı�������**˫��֧**ң�б仯����ܡ�

## ���Ĵ���

�� V4 ʵ�������Ļ����ϣ���������ָ��֧�������ΧĿ�꣨�ܵ���ͣ��ƺ���ֲ����޷���ȷ��λ�����⡣

| ���� | ˵�� |
|------|------|
| ˫��֧������ | ʵ����⣨СĿ�꣩+ ����ָ��/����Ŀ�꣩���� |
| Task-Specific Adapters | ��������㣨~100K �������������ݶȳ�ͻ |
| DatasetConfig ��֧·�� | YAML ����ָ��ÿ��������ĸ���֧ |
| Masked Dual-Branch Loss | ʵ����ʧ + Masked CE + Masked Dice��ֻ������Ŀ��������㣬�������ش��ݶȣ� |
| SD-SSM + SparseChangeGate | V4 �̳У���ʽ��ģ˫ʱ���ֵ + ϡ���ſ� |
| CLIP �ı����� | ���׶�ѵ����������ⶳ |

## ��֧·��

| ��֧ | ������� | �����ʽ |
|------|---------|---------|
| ʵ����� | Building, Aircraft, Tank, Vessel, Crater, Playground | bbox + target + state |
| ����ָ� | Runway, Taxiway, Apron, Highway, Farmland, Non-veg ground, Trees, Low veg, Water | ���ؼ� target_map + state_map |

�������·��ͨ�� YAML �����ļ������ݼ�ָ����ģ�Ͳ�д���������

## ֧�ֵ����ݼ�

| ���ݼ� | ����� | ״̬�� | ʵ����֧ | �����֧ |
|--------|--------|--------|----------|----------|
| 0617final | 10 | 6 | Building, Aircraft, Tank, Vessel, Crater | Runway, Taxiway, Apron, Highway, Farmland |
| SECOND | 6 | 4 | Building, Playground | Non-veg ground, Trees, Low veg, Water |
| xbd | - | - | ������ | ������ |

## ���ٿ�ʼ

```bash
cd /mnt/f/mambacd/home
export PYTHONPATH="/mnt/f/mambacd/home:$PYTHONPATH"
source ~/miniconda/bin/activate && conda activate mamba

# ѵ�� 0617final
python HICD_v5/changedetection/script/train_full_v5.py \
    --dataset 0617final \
    --data_dir HICD/0617final \
    --batch_size 4 --grad_accum 4 \
    --learning_rate 3e-4 --max_epochs 100 \
    --w_instance 3.0 --w_semantic 1.0 \
    --clip_mode target --clip_unfreeze_epoch 20 \
    --use_amp --exp_name v5_0617final

# ѵ�� SECOND����Ҫ��ת�����ݼ���
python HICD_v5/changedetection/datasets/convert_second_v5.py  # ���� SECOND_hicd_v5/
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

## ��Ŀ�ṹ

```
HICD_v5/
������ changedetection/
��   ������ configs/
��   ��   ������ config.py
��   ��   ������ datasets/
��   ��       ������ 0617final.yaml    # 10��, 6״̬
��   ��       ������ second.yaml       # 6��, 4״̬
��   ��       ������ xbd.yaml
��   ������ datasets/
��   ��   ������ dataset_v5.py         # ˫��֧���ݼ�
��   ��   ������ convert_second_v5.py  # SECOND ���ݼ�ת��
��   ��   ������ imutils.py
��   ������ models/
��   ��   ������ HICD_v5.py            # ��ģ�ͣ�˫��֧��
��   ��   ������ TaskAdapter.py        # �����ض�������
��   ��   ������ SemanticSegmentationHead.py  # ����ָ�ͷ
��   ��   ������ DualBranchLoss.py     # ˫��֧������ʧ��Masked CE + Masked Dice��
��   ��   ������ HierarchicalInstanceHead.py  # ʵ�����ͷ������ V4��
��   ��   ������ HierarchicalInstanceLoss.py  # ʵ����ʧ������ V4��
��   ��   ������ ChangeDecoder.py      # SD-SSM + SparseChangeGate
��   ��   ������ CLIPTextEncoder.py
��   ��   ������ CrossAttentionFusion.py
��   ��   ������ Mamba_backbone.py
��   ��   ������ class_mapping.py      # DatasetConfig + ��֧·�� + train_id_map
��   ������ script/
��       ������ train_full_v5.py      # V5 ѵ���ű�
��       ������ metrics.py
������ README.md
```

## ʵ�����طֲ�����ʧ���

**0617final Airports��**

| ��� | ����ռ�� | ��֧ |
|------|---------|------|
| Background | 56.98% | �� |
| Taxiway | 20.81% | ���� |
| Runway | 15.73% | ���� |
| Apron | 3.98% | ���� |
| Building | 2.00% | ʵ�� |
| Farmland | 0.36% | ���� |
| Highway | 0.09% | ���� |
| Crater | 0.03% | ʵ�� |
| Aircraft | 0.02% | ʵ�� |

**SECOND��**

| ��� | ״̬ | ��֧ |
|------|------|------|
| Building | Disappeared/Appeared | ʵ�� (42,426 instances) |
| Playground | Disappeared/Appeared | ʵ�� (321 instances) |
| Non-veg ground | �仯 | ���� |
| Trees | �仯 | ���� |
| Low vegetation | �仯 | ���� |
| Water | �仯 | ���� |

**��ʧ���**��
- **Masked Loss**���ų�������ֻ������Ŀ��������� CE + Dice
- **GT �²���**�������֧��� H/4 �ֱ��ʣ�GT �Զ��²�������
- **Loss Ȩ��**��`w_instance=3.0, w_semantic=1.0`


## ICD-Eval V5 ����Э��

˫��֧�����������ְ֧��

**ʵ������ICD-Instance��**��bbox IoU ƥ�� �� mAP��P/R/F1��Target-Acc��State-Acc
**���ؼ���ICD-Pixel��**�������ط��� �� mIoU��Pixel P/R/F1��State-mIoU
**���壨ICD-Overall��**������֧�������Ȩƽ��

��ϸ��Ƽ� changedetection/script/icd_eval_v5.py��
## �汾��ʷ

| �汾 | ���� | ��Ҫ�Ķ� |
|------|------|----------|
| V1-V3 | 2026-07-28~30 | �� [HICD](https://github.com/Ruikunw726/HICD) |
| V4 | 2026-07-30 | SD-SSM, Context-SSM, Pair-weighted Loss |
| V4.2 | 2026-07-31 | SparseChangeGate |
| V5 | 2026-08-03 | ˫��֧��������ʵ����� + ����ָTask Adapters��DatasetConfig ·�ɣ�Dual-Branch Loss |
| V5.1 | 2026-08-03 | Masked Dual-Branch Loss + ʵ�����طֲ��������ų�������w_instance=3.0 ����ʵ����֧ |
| V5.2 | 2026-08-03 | SECOND ���ݼ�֧�֣�6 ����� + 4 �ֱ仯״̬��NoChange/Disappeared/Appeared/Transitioned����RGB ��ǩ��train_id ���룬ʵ��+����˫��֧��ǩ���� |

## �ȿӼ�¼

1. **����·��**��V4 �����в��� `from HICD.` / `from MambaCD.`����ȫ����Ϊ `from HICD_v5.`
2. **TARGET_VALID_STATES**��V5 class_mapping ���� DatasetConfig���� HierarchicalInstanceHead ��ֱ�ӵ��� �� ���Ӽ����Գ���
3. **����·��˫��ƴ��**��`scene_dir` �Ѻ� split ·����dataset ��Ӧ��ƴ `split`
4. **target_state_mask ά��**��Ĭ�� 10��6�������ʵ�� num_targets/num_states ��̬����
5. **YAML ����**��PowerShell �� `Set-Content` ��� BOM ��� GBK �� �� Python д�ļ����ڷ������� `cat` д��
6. **Loss �ֱ��ʶ���**������ͷ��� H/4��GT ��ԭʼ�ֱ��� �� ���� loss ���²��� GT
7. **Dice/CE mask ά��**��pred (B,C,H,W) vs mask (B,H,W) �� �� `mask.unsqueeze(1).expand_as(pred)` �� `pred.permute(0,2,3,1)[mask]`

## ��л

- [VMamba](https://github.com/MzeroMiko/VMamba) �� VSSM �Ǹ�����
- [OpenCLIP](https://github.com/mlfoundations/open_clip) �� �ı�������
- [HICD](https://github.com/Ruikunw726/HICD) �� V1-V4 �����ܹ�
- [SECOND](https://captain-whu.github.io/SCD/) �� ����仯������ݼ�
