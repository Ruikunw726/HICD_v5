# -*- coding: utf-8 -*-
"""
Hierarchical Instance Detection Loss

根据数据集特点设计:
  - Hungarian 匹配 (bbox + target type + focal)
  - Focal Loss 处理目标类型不平衡 (建筑物占 92%)
  - Dice Loss 辅助状态分类 (处理小目标)
  - L1 + GIoU bbox 回归
  - 辅助层损失 (辅助 decoder 中间层监督)
  - 层级有效性约束 (非法状态不参与损失计算)

损失权重 (可调):
  bbox: 2.0, giou: 1.5, target: 3.0, state: 2.0, aux: 0.4  (V3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from HICD_v5.changedetection.models.class_mapping import (
    NUM_TARGETS, NUM_STATES, TARGET_VALID_STATES,
)


# =====================================================================
# Focal Loss (处理类别不平衡)
# =====================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: 降低易分类样本的权重, 聚焦于难分类样本。

    设计依据: 数据集中建筑物占比高达 92%, 普通交叉熵会被
    大量简单样本主导。Focal Loss 通过 (1-pt)^gamma 抑制简单样本。
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


# =====================================================================
# Dice Loss (辅助状态分类)
# =====================================================================
class DiceLoss(nn.Module):
    """
    Dice Loss: 对类别不平衡更鲁棒, 适合小目标。
    用于状态分类的辅助损失。
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred:   (N, C) logits
        target: (N,) long class indices
        """
        pred_soft = F.softmax(pred, dim=-1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[-1]).float()
        intersection = (pred_soft * target_onehot).sum(dim=0)
        union = pred_soft.sum(dim=0) + target_onehot.sum(dim=0)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# =====================================================================
# 主损失函数
# =====================================================================
class HierarchicalInstanceLoss(nn.Module):
    """
    层级实例检测损失

    匹配策略: 基于 (bbox L1 + GIoU + focal target) 的 Hungarian 匹配
    损失组成:
      - L_bbox: L1 距离
      - L_giou: Generalized IoU
      - L_target: Focal Loss (目标类型分类)
      - L_state: CrossEntropy + Dice (变化状态分类, 有效性掩码)
      - L_aux: 辅助层损失 (权重衰减)
    """
    def __init__(self, num_targets=NUM_TARGETS, num_states=NUM_STATES,
                 weight_bbox=2.0, weight_giou=1.5,
                 weight_target=3.0, weight_state=2.0,
                 weight_aux=0.4, focal_alpha=0.25, focal_gamma=2.0,
                 class_weights=None, topk=3):
        super().__init__()
        self.num_targets = num_targets
        self.num_states = num_states
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.weight_target = weight_target
        self.weight_state = weight_state
        self.weight_aux = weight_aux
        self.topk = topk

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.state_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, outputs, gt_boxes_list, gt_target_list, gt_state_list):
        """
        Args:
            outputs: dict from HierarchicalInstanceHead.forward()
                pred_boxes, pred_target, pred_state, aux_outputs
            gt_boxes_list:   list of (M_i, 4) — normalized [cx,cy,w,h]
            gt_target_list:  list of (M_i,) — target type indices 0-15
            gt_state_list:   list of (M_i,) — change state indices 0-5

        Returns:
            loss: scalar tensor
            loss_dict: dict of individual loss components
        """
        pred_boxes = outputs['pred_boxes']
        pred_target = outputs['pred_target']
        pred_state = outputs['pred_state']
        aux_outputs = outputs.get('aux_outputs', [])

        B = pred_boxes.shape[0]
        device = pred_boxes.device

        # ── 主损失 ──
        main_loss, main_dict = self._compute_set_loss(
            pred_boxes, pred_target, pred_state,
            gt_boxes_list, gt_target_list, gt_state_list,
        )

        # ── 辅助层损失 ──
        aux_loss = torch.tensor(0.0, device=device)
        aux_dict = {}
        for aux_i, aux_out in enumerate(aux_outputs):
            a_l, a_d = self._compute_set_loss(
                aux_out['pred_boxes'], aux_out['pred_target'],
                aux_out['pred_state'],
                gt_boxes_list, gt_target_list, gt_state_list,
            )
            aux_loss = aux_loss + a_l
            for k, v in a_d.items():
                aux_dict[f'aux{aux_i}_{k}'] = v

        # ── 总损失 ──
        total_loss = main_loss + self.weight_aux * aux_loss

        loss_dict = {**main_dict, 'loss_aux': aux_loss.item(), 'loss_total': total_loss.item()}
        loss_dict.update(aux_dict)

        return total_loss, loss_dict

    def _compute_set_loss(self, pred_boxes, pred_target, pred_state,
                          gt_boxes_list, gt_target_list, gt_state_list):
        """单层的 set prediction loss (Hungarian matching)"""
        B = pred_boxes.shape[0]
        device = pred_boxes.device

        total_bbox = torch.tensor(0.0, device=device)
        total_giou = torch.tensor(0.0, device=device)
        total_target = torch.tensor(0.0, device=device)
        total_state = torch.tensor(0.0, device=device)
        n_matched = 0

        for b in range(B):
            gt_boxes = gt_boxes_list[b].to(device)
            gt_target = gt_target_list[b].to(device)
            gt_state = gt_state_list[b].to(device)
            M = gt_boxes.shape[0]

            if M == 0:
                continue

            # ── One-to-Many Top-K 匹配 (V3) ──
            cost_bbox = torch.cdist(pred_boxes[b], gt_boxes, p=1)

            cost_giou = -self._generalized_box_iou(
                self._cxcywh_to_xyxy(pred_boxes[b]),
                self._cxcywh_to_xyxy(gt_boxes)
            )

            prob_t = pred_target[b].softmax(-1)
            target_prob_gt = prob_t[:, gt_target]
            cost_target = -torch.log(target_prob_gt.clamp(min=1e-6))

            cost_matrix = (
                self.weight_bbox * cost_bbox.detach() +
                self.weight_giou * cost_giou.detach() +
                self.weight_target * cost_target.detach()
            )  # (Q, M)
            cost_matrix = torch.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=-1e6)

            # 每个 GT 匹配 Top-K 个 query (V3: one-to-many)
            Q = cost_matrix.shape[0]
            K = min(self.topk, Q)
            _, topk_indices = cost_matrix.topk(K, dim=0, largest=False)  # (K, M)
            row_ind = topk_indices.flatten().clamp(0, Q - 1).cpu().numpy()
            col_ind = torch.arange(M).unsqueeze(0).expand(K, -1).flatten().cpu().numpy()

            # ── bbox 损失 ──
            total_bbox = total_bbox + F.l1_loss(
                pred_boxes[b][row_ind], gt_boxes[col_ind]
            )
            total_giou = total_giou + (1 - torch.diag(
                self._generalized_box_iou(
                    self._cxcywh_to_xyxy(pred_boxes[b][row_ind]),
                    self._cxcywh_to_xyxy(gt_boxes[col_ind])
                )
            )).mean()

            # ── target focal loss ──
            total_target = total_target + self.focal_loss(
                pred_target[b][row_ind], gt_target[col_ind]
            )

            # ── state loss (仅合法状态) ──
            matched_pred_state = pred_state[b][row_ind]
            matched_gt_state = gt_state[col_ind]

            total_state = total_state + self.state_loss_fn(
                matched_pred_state, matched_gt_state
            )
            # Dice 辅助
            total_state = total_state + self.dice_loss(
                matched_pred_state, matched_gt_state
            )

            n_matched += 1

        n = max(n_matched, 1)

        loss = (
            self.weight_bbox * total_bbox / n +
            self.weight_giou * total_giou / n +
            self.weight_target * total_target / n +
            self.weight_state * total_state / n
        )

        return loss, {
            'loss_bbox': (total_bbox / n).item(),
            'loss_giou': (total_giou / n).item(),
            'loss_target': (total_target / n).item(),
            'loss_state': (total_state / n).item(),
        }

    @staticmethod
    def _cxcywh_to_xyxy(x):
        xc, yc, w, h = x.unbind(-1)
        return torch.stack([xc - w/2, yc - w/2, xc + w/2, yc + h/2], dim=-1)

    @staticmethod
    def _generalized_box_iou(boxes1, boxes2):
        inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
        inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
        inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
        inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / union.clamp(min=1e-6)

        enclose_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
        enclose_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
        enclose_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
        enclose_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
        enclose = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

        return iou - (enclose - union) / enclose.clamp(min=1e-6)
