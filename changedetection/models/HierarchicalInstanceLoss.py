# -*- coding: utf-8 -*-
"""
Hierarchical Instance Detection Loss

?????????:
  - Top-K One-to-Many ?? (bbox + target type + focal)
  - Focal Loss ????????? (???? 92%)
  - Dice Loss ?????? (?????)
  - L1 + GIoU bbox ??
  - ????? (?? decoder ?????)
  - ??????? (???????????)

???? (??):
  bbox: 2.0, giou: 1.5, target: 3.0, state: 2.0, aux: 0.4  (V3)
  V4: pair-weighted state loss, ??(target, state)???????
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from HICD.changedetection.models.class_mapping import (
    DatasetConfig, NUM_TARGETS, NUM_STATES, TARGET_VALID_STATES,
)


# =====================================================================
# Focal Loss (???????)
# =====================================================================
class FocalLoss(nn.Module):
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
# Dice Loss (??????)
# =====================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=-1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[-1]).float()
        intersection = (pred_soft * target_onehot).sum(dim=0)
        union = pred_soft.sum(dim=0) + target_onehot.sum(dim=0)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# =====================================================================
# ?????
# =====================================================================
class HierarchicalInstanceLoss(nn.Module):
    def __init__(self, num_targets=NUM_TARGETS, num_states=NUM_STATES, dataset_config=None,
                 weight_bbox=2.0, weight_giou=1.5,
                 weight_target=3.0, weight_state=2.0,
                 weight_aux=0.4, focal_alpha=0.25, focal_gamma=2.0,
                 class_weights=None, topk=3):
        super().__init__()
        if dataset_config:
            num_targets = dataset_config.num_targets
            num_states = dataset_config.num_states
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

        # V4: per (target, state) pair weighting for rare combinations
        self.register_buffer('pair_weights', self._default_pair_weights())

    def forward(self, outputs, gt_boxes_list, gt_target_list, gt_state_list):
        pred_boxes = outputs['pred_boxes']
        pred_target = outputs['pred_target']
        pred_state = outputs['pred_state']
        aux_outputs = outputs.get('aux_outputs', [])

        B = pred_boxes.shape[0]
        device = pred_boxes.device

        main_loss, main_dict = self._compute_set_loss(
            pred_boxes, pred_target, pred_state,
            gt_boxes_list, gt_target_list, gt_state_list,
        )

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

        total_loss = main_loss + self.weight_aux * aux_loss

        loss_dict = {**main_dict, 'loss_aux': aux_loss.item(), 'loss_total': total_loss.item()}
        loss_dict.update(aux_dict)

        return total_loss, loss_dict

    def _compute_set_loss(self, pred_boxes, pred_target, pred_state,
                          gt_boxes_list, gt_target_list, gt_state_list):
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
            N = pred_boxes.shape[1]

            if M == 0:
                continue

            # ?? ????: bbox L1 + GIoU + focal target ??
            pred_b = pred_boxes[b]  # (N, 4)
            tgt_b = gt_boxes        # (M, 4)

            # L1 cost
            cost_bbox = torch.cdist(pred_b, tgt_b, p=1)  # (N, M)

            # GIoU cost
            pred_xyxy = self._cxcywh_to_xyxy(pred_b)
            gt_xyxy = self._cxcywh_to_xyxy(tgt_b)
            giou = self._generalized_box_iou(pred_xyxy, gt_xyxy)  # (N, M)
            cost_giou = -giou

            # Focal target cost
            with torch.no_grad():
                tgt_logits = pred_target[b]  # (N, num_targets)
                prob = tgt_logits.softmax(-1)  # (N, C)
                cost_target = -prob[:, gt_target]  # (N, M)

            # ???
            C = (self.weight_bbox * cost_bbox +
                 self.weight_giou * cost_giou +
                 self.weight_target * cost_target)
            C = C.detach().cpu()

            # ?? Top-K One-to-Many ?? ??
            # ??? GT, ? cost ??? top-K ? query
            K = min(self.topk, N)
            topk_vals, topk_idx = C.topk(K, dim=0, largest=False)  # (K, M)

            # ????????: K*M ?
            cost_k = topk_vals.T.reshape(-1)  # (M*K,)
            gt_rep = torch.arange(M).repeat_interleave(K)  # (M*K,)
            q_rep = topk_idx.T.reshape(-1)  # (M*K,)

            # ? Hungarian ?????????
            # (M*K ????? M, ?????????????)
            # ?????????? (query, gt) ????
            # ?????, ??? topk ???? (? GT ??? query)
            # ?????? query, ?????
            if K >= 1:
                # ????: ?? GT ?? cost ??? query
                row_ind = topk_idx[0]  # (M,) ?? query indices
                col_ind = torch.arange(M)  # (0..M-1) GT indices
                # ??: ???? GT ???? query, ??? cost ???
                unique_q, inv = row_ind.unique(return_inverse=True)
                keep = []
                for q in unique_q:
                    mask = (row_ind == q)
                    costs_q = C[q, col_ind[mask]]
                    best_gt = col_ind[mask][costs_q.argmin()]
                    keep.append((q.item(), best_gt.item()))
                if len(keep) > 0:
                    row_ind = torch.tensor([k[0] for k in keep], dtype=torch.long)
                    col_ind = torch.tensor([k[1] for k in keep], dtype=torch.long)
                else:
                    n_matched += 0
                    continue

            # ?? ?????? ??
            matched_pred_boxes = pred_b[row_ind]  # (K', 4)
            matched_gt_boxes = tgt_b[col_ind]     # (K', 4)

            # Bbox L1 loss
            total_bbox = total_bbox + F.l1_loss(matched_pred_boxes, matched_gt_boxes, reduction='mean')

            # GIoU loss
            mp_xyxy = self._cxcywh_to_xyxy(matched_pred_boxes)
            mg_xyxy = self._cxcywh_to_xyxy(matched_gt_boxes)
            giou_matched = self._generalized_box_iou(mp_xyxy, mg_xyxy)
            total_giou = total_giou + (1 - torch.diag(giou_matched)).mean()

            # Target classification loss (focal)
            matched_pred_target = pred_target[b][row_ind]  # (K', num_targets)
            matched_gt_target_for_cls = gt_target[col_ind]  # (K',)
            total_target = total_target + self.focal_loss(matched_pred_target, matched_gt_target_for_cls)

            # V4: pair-weighted state loss
            matched_pred_state = pred_state[b][row_ind]
            matched_gt_state = gt_state[col_ind]
            matched_gt_target = gt_target[col_ind]

            pw = self.pair_weights[matched_gt_target, matched_gt_state]
            ce_per_sample = F.cross_entropy(matched_pred_state, matched_gt_state, reduction='none')
            total_state = total_state + (pw * ce_per_sample).mean()

            # Dice
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
    def _default_pair_weights():
        w = torch.ones(10, 6)
        # Damaged/Reduced/Added/Extended: moderate boost
        w[:, 1] = 1.5
        w[:, 2] = 1.5
        w[:, 3] = 1.5
        w[:, 4] = 1.5
        # Replaced (Aircraft=7, Vessel=8): high weight, rare
        w[7, 5] = 3.0
        w[8, 5] = 3.0
        return w

    @staticmethod
    def _cxcywh_to_xyxy(x):
        xc, yc, w, h = x.unbind(-1)
        return torch.stack([xc - w/2, yc - h/2, xc + w/2, yc + h/2], dim=-1)

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


