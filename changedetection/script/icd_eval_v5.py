# -*- coding: utf-8 -*-
"""
ICD-Eval V5: Dual-Branch Change Detection Evaluation
=====================================================

Three evaluation levels:
  1. ICD-Instance: bbox-based evaluation for instance branch targets
  2. ICD-Pixel: pixel-level evaluation for semantic branch targets
  3. ICD-Overall: weighted combination of both

Usage:
    evaluator = ICDEvalV5(dataset_config)
    evaluator.reset()
    for batch in dataloader:
        evaluator.update(outputs, gt_data)
    results = evaluator.compute()
    print(evaluator.format(results))
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


# =====================================================================
# Instance-Level Evaluation (bbox-based)
# =====================================================================

def box_iou_cxcywh(boxes1, boxes2):
    """IoU between two sets of boxes in cxcywh format."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(boxes1.shape[0], boxes2.shape[0])

    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    b1 = torch.stack([
        boxes1[:, 0] - boxes1[:, 2] / 2, boxes1[:, 1] - boxes1[:, 3] / 2,
        boxes1[:, 0] + boxes1[:, 2] / 2, boxes1[:, 1] + boxes1[:, 3] / 2,
    ], dim=-1)
    b2 = torch.stack([
        boxes2[:, 0] - boxes2[:, 2] / 2, boxes2[:, 1] - boxes2[:, 3] / 2,
        boxes2[:, 0] + boxes2[:, 2] / 2, boxes2[:, 1] + boxes2[:, 3] / 2,
    ], dim=-1)

    inter_x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


class InstanceEvaluator:
    """Instance-level ICD evaluation via bbox IoU matching."""

    def __init__(self, num_targets, num_states, target_names, state_names,
                 iou_thresholds=None):
        self.num_targets = num_targets
        self.num_states = num_states
        self.target_names = target_names
        self.state_names = state_names
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.reset()

    def reset(self):
        self.gt_boxes = []
        self.gt_targets = []
        self.gt_states = []
        self.pred_boxes = []
        self.pred_targets = []
        self.pred_states = []
        self.pred_scores = []

    def update(self, pred_boxes, pred_target_logits, pred_state_logits,
               gt_boxes, gt_targets, gt_states):
        probs = torch.softmax(pred_target_logits, dim=-1)
        scores, targets = probs.max(dim=-1)
        states = pred_state_logits.argmax(dim=-1)

        self.gt_boxes.append(gt_boxes.cpu())
        self.gt_targets.append(gt_targets.cpu())
        self.gt_states.append(gt_states.cpu())
        self.pred_boxes.append(pred_boxes.cpu())
        self.pred_targets.append(targets.cpu())
        self.pred_states.append(states.cpu())
        self.pred_scores.append(scores.cpu())

    def compute(self):
        results = {}
        aps = {t: [] for t in self.iou_thresholds}
        total_tp = {t: 0 for t in self.iou_thresholds}
        total_fp = {t: 0 for t in self.iou_thresholds}
        total_fn = {t: 0 for t in self.iou_thresholds}
        correct_target = 0
        correct_state = 0
        total_matched = 0

        for i in range(len(self.gt_boxes)):
            gt_b, gt_t, gt_s = self.gt_boxes[i], self.gt_targets[i], self.gt_states[i]
            pred_b, pred_t, pred_s, pred_sc = (
                self.pred_boxes[i], self.pred_targets[i],
                self.pred_states[i], self.pred_scores[i]
            )

            if gt_b.shape[0] == 0 and pred_b.shape[0] == 0:
                continue
            if gt_b.shape[0] == 0:
                for t in self.iou_thresholds:
                    total_fp[t] += pred_b.shape[0]
                continue
            if pred_b.shape[0] == 0:
                for t in self.iou_thresholds:
                    total_fn[t] += gt_b.shape[0]
                continue

            sorted_idx = pred_sc.argsort(descending=True)
            pred_b = pred_b[sorted_idx]
            pred_t = pred_t[sorted_idx]
            pred_s = pred_s[sorted_idx]
            pred_sc = pred_sc[sorted_idx]

            iou_mat = box_iou_cxcywh(pred_b, gt_b)

            for iou_t in self.iou_thresholds:
                tp = torch.zeros(pred_b.shape[0])
                matched_gt = set()

                for j in range(pred_b.shape[0]):
                    best_iou, best_gt = iou_mat[j].max(dim=0)
                    if best_iou >= iou_t and best_gt.item() not in matched_gt:
                        tp[j] = 1
                        matched_gt.add(best_gt.item())

                        if iou_t == 0.5:
                            total_matched += 1
                            if pred_t[j] == gt_t[best_gt]:
                                correct_target += 1
                            if pred_s[j] == gt_s[best_gt]:
                                correct_state += 1

                fp = 1 - tp
                fn = gt_b.shape[0] - int(tp.sum().item())

                total_tp[iou_t] += int(tp.sum().item())
                total_fp[iou_t] += int(fp.sum().item())
                total_fn[iou_t] += fn

                tp_cumsum = tp.cumsum(0)
                fp_cumsum = fp.cumsum(0)
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
                recall = tp_cumsum / max(gt_b.shape[0], 1)
                ap = 0
                for r_threshold in np.arange(0, 1.1, 0.1):
                    prec_at_recall = precision[recall >= r_threshold]
                    if len(prec_at_recall) > 0:
                        ap += prec_at_recall.max().item()
                ap /= 11
                aps[iou_t].append(ap)

        for t in self.iou_thresholds:
            results[f'mAP@{t}'] = np.mean(aps[t]) if aps[t] else 0.0

        results['mAP@[0.5:0.95]'] = np.mean([results[f'mAP@{t}'] for t in self.iou_thresholds])

        tp5, fp5, fn5 = total_tp[0.5], total_fp[0.5], total_fn[0.5]
        results['inst_precision'] = tp5 / max(tp5 + fp5, 1)
        results['inst_recall'] = tp5 / max(tp5 + fn5, 1)
        p, r = results['inst_precision'], results['inst_recall']
        results['inst_f1'] = 2 * p * r / max(p + r, 1e-6)

        results['inst_target_acc'] = correct_target / max(total_matched, 1)
        results['inst_state_acc'] = correct_state / max(total_matched, 1)
        results['inst_matched'] = total_matched

        return results


# =====================================================================
# Pixel-Level Evaluation (semantic segmentation)
# =====================================================================

class PixelEvaluator:
    """Pixel-level ICD evaluation for semantic branch."""

    def __init__(self, num_targets, num_states, target_names, state_names):
        self.num_targets = num_targets
        self.num_states = num_states
        self.target_names = target_names
        self.state_names = state_names
        self.reset()

    def reset(self):
        self.target_cm = np.zeros((self.num_targets + 1, self.num_targets + 1), dtype=np.int64)
        self.state_cm = np.zeros((self.num_states, self.num_states), dtype=np.int64)
        self.total_pixels = 0
        self.correct_pixels = 0

    def update(self, pred_target_map, pred_state_map, gt_target_map, gt_state_map):
        if pred_target_map.shape[-2:] != gt_target_map.shape[-2:]:
            pred_target_map = F.interpolate(
                pred_target_map, size=gt_target_map.shape[-2:],
                mode='bilinear', align_corners=False
            )
            pred_state_map = F.interpolate(
                pred_state_map, size=gt_state_map.shape[-2:],
                mode='bilinear', align_corners=False
            )

        pred_t = pred_target_map.argmax(dim=1)
        pred_s = pred_state_map.argmax(dim=1)

        mask = gt_target_map > 0
        if mask.sum() == 0:
            return

        pred_t_valid = pred_t[mask].cpu().numpy()
        gt_t_valid = gt_target_map[mask].cpu().numpy()
        pred_s_valid = pred_s[mask].cpu().numpy()
        gt_s_valid = gt_state_map[mask].cpu().numpy()

        for p, g in zip(pred_t_valid, gt_t_valid):
            self.target_cm[g, p] += 1

        for p, g in zip(pred_s_valid, gt_s_valid):
            self.state_cm[g, p] += 1

        self.correct_pixels += (pred_t_valid == gt_t_valid).sum()
        self.total_pixels += len(gt_t_valid)

    def compute(self):
        results = {}

        cm = self.target_cm
        ious, precisions, recalls, f1s = [], [], [], []
        for c in range(1, self.num_targets + 1):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            union = tp + fp + fn

            iou = tp / max(union, 1)
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-6)

            ious.append(iou)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        results['pixel_mIoU'] = np.mean(ious)
        results['pixel_precision'] = np.mean(precisions)
        results['pixel_recall'] = np.mean(recalls)
        results['pixel_f1'] = np.mean(f1s)
        results['pixel_oa'] = self.correct_pixels / max(self.total_pixels, 1)

        for c in range(1, self.num_targets + 1):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            results[f'pixel_iou_{self.target_names[c-1]}'] = tp / max(tp + fp + fn, 1)

        scm = self.state_cm
        state_ious = []
        for s in range(self.num_states):
            tp = scm[s, s]
            fp = scm[:, s].sum() - tp
            fn = scm[s, :].sum() - tp
            state_ious.append(tp / max(tp + fp + fn, 1))

        results['state_mIoU'] = np.mean(state_ious)
        results['total_pixels'] = self.total_pixels

        return results


# =====================================================================
# Combined Evaluator
# =====================================================================

class ICDEvalV5:
    """Unified dual-branch evaluator for V5."""

    def __init__(self, dataset_config):
        self.config = dataset_config

        self.instance_targets = []
        self.semantic_targets = []
        for t_name, branch in dataset_config.branch_routing.items():
            if branch == 'instance':
                self.instance_targets.append(t_name)
            else:
                self.semantic_targets.append(t_name)

        self.instance_eval = InstanceEvaluator(
            num_targets=dataset_config.num_targets,
            num_states=dataset_config.num_states,
            target_names=dataset_config.target_names,
            state_names=dataset_config.state_names,
        )

        self.pixel_eval = PixelEvaluator(
            num_targets=dataset_config.num_targets,
            num_states=dataset_config.num_states,
            target_names=dataset_config.target_names,
            state_names=dataset_config.state_names,
        )

        self.n_inst = max(len(self.instance_targets), 1)
        self.n_sem = max(len(self.semantic_targets), 1)

    def reset(self):
        self.instance_eval.reset()
        self.pixel_eval.reset()

    def update(self, outputs, gt_data):
        B = outputs['target_map'].shape[0]

        inst = outputs['instance_outputs']
        for b in range(B):
            self.instance_eval.update(
                inst['pred_boxes'][b],
                inst['pred_target'][b],
                inst['pred_state'][b],
                gt_data['gt_boxes_list'][b],
                gt_data['gt_target_list'][b],
                gt_data['gt_state_list'][b],
            )

        self.pixel_eval.update(
            outputs['target_map'],
            outputs['state_map'],
            gt_data['gt_target_mask'],
            gt_data['gt_state_mask'],
        )

    def compute(self):
        inst_results = self.instance_eval.compute()
        pixel_results = self.pixel_eval.compute()

        results = {}
        results.update({f'inst_{k}': v for k, v in inst_results.items()})
        results.update(pixel_results)

        inst_f1 = inst_results.get('inst_f1', 0)
        pixel_f1 = pixel_results.get('pixel_f1', 0)
        n_inst = self.n_inst
        n_sem = self.n_sem
        results['icd_instance_f1'] = inst_f1
        results['icd_pixel_f1'] = pixel_f1
        results['icd_overall'] = (inst_f1 * n_inst + pixel_f1 * n_sem) / (n_inst + n_sem)

        inst_sa = inst_results.get('inst_state_acc', 0)
        pixel_sa = pixel_results.get('state_mIoU', 0)
        results['icd_state_acc'] = (inst_sa * n_inst + pixel_sa * n_sem) / (n_inst + n_sem)

        return results

    def format(self, results):
        lines = []
        lines.append("=" * 60)
        lines.append("  ICD-Eval V5 Results")
        lines.append("=" * 60)

        lines.append("\n  [Instance Branch]")
        if 'inst_mAP@[0.5:0.95]' in results:
            lines.append(f"    mAP@[0.5:0.95] = {results['inst_mAP@[0.5:0.95]']:.4f}")
            lines.append(f"    mAP@0.5        = {results.get('inst_mAP@0.5', 0):.4f}")
            lines.append(f"    mAP@0.75       = {results.get('inst_mAP@0.75', 0):.4f}")
        lines.append(f"    P@0.5  = {results.get('inst_precision', 0):.4f}")
        lines.append(f"    R@0.5  = {results.get('inst_recall', 0):.4f}")
        lines.append(f"    F1@0.5 = {results.get('inst_f1', 0):.4f}")
        lines.append(f"    Target-Acc = {results.get('inst_target_acc', 0):.4f}")
        lines.append(f"    State-Acc  = {results.get('inst_state_acc', 0):.4f}")

        lines.append("\n  [Semantic Branch]")
        lines.append(f"    mIoU     = {results.get('pixel_mIoU', 0):.4f}")
        lines.append(f"    P (avg)  = {results.get('pixel_precision', 0):.4f}")
        lines.append(f"    R (avg)  = {results.get('pixel_recall', 0):.4f}")
        lines.append(f"    F1 (avg) = {results.get('pixel_f1', 0):.4f}")
        lines.append(f"    OA       = {results.get('pixel_oa', 0):.4f}")
        lines.append(f"    State-mIoU = {results.get('state_mIoU', 0):.4f}")

        lines.append("\n    Per-class IoU:")
        for t_name in self.config.target_names:
            key = f'pixel_iou_{t_name}'
            if key in results:
                lines.append(f"      {t_name:25s} = {results[key]:.4f}")

        lines.append("\n  [Overall]")
        lines.append(f"    ICD-Instance-F1 = {results.get('icd_instance_f1', 0):.4f}")
        lines.append(f"    ICD-Pixel-F1   = {results.get('icd_pixel_f1', 0):.4f}")
        lines.append(f"    ICD-Overall    = {results.get('icd_overall', 0):.4f}")
        lines.append(f"    ICD-State-Acc  = {results.get('icd_state_acc', 0):.4f}")
        lines.append("=" * 60)

        return "\n".join(lines)
