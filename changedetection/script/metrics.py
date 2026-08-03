# -*- coding: utf-8 -*-
"""
Instance-Level Metrics for Hierarchical Change Detection

Metrics:
  - mAP@0.5, mAP@0.75, mAP@[0.5:0.95]  (based on IoU matching)
  - Per-class and overall Precision / Recall / F1 for target type and state
  - Training speed (samples/sec)
  - Inference speed (ms/image)
  - Model complexity (FLOPs, params)
"""
import torch
import numpy as np
from collections import defaultdict


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes in cxcywh format."""
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    b1_xyxy = torch.stack([
        boxes1[:, 0] - boxes1[:, 2] / 2, boxes1[:, 1] - boxes1[:, 3] / 2,
        boxes1[:, 0] + boxes1[:, 2] / 2, boxes1[:, 1] + boxes1[:, 3] / 2,
    ], dim=-1)
    b2_xyxy = torch.stack([
        boxes2[:, 0] - boxes2[:, 2] / 2, boxes2[:, 1] - boxes2[:, 3] / 2,
        boxes2[:, 0] + boxes2[:, 2] / 2, boxes2[:, 1] + boxes2[:, 3] / 2,
    ], dim=-1)

    inter_x1 = torch.max(b1_xyxy[:, None, 0], b2_xyxy[None, :, 0])
    inter_y1 = torch.max(b1_xyxy[:, None, 1], b2_xyxy[None, :, 1])
    inter_x2 = torch.min(b1_xyxy[:, None, 2], b2_xyxy[None, :, 2])
    inter_y2 = torch.min(b1_xyxy[:, None, 3], b2_xyxy[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


class InstanceMetrics:
    """Accumulate predictions across batches and compute detection metrics."""

    def __init__(self, num_targets, num_states, target_names, state_names,
                 iou_thresholds=None):
        self.num_targets = num_targets
        self.num_states = num_states
        self.target_names = target_names
        self.state_names = state_names
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        self.reset()

    def reset(self):
        self.gt_boxes_all = []
        self.gt_targets_all = []
        self.gt_states_all = []
        self.pred_boxes_all = []
        self.pred_targets_all = []
        self.pred_states_all = []
        self.pred_scores_all = []

        self.train_time = 0
        self.train_samples = 0
        self.infer_time = 0
        self.infer_samples = 0

    def update(self, outputs, gt_boxes_list, gt_target_list, gt_state_list):
        pred_boxes = outputs['pred_boxes'].detach().cpu()
        pred_target = outputs['pred_target'].detach().cpu()
        pred_state = outputs['pred_state'].detach().cpu()

        B = pred_boxes.shape[0]
        for b in range(B):
            probs = torch.softmax(pred_target[b], dim=-1)
            scores, targets = probs.max(dim=-1)
            states = pred_state[b].argmax(dim=-1)

            self.gt_boxes_all.append(gt_boxes_list[b].cpu())
            self.gt_targets_all.append(gt_target_list[b].cpu())
            self.gt_states_all.append(gt_state_list[b].cpu())
            self.pred_boxes_all.append(pred_boxes[b])
            self.pred_targets_all.append(targets)
            self.pred_states_all.append(states)
            self.pred_scores_all.append(scores)

    def compute(self):
        results = {}

        # ── mAP (IoU-based) ──
        results.update(self._compute_map())

        # ── F1 / Precision / Recall ──
        results.update(self._compute_f1())

        # ── Speed ──
        if self.train_samples > 0 and self.train_time > 0:
            results['train_samples_per_sec'] = self.train_samples / self.train_time
            results['train_ms_per_sample'] = self.train_time / self.train_samples * 1000
        if self.infer_samples > 0 and self.infer_time > 0:
            results['infer_samples_per_sec'] = self.infer_samples / self.infer_time
            results['infer_ms_per_sample'] = self.infer_time / self.infer_samples * 1000

        return results

    def _compute_map(self):
        aps = {iou_t: [] for iou_t in self.iou_thresholds}
        per_class_ap = defaultdict(list)

        for sample_idx in range(len(self.gt_boxes_all)):
            gt_boxes = self.gt_boxes_all[sample_idx]
            gt_t = self.gt_targets_all[sample_idx]
            pred_boxes = self.pred_boxes_all[sample_idx]
            pred_t = self.pred_targets_all[sample_idx]
            pred_s = self.pred_scores_all[sample_idx]

            if gt_boxes.shape[0] == 0:
                continue
            if pred_boxes.shape[0] == 0:
                for iou_t in self.iou_thresholds:
                    aps[iou_t].append(0.0)
                continue

            iou_mat = box_iou(pred_boxes, gt_boxes)

            for iou_t in self.iou_thresholds:
                tp = torch.zeros(pred_boxes.shape[0])
                fp = torch.zeros(pred_boxes.shape[0])
                matched = set()

                sorted_idx = pred_s.argsort(descending=True)
                for rank, pi in enumerate(sorted_idx):
                    best_iou, best_gi = 0, -1
                    for gi in range(gt_boxes.shape[0]):
                        if gi in matched:
                            continue
                        if iou_mat[pi, gi] > best_iou:
                            best_iou = iou_mat[pi, gi]
                            best_gi = gi
                    if best_iou >= iou_t and best_gi >= 0 and pred_t[pi] == gt_t[best_gi]:
                        tp[rank] = 1
                        matched.add(best_gi)
                    else:
                        fp[rank] = 1

                tp_cum = tp.cumsum(0)
                fp_cum = fp.cumsum(0)
                precision = tp_cum / (tp_cum + fp_cum)
                recall = tp_cum / gt_boxes.shape[0]

                ap = self._voc_ap(recall.numpy(), precision.numpy())
                aps[iou_t].append(ap)
                per_class_ap[self.target_names[gt_t[0].item()] if gt_t.shape[0] > 0 else 'unknown'].append(ap)

        results = {}
        for iou_t in self.iou_thresholds:
            if aps[iou_t]:
                results[f'AP@{iou_t}'] = np.mean(aps[iou_t])

        if aps[0.5] and aps[0.75]:
            results['mAP@0.5'] = np.mean(aps[0.5])
            results['mAP@0.75'] = np.mean(aps[0.75])
            all_aps = []
            for iou_t in self.iou_thresholds:
                all_aps.extend(aps[iou_t])
            results['mAP@[0.5:0.95]'] = np.mean(all_aps) if all_aps else 0.0

        return results

    def _compute_f1(self):
        results = {}

        # Target type
        all_gt_t, all_pred_t = [], []
        for i in range(len(self.gt_targets_all)):
            gt_t = self.gt_targets_all[i].numpy()
            pred_t = self.pred_targets_all[i].numpy()

            if len(gt_t) == 0:
                continue
            if len(pred_t) >= len(gt_t):
                iou_mat = box_iou(self.pred_boxes_all[i], self.gt_boxes_all[i])
                for gi in range(len(gt_t)):
                    best_pi = iou_mat[:, gi].argmax().item()
                    all_gt_t.append(gt_t[gi])
                    all_pred_t.append(pred_t[best_pi])
            else:
                for gi in range(len(gt_t)):
                    all_gt_t.append(gt_t[gi])
                    all_pred_t.append(pred_t[gi % len(pred_t)] if len(pred_t) > 0 else -1)

        if all_gt_t:
            gt_arr = np.array(all_gt_t)
            pred_arr = np.array(all_pred_t)
            valid = pred_arr >= 0
            gt_arr = gt_arr[valid]
            pred_arr = pred_arr[valid]

            correct = (gt_arr == pred_arr).sum()
            total = len(gt_arr)
            acc = correct / max(total, 1)
            results['target_acc'] = acc

            # Per-class F1
            f1s = []
            precisions = []
            recalls = []
            for c in range(self.num_targets):
                tp = ((pred_arr == c) & (gt_arr == c)).sum()
                fp = ((pred_arr == c) & (gt_arr != c)).sum()
                fn = ((pred_arr != c) & (gt_arr == c)).sum()
                p = tp / max(tp + fp, 1)
                r = tp / max(tp + fn, 1)
                f1 = 2 * p * r / max(p + r, 1e-6)
                if (gt_arr == c).sum() > 0:
                    f1s.append(f1)
                    precisions.append(p)
                    recalls.append(r)
            results['target_macro_f1'] = np.mean(f1s) if f1s else 0
            results['target_macro_precision'] = np.mean(precisions) if precisions else 0
            results['target_macro_recall'] = np.mean(recalls) if recalls else 0

        # State
        all_gt_s, all_pred_s = [], []
        for i in range(len(self.gt_states_all)):
            gt_s = self.gt_states_all[i].numpy()
            pred_s = self.pred_states_all[i].numpy()

            if len(gt_s) == 0:
                continue
            if len(pred_s) >= len(gt_s):
                iou_mat = box_iou(self.pred_boxes_all[i], self.gt_boxes_all[i])
                for gi in range(len(gt_s)):
                    best_pi = iou_mat[:, gi].argmax().item()
                    all_gt_s.append(gt_s[gi])
                    all_pred_s.append(pred_s[best_pi])
            else:
                for gi in range(len(gt_s)):
                    all_gt_s.append(gt_s[gi])
                    all_pred_s.append(pred_s[gi % len(pred_s)] if len(pred_s) > 0 else -1)

        if all_gt_s:
            gt_arr = np.array(all_gt_s)
            pred_arr = np.array(all_pred_s)
            valid = pred_arr >= 0
            gt_arr = gt_arr[valid]
            pred_arr = pred_arr[valid]

            correct = (gt_arr == pred_arr).sum()
            total = len(gt_arr)
            results['state_acc'] = correct / max(total, 1)

            f1s, precisions, recalls = [], [], []
            for c in range(self.num_states):
                tp = ((pred_arr == c) & (gt_arr == c)).sum()
                fp = ((pred_arr == c) & (gt_arr != c)).sum()
                fn = ((pred_arr != c) & (gt_arr == c)).sum()
                p = tp / max(tp + fp, 1)
                r = tp / max(tp + fn, 1)
                f1 = 2 * p * r / max(p + r, 1e-6)
                if (gt_arr == c).sum() > 0:
                    f1s.append(f1)
                    precisions.append(p)
                    recalls.append(r)
            results['state_macro_f1'] = np.mean(f1s) if f1s else 0
            results['state_macro_precision'] = np.mean(precisions) if precisions else 0
            results['state_macro_recall'] = np.mean(recalls) if recalls else 0

        return results

    @staticmethod
    def _voc_ap(recall, precision):
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def format_results(self, results):
        lines = []
        if 'mAP@0.5' in results:
            lines.append(f"  mAP@0.5={results['mAP@0.5']:.4f}  mAP@0.75={results['mAP@0.75']:.4f}  mAP@[.5:.95]={results['mAP@[0.5:0.95]']:.4f}")
        if 'target_macro_f1' in results:
            lines.append(f"  Target  P={results['target_macro_precision']:.4f} R={results['target_macro_recall']:.4f} F1={results['target_macro_f1']:.4f} Acc={results['target_acc']:.4f}")
        if 'state_macro_f1' in results:
            lines.append(f"  State   P={results['state_macro_precision']:.4f} R={results['state_macro_recall']:.4f} F1={results['state_macro_f1']:.4f} Acc={results['state_acc']:.4f}")
        if 'train_samples_per_sec' in results:
            lines.append(f"  Train   {results['train_samples_per_sec']:.1f} samples/s  {results['train_ms_per_sample']:.1f} ms/sample")
        if 'infer_samples_per_sec' in results:
            lines.append(f"  Infer   {results['infer_samples_per_sec']:.1f} samples/s  {results['infer_ms_per_sample']:.1f} ms/sample")
        return "\n".join(lines)


def compute_model_stats(model, input_size=(1, 3, 512, 512), device='cpu'):
    """Compute model FLOPs, params, and inference latency."""
    results = {}

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results['total_params_M'] = total_params / 1e6
    results['trainable_params_M'] = trainable_params / 1e6

    # FLOPs via fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        dummy_pre = torch.randn(input_size, device=device)
        dummy_post = torch.randn(input_size, device=device)
        flops = FlopCountAnalysis(model, (dummy_pre, dummy_post))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        results['flops_G'] = flops.total() / 1e9
    except Exception as e:
        results['flops_G'] = -1
        print(f"  [FLOPs computation failed: {e}]")

    return results