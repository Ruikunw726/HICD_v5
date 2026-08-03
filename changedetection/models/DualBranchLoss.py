import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedDiceLoss(nn.Module):
    """带 mask 的 Dice Loss：只在有效区域计算，忽略背景像素。"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, mask):
        """
        Args:
            pred:  (B, C, H, W) logits
            target: (B, H, W) class indices
            mask:  (B, H, W) bool, True = 有效像素（语义目标区域）
        """
        # 只在有效区域计算
        C = pred.shape[1]
        mask4d = mask.unsqueeze(1).expand_as(pred)  # (B,C,H,W)
        pred_masked = pred[mask4d].reshape(-1, C)    # (N,C)
        target_masked = target[mask]   # (N,)
        if target_masked.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred_soft = F.softmax(pred_masked, dim=-1)            # (N, C)
        target_onehot = F.one_hot(target_masked, num_classes=pred.shape[1]).float()  # (N, C)

        intersection = (pred_soft * target_onehot).sum(dim=0)  # (C,)
        union = pred_soft.sum(dim=0) + target_onehot.sum(dim=0)  # (C,)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class MaskedCELoss(nn.Module):
    """带 mask 的 CrossEntropy：只在有效像素上计算，忽略背景区域。"""
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, pred, target, mask):
        """
        Args:
            pred:   (B, C, H, W) logits
            target: (B, H, W) class indices
            mask:   (B, H, W) bool, True = 有效像素
        """
        loss_map = self.ce(pred, target)  # (B, H, W)
        masked_loss = loss_map[mask]
        if masked_loss.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        return masked_loss.mean()


class DualBranchLoss(nn.Module):
    """双分支联合损失：实例检测 + 语义分割。

    语义分支只在有语义目标的区域计算损失，背景区域不回传梯度。
    避免 91% 建筑像素主导分割 loss，让模型聚焦学习 Runway/Taxiway 等。

    L_total = w_instance * L_instance + w_semantic * (L_CE + L_Dice)
    """
    def __init__(self, instance_loss, num_target_classes=5, num_state_classes=6,
                 class_weights=None, w_instance=3.0, w_semantic=1.0):
        super().__init__()
        self.instance_loss = instance_loss
        self.w_instance = w_instance
        self.w_semantic = w_semantic

        # 带 mask 的语义分割损失
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.seg_target_ce = MaskedCELoss(weight=weight)
        self.seg_state_ce = MaskedCELoss()
        self.seg_target_dice = MaskedDiceLoss()
        self.seg_state_dice = MaskedDiceLoss()

    def forward(self, outputs, gt_data):
        """
        Args:
            outputs: dict from model containing:
                - instance_outputs: dict (pred_boxes, pred_target, pred_state, aux_outputs)
                - target_map: (B, N_target_cls, H/4, W/4)
                - state_map: (B, N_states, H/4, W/4)
            gt_data: dict containing:
                - gt_boxes_list, gt_target_list, gt_state_list  (实例分支)
                - gt_target_mask: (B, H, W)  语义分支目标 mask
                - gt_state_mask: (B, H, W)  语义分支状态 mask
        Returns:
            total_loss, loss_dict
        """
        loss_dict = {}

        # ── 分支 A：实例检测损失 ──
        loss_instance, inst_dict = self.instance_loss(
            outputs['instance_outputs'],
            gt_data['gt_boxes_list'],
            gt_data['gt_target_list'],
            gt_data['gt_state_list'],
        )
        loss_dict.update(inst_dict)
        loss_dict['loss_instance'] = loss_instance.item()

        # ── 分支 B：语义分割损失（带 mask）──
        target_map = outputs['target_map']
        state_map = outputs['state_map']
        gt_target_mask = gt_data['gt_target_mask']
        gt_state_mask = gt_data['gt_state_mask']

        # 语义 mask：gt_target_mask > 0 的像素才是语义目标区域
        # Downsample GT to match semantic head output resolution
        if gt_target_mask.shape[-1] != target_map.shape[-1]:
            import torch.nn.functional as F
            gt_target_mask = F.interpolate(gt_target_mask.float().unsqueeze(1), size=target_map.shape[2:], mode='nearest').squeeze(1).long()
            gt_state_mask = F.interpolate(gt_state_mask.float().unsqueeze(1), size=state_map.shape[2:], mode='nearest').squeeze(1).long()

        sem_mask = (gt_target_mask > 0)  # (B, H, W)，True = 有语义目标

        # Target segmentation（只在语义目标区域计算）
        loss_target_ce = self.seg_target_ce(target_map, gt_target_mask, sem_mask)
        loss_target_dice = self.seg_target_dice(target_map, gt_target_mask, sem_mask)

        # State segmentation（同样只在语义目标区域计算）
        loss_state_ce = self.seg_state_ce(state_map, gt_state_mask, sem_mask)
        loss_state_dice = self.seg_state_dice(state_map, gt_state_mask, sem_mask)

        loss_semantic = loss_target_ce + loss_target_dice + loss_state_ce + loss_state_dice
        loss_dict['loss_seg_target_ce'] = loss_target_ce.item()
        loss_dict['loss_seg_target_dice'] = loss_target_dice.item()
        loss_dict['loss_seg_state_ce'] = loss_state_ce.item()
        loss_dict['loss_seg_state_dice'] = loss_state_dice.item()
        loss_dict['loss_semantic'] = loss_semantic.item()
        loss_dict['sem_pixel_ratio'] = sem_mask.float().mean().item()  # 记录语义像素占比

        # ── 总损失 ──
        total_loss = self.w_instance * loss_instance + self.w_semantic * loss_semantic
        loss_dict['loss_total'] = total_loss.item()

        return total_loss, loss_dict
