import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        """
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (pred_soft * target_onehot).sum(dim=(0, 2, 3))
        union = pred_soft.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DualBranchLoss(nn.Module):
    """双分支联合损失：实例检测 + 语义分割。
    
    L_total = w_instance * L_instance + w_semantic * (L_CE + L_Dice)
    """
    def __init__(self, instance_loss, num_target_classes=5, num_state_classes=6,
                 class_weights=None, w_instance=1.0, w_semantic=1.0):
        super().__init__()
        self.instance_loss = instance_loss
        self.w_instance = w_instance
        self.w_semantic = w_semantic

        # 语义分割损失
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.seg_target_ce = nn.CrossEntropyLoss(weight=weight)
        self.seg_state_ce = nn.CrossEntropyLoss()
        self.seg_dice = DiceLoss()

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

        # ── 分支 B：语义分割损失 ──
        target_map = outputs['target_map']
        state_map = outputs['state_map']
        gt_target_mask = gt_data['gt_target_mask']
        gt_state_mask = gt_data['gt_state_mask']

        # Target segmentation CE + Dice
        loss_target_ce = self.seg_target_ce(target_map, gt_target_mask)
        loss_target_dice = self.seg_dice(target_map, gt_target_mask)

        # State segmentation CE + Dice
        loss_state_ce = self.seg_state_ce(state_map, gt_state_mask)
        loss_state_dice = self.seg_dice(state_map, gt_state_mask)

        loss_semantic = loss_target_ce + loss_target_dice + loss_state_ce + loss_state_dice
        loss_dict['loss_seg_target_ce'] = loss_target_ce.item()
        loss_dict['loss_seg_target_dice'] = loss_target_dice.item()
        loss_dict['loss_seg_state_ce'] = loss_state_ce.item()
        loss_dict['loss_seg_state_dice'] = loss_state_dice.item()
        loss_dict['loss_semantic'] = loss_semantic.item()

        # ── 总损失 ──
        total_loss = self.w_instance * loss_instance + self.w_semantic * loss_semantic
        loss_dict['loss_total'] = total_loss.item()

        return total_loss, loss_dict
