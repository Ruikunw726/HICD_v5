import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSegmentationHead(nn.Module):
    """语义分割解码器：FPN融合 + 像素级双头分类。
    
    负责大/线性目标（Runway、Taxiway、Apron、Highway）的像素级变化检测。
    输入 ChangeDecoder 输出的 3 个尺度特征，输出 H/4 分辨率的分割图。
    """
    def __init__(self, in_dim=128, num_target_classes=5, num_state_classes=6):
        super().__init__()
        self.num_target_classes = num_target_classes
        self.num_state_classes = num_state_classes

        # ── FPN 自顶向下融合 ──
        # p3 (1/16) → lateral → upsample → fuse with p2 (1/8)
        # p2_fused → upsample → fuse with p1 (1/4)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_dim, in_dim, kernel_size=1) for _ in range(3)
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1) for _ in range(3)
        ])

        # ── Target Head（目标区域分割）──
        self.target_head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 2, num_target_classes, kernel_size=1),
        )

        # ── State Head（变化状态分割）──
        self.state_head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 2, num_state_classes, kernel_size=1),
        )

    def forward(self, p1, p2, p3):
        """
        Args:
            p1: (B, 128, H/4, W/4)
            p2: (B, 128, H/8, W/8)
            p3: (B, 128, H/16, W/16)
        Returns:
            target_map: (B, num_target_classes, H/4, W/4)
            state_map:  (B, num_state_classes, H/4, W/4)
        """
        # FPN: 自顶向下
        c3 = self.lateral_convs[2](p3)  # (B, 128, H/16, W/16)
        c2 = self.lateral_convs[1](p2)  # (B, 128, H/8, W/8)
        c1 = self.lateral_convs[0](p1)  # (B, 128, H/4, W/4)

        # p3 → upsample → fuse with p2
        p2_fused = c2 + F.interpolate(c3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p2_fused = self.fpn_convs[1](p2_fused)

        # p2 → upsample → fuse with p1
        p1_fused = c1 + F.interpolate(p2_fused, size=c1.shape[2:], mode='bilinear', align_corners=False)
        p1_fused = self.fpn_convs[0](p1_fused)

        # 双头输出（H/4 分辨率）
        target_map = self.target_head(p1_fused)
        state_map = self.state_head(p1_fused)

        return target_map, state_map
