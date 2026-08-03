import torch
import torch.nn as nn


class TaskAdapter(nn.Module):
    """任务特定适配器：让实例/语义分支看到不同的特征视图。
    
    LayerNorm + 1×1 Conv，极轻量（~50K 参数）。
    避免两个分支共享相同特征导致的梯度冲突。
    """
    def __init__(self, dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        # LayerNorm 需要 channel-last
        x_perm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_normed = self.norm(x_perm)
        x_perm = x_normed.permute(0, 3, 1, 2)  # (B, C, H, W)
        return self.act(self.conv(x_perm))
