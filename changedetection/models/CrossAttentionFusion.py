import torch
import torch.nn as nn
import torch.nn.functional as F


class TextVisualCrossAttention(nn.Module):
    """
    交叉注意力：文本语义注入视觉特征。
    
    视觉特征作为 Query，文本特征作为 Key/Value。
    通过注意力权重，将文本语义信息聚合到视觉特征中。
    使用门控机制控制文本信息注入强度。
    """
    def __init__(self, visual_dim=128, text_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 文本特征投影到视觉空间
        self.text_proj = nn.Linear(text_dim, visual_dim)
        
        # QKV 投影
        self.q_proj = nn.Linear(visual_dim, visual_dim)
        self.k_proj = nn.Linear(visual_dim, visual_dim)
        self.v_proj = nn.Linear(visual_dim, visual_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(visual_dim, visual_dim)
        self.norm = nn.LayerNorm(visual_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 门控机制：控制文本信息注入强度
        self.gate = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim),
            nn.Sigmoid()
        )
    
    def forward(self, visual_feat, text_feat):
        """
        Args:
            visual_feat: (B, C, H, W) - 像素级视觉特征
            text_feat: (N, text_dim) - 文本语义特征 (N个类别)
        Returns:
            fused_feat: (B, C, H, W) - 融合后的视觉特征
        """
        B, C, H, W = visual_feat.shape
        N = text_feat.shape[0]
        
        # 视觉特征展平: (B, H*W, C)
        v = visual_feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        
        # 文本特征投影并扩展: (1, N, C) -> (B, N, C)
        t = self.text_proj(text_feat).unsqueeze(0).expand(B, -1, -1)
        
        # QKV
        q = self.q_proj(v)  # (B, HW, C)
        k = self.k_proj(t)  # (B, N, C)
        v_proj = self.v_proj(t)  # (B, N, C)
        
        # Multi-head reshape
        q = q.reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_proj = v_proj.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention: (B, heads, HW, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 加权聚合文本信息: (B, heads, HW, head_dim)
        out = attn @ v_proj
        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        out = self.out_proj(out)
        
        # 门控融合
        gate_input = torch.cat([v, out], dim=-1)  # (B, HW, 2C)
        gate_weight = self.gate(gate_input)  # (B, HW, C)
        fused = gate_weight * out + (1 - gate_weight) * v
        
        # 残差连接 + 归一化
        fused = self.norm(fused + v)
        
        # 恢复空间形状
        fused = fused.permute(0, 2, 1).reshape(B, C, H, W)
        return fused
