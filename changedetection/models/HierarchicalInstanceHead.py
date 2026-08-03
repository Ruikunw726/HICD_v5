# -*- coding: utf-8 -*-
"""
Hierarchical Instance Detection Head

鏍规嵁 0617final 鏁版嵁闆嗙壒鐐归噸鏂拌璁?
  - 16 绉嶇洰鏍囩被鍨? 6 绉嶅彉鍖栫姸鎬? 灞傜骇鏈夋晥鎬х害鏉?
  - 鏋佺灏哄害宸紓: 寮瑰潙 ~50px ? 鍐滅敯 ~262K px
  - 涓ラ噸绫诲埆涓嶅钩琛? 寤虹瓚鐗╁崰姣旈珮杈?92%
  - 3 涓満鏅? 鏈哄満/娓彛/鍩庝埂

鏋舵瀯:
  pixel_features (B, 128, H/4, W/4)
    鈫?ScaleFPN (3 绾ч噾瀛楀: P3@1x, P4@2x, P5@4x)
    鈫?Transformer Decoder (6 灞? 涓棿灞傝緟鍔╄緭鍑?
    鈫?Scale-aware Query Embedding (34 queries 脳 3 scales)
    鈫?棰勬祴澶?
        bbox_head     鈫?(B, Q, 4)        [cx, cy, w, h] 鈭?[0,1]
        target_head   鈫?(B, Q, 16)       鐩爣绫诲瀷 logits
        state_head    鈫?(B, Q, 6)        鍙樺寲鐘舵€?logits (鏈夋晥鎬ф帺鐮?
"""
import math
import numpy as np
import torch
import numpy as np
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from HICD.changedetection.models.class_mapping import (
    DatasetConfig,
    TARGET_NAMES, STATE_NAMES, NUM_TARGETS, NUM_STATES,
    CLIP_TEXT_PROMPTS, TARGET_VALID_STATES, get_valid_state_mask,
)


# =====================================================================
# 澶氬昂搴︾壒寰侀噾瀛楀 (FPN)
# =====================================================================
class ScaleFPN(nn.Module):
    """
    浠?ChangeDecoder 杈撳嚭鐨勫崟灏哄害鐗瑰緛鏋勫缓 3 绾ч噾瀛楀銆?

    璁捐渚濇嵁: 鏁版嵁闆嗕腑鐩爣灏哄害璺ㄥ害杈?5000 鍊?
    闇€瑕佸灏哄害鐗瑰緛鏉ヨ鐩栦笉鍚屽ぇ灏忕殑鐩爣銆?

      P3: 128ch @ H/4 脳 W/4   鈥?灏忕洰鏍?(寮瑰潙銆佸鍙?
      P4: 128ch @ H/8 脳 W/8   鈥?涓洰鏍?(寤虹瓚銆佽溅杈?
      P5: 128ch @ H/16 脳 W/16 鈥?澶х洰鏍?(鍐滅敯銆佽窇閬?

    浣跨敤鑷《鍚戜笅璺緞 + 渚у悜杩炴帴杩涜鐗瑰緛铻嶅悎銆?
    """
    def __init__(self, in_channels=128, out_channels=128):
        super().__init__()
        self.down_2x = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.down_4x = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.lateral_5 = nn.Conv2d(out_channels, out_channels, 1)
        self.lateral_4 = nn.Conv2d(out_channels, out_channels, 1)
        self.lateral_3 = nn.Conv2d(out_channels, out_channels, 1)
        self.smooth_5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        c3 = x
        c4 = self.down_2x(x)
        c5 = self.down_4x(c4)
        p5 = self.smooth_5(self.lateral_5(c5))
        p4 = self.smooth_4(self._upsample_add(p5, self.lateral_4(c4)))
        p3 = self.smooth_3(self._upsample_add(p4, self.lateral_3(c3)))
        return [p3, p4, p5]


# =====================================================================
# 灏哄害鎰熺煡鏌ヨ宓屽叆
# =====================================================================
class ScaleAwareQueryEmbedding(nn.Module):
    """
    涓烘瘡涓壒寰佸昂搴︾敓鎴愮嫭绔嬬殑鏌ヨ宓屽叆銆?
    涓嶅悓灏哄害鐨勭壒寰佸浘瀵瑰簲涓嶅悓澶у皬鐨勭洰鏍? 鏌ヨ闇€瑕佹劅鐭ヨ嚜韬昂搴︺€?
    """
    def __init__(self, hidden_dim=128, num_queries_per_scale=34):
        super().__init__()
        self.num_queries_per_scale = num_queries_per_scale
        self.hidden_dim = hidden_dim
        self.scale_embed = nn.Embedding(3, hidden_dim)
        self.query_embed = nn.Embedding(num_queries_per_scale, hidden_dim)

    def forward(self, device):
        queries = []
        for scale_idx in range(3):
            scale_emb = self.scale_embed(torch.tensor(scale_idx, device=device))
            q = self.query_embed.weight + scale_emb.unsqueeze(0)
            queries.append(q)
        return torch.cat(queries, dim=0)  # (3*N, D)


# =====================================================================
# 瀹炰緥绾ф娴嬪ご
# =====================================================================

class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for feature maps."""
    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, x):
        """x: (B, C, H, W) -> adds positional encoding."""
        B, C, H, W = x.shape
        device = x.device
        
        pe = torch.zeros(C, H, W, device=device)
        half = C // 2
        n_freq = half // 2
        
        y_pos = torch.arange(H, device=device).float()
        div_term = torch.exp(torch.arange(0, n_freq, device=device).float() *
                             -(np.log(10000.0) / n_freq))
        sin_y = torch.sin(y_pos[:, None] * div_term[None, :])
        cos_y = torch.cos(y_pos[:, None] * div_term[None, :])
        pe[0:half:2, :, :] = sin_y.T.unsqueeze(2).expand(-1, -1, W)
        pe[1:half:2, :, :] = cos_y.T.unsqueeze(2).expand(-1, -1, W)
        
        x_pos = torch.arange(W, device=device).float()
        sin_x = torch.sin(x_pos[:, None] * div_term[None, :])
        cos_x = torch.cos(x_pos[:, None] * div_term[None, :])
        pe[half::2, :, :]   = sin_x.T.unsqueeze(1).expand(-1, H, -1)
        pe[half+1::2, :, :] = cos_x.T.unsqueeze(1).expand(-1, H, -1)
        
        return x + pe.unsqueeze(0)


class HierarchicalInstanceHead(nn.Module):
    """
    灞傜骇瀹炰緥妫€娴嬪ご: FPN 鈫?Transformer Decoder 鈫?鍒嗗眰棰勬祴

    鍒嗗眰棰勬祴閫昏緫:
      1. target_head 棰勬祴 16 绉嶇洰鏍囩被鍨?
      2. state_head 棰勬祴 6 绉嶅彉鍖栫姸鎬?
      3. 閫氳繃 target_state_mask 纭繚鍙湁鍚堟硶鐨勭姸鎬佺粍鍚堣閫変腑

    杈呭姪杈撳嚭:
      涓棿 decoder 灞?(layer 2, 4) 涔熶骇鐢熼娴? 鐢ㄤ簬杈呭姪鎹熷け銆?
      璁粌鏃惰繑鍥炴墍鏈夊眰鐨勯娴? 鎺ㄧ悊鏃跺彧鐢ㄦ渶鍚庝竴灞傘€?
    """
    def __init__(self, visual_dim=128, num_queries_per_scale=34, dataset_config=None,
                 num_targets=NUM_TARGETS, num_states=NUM_STATES,
                 num_decoder_layers=6, nhead=8, dropout=0.1,
                 num_aux_layers=2):
        super().__init__()
        self.visual_dim = visual_dim
        self.num_queries_per_scale = num_queries_per_scale
        self.num_targets = num_targets
        self.num_states = num_states
        self.num_decoder_layers = num_decoder_layers
        self.num_aux_layers = num_aux_layers

        # 鈹€鈹€ FPN 鈹€鈹€
        # V2: 2D sinusoidal positional encoding
        self.pos_enc = PositionalEncoding2D(visual_dim)
        self.fpn = ScaleFPN(in_channels=visual_dim, out_channels=visual_dim)

        # 鈹€鈹€ 澶氬昂搴︾壒寰佹姇褰?鈹€鈹€
        self.scale_proj = nn.ModuleList([
            nn.Linear(visual_dim, visual_dim) for _ in range(3)
        ])

        # 鈹€鈹€ 灏哄害鎰熺煡鏌ヨ宓屽叆 鈹€鈹€
        self.query_embedding = ScaleAwareQueryEmbedding(
            hidden_dim=visual_dim,
            num_queries_per_scale=num_queries_per_scale
        )

        # 鈹€鈹€ Transformer Decoder Layers (鎵嬪姩閬嶅巻浠ヨ幏鍙栦腑闂磋緭鍑? 鈹€鈹€
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=visual_dim, nhead=nhead,
                dim_feedforward=visual_dim * 4,
                dropout=dropout, batch_first=True,
                activation='gelu',
            )
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(visual_dim)

        # 鈹€鈹€ 杈呭姪灞傜储寮?鈹€鈹€
        aux_indices = []
        if num_aux_layers >= 1:
            aux_indices.append(num_decoder_layers // 3)
        if num_aux_layers >= 2:
            aux_indices.append(2 * num_decoder_layers // 3)
        self.aux_layer_indices = aux_indices

        # 鈹€鈹€ 杈呭姪棰勬祴澶?鈹€鈹€
        self.aux_heads = nn.ModuleList()
        for _ in self.aux_layer_indices:
            self.aux_heads.append(nn.ModuleDict({
                'bbox': nn.Sequential(
                    nn.Linear(visual_dim, visual_dim), nn.GELU(),
                    nn.Linear(visual_dim, 4), nn.Sigmoid()
                ),
                'target': nn.Sequential(
                    nn.Linear(visual_dim, visual_dim), nn.GELU(),
                    nn.Linear(visual_dim, num_targets)
                ),
                'state': nn.Sequential(
                    nn.Linear(visual_dim, visual_dim), nn.GELU(),
                    nn.Linear(visual_dim, num_states)
                ),
            }))

        # 鈹€鈹€ 鏈€缁堥娴嬪ご 鈹€鈹€
        self.bbox_head = nn.Sequential(
            nn.Linear(visual_dim, visual_dim), nn.GELU(),
            nn.Linear(visual_dim, visual_dim), nn.GELU(),
            nn.Linear(visual_dim, 4), nn.Sigmoid()
        )
        self.target_head = nn.Sequential(
            nn.Linear(visual_dim, visual_dim), nn.GELU(),
            nn.Linear(visual_dim, visual_dim), nn.GELU(),
            nn.Linear(visual_dim, num_targets)
        )
        self.state_head = nn.Sequential(
            nn.Linear(visual_dim, visual_dim), nn.GELU(),
            nn.Linear(visual_dim, num_states)
        )

        # V3: Change attention for state classification
        # 璺戦亾鍗犳弧鐢婚潰浣嗗脊鍧戝彧鏈夊嚑鍗佸儚绱? 鍏ㄥ眬骞冲潎浼氭饭娌℃崯鍧忎俊鍙枫€?
        # 鐢?cross-attention 璁╂瘡涓疄渚嬫煡璇㈣仛鐒﹀埌鍙樺寲鏈€鍓х儓鐨勫尯鍩熴€?
        self.change_attn = nn.MultiheadAttention(visual_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.change_gate = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim), nn.GELU(),
            nn.Linear(visual_dim, 1), nn.Sigmoid()
        )
        self.state_head_v3 = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim), nn.GELU(),
            nn.Linear(visual_dim, num_states)
        )

        # 鈹€鈹€ 灞傜骇鏈夋晥鎬х煩闃?鈹€鈹€
        mask = dataset_config.get_valid_state_mask() if dataset_config else get_valid_state_mask()
        self.register_buffer("target_state_mask", mask)

        self._pos_cache = {}

    def _get_sincos_pos_embed(self, length, dim, device):
        key = (length, dim)
        if key not in self._pos_cache:
            pe = torch.zeros(length, dim, device=device)
            position = torch.arange(0, length, device=device).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, dim, 2, device=device).float() * -(math.log(10000.0) / dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self._pos_cache[key] = pe
        return self._pos_cache[key]

    def _flatten_scale(self, feat, scale_idx):
        B, C, H, W = feat.shape
        memory = feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        memory = self.scale_proj[scale_idx](memory)
        pos = self._get_sincos_pos_embed(H * W, C, feat.device)
        memory = memory + pos.unsqueeze(0)
        return memory

    def _apply_state_mask(self, pred_target, pred_state):
        target_prob = F.softmax(pred_target, dim=-1)
        valid_mask = torch.matmul(target_prob, self.target_state_mask)
        pred_state = pred_state + (1 - valid_mask).clamp(min=1e-6).log()
        return pred_state

    def forward(self, pixel_features, text_features=None, multi_scale=False, state_text_features=None):
        """
        Args:
            pixel_features: (B, 128, H/4, W/4)
            text_features:  (16, text_dim) 鍙€?

        Returns:
            dict:
                pred_boxes:    (B, Q, 4)
                pred_target:  (B, Q, 16)
                pred_state:   (B, Q, 6)
                query_feats:  (B, Q, 128)
                aux_outputs:  list of dict (杈呭姪灞傞娴?
        """
        # V2: Handle multi-scale input from ChangeDecoder
        if multi_scale and isinstance(pixel_features, (list, tuple)):
            B = pixel_features[0].shape[0]
            scales = []
            for feat in pixel_features:
                scales.append(self.pos_enc(feat))
            pixel_features = pixel_features[0]
        else:
            B = pixel_features.shape[0]
            pixel_features = self.pos_enc(pixel_features)
            scales = self.fpn(pixel_features)


        # 2. 灞曞钩鎵€鏈夊昂搴?
        memories = []
        for scale_idx, feat in enumerate(scales):
            memories.append(self._flatten_scale(feat, scale_idx))
        memory = torch.cat(memories, dim=1)  # (B, total_HW, C)

        # 3. 鐢熸垚鏌ヨ
        queries = self.query_embedding(pixel_features.device)
        queries = queries.unsqueeze(0).expand(B, -1, -1)

        # 4. 鎵嬪姩閬嶅巻 Decoder 灞?(鑾峰彇涓棿杈撳嚭)
        instance_feats = queries
        aux_outputs = []

        for layer_idx, decoder_layer in enumerate(self.decoder_layers):
            instance_feats = decoder_layer(tgt=instance_feats, memory=memory)

            if layer_idx in self.aux_layer_indices:
                aux_i = self.aux_layer_indices.index(layer_idx)
                aux_normed = self.decoder_norm(instance_feats)
                aux_bbox = self.aux_heads[aux_i]['bbox'](aux_normed)
                aux_target = self.aux_heads[aux_i]['target'](aux_normed)
                aux_state = self.aux_heads[aux_i]['state'](aux_normed)
                aux_state = self._apply_state_mask(aux_target, aux_state)
                aux_outputs.append({
                    'pred_boxes': aux_bbox,
                    'pred_target': aux_target,
                    'pred_state': aux_state,
                })

        # 5. 鏈€缁堝眰
        instance_feats = self.decoder_norm(instance_feats)
        pred_boxes = self.bbox_head(instance_feats)
        pred_target = self.target_head(instance_feats)

        # V3: change attention enhances state classification
        change_ctx, _ = self.change_attn(
            query=instance_feats, key=memory, value=memory
        )
        gate = self.change_gate(torch.cat([instance_feats, change_ctx], dim=-1))
        fused = torch.cat([instance_feats, gate * change_ctx], dim=-1)
        pred_state = self.state_head_v3(fused)
        pred_state = self._apply_state_mask(pred_target, pred_state)

        # 6. CLIP 鏂囨湰澧炲己
        if text_features is not None and text_features.shape[0] == self.num_targets:
            inst_norm = F.normalize(instance_feats, dim=-1)
            txt_norm = F.normalize(text_features, dim=-1)
            cosine_sim = torch.matmul(inst_norm, txt_norm.t())
            pred_target = (pred_target + cosine_sim * 2.0) / 3.0

        return {
            'pred_boxes': pred_boxes,
            'pred_target': pred_target,
            'pred_state': pred_state,
            'query_feats': instance_feats,
            'aux_outputs': aux_outputs,
        }

    @property
    def total_queries(self):
        return 3 * self.num_queries_per_scale



