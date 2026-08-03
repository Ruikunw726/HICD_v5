import torch
import torch.nn as nn
import torch.nn.functional as F

from HICD_v5.changedetection.models.Mamba_backbone import Backbone_VSSM
from HICD_v5.classification.models.vmamba import LayerNorm2d
from HICD_v5.changedetection.models.ChangeDecoder import ChangeDecoder
from HICD_v5.changedetection.models.CLIPTextEncoder import CLIPTextEncoder
from HICD_v5.changedetection.models.CrossAttentionFusion import TextVisualCrossAttention
from HICD_v5.changedetection.models.HierarchicalInstanceHead import HierarchicalInstanceHead
from HICD_v5.changedetection.models.TaskAdapter import TaskAdapter
from HICD_v5.changedetection.models.SemanticSegmentationHead import SemanticSegmentationHead
from HICD_v5.changedetection.models.class_mapping import (
    DatasetConfig, TARGET_NAMES, STATE_NAMES, NUM_TARGETS, NUM_STATES,
)


class HICD_v5(nn.Module):
    """HICD V5: 双分支变化检测。
    
    分支 A（实例检测）：小/中目标 — Building, Aircraft, Tank, Vessel, Crater
    分支 B（语义分割）：大/线性目标 — Runway, Taxiway, Apron, Highway
    
    共享编码器和 ChangeDecoder，通过 Task-Specific Adapter 分流。
    """
    def __init__(self, pretrained, num_queries_per_scale=17,
                 dataset_config=None, clip_mode='both',
                 clip_model="ViT-B-16", clip_weights_path=None,
                 **kwargs):
        super().__init__()
        self.dataset_config = dataset_config
        self.clip_mode = clip_mode

        # ── 1. Siamese VSSM Backbone ──
        self.encoder = Backbone_VSSM(
            out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs
        )

        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)

        self.channel_first = self.encoder.channel_first
        norm_layer = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)
        clean_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']
        }

        # ── 2. ChangeDecoder ──
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        # ── 3. CLIP Text Encoder ──
        self.clip_text_encoder = CLIPTextEncoder(
            clip_model=clip_model, embed_dim=128,
            freeze=True, pretrained_path=clip_weights_path
        )

        # ── 4. Cross-Attention ──
        self.cross_attn = TextVisualCrossAttention(
            visual_dim=128, text_dim=128, num_heads=8, dropout=0.1
        )

        # ── 5. Task-Specific Adapters ──
        self.adapter_instance = TaskAdapter(dim=128)
        self.adapter_semantic = TaskAdapter(dim=128)

        # ── 6A. Instance Detection Head ──
        num_targets = dataset_config.num_targets if dataset_config else NUM_TARGETS
        num_states = dataset_config.num_states if dataset_config else NUM_STATES
        self.instance_head = HierarchicalInstanceHead(
            visual_dim=128,
            num_queries_per_scale=num_queries_per_scale,
            dataset_config=dataset_config,
            num_targets=num_targets,
            num_states=num_states,
            num_decoder_layers=6,
            nhead=8,
        )

        # ── 6B. Semantic Segmentation Head ──
        # 语义分支只负责路由到 semantic 的类别
        if dataset_config and hasattr(dataset_config, 'branch_routing'):
            semantic_targets = [t for t, b in dataset_config.branch_routing.items() if b == 'semantic']
            num_semantic_targets = len(semantic_targets) if semantic_targets else 4
        else:
            num_semantic_targets = 4  # 默认: Runway, Taxiway, Apron, Highway

        self.semantic_head = SemanticSegmentationHead(
            in_dim=128,
            num_target_classes=num_semantic_targets + 1,  # +1 for background
            num_state_classes=num_states,
        )

    def forward(self, pre_data, post_data):
        """
        Args:
            pre_data:  (B, 3, H, W)
            post_data: (B, 3, H, W)
        Returns:
            dict with instance_outputs, target_map, state_map
        """
        # 1. Backbone
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # 2. ChangeDecoder
        pixel_features = self.decoder(pre_features, post_features)  # (p1, p2, p3)
        p1, p2, p3 = pixel_features

        # 3. CLIP
        text_features = self.clip_text_encoder(self.dataset_config.clip_text_prompts)

        # 4. Cross-Attention
        enhanced_p1 = self.cross_attn(p1, text_features)
        enhanced_p2 = self.cross_attn(p2, text_features)
        enhanced_p3 = self.cross_attn(p3, text_features)

        # 5. Task-Specific Adapters
        f_inst = [self.adapter_instance(enhanced_p1),
                  self.adapter_instance(enhanced_p2),
                  self.adapter_instance(enhanced_p3)]
        f_sem = [self.adapter_semantic(enhanced_p1),
                 self.adapter_semantic(enhanced_p2),
                 self.adapter_semantic(enhanced_p3)]

        # 6A. Instance Detection
        instance_outputs = self.instance_head(f_inst, text_features, multi_scale=True)

        # 6B. Semantic Segmentation
        target_map, state_map = self.semantic_head(f_sem[0], f_sem[1], f_sem[2])

        return {
            'instance_outputs': instance_outputs,
            'target_map': target_map,
            'state_map': state_map,
        }

    @torch.no_grad()
    def inference(self, pre_data, post_data, confidence_threshold=0.3):
        """推理接口。"""
        outputs = self.forward(pre_data, post_data)
        # TODO: 合并实例分支和语义分支的推理结果
        return outputs
