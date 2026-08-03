# -*- coding: utf-8 -*-
from HICD.classification.models.vmamba import VSSM, LayerNorm2d

import torch
import torch.nn as nn


class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', **kwargs):
        # norm_layer='ln'
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Skip VSSM's _load_from_state_dict which renames norm->classifier.norm and head->classifier.head
        # We handle key remapping ourselves in _remap_keys
        nn.Module._load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.load_pretrained(pretrained)

    def _remap_keys(self, state_dict):
        """Remap old-style VSSM checkpoint keys to match current model.
        
        Old checkpoint: self_attention.*, ln_1.*
        Current model:  op.*, norm.*
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            # Remove classifier head keys (model deleted self.classifier)
            if k.startswith('head.'):
                continue
            # Remap block internals: self_attention -> op
            if '.self_attention.' in new_k:
                new_k = new_k.replace('.self_attention.', '.op.')
            # Remap block norm: ln_1 -> norm
            if '.ln_1.' in new_k:
                new_k = new_k.replace('.ln_1.', '.norm.')
            new_state_dict[new_k] = v
        return new_state_dict

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            
            if isinstance(_ckpt, dict) and key in _ckpt:
                raw_state = _ckpt[key]
            elif isinstance(_ckpt, dict) and 'state_dict' in _ckpt:
                raw_state = _ckpt['state_dict']
            else:
                raw_state = _ckpt
            
            # Remap keys for compatibility
            remapped_state = self._remap_keys(raw_state)
            
            incompatibleKeys = self.load_state_dict(remapped_state, strict=False)
            print(f"Pretrained weights loaded with {len(incompatibleKeys.missing_keys)} missing, "
                  f"{len(incompatibleKeys.unexpected_keys)} unexpected keys")
            if incompatibleKeys.missing_keys:
                print(f"  Missing (first 5): {incompatibleKeys.missing_keys[:5]}")
            if incompatibleKeys.unexpected_keys:
                print(f"  Unexpected (first 5): {incompatibleKeys.unexpected_keys[:5]}")
        except Exception as e:
            print(f"Failed loading checkpoint from {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs
