import torch
import torch.nn as nn


class CLIPTextEncoder(nn.Module):
    """
    CLIP ViT-B/16 文本编码器，支持两阶段训练:
      - 阶段1 (freeze=True): 全部冻结，用 torch.no_grad 推理
      - 阶段2 (调用 unfreeze()): 解冻最后 N 层 + text_projection，参与梯度更新
    """
    def __init__(self, clip_model="ViT-B-16", embed_dim=512, freeze=True,
                 pretrained_path=None):
        super().__init__()

        import open_clip
        import logging
        logging.disable(logging.WARNING)

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained=""
            )
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            self.clip_model = model
        else:
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="openai"
            )

        logging.disable(logging.NOTSET)

        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.text_projection = nn.Linear(512, embed_dim)

        self._frozen = True
        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.text_projection.requires_grad_(True)  # projection 始终可训练

    def unfreeze(self, n_layers=2):
        """解冻最后 n_layers 个 transformer 层 + text_projection"""
        for param in self.clip_model.parameters():
            param.requires_grad = False  # 先全部冻结

        if hasattr(self.clip_model, "transformer"):
            layers = list(self.clip_model.transformer.resblocks)
            for layer in layers[-n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.text_projection.requires_grad_(True)
        self._frozen = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[CLIP] Unfroze last {n_layers} layers. "
              f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    @property
    def is_frozen(self):
        return self._frozen

    def forward(self, text_list):
        tokens = self.tokenizer(text_list).to(next(self.parameters()).device)
        has_trainable = any(p.requires_grad for p in self.clip_model.parameters())
        if has_trainable:
            text_features = self.clip_model.encode_text(tokens).float()
        else:
            with torch.no_grad():
                text_features = self.clip_model.encode_text(tokens).float()
        text_features = self.text_projection(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
