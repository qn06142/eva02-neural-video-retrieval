"""SigLIP with an optional permutation-equivariant residual frame set encoder."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from transformers import SiglipModel


class ResidualFrameMixer(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=256, layers=2, heads=4):
        super().__init__()
        if min(feature_dim, hidden_dim, layers, heads) < 1 or hidden_dim % heads:
            raise ValueError("Positive dimensions required; hidden dimension must divide into attention heads")
        self.input_norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, hidden_dim)
        block = nn.TransformerEncoderLayer(hidden_dim, heads, 4*hidden_dim,
            dropout=0., activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.up = nn.Linear(hidden_dim, feature_dim)
        # A zero gate preserves baseline behavior; random branch weights let the
        # gate learn on the first update, then the rest of the branch can learn.
        self.gate = nn.Parameter(torch.zeros(()))
        for param in self.encoder.parameters():
            if param.ndim > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, features):
        if features.ndim != 3 or features.shape[1] < 1 or features.shape[-1] != self.down.in_features:
            raise ValueError("Expected frame features shaped (bags, frames, feature_dim)")
        features = F.normalize(features.float(), dim=-1)
        branch_input = features.to(dtype=self.down.weight.dtype)
        residual = self.up(self.output_norm(self.encoder(self.down(self.input_norm(branch_input)))))
        return F.normalize(features + self.gate.tanh() * residual.float(), dim=-1)


class SiglipBagModel(SiglipModel):
    def __init__(self, config):
        super().__init__(config)
        spec = getattr(config, "frame_mixer", None)
        self.frame_mixer = ResidualFrameMixer(**spec) if spec else None
        self._backbone_frozen = False

    def enable_frame_mixer(self, hidden_dim=256, layers=2, heads=4):
        spec = dict(feature_dim=self.config.vision_config.hidden_size,
                    hidden_dim=hidden_dim, layers=layers, heads=heads)
        if self.frame_mixer is not None:
            if self.config.frame_mixer != spec:
                raise ValueError("Requested frame mixer differs from the saved architecture")
            return
        self.frame_mixer = ResidualFrameMixer(**spec).to(self.logit_scale.device)
        self.config.frame_mixer = spec
        self.config.architectures = ["SiglipBagModel"]

    def mix_frame_features(self, features):
        return self.frame_mixer(features) if self.frame_mixer is not None else features

    def freeze_backbone(self):
        if self.frame_mixer is None:
            raise ValueError("A frame mixer is required when freezing the backbone")
        self.requires_grad_(False)
        self.frame_mixer.requires_grad_(True)
        self._backbone_frozen = True
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "_backbone_frozen", False):
            self.vision_model.eval()
            self.text_model.eval()
        return self
