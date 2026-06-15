"""Hierarchical Boundary-Aware Decoder (HiBoDec)."""

import torch
import torch.nn as nn


class LayerAggregator(nn.Module):
    """Agg(.): learned weighted average of patch tokens within a group,
    reshaped to a 2D spatial feature map (Sec. III-C, architecture details)."""

    def __init__(self, dim, num_layers=3):
        super().__init__()
        self.weight_proj = nn.ModuleList(
            [nn.Linear(dim, 1) for _ in range(num_layers)]
        )

    def forward(self, layer_tokens, h, w):
        """layer_tokens: list of (B, N, d) tensors (patch tokens, CLS removed)."""
        weighted = []
        for tokens, proj in zip(layer_tokens, self.weight_proj):
            score = torch.sigmoid(proj(tokens))  # (B, N, 1)
            weighted.append(score * tokens)
        agg = torch.stack(weighted, dim=0).sum(0)  # (B, N, d)
        b, n, d = agg.shape
        agg = agg.transpose(1, 2).reshape(b, d, h, w)
        return agg


class StageProjection(nn.Module):
    """phi_k(.): 1x1 conv projecting aggregated features to 256 channels."""

    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


class EdgePredictor(nn.Module):
    """F_edge(.): two 3x3 conv (BN+ReLU) + 1x1 output conv -> edge map (Eq. 7)."""

    def __init__(self, in_dim=256, mid_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 3, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, 1, kernel_size=1),
        )

    def forward(self, F3):
        return torch.sigmoid(self.net(F3))


class SpatialGate(nn.Module):
    """MLP (1x1 conv) + sigmoid producing gate G (Eq. 8)."""

    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, F_ctx):
        return torch.sigmoid(self.proj(F_ctx))


class SegmentationHead(nn.Module):
    """Lightweight head: two 3x3 conv layers + bilinear upsampling."""

    def __init__(self, in_dim=256, mid_dim=256, num_classes=21):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 3, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(mid_dim, num_classes, kernel_size=1)

    def forward(self, x, out_size):
        x = self.conv(x)
        x = self.classifier(x)
        return nn.functional.interpolate(
            x, size=out_size, mode='bilinear', align_corners=False
        )


class HiBoDec(nn.Module):
    """Hierarchical Boundary-Aware Decoder.

    Expects 12 layers of ViT-B/16 patch tokens grouped into 4 stages:
        group1: layers 1-3  -> F3  (boundary/texture cues)
        group2: layers 4-6  -> F6  (intermediate detail)
        group3: layers 7-9  -> F9  (deep semantics)
        group4: layers 10-12 -> F11 (deep semantics)
    """

    def __init__(self, vit_dim=768, proj_dim=256, num_classes=21):
        super().__init__()
        self.agg1 = LayerAggregator(vit_dim)
        self.agg2 = LayerAggregator(vit_dim)
        self.agg3 = LayerAggregator(vit_dim)
        self.agg4 = LayerAggregator(vit_dim)

        self.phi1 = StageProjection(vit_dim, proj_dim)
        self.phi2 = StageProjection(vit_dim, proj_dim)
        self.phi3 = StageProjection(vit_dim, proj_dim)
        self.phi4 = StageProjection(vit_dim, proj_dim)

        self.edge_predictor = EdgePredictor(proj_dim)

        # Eq. 5: F_ctx = Conv(F9 (+) F11) -- concat then 1x1 conv fusion
        self.ctx_fuse = nn.Conv2d(proj_dim * 2, proj_dim, kernel_size=1)
        self.spatial_gate = SpatialGate(proj_dim)

        self.beta = nn.Parameter(torch.tensor(1.0))  # learnable scaling (Eq. 9)

        self.seg_head = SegmentationHead(proj_dim, proj_dim, num_classes)

    def forward(self, layer_tokens, h, w, out_size):
        """
        layer_tokens: list of 12 tensors, each (B, N, vit_dim), patch tokens
                      (CLS token removed) from ViT layers 1..12.
        h, w: spatial grid size (e.g. 32x32 for 512 input with patch 16).
        out_size: (H_img, W_img) target output resolution.
        """
        g1 = layer_tokens[0:3]
        g2 = layer_tokens[3:6]
        g3 = layer_tokens[6:9]
        g4 = layer_tokens[9:12]

        F3 = self.phi1(self.agg1(g1, h, w))
        F6 = self.phi2(self.agg2(g2, h, w))
        F9 = self.phi3(self.agg3(g3, h, w))
        F11 = self.phi4(self.agg4(g4, h, w))

        # Eq. 7: class-agnostic edge map from shallow feature F3
        m_edge = self.edge_predictor(F3)

        # Eq. 5: contextual feature from deep semantics
        F_ctx = self.ctx_fuse(torch.cat([F9, F11], dim=1))

        # Eq. 8: spatial gate
        G = self.spatial_gate(F_ctx)

        # Eq. 6: purified semantic feature
        F_clean = F_ctx + G * F6

        # Eq. 9: incorporate boundary prior
        F_out = F_clean * (1 + self.beta * m_edge)

        m_pred = self.seg_head(F_out, out_size)
        m_edge_up = nn.functional.interpolate(
            m_edge, size=out_size, mode='bilinear', align_corners=False
        )

        return {'pred': m_pred, 'edge': m_edge_up}


def loss_edge(pred_edge, target_edge):
    """Binary cross-entropy edge loss (L_edge)."""
    return nn.functional.binary_cross_entropy(pred_edge, target_edge)
