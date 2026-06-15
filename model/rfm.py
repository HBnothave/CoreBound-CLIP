"""Residual Fusion Module (RFM), retained from WeCLIP as an auxiliary
CAM-enhancement branch (Sec. III-A, Overview of CoreBound-CLIP):

  "we retain the residual fusion module (RFM) from WeCLIP as an auxiliary
   CAM-enhancement branch."

WeCLIP's RFM refines the frozen CLIP dense feature map used for the CAM
similarity computation by (1) re-weighting and fusing intermediate
transformer-layer features through a small set of learnable 1x1
convolutions, and (2) adding the result back to the final-layer feature map
as a residual correction:

    F_rfm = F_final + Conv_fuse( sum_l w_l * Conv_l(V_l) )

where V_l are the intermediate patch-token feature maps (reshaped to 2D),
Conv_l are per-layer 1x1 projections to a common channel width, w_l are
learned per-layer scalar weights (softmax-normalized), and Conv_fuse
projects back to the CLIP embedding dimension d so the result can be used
directly in Eq. 5 (CeSePro CAM similarity) in place of the raw final-layer
feature map.

Only RFM's parameters are trained; the CLIP backbone itself remains frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFusionModule(nn.Module):
    """WeCLIP-style residual fusion of multi-layer CLIP features.

    Args:
        token_dim: width of ViT patch tokens (e.g. 768 for ViT-B/16).
        embed_dim: CLIP joint embedding dimension d (e.g. 512), matching the
                   dense feature map F used in Eq. 5.
        proj_dim: shared channel width for per-layer projections.
        layer_indices: which transformer-layer outputs (0-indexed) to fuse;
                       WeCLIP uses the deeper layers, here the last 6 of 12.
    """

    def __init__(self, token_dim=768, embed_dim=512, proj_dim=256,
                 layer_indices=(6, 7, 8, 9, 10, 11)):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.layer_projs = nn.ModuleList([
            nn.Conv2d(token_dim, proj_dim, kernel_size=1)
            for _ in self.layer_indices
        ])
        # Learnable per-layer fusion weights (softmax-normalized).
        self.layer_weights = nn.Parameter(torch.ones(len(self.layer_indices)))
        self.fuse = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, embed_dim, kernel_size=1),
        )

    def forward(self, layer_tokens, F_final, h, w):
        """
        Args:
            layer_tokens: list of 12 tensors (B, N, token_dim), patch tokens
                          (CLS removed) from each ViT transformer layer.
            F_final: (B, h, w, embed_dim) final-layer dense feature map,
                     used as the residual base (the F in Eq. 5).
            h, w: patch grid size.

        Returns:
            F_rfm: (B, h, w, embed_dim) CAM-enhancement feature map, to be
                   used in place of F_final for the CeSePro CAM similarity
                   (Eq. 5).
        """
        weights = F.softmax(self.layer_weights, dim=0)

        fused = None
        for idx, proj, wgt in zip(self.layer_indices, self.layer_projs, weights):
            tok = layer_tokens[idx]  # (B, N, token_dim)
            B, N, C = tok.shape
            feat = tok.transpose(1, 2).reshape(B, C, h, w)  # (B, C, h, w)
            feat = proj(feat)  # (B, proj_dim, h, w)
            fused = feat * wgt if fused is None else fused + feat * wgt

        residual = self.fuse(fused)  # (B, embed_dim, h, w)
        residual = residual.permute(0, 2, 3, 1)  # (B, h, w, embed_dim)

        return F_final + residual
