"""Core-to-Extent Semantic Prompting (CeSePro)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """Two-layer MLP expert transformation f_core / f_ext (Eq. 1)."""

    def __init__(self, dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class GatingMLP(nn.Module):
    """Image-conditioned gating coefficient alpha (Eq. 4)."""

    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, v_img):
        return torch.sigmoid(self.fc(v_img)).squeeze(-1)  # (B,)


class CeSePro(nn.Module):
    """Dual-expert prompt learning with image-conditioned fusion.

    Inputs:
        T_txt:  (C, d) frozen CLIP text embeddings, one per class.
        v_img:  (B, d) global image [CLS] embedding from frozen CLIP.
        F_vis:  (B, H, W, d) dense visual feature map from frozen CLIP.

    Outputs:
        cams:     (B, C, H, W) initial activation maps (Eq. 5).
        s_core:   (B, C) core-expert classification logits (scaled).
        s_ext:    (B, C) extent-expert classification logits (scaled).
        T_core/T_ext: (C, d) expert prototypes for divergence loss.
    """

    def __init__(self, dim=512, hidden=256, temperature=0.07):
        super().__init__()
        self.f_core = ExpertMLP(dim, hidden)
        self.f_ext = ExpertMLP(dim, hidden)
        self.gate = GatingMLP(dim)
        self.tau = 1.0 / temperature

    @staticmethod
    def _l2norm(x, dim=-1):
        return F.normalize(x, dim=dim)

    def forward(self, T_txt, v_img, F_vis):
        # Eq. 1: dual expert text prototypes
        T_core = self.f_core(T_txt)  # (C, d)
        T_ext = self.f_ext(T_txt)    # (C, d)

        # Eq. 2: temperature-scaled cosine similarity for image-level labels
        v_n = self._l2norm(v_img)            # (B, d)
        core_n = self._l2norm(T_core)        # (C, d)
        ext_n = self._l2norm(T_ext)          # (C, d)
        s_core = self.tau * (v_n @ core_n.t())  # (B, C)
        s_ext = self.tau * (v_n @ ext_n.t())    # (B, C)

        # Eq. 4: image-conditioned fusion coefficient alpha (shared across classes)
        alpha = self.gate(v_img)  # (B,)

        # Eq. 3: fuse expert prototypes per image
        # T_final: (B, C, d)
        T_final = (alpha.view(-1, 1, 1) * T_core.unsqueeze(0)
                   + (1 - alpha.view(-1, 1, 1)) * T_ext.unsqueeze(0))
        T_final_n = self._l2norm(T_final, dim=-1)

        # Eq. 5: dense cosine-similarity activation maps
        F_n = self._l2norm(F_vis, dim=-1)  # (B, H, W, d)
        cams = torch.einsum('bhwd,bcd->bchw', F_n, T_final_n)

        return {
            'cams': cams,
            's_core': s_core,
            's_ext': s_ext,
            'T_core': T_core,
            'T_ext': T_ext,
            'alpha': alpha,
        }


def loss_inj(s_core, s_ext, labels):
    """Eq. 3: image-level BCE supervision for both experts.

    labels: (B, C) in {0,1}.
    """
    bce = nn.functional.binary_cross_entropy_with_logits
    return bce(s_core, labels.float()) + bce(s_ext, labels.float())


def loss_div(T_core, T_ext):
    """Eq. 6: divergence regularization between core and extent prototypes."""
    sim = F.cosine_similarity(T_core, T_ext, dim=-1)
    return sim.sum()
