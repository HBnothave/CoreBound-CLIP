"""CoreBound-CLIP: full model combining CeSePro and HiBoDec on a frozen CLIP backbone.

The frozen CLIP image encoder also feeds the Residual Fusion Module (RFM,
retained from WeCLIP) as an auxiliary CAM-enhancement branch (Sec. III-A):

    "we retain the residual fusion module (RFM) from WeCLIP as an auxiliary
     CAM-enhancement branch."

RFM refines the dense feature map used in CeSePro's CAM similarity (Eq. 5)
by fusing intermediate-layer CLIP features as a residual correction on top
of the final-layer feature map.
"""

import torch
import torch.nn as nn

from .clip_backbone import CLIPBackbone
from .cesepro import CeSePro
from .hibodec import HiBoDec
from .rfm import ResidualFusionModule


class CoreBoundCLIP(nn.Module):
    def __init__(self, class_names, clip_checkpoint=None, vit_dim=768,
                 text_dim=512, proj_dim=256, num_classes=21, device="cuda",
                 rfm_layer_indices=(6, 7, 8, 9, 10, 11)):
        super().__init__()
        self.class_names = class_names
        self.clip = CLIPBackbone(checkpoint_path=clip_checkpoint, device=device)
        self.cesepro = CeSePro(dim=text_dim)
        self.hibodec = HiBoDec(vit_dim=vit_dim, proj_dim=proj_dim, num_classes=num_classes)

        # WeCLIP's residual fusion module, retained as an auxiliary
        # CAM-enhancement branch (Sec. III-A).
        self.rfm = ResidualFusionModule(
            token_dim=vit_dim, embed_dim=text_dim, proj_dim=proj_dim,
            layer_indices=rfm_layer_indices,
        )

        # Precompute frozen CLIP text embeddings for all classes (recomputed
        # at init since the text encoder is frozen).
        prompts = [f"a photo of a {c}" for c in class_names]
        with torch.no_grad():
            self.register_buffer("T_txt", self.clip.encode_text(prompts).float())

    def forward(self, image):
        """
        image: (B, 3, H, W) CLIP-normalized input.

        Returns dict with:
          cams        (B, C, h, w)   CeSePro activation maps (RFM-enhanced)
          seg_pred    (B, num_classes, H, W) HiBoDec segmentation logits
          edge_pred   (B, 1, H, W)   predicted edge map
          s_core/s_ext, T_core/T_ext, alpha : for CeSePro losses
        """
        v_img, F_vis, layer_tokens, (h, w) = self.clip.encode_image(image)

        # RFM: residual-fuse intermediate CLIP layer features into the dense
        # feature map used for CAM similarity (Eq. 5), as an auxiliary
        # CAM-enhancement branch retained from WeCLIP.
        F_rfm = self.rfm(layer_tokens, F_vis.float(), h, w)

        cese_out = self.cesepro(self.T_txt.to(image.dtype), v_img.float(), F_rfm)

        out_size = image.shape[-2:]
        hibo_out = self.hibodec(layer_tokens, h, w, out_size)

        return {
            'cams': cese_out['cams'],
            'seg_pred': hibo_out['pred'],
            'edge_pred': hibo_out['edge'],
            's_core': cese_out['s_core'],
            's_ext': cese_out['s_ext'],
            'T_core': cese_out['T_core'],
            'T_ext': cese_out['T_ext'],
            'alpha': cese_out['alpha'],
        }

    def trainable_parameters(self):
        """Only RFM, CeSePro, and HiBoDec are trained; CLIP stays frozen."""
        params = (list(self.rfm.parameters())
                  + list(self.cesepro.parameters())
                  + list(self.hibodec.parameters()))
        return params
