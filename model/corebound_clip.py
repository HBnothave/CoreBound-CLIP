"""CoreBound-CLIP: full model combining CeSePro and HiBoDec on a frozen CLIP backbone."""

import torch
import torch.nn as nn

from .clip_backbone import CLIPBackbone
from .cesepro import CeSePro
from .hibodec import HiBoDec


class CoreBoundCLIP(nn.Module):
    def __init__(self, class_names, clip_checkpoint=None, vit_dim=768,
                 text_dim=512, proj_dim=256, num_classes=21, device="cuda"):
        super().__init__()
        self.class_names = class_names
        self.clip = CLIPBackbone(checkpoint_path=clip_checkpoint, device=device)
        self.cesepro = CeSePro(dim=text_dim)
        self.hibodec = HiBoDec(vit_dim=vit_dim, proj_dim=proj_dim, num_classes=num_classes)

        # Precompute frozen CLIP text embeddings for all classes (recomputed
        # at init since the text encoder is frozen).
        prompts = [f"a photo of a {c}" for c in class_names]
        with torch.no_grad():
            self.register_buffer("T_txt", self.clip.encode_text(prompts).float())

    def forward(self, image):
        """
        image: (B, 3, H, W) CLIP-normalized input.

        Returns dict with:
          cams        (B, C, h, w)   CeSePro activation maps
          seg_pred    (B, num_classes, H, W) HiBoDec segmentation logits
          edge_pred   (B, 1, H, W)   predicted edge map
          s_core/s_ext, T_core/T_ext, alpha : for CeSePro losses
        """
        v_img, F_vis, layer_tokens, (h, w) = self.clip.encode_image(image)

        cese_out = self.cesepro(self.T_txt.to(image.dtype), v_img.float(), F_vis.float())

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
        """Only CeSePro and HiBoDec are trained; CLIP stays frozen."""
        params = list(self.cesepro.parameters()) + list(self.hibodec.parameters())
        return params
