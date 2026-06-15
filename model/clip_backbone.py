"""Frozen CLIP ViT-B/16 backbone wrapper.

Exposes:
  - global image [CLS] embedding v_img
  - dense visual feature map F (for CeSePro CAM, Eq. 5)
  - per-layer patch tokens (for HiBoDec, layers 1-12)
  - text embeddings T_txt for a list of class prompts

Built on top of the official OpenAI CLIP repo (or open_clip). Place the
ViT-B/16 checkpoint as instructed in the README.
"""

import torch
import torch.nn as nn
import clip  # https://github.com/openai/CLIP


class CLIPBackbone(nn.Module):
    def __init__(self, model_name="ViT-B/16", checkpoint_path=None, device="cuda"):
        super().__init__()
        # Load from local checkpoint if provided, else CLIP's default cache.
        self.model, _ = clip.load(
            checkpoint_path if checkpoint_path else model_name,
            device=device,
            jit=False,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.visual = self.model.visual
        self.patch_size = self.visual.conv1.kernel_size[0]
        self._layer_outputs = []
        self._register_hooks()

    def _register_hooks(self):
        # ViT-B/16 transformer has 12 resblocks.
        for blk in self.visual.transformer.resblocks:
            blk.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._layer_outputs.append(out)

    @torch.no_grad()
    def encode_image(self, image):
        """
        image: (B, 3, H, W), already CLIP-normalized.

        Returns:
            v_img: (B, d) global [CLS] embedding (post visual.ln_post + proj).
            F_vis: (B, h, w, d_token) dense patch feature map (pre-projection,
                   token width matches the ViT width, e.g. 768 for ViT-B/16).
            layer_tokens: list of 12 tensors (B, N, d_token), patch tokens
                          (CLS removed) for each transformer layer.
            grid_hw: (h, w) patch grid size.
        """
        self._layer_outputs = []
        x = self.visual.conv1(image)  # (B, width, h, w)
        B, C, h, w = x.shape
        x = x.reshape(B, C, h * w).permute(0, 2, 1)  # (B, N, width)
        cls = self.visual.class_embedding.to(x.dtype) + torch.zeros(
            B, 1, C, dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND for transformer
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        cls_out = self.visual.ln_post(x[:, 0, :])
        v_img = cls_out @ self.visual.proj  # (B, d)

        patch_tokens_final = x[:, 1:, :]  # (B, N, width)
        F_vis_tok = self.visual.ln_post(patch_tokens_final)
        F_vis = (F_vis_tok @ self.visual.proj).reshape(B, h, w, -1)  # (B, h, w, d)

        # Per-layer patch tokens (LND -> NLD, drop CLS)
        layer_tokens = []
        for out in self._layer_outputs:
            out_nld = out.permute(1, 0, 2)  # (B, N+1, width)
            layer_tokens.append(out_nld[:, 1:, :])

        return v_img, F_vis, layer_tokens, (h, w)

    @torch.no_grad()
    def encode_text(self, prompts):
        """prompts: list[str]. Returns (C, d) text embeddings."""
        tokens = clip.tokenize(prompts).to(next(self.model.parameters()).device)
        return self.model.encode_text(tokens)
