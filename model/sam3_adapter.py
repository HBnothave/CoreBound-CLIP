"""Adapter that builds a concept-prompted mask predictor for CoReSAM3.

CoReSAM3 (model/coresam3.py) calls `predictor.predict(image, box, text)` and
expects a binary (H, W) mask. This module provides:

  1. `Sam3Predictor` — adapter for the official SAM3 release. SAM3 supports
     joint box + text ("concept") prompts (Eq. 10-11 in the paper). Adapt
     `_load_model` / `_run_inference` below to match the installed SAM3
     package's actual API once available.

  2. `Sam2BoxPredictor` — fallback using SAM (1/2)'s box-prompted predictor
     plus a CLIP-based text/box consistency check. SAM (1/2) does not accept
     text prompts directly, so the "concept" prompt is enforced by scoring
     the resulting mask crop against the text with CLIP and rejecting
     low-similarity masks. This lets the pipeline run end-to-end with widely
     available checkpoints (segment-anything / SAM2) while approximating
     CoReSAM3's concept-guided behavior; swap in `Sam3Predictor` for the full
     paper setup once SAM3 weights/code are available.

`build_sam3_predictor(checkpoint_path, device)` picks whichever backend is
importable and returns a ready-to-use predictor object.
"""

import numpy as np
import torch


class Sam3Predictor:
    """Adapter for the official SAM3 concept-prompted predictor.

    Expected SAM3 usage (adapt to the actual installed API):

        from sam3.build_sam import build_sam3
        from sam3.predictor import SAM3Predictor

        model = build_sam3(checkpoint=checkpoint_path)
        predictor = SAM3Predictor(model)
        predictor.set_image(image)
        masks = predictor.predict(box=box, text=text)
    """

    def __init__(self, checkpoint_path, device="cuda"):
        from sam3.build_sam import build_sam3          # noqa: F401
        from sam3.predictor import SAM3Predictor as _SAM3Predictor

        model = build_sam3(checkpoint=checkpoint_path)
        model.to(device)
        self._predictor = _SAM3Predictor(model)
        self._device = device
        self._cached_image_id = None

    def predict(self, image, box, text):
        """
        image: (H, W, 3) uint8 RGB numpy array.
        box: [x1, y1, x2, y2] pixel coordinates.
        text: concept prompt string, e.g. "a photo of a dog".

        Returns: (H, W) binary mask (uint8).
        """
        img_id = id(image)
        if img_id != self._cached_image_id:
            self._predictor.set_image(image)
            self._cached_image_id = img_id

        masks, scores, _ = self._predictor.predict(
            box=np.array(box), text=text, multimask_output=False,
        )
        mask = masks[0] if masks.ndim == 3 else masks
        return mask.astype(np.uint8)


class Sam2BoxPredictor:
    """Fallback: SAM(1/2) box-prompted predictor + CLIP concept check.

    Since SAM1/2 lack native text prompts, the "concept" half of the
    box+text prompt (Eq. 10-11) is approximated by computing CLIP similarity
    between the masked image crop and the text prompt, and rejecting the
    mask (returning all-zeros) if similarity falls below `clip_thresh`.
    This keeps the refinement "concept-guided" in spirit while remaining
    runnable with public SAM checkpoints.
    """

    def __init__(self, checkpoint_path, device="cuda", model_type="vit_b",
                 clip_thresh=0.20, clip_model_name="ViT-B/16"):
        from segment_anything import sam_model_registry, SamPredictor
        import clip

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device)
        self._predictor = SamPredictor(sam)

        self._clip_model, self._clip_preprocess = clip.load(clip_model_name, device=device)
        self._clip_model.eval()
        self._device = device
        self.clip_thresh = clip_thresh
        self._cached_image_id = None

    def predict(self, image, box, text):
        img_id = id(image)
        if img_id != self._cached_image_id:
            self._predictor.set_image(image)
            self._cached_image_id = img_id

        box_arr = np.array(box)
        masks, scores, _ = self._predictor.predict(box=box_arr, multimask_output=False)
        mask = masks[0].astype(np.uint8)

        if mask.sum() == 0:
            return mask

        if self._concept_score(image, mask, text) < self.clip_thresh:
            return np.zeros_like(mask)
        return mask

    @torch.no_grad()
    def _concept_score(self, image, mask, text):
        from PIL import Image as PILImage

        ys, xs = np.where(mask > 0)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        crop = image[y1:y2 + 1, x1:x2 + 1]
        crop_masked = crop.copy()
        crop_mask = mask[y1:y2 + 1, x1:x2 + 1]
        crop_masked[crop_mask == 0] = 0

        pil_img = PILImage.fromarray(crop_masked)
        img_t = self._clip_preprocess(pil_img).unsqueeze(0).to(self._device)
        text_t = clip_tokenize([text]).to(self._device)

        img_feat = self._clip_model.encode_image(img_t)
        txt_feat = self._clip_model.encode_text(text_t)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        return (img_feat @ txt_feat.t()).item()


def clip_tokenize(texts):
    import clip
    return clip.tokenize(texts)


def build_sam3_predictor(checkpoint_path, device="cuda", model_type="vit_b"):
    """Return the best-available concept-prompted predictor.

    Tries the official SAM3 package first; falls back to SAM(1/2) box
    prompting + CLIP concept scoring if SAM3 is not installed.

    `checkpoint_path` should point at the SAM3 checkpoint when using
    `Sam3Predictor`, or the SAM/SAM2 checkpoint when falling back to
    `Sam2BoxPredictor`. `model_type` selects the SAM backbone for the
    fallback (e.g. 'vit_b', 'vit_l', 'vit_h').
    """
    try:
        return Sam3Predictor(checkpoint_path, device=device)
    except ImportError:
        try:
            return Sam2BoxPredictor(checkpoint_path, device=device, model_type=model_type)
        except ImportError as e:
            raise ImportError(
                "Neither the SAM3 package nor `segment_anything` is "
                "installed. Install one of:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
                "or place the official SAM3 package on PYTHONPATH. "
                "See README.md for checkpoint download instructions."
            ) from e
