"""Concept-Guided Refinement with SAM3 (CoReSAM3)."""

import numpy as np
import torch
import cv2


def cams_to_boxes(cam, fg_thresh=0.35, bg_thresh=0.15, min_area=64, nms_iou=0.5):
    """Connected-component box extraction from a single-class CAM
    (Sec. III, 'Pseudo-label thresholding and multi-instance handling').

    Args:
        cam: (H, W) numpy array, normalized activation map in [0, 1].
        fg_thresh: foreground threshold.
        bg_thresh: background threshold (pixels below are background;
                   in-between pixels are ignored during training).
        min_area: discard connected components smaller than this (pixels).
        nms_iou: IoU threshold above which overlapping boxes are NMS'd,
                 keeping the box with higher mean activation.

    Returns:
        List of boxes [x1, y1, x2, y2] in pixel coordinates.
    """
    fg_mask = (cam >= fg_thresh).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(fg_mask)

    boxes, scores = [], []
    for lbl in range(1, num_labels):
        comp = (labels == lbl)
        area = comp.sum()
        if area < min_area:
            continue
        ys, xs = np.where(comp)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        scores.append(float(cam[comp].mean()))

    return _nms_boxes(boxes, scores, nms_iou)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_boxes(boxes, scores, iou_thresh):
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(boxes[i])
        order = [j for j in order if _iou(boxes[i], boxes[j]) <= iou_thresh]
    return keep


def mask_to_box(mask, min_area=64):
    """Derive a single bounding box from a binary prediction mask
    (used in Phase 2, box from HiBoDec prediction)."""
    mask = mask.astype(np.uint8)
    if mask.sum() < min_area:
        return None
    ys, xs = np.where(mask > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


class CoReSAM3:
    """Wrapper around a SAM3 predictor for concept-guided mask refinement
    (Eq. 10 / Eq. 11).

    This is a thin interface: plug in the official SAM3 checkpoint/predictor.
    See README for download instructions.
    """

    def __init__(self, sam3_predictor):
        """
        sam3_predictor: an object exposing
            predict(image, box=[x1,y1,x2,y2], text="a photo of a {class}") -> (H, W) binary mask
        This should wrap the official SAM3 release's concept-prompted API.
        """
        self.predictor = sam3_predictor

    def refine(self, image, boxes, class_text):
        """Refine a set of boxes for one class into masks and merge them.

        Args:
            image: HxWx3 numpy array (RGB).
            boxes: list of [x1,y1,x2,y2] for this class.
            class_text: concept prompt string, e.g. "a photo of a {class}".

        Returns:
            (H, W) binary refined mask for this class.
        """
        H, W = image.shape[:2]
        merged = np.zeros((H, W), dtype=np.uint8)
        for box in boxes:
            m = self.predictor.predict(image, box=box, text=class_text)
            merged = np.logical_or(merged, m.astype(bool)).astype(np.uint8)
        return merged

    def phase1_refine(self, image, cams, class_names, prompt_template="a photo of a {}",
                       fg_thresh=0.35, bg_thresh=0.15, min_area=64, nms_iou=0.5):
        """Phase 1: refine CeSePro CAMs into preliminary pseudo-labels (Eq. 10).

        Args:
            image: HxWx3 numpy array.
            cams: dict {class_idx: (H, W) numpy CAM} for classes present in the image.
            class_names: dict {class_idx: str}.

        Returns:
            dict {class_idx: (H, W) binary refined mask}.
        """
        refined = {}
        for c, cam in cams.items():
            boxes = cams_to_boxes(cam, fg_thresh, bg_thresh, min_area, nms_iou)
            if not boxes:
                continue
            text = prompt_template.format(class_names[c])
            refined[c] = self.refine(image, boxes, text)
        return refined

    def phase2_refine(self, image, pred_masks, class_names, prompt_template="a photo of a {}",
                       min_area=64):
        """Phase 2: refine HiBoDec predictions into final pseudo-labels (Eq. 11).

        Args:
            pred_masks: dict {class_idx: (H, W) binary mask} from HiBoDec softmax.
        """
        refined = {}
        for c, mask in pred_masks.items():
            box = mask_to_box(mask, min_area)
            if box is None:
                continue
            text = prompt_template.format(class_names[c])
            refined[c] = self.refine(image, [box], text)
        return refined


def assemble_pseudo_label(refined_masks, ignore_index=255, bg_index=0):
    """Combine per-class refined binary masks into a single label map.

    refined_masks: dict {class_idx: (H, W) binary mask}.
    Pixels covered by no class become background; overlaps resolved by
    class index priority (later classes overwrite earlier ones).
    """
    if not refined_masks:
        return None
    h, w = next(iter(refined_masks.values())).shape
    label = np.full((h, w), bg_index, dtype=np.uint8)
    for c, mask in refined_masks.items():
        label[mask > 0] = c
    return label
