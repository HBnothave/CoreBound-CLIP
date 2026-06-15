"""Utility functions: Sobel edge targets, CRF post-processing, mIoU metric."""

import numpy as np
import torch
import torch.nn.functional as F


def sobel_edge_target(label, ignore_index=255, threshold=0.1):
    """Derive a class-agnostic binary edge map from a label map using the
    Sobel operator (used to supervise HiBoDec's edge branch, L_edge).

    label: (B, H, W) long tensor of class indices (255 = ignore).
    Returns: (B, 1, H, W) float tensor in {0, 1}.
    """
    valid = (label != ignore_index).float()
    lbl = label.clone()
    lbl[label == ignore_index] = 0
    lbl = lbl.float().unsqueeze(1)  # (B,1,H,W)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32, device=lbl.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32, device=lbl.device).view(1, 1, 3, 3)

    gx = F.conv2d(lbl, sobel_x, padding=1)
    gy = F.conv2d(lbl, sobel_y, padding=1)
    grad = torch.sqrt(gx ** 2 + gy ** 2)

    edge = (grad > threshold).float() * valid.unsqueeze(1)
    return edge


def compute_miou(pred, label, num_classes, ignore_index=255):
    """pred, label: (H, W) numpy arrays of class indices."""
    mask = (label != ignore_index)
    pred = pred[mask]
    label = label[mask]
    hist = np.bincount(
        num_classes * label.astype(int) + pred.astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)

    ious = []
    for c in range(num_classes):
        tp = hist[c, c]
        fp = hist[:, c].sum() - tp
        fn = hist[c, :].sum() - tp
        denom = tp + fp + fn
        if denom == 0:
            continue
        ious.append(tp / denom)
    return float(np.mean(ious)), hist


try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    def apply_dense_crf(image, probs):
        """image: (H,W,3) uint8 RGB; probs: (C,H,W) softmax. Returns (H,W) labels."""
        C, H, W = probs.shape
        d = dcrf.DenseCRF2D(W, H, C)
        U = unary_from_softmax(probs)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
        Q = d.inference(5)
        return np.argmax(Q, axis=0).reshape(H, W)

except ImportError:
    def apply_dense_crf(image, probs):
        raise ImportError("pydensecrf not installed. `pip install pydensecrf`")
