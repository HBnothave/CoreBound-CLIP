"""Evaluation: multi-scale inference + DenseCRF post-processing (Sec. III, Implementation details).

Usage:
    python scripts/eval.py --config configs/voc.yaml --checkpoint work_dirs/voc/phase2_final.pth
"""

import argparse
import os
import sys
import yaml
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import CoreBoundCLIP
from datasets import VOCWSSSDataset, VOC_CLASSES, COCOWSSSDataset, COCO_CLASSES
from utils import compute_miou, apply_dense_crf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg["dataset"] == "voc":
        dataset = VOCWSSSDataset(cfg["data_root"], split="val",
                                  crop_size=cfg["crop_size"], train=False)
        class_names = VOC_CLASSES
    else:
        dataset = COCOWSSSDataset(cfg["data_root"], split="val",
                                   crop_size=cfg["crop_size"], train=False)
        class_names = COCO_CLASSES

    model = CoreBoundCLIP(
        class_names=class_names,
        clip_checkpoint=cfg["clip_checkpoint"],
        vit_dim=cfg["vit_dim"],
        text_dim=cfg["text_dim"],
        proj_dim=cfg["proj_dim"],
        num_classes=cfg["num_classes"],
        device=device,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.cesepro.load_state_dict(state["cesepro"])
    model.hibodec.load_state_dict(state["hibodec"])
    model.eval()

    scales = cfg.get("eval_scales", [1.0])
    use_crf = cfg.get("use_dense_crf", False)
    num_classes = cfg["num_classes"]

    hist_total = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["label"].numpy()
            H, W = image.shape[-2:]

            agg_probs = torch.zeros(1, num_classes, H, W, device=device)
            for s in scales:
                sh, sw = int(H * s), int(W * s)
                img_s = F.interpolate(image, size=(sh, sw), mode="bilinear", align_corners=False)
                out = model(img_s)
                probs = F.softmax(out["seg_pred"], dim=1)
                probs = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)
                agg_probs += probs
            agg_probs /= len(scales)

            probs_np = agg_probs[0].cpu().numpy()

            if use_crf:
                # Recover RGB image for CRF (undo CLIP normalization)
                from datasets.voc import CLIP_MEAN, CLIP_STD
                img_t = image[0].cpu()
                mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
                std = torch.tensor(CLIP_STD).view(3, 1, 1)
                rgb = ((img_t * std + mean) * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                pred = apply_dense_crf(np.ascontiguousarray(rgb), probs_np)
            else:
                pred = probs_np.argmax(0)

            _, hist = compute_miou(pred, label, num_classes)
            hist_total += hist

            if idx % 100 == 0:
                print(f"[{idx}/{len(dataset)}]")

    ious = []
    for c in range(num_classes):
        tp = hist_total[c, c]
        fp = hist_total[:, c].sum() - tp
        fn = hist_total[c, :].sum() - tp
        denom = tp + fp + fn
        if denom == 0:
            continue
        ious.append(tp / denom)
        print(f"{class_names[c]:>15s}: IoU = {ious[-1]*100:.2f}")

    print(f"mIoU = {np.mean(ious)*100:.2f}")


if __name__ == "__main__":
    main()
