"""Generate CoReSAM3-refined pseudo-labels for Phase 1 or Phase 2 (Sec. III-D).

Phase 1: derives boxes from CeSePro CAMs -> SAM3 -> PseudoLabelsPhase1/
Phase 2: derives boxes from trained HiBoDec predictions -> SAM3 -> PseudoLabelsPhase2/

Usage:
    python scripts/gen_pseudo_labels.py --config configs/voc.yaml --phase 1
    python scripts/gen_pseudo_labels.py --config configs/voc.yaml --phase 2 \
        --checkpoint work_dirs/voc/phase1_final.pth
"""

import argparse
import os
import sys
import yaml
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import CoreBoundCLIP, CoReSAM3, assemble_pseudo_label
from datasets import VOCWSSSDataset, VOC_CLASSES, COCOWSSSDataset, COCO_CLASSES


def load_sam3_predictor(checkpoint_path, device):
    """Load the official SAM3 concept-prompted predictor.

    Replace this with the actual SAM3 API. See README for download
    instructions and the expected predictor interface
    (predict(image, box, text) -> binary mask).
    """
    try:
        from sam3 import build_sam3_predictor  # placeholder import
    except ImportError as e:
        raise ImportError(
            "SAM3 package not found. Install the official SAM3 release "
            "and place the checkpoint as described in README.md."
        ) from e
    return build_sam3_predictor(checkpoint_path, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument("--checkpoint", default=None,
                         help="Phase-1 model checkpoint (required for phase 2)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg["dataset"] == "voc":
        dataset = VOCWSSSDataset(cfg["data_root"], split="train_aug",
                                  crop_size=cfg["crop_size"], train=False)
        class_names = VOC_CLASSES
    else:
        dataset = COCOWSSSDataset(cfg["data_root"], split="train",
                                   crop_size=cfg["crop_size"], train=False)
        class_names = COCO_CLASSES

    out_dir = os.path.join(cfg["data_root"],
                            "PseudoLabelsPhase1" if args.phase == 1 else "PseudoLabelsPhase2")
    os.makedirs(out_dir, exist_ok=True)

    model = CoreBoundCLIP(
        class_names=class_names,
        clip_checkpoint=cfg["clip_checkpoint"],
        vit_dim=cfg["vit_dim"],
        text_dim=cfg["text_dim"],
        proj_dim=cfg["proj_dim"],
        num_classes=cfg["num_classes"],
        device=device,
    ).to(device)
    model.eval()

    if args.phase == 2:
        if args.checkpoint is None:
            raise ValueError("Phase 2 requires --checkpoint from phase 1 training")
        state = torch.load(args.checkpoint, map_location=device)
        model.cesepro.load_state_dict(state["cesepro"])
        model.hibodec.load_state_dict(state["hibodec"])

    sam3 = CoReSAM3(load_sam3_predictor(cfg["sam3_checkpoint"], device))
    prompt_template = "a photo of a {}"

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            img_id = sample["img_id"]
            cls_label = sample["cls_label"]
            present = [c for c in range(len(class_names)) if cls_label[c] > 0 and c != 0]
            if not present:
                continue

            out = model(image)

            # Reconstruct RGB image for SAM3 input
            raw = Image.open(os.path.join(
                dataset.root,
                "JPEGImages" if cfg["dataset"] == "voc" else dataset.img_dir,
                f"{img_id}.jpg")).convert("RGB")
            raw_np = np.array(raw.resize((cfg["crop_size"], cfg["crop_size"])))

            if args.phase == 1:
                cams = {}
                cam_maps = out["cams"][0]  # (C, h, w)
                cam_up = torch.nn.functional.interpolate(
                    cam_maps.unsqueeze(0), size=(cfg["crop_size"], cfg["crop_size"]),
                    mode="bilinear", align_corners=False)[0]
                cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-8)
                for c in present:
                    cams[c] = cam_up[c].cpu().numpy()
                refined = sam3.phase1_refine(
                    raw_np, cams, class_names, prompt_template,
                    fg_thresh=cfg["fg_thresh"], bg_thresh=cfg["bg_thresh"],
                    min_area=cfg["min_area"], nms_iou=cfg["nms_iou"])
            else:
                pred = out["seg_pred"][0].softmax(0)  # (num_classes, H, W)
                pred_cls = pred.argmax(0).cpu().numpy()
                pred_masks = {c: (pred_cls == c).astype(np.uint8) for c in present}
                refined = sam3.phase2_refine(
                    raw_np, pred_masks, class_names, prompt_template,
                    min_area=cfg["min_area"])

            label_map = assemble_pseudo_label(refined, bg_index=0)
            if label_map is None:
                label_map = np.zeros((cfg["crop_size"], cfg["crop_size"]), dtype=np.uint8)

            Image.fromarray(label_map).save(os.path.join(out_dir, f"{img_id}.png"))

            if idx % 200 == 0:
                print(f"[{idx}/{len(dataset)}] {img_id} -> {out_dir}")


if __name__ == "__main__":
    main()
