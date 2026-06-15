"""Two-phase training of CoreBound-CLIP (Sec. III, Overview).

Phase 1: CeSePro CAMs -> CoReSAM3 -> preliminary pseudo-labels -> train HiBoDec.
Phase 2: HiBoDec predictions -> CoReSAM3 -> final pseudo-labels -> fine-tune HiBoDec.

Usage:
    python scripts/train.py --config configs/voc.yaml --phase 1
    python scripts/train.py --config configs/voc.yaml --phase 2
"""

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import CoreBoundCLIP, loss_inj, loss_div, loss_edge
from datasets import VOCWSSSDataset, VOC_CLASSES, COCOWSSSDataset, COCO_CLASSES
from utils import sobel_edge_target


def build_dataset(cfg, train=True, label_dir="SegmentationClassAug"):
    if cfg["dataset"] == "voc":
        split = "train_aug" if train else "val"
        return VOCWSSSDataset(cfg["data_root"], split=split, crop_size=cfg["crop_size"],
                               label_dir=label_dir, train=train), VOC_CLASSES
    elif cfg["dataset"] == "coco":
        split = "train" if train else "val"
        return COCOWSSSDataset(cfg["data_root"], split=split, crop_size=cfg["crop_size"],
                                label_dir=label_dir, train=train), COCO_CLASSES
    raise ValueError(cfg["dataset"])


def poly_lr(base_lr, it, max_it, power=1.0):
    return base_lr * (1 - it / max_it) ** power


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument("--resume", default=None, help="checkpoint to resume HiBoDec/CeSePro weights from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["log_dir"], exist_ok=True)

    # Phase 1 trains on CAM-derived pseudo-labels stored under
    # 'PseudoLabelsPhase1'; Phase 2 trains on HiBoDec-refined labels under
    # 'PseudoLabelsPhase2'. Generate these with scripts/gen_pseudo_labels.py
    # before running each phase.
    label_dir = "PseudoLabelsPhase1" if args.phase == 1 else "PseudoLabelsPhase2"
    train_set, class_names = build_dataset(cfg, train=True, label_dir=label_dir)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=4, drop_last=True)

    model = CoreBoundCLIP(
        class_names=class_names,
        clip_checkpoint=cfg["clip_checkpoint"],
        vit_dim=cfg["vit_dim"],
        text_dim=cfg["text_dim"],
        proj_dim=cfg["proj_dim"],
        num_classes=cfg["num_classes"],
        device=device,
    ).to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.cesepro.load_state_dict(state["cesepro"])
        model.hibodec.load_state_dict(state["hibodec"])
        if "rfm" in state:
            model.rfm.load_state_dict(state["rfm"])

    optimizer = torch.optim.AdamW([
        {"params": list(model.cesepro.parameters()) + list(model.rfm.parameters()),
         "lr": cfg["lr_cesepro"]},
        {"params": model.hibodec.parameters(), "lr": cfg["lr_hibodec"]},
    ], weight_decay=cfg["weight_decay"])

    max_iters = cfg["iters_phase1"] if args.phase == 1 else cfg["iters_phase2"]
    seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    it = 0
    while it < max_iters:
        for batch in train_loader:
            if it >= max_iters:
                break

            image = batch["image"].to(device)
            label = batch["label"].to(device)
            cls_label = batch["cls_label"].to(device)

            out = model(image)

            # L_seg: pixel-wise CE on (refined) pseudo-labels
            l_seg = seg_criterion(out["seg_pred"], label)

            # L_CeSePro = L_inj + lambda_div * L_div  (Eq. 7-9)
            l_inj = loss_inj(out["s_core"], out["s_ext"], cls_label)
            l_div = loss_div(out["T_core"], out["T_ext"])
            l_cesepro = l_inj + cfg["lambda_div"] * l_div

            # L_edge: BCE against Sobel-derived edge target (Eq. 7)
            edge_target = sobel_edge_target(label, ignore_index=255)
            l_edge = loss_edge(out["edge_pred"], edge_target)

            # L_total (Eq. 10)
            loss = (l_seg
                    + cfg["lambda_cesepro"] * l_cesepro
                    + cfg["lambda_edge"] * l_edge)

            for g, base_lr in zip(optimizer.param_groups,
                                   [cfg["lr_cesepro"], cfg["lr_hibodec"]]):
                g["lr"] = poly_lr(base_lr, it, max_iters, cfg["poly_power"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % 50 == 0:
                print(f"[phase {args.phase}] iter {it}/{max_iters} "
                      f"loss={loss.item():.4f} seg={l_seg.item():.4f} "
                      f"cesepro={l_cesepro.item():.4f} edge={l_edge.item():.4f}")

            if it % 5000 == 0 and it > 0:
                ckpt = {"cesepro": model.cesepro.state_dict(),
                        "hibodec": model.hibodec.state_dict(),
                        "rfm": model.rfm.state_dict()}
                torch.save(ckpt, os.path.join(cfg["log_dir"], f"phase{args.phase}_iter{it}.pth"))

            it += 1

    ckpt = {"cesepro": model.cesepro.state_dict(),
            "hibodec": model.hibodec.state_dict(),
            "rfm": model.rfm.state_dict()}
    torch.save(ckpt, os.path.join(cfg["log_dir"], f"phase{args.phase}_final.pth"))
    print(f"Saved final phase {args.phase} checkpoint to {cfg['log_dir']}")


if __name__ == "__main__":
    main()
