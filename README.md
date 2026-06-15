# CoreBound-CLIP

Reference implementation of **CoreBound-CLIP: Core-to-Extent Semantic Prompting with
Boundary-Aware Decoding for Weakly-Supervised Semantic Segmentation**.

Single-stage WSSS framework built on a frozen CLIP ViT-B/16 backbone with three
components:

- **CeSePro** — Core-to-Extent Semantic Prompting: dual-expert (core/extent) text
  prototypes fused via an image-conditioned gate `alpha`.
- **HiBoDec** — Hierarchical Boundary-Aware Decoder: fuses multi-level CLIP ViT
  features (4 stage groups from 12 layers) with a shallow-feature edge branch.
- **CoReSAM3** — Concept-Guided Refinement with SAM3: refines CAM-/prediction-derived
  boxes with SAM3 concept prompts into pseudo-labels (two-phase iterative training).

## Repository layout

```
CoreBound-CLIP/
├── model/
│   ├── clip_backbone.py   # frozen CLIP ViT-B/16 wrapper, multi-level features
│   ├── cesepro.py          # CeSePro dual-expert prompting (Eq. 1-6)
│   ├── hibodec.py          # HiBoDec hierarchical decoder (Eq. 5-9)
│   ├── coresam3.py          # CoReSAM3 box extraction + SAM3 refinement (Eq. 10-11)
│   └── corebound_clip.py   # full model
├── datasets/                # VOC / COCO WSSS dataset loaders
├── utils/                   # Sobel edge targets, mIoU, DenseCRF
├── configs/                 # voc.yaml, coco.yaml
├── scripts/
│   ├── gen_pseudo_labels.py # CoReSAM3 pseudo-label generation (phase 1 / 2)
│   ├── train.py              # two-phase training loop
│   └── eval.py                # multi-scale + DenseCRF evaluation
└── pretrained/               # place downloaded checkpoints here (see below)
```

## Installation

```bash
pip install -r requirements.txt
```

## Required pretrained models

Download the following checkpoints and place them under `pretrained/` as shown.
These are **not** included in this repository.

| Model | Purpose | Download | Location |
|---|---|---|---|
| CLIP ViT-B/16 | Frozen vision-language backbone | [OpenAI CLIP](https://github.com/openai/CLIP) (`clip.load("ViT-B/16")` auto-downloads, or download `ViT-B-16.pt` manually) | `pretrained/clip/ViT-B-16.pt` |
| SAM3 | Concept-guided pseudo-label refinement (CoReSAM3) | Official SAM3 release (Meta) | `pretrained/sam3/sam3_checkpoint.pt` |

Update `clip_checkpoint` / `sam3_checkpoint` paths in `configs/voc.yaml` and
`configs/coco.yaml` if you place checkpoints elsewhere.

`scripts/gen_pseudo_labels.py` expects a SAM3 Python package exposing
`build_sam3_predictor(checkpoint_path, device)` returning an object with
`predict(image, box=[x1,y1,x2,y2], text="a photo of a {class}") -> binary mask`.
Adapt `load_sam3_predictor()` in that script to match the official SAM3 API.

## Data preparation

Download and arrange datasets following standard WSSS conventions:

- **PASCAL VOC 2012 (augmented)**: `data/VOCdevkit/VOC2012/{JPEGImages, ImageSets/Segmentation, SegmentationClassAug}`
- **MS COCO 2014**: `data/COCO2014/{train2014, val2014, ImageSets, SegmentationClassAug}`

Update `data_root` in `configs/voc.yaml` / `configs/coco.yaml` accordingly.

## Training (two-phase, per Sec. III)

```bash
# Phase 1: CeSePro CAMs -> CoReSAM3 -> preliminary pseudo-labels -> train HiBoDec
python scripts/gen_pseudo_labels.py --config configs/voc.yaml --phase 1
python scripts/train.py --config configs/voc.yaml --phase 1

# Phase 2: trained HiBoDec predictions -> CoReSAM3 -> final pseudo-labels -> fine-tune
python scripts/gen_pseudo_labels.py --config configs/voc.yaml --phase 2 \
    --checkpoint work_dirs/voc/phase1_final.pth
python scripts/train.py --config configs/voc.yaml --phase 2 \
    --resume work_dirs/voc/phase1_final.pth
```

For COCO, use `configs/coco.yaml`.

## Evaluation

```bash
python scripts/eval.py --config configs/voc.yaml --checkpoint work_dirs/voc/phase2_final.pth
```

Performs multi-scale testing (scales 0.75/1.0/2.0) with DenseCRF post-processing,
matching the paper's evaluation protocol.

## Hyperparameters (Sec. III)

- Optimizer: AdamW, polynomial LR decay
- LR: `1e-3` (CeSePro), `1e-4` (HiBoDec)
- Iterations: 30k (VOC), 80k (COCO), batch size 4
- Input: 512×512 random-resize-crop
- Loss weights: `lambda_div=0.1`, `lambda_1=1.0` (CeSePro), `lambda_2=0.5` (edge)
- Pseudo-label thresholds: fg=0.35, bg=0.15, min component area=64px, NMS IoU=0.5
