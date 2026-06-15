# CoreBound-CLIP

Reference implementation of **CoreBound-CLIP: Core-to-Extent Semantic Prompting with
Boundary-Aware Decoding for Weakly-Supervised Semantic Segmentation**.

Single-stage WSSS framework built on a frozen CLIP ViT-B/16 backbone with three
components:

- **CeSePro** — Core-to-Extent Semantic Prompting: dual-expert (core/extent) text
  prototypes fused via an image-conditioned gate `alpha`.
- **RFM** — Residual Fusion Module, retained from WeCLIP as an auxiliary
  CAM-enhancement branch that refines the dense feature map used in CeSePro's
  CAM similarity.
- **HiBoDec** — Hierarchical Boundary-Aware Decoder: fuses multi-level CLIP ViT
  features (4 stage groups from 12 layers) with a shallow-feature edge branch.
- **CoReSAM3** — Concept-Guided Refinement with SAM3: refines CAM-/prediction-derived
  boxes with SAM3 concept prompts into pseudo-labels (two-phase iterative training).

## Repository layout

```
CoreBound-CLIP/
├── model/
│   ├── clip_backbone.py   # frozen CLIP ViT-B/16 wrapper, multi-level features
│   ├── rfm.py               # WeCLIP residual fusion module (CAM-enhancement branch)
│   ├── cesepro.py          # CeSePro dual-expert prompting (Eq. 1-6)
│   ├── hibodec.py          # HiBoDec hierarchical decoder (Eq. 5-9)
│   ├── coresam3.py          # CoReSAM3 box extraction + refinement logic (Eq. 10-11)
│   ├── sam3_adapter.py     # SAM3 predictor adapter (+ SAM1/2 fallback)
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
| SAM3 (preferred) | Concept-guided pseudo-label refinement (CoReSAM3) | Official SAM3 release (Meta) | `pretrained/sam3/sam3_checkpoint.pt` |
| SAM / SAM2 (fallback) | Used automatically if SAM3 is unavailable | [segment-anything](https://github.com/facebookresearch/segment-anything) or [SAM2](https://github.com/facebookresearch/segment-anything-2) checkpoints | `pretrained/sam3/sam3_checkpoint.pt` (point `sam3_checkpoint` at this file instead) |

Update `clip_checkpoint` / `sam3_checkpoint` / `sam_model_type` paths in
`configs/voc.yaml` and `configs/coco.yaml` if you place checkpoints
elsewhere.

### SAM3 integration

`model/sam3_adapter.py` provides `build_sam3_predictor(checkpoint_path, device, model_type)`,
used by `scripts/gen_pseudo_labels.py`:

- **If the official SAM3 package is installed** (importable as `sam3`,
  providing `sam3.build_sam.build_sam3` and `sam3.predictor.SAM3Predictor`),
  it is used directly with joint box + text concept prompts, matching
  Eq. 10-11 of the paper. Adjust the import paths in `Sam3Predictor` if the
  released package's module layout differs.
- **Otherwise**, the adapter falls back to `segment-anything` (SAM1/2) box
  prompting plus a CLIP-based concept-consistency check (rejecting masks
  whose crop doesn't match the class text above a similarity threshold).
  This lets the full pipeline run end-to-end on widely available checkpoints
  while approximating CoReSAM3's concept-guided behavior. Set
  `sam_model_type` in the config to match the fallback checkpoint
  (`vit_b` / `vit_l` / `vit_h`).

For results matching the paper, use the official SAM3 release once available.

## Data preparation

Download and arrange datasets following standard WSSS conventions:

- **PASCAL VOC 2012 (augmented)**: `data/VOCdevkit/VOC2012/{JPEGImages, ImageSets/Segmentation, SegmentationClassAug}`
- **MS COCO 2014**: `data/COCO2014/{train2014, val2014, ImageSets, SegmentationClassAug}`

Update `data_root` in `configs/voc.yaml` / `configs/coco.yaml` accordingly.

## Training

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

