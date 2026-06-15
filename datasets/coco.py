"""MS COCO 2014 dataset for WSSS training/evaluation."""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .voc import CLIP_MEAN, CLIP_STD  # reuse CLIP normalization

# 80 foreground classes + background, indices 0..80 matching SegmentationClassAug-style maps.
COCO_CLASSES = ["background"] + [f"class_{i}" for i in range(1, 81)]  # replace with real names


class COCOWSSSDataset(Dataset):
    """Expects:

    root/
      train2014/*.jpg, val2014/*.jpg
      SegmentationClassAug/*.png   (pseudo-labels, 0=bg .. 80=class, 255=ignore)
      ImageSets/{train,val}.txt
    """

    def __init__(self, root, split="train", crop_size=512,
                 label_dir="SegmentationClassAug", train=True):
        self.root = root
        self.train = train
        self.crop_size = crop_size
        self.label_dir = label_dir
        self.img_dir = "train2014" if train else "val2014"

        list_path = os.path.join(root, "ImageSets", f"{split}.txt")
        with open(list_path) as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def _load_image(self, img_id):
        path = os.path.join(self.root, self.img_dir, f"{img_id}.jpg")
        return Image.open(path).convert("RGB")

    def _load_label(self, img_id):
        path = os.path.join(self.root, self.label_dir, f"{img_id}.png")
        if os.path.exists(path):
            return Image.open(path)
        return None

    def _image_level_labels(self, label):
        y = torch.zeros(len(COCO_CLASSES), dtype=torch.float32)
        if label is None:
            return y
        arr = np.array(label)
        for c in np.unique(arr):
            if c == 255 or c >= len(COCO_CLASSES):
                continue
            y[c] = 1.0
        return y

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image = self._load_image(img_id)
        label = self._load_label(img_id)

        if self.train:
            scale = np.random.uniform(0.5, 2.0)
            w, h = image.size
            nw, nh = int(w * scale), int(h * scale)
            image = image.resize((nw, nh), Image.BILINEAR)
            if label is not None:
                label = label.resize((nw, nh), Image.NEAREST)

            cs = self.crop_size
            pad_w, pad_h = max(cs - nw, 0), max(cs - nh, 0)
            if pad_w > 0 or pad_h > 0:
                image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
                if label is not None:
                    label = TF.pad(label, [0, 0, pad_w, pad_h], fill=255)
            w, h = image.size
            x1 = np.random.randint(0, w - cs + 1)
            y1 = np.random.randint(0, h - cs + 1)
            image = image.crop((x1, y1, x1 + cs, y1 + cs))
            if label is not None:
                label = label.crop((x1, y1, x1 + cs, y1 + cs))
            if np.random.rand() < 0.5:
                image = TF.hflip(image)
                if label is not None:
                    label = TF.hflip(label)
        else:
            image = TF.resize(image, [self.crop_size, self.crop_size])
            if label is not None:
                label = TF.resize(label, [self.crop_size, self.crop_size],
                                   interpolation=TF.InterpolationMode.NEAREST)

        img_t = TF.normalize(TF.to_tensor(image), CLIP_MEAN, CLIP_STD)

        if label is not None:
            label_t = torch.from_numpy(np.array(label)).long()
        else:
            label_t = torch.full((self.crop_size, self.crop_size), 255, dtype=torch.long)

        cls_label = self._image_level_labels(label)

        return {
            "image": img_t,
            "label": label_t,
            "cls_label": cls_label,
            "img_id": img_id,
        }
