"""
dataset.py -- XRayDataset, data splitting, augmentation, WeightedRandomSampler.
Compatible with SIXray folder layout:
    data/raw/positive/   <- images with prohibited items  (label = 1)
    data/raw/negative/   <- safe bag images               (label = 0)
"""
import os
import csv
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# --- Transforms --------------------------------------------------------------

def get_transforms(split: str, image_size: int = 224) -> A.Compose:
    """Return Albumentations pipeline for train / val / test."""
    mean = [0.485, 0.456, 0.406]   # ImageNet stats (works well for X-ray too)
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:  # val / test -- no augmentation
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


# --- Dataset -----------------------------------------------------------------

class XRayDataset(Dataset):
    """
    Loads (image_path, label) pairs from a CSV manifest.
    label 0 = safe  |  label 1 = prohibited
    """

    def __init__(self, manifest_csv: str, transform: A.Compose):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        with open(manifest_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["path"], int(row["label"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        aug = self.transform(image=img_np)
        tensor = aug["image"]                   # float32 CHW
        return tensor, label, path              # path for Grad-CAM tracing

    @property
    def labels(self) -> List[int]:
        return [s[1] for s in self.samples]


# --- Split builder -----------------------------------------------------------

def build_splits(raw_root: str,
                 processed_dir: str,
                 train_ratio: float = 0.70,
                 val_ratio:   float = 0.15,
                 seed: int = 42) -> dict:
    """
    Scans positive/ and negative/ under raw_root, creates stratified splits,
    writes three CSVs to processed_dir, returns paths.
    """
    random.seed(seed)
    pos_dir = Path(raw_root) / "positive"
    neg_dir = Path(raw_root) / "negative"
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    pos_files = sorted(pos_dir.glob("*"))
    neg_files = sorted(neg_dir.glob("*"))

    # Filter to image extensions only
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    pos_files = [p for p in pos_files if p.suffix.lower() in exts]
    neg_files = [p for p in neg_files if p.suffix.lower() in exts]

    print(f"[Dataset] Found {len(pos_files)} positive, "
          f"{len(neg_files)} negative images.")

    def split_list(items):
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        return (items[:n_train],
                items[n_train:n_train + n_val],
                items[n_train + n_val:])

    pos_train, pos_val, pos_test = split_list(pos_files)
    neg_train, neg_val, neg_test = split_list(neg_files)

    splits = {
        "train": (pos_train + neg_train, [1]*len(pos_train) + [0]*len(neg_train)),
        "val":   (pos_val   + neg_val,   [1]*len(pos_val)   + [0]*len(neg_val)),
        "test":  (pos_test  + neg_test,  [1]*len(pos_test)  + [0]*len(neg_test)),
    }

    csv_paths = {}
    for split, (paths, labels) in splits.items():
        csv_path = os.path.join(processed_dir, f"{split}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            for p, l in zip(paths, labels):
                writer.writerow({"path": str(p), "label": l})
        csv_paths[split] = csv_path
        print(f"  [{split:5s}] {len(paths):5d} samples "
              f"({sum(labels)} pos / {len(labels)-sum(labels)} neg) -> {csv_path}")

    return csv_paths


# --- WeightedRandomSampler ---------------------------------------------------

def get_weighted_sampler(dataset: XRayDataset) -> WeightedRandomSampler:
    """Over-sample the minority (positive) class each epoch."""
    labels = dataset.labels
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# --- DataLoader factory ------------------------------------------------------

def get_dataloaders(csv_paths: dict,
                    batch_size: int = 32,
                    image_size: int = 224,
                    num_workers: int = 0,
                    use_weighted_sampler: bool = True) -> dict:
    loaders = {}
    for split, csv_path in csv_paths.items():
        transform = get_transforms(split, image_size)
        ds = XRayDataset(csv_path, transform)
        if split == "train" and use_weighted_sampler:
            sampler = get_weighted_sampler(ds)
            loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                                num_workers=num_workers, pin_memory=True)
        else:
            shuffle = (split == "train")
            loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=True)
        loaders[split] = loader
    return loaders
