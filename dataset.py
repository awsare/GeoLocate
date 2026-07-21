"""
PyTorch Dataset for GeoLocate: reads data/manifest.csv, filters to a split,
and loads/transforms images for training a country classifier.

Usage:
    python dataset.py

Requirements:
    pip install torch torchvision pillow
"""

import json
import os

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_MAP_PATH,
    MANIFEST_PATH,
    REQUIRED_COLUMNS,
    SECTOR_GRANULARITY,
    TRAIN_NUM_WORKERS,
)
from sectors import SECTOR_MAPS


def _build_all_label_maps():
    """Build {granularity: {sector: index}} for all known granularities."""
    all_maps = {}
    for granularity, country_to_sector in SECTOR_MAPS.items():
        sectors = sorted(set(country_to_sector.values()))
        all_maps[granularity] = {sector: idx for idx, sector in enumerate(sectors)}
    return all_maps


def _build_active_label_map(manifest):
    """Build active {sector: index} mapping from sectors present in manifest."""
    sectors = sorted(manifest["sector"].unique())
    return {sector: idx for idx, sector in enumerate(sectors)}


def _is_flat_label_map(payload):
    """Return True if payload is the legacy single-map format."""
    return (
        isinstance(payload, dict)
        and payload
        and all(isinstance(k, str) and isinstance(v, int) for k, v in payload.items())
    )


def validate_manifest(manifest_path):
    """Load and validate manifest schema before dataset construction."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found. Run prepare_dataset.py first."
        )

    manifest = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise ValueError(f"Manifest is missing required columns: {cols}")
    if manifest.empty:
        raise ValueError(
            f"{manifest_path} has no rows. Re-run prepare_dataset.py."
        )
    return manifest


def build_label_map(manifest):
    """Return a {sector: index} mapping, sorted by sector name.

    LABEL_MAP_PATH now stores mappings for both granularities:
    {
      "continent": {...},
      "subregion": {...}
    }

    The active map is selected by SECTOR_GRANULARITY. Legacy single-map files
    are migrated in place to the new multi-granularity format.
    """
    all_label_maps = _build_all_label_maps()
    active_label_map = _build_active_label_map(manifest)

    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH) as f:
            payload = json.load(f)

        if _is_flat_label_map(payload):
            # Legacy format: migrate file and use the active granularity map.
            all_label_maps[SECTOR_GRANULARITY] = active_label_map
            os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
            with open(LABEL_MAP_PATH, "w") as f:
                json.dump(all_label_maps, f, indent=2)
            return active_label_map

        if (
            isinstance(payload, dict)
            and SECTOR_GRANULARITY in payload
            and _is_flat_label_map(payload[SECTOR_GRANULARITY])
        ):
            payload[SECTOR_GRANULARITY] = active_label_map
            os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
            with open(LABEL_MAP_PATH, "w") as f:
                json.dump(payload, f, indent=2)
            return active_label_map

        raise ValueError(
            f"{LABEL_MAP_PATH} has an unexpected format. Delete it and retry."
        )

    os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
    all_label_maps[SECTOR_GRANULARITY] = active_label_map
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(all_label_maps, f, indent=2)

    active_sectors = set(manifest["sector"].unique())
    missing_from_active_map = active_sectors - set(active_label_map)
    if missing_from_active_map:
        missing_names = ", ".join(sorted(missing_from_active_map))
        raise ValueError(
            "Active sector map is missing sectors found in manifest. "
            f"Granularity: {SECTOR_GRANULARITY}. Missing: {missing_names}."
        )

    return active_label_map


def build_transforms(split):
    """Return the torchvision transform pipeline for the given split.

    train gets augmentation (random crop/flip/color jitter); val/test get a
    deterministic resize + center crop. Both normalize to ImageNet
    statistics since the eventual model is a pretrained backbone.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )


class GeoLocateDataset(Dataset):
    """Images + sector labels for one split (train/val/test) of the manifest."""

    def __init__(self, split, manifest_path=MANIFEST_PATH, transform=None):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{split}'. Expected train/val/test.")

        manifest = validate_manifest(manifest_path)
        # Build the label map from the full manifest, not the filtered split,
        # so train/val/test datasets always agree on indices even if a rare
        # country is missing from one split.
        self.label_map = build_label_map(manifest)
        missing_sectors = set(manifest["sector"].unique()) - set(self.label_map)
        if missing_sectors:
            sector_names = ", ".join(sorted(missing_sectors))
            raise ValueError(
                "label_map.json is out of sync with the manifest. "
                f"Missing sectors: {sector_names}. Active granularity: {SECTOR_GRANULARITY}. "
                f"Delete {LABEL_MAP_PATH} and retry."
            )
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(
                f"Manifest contains no rows for split '{split}'. "
                "Re-run prepare_dataset.py."
            )
        self.transform = transform if transform is not None else build_transforms(split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        image = self.transform(image)
        label = self.label_map[row["sector"]]
        return image, label


def main():
    datasets = {split: GeoLocateDataset(split) for split in ("train", "val", "test")}

    num_classes = len(datasets["train"].label_map)
    print(f"Classes: {num_classes}")
    for split, ds in datasets.items():
        print(f"{split}: {len(ds)} images")

    loader_kwargs = {"num_workers": TRAIN_NUM_WORKERS}
    if TRAIN_NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(
        datasets["train"],
        batch_size=8,
        shuffle=True,
        **loader_kwargs,
    )
    images, labels = next(iter(loader))
    print(f"Batch image shape: {tuple(images.shape)}")
    print(f"Batch label shape: {tuple(labels.shape)}")


if __name__ == "__main__":
    main()
