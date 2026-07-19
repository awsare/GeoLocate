"""
Filters out countries with too few images, splits the rest into
train/val/test per country, and writes a manifest of the remaining
(filepath, country, split) rows for use in model training.

Usage:
    python prepare_dataset.py

Output:
    data/manifest.csv
"""

import os

import pandas as pd

from config import MANIFEST_PATH, MIN_IMAGES_PER_SECTOR, SPLIT_RATIOS, SPLIT_SEED
from download_dataset import find_cached_download
from sectors import get_sector_map


def find_country_level_dir(dataset_root):
    """Return the directory holding the per-country folders.

    The country folders live under a wrapper dir (e.g. compressed_dataset/)
    whose name/depth may vary, so find it generically: it's the directory
    in the tree with the most immediate subdirectories.
    """
    country_level_dir, subdirs = max(
        ((path, dirs) for path, dirs, _ in os.walk(dataset_root)),
        key=lambda entry: len(entry[1]),
    )
    if not subdirs:
        raise RuntimeError(
            f"No country folders found under dataset root: {dataset_root}. "
            "Re-run download_dataset.py to refresh the Kaggle cache."
        )
    return country_level_dir


def build_manifest(country_level_dir):
    """Return a DataFrame of (filepath, country, sector) rows for every
    country folder, then drop whole sectors with fewer than
    MIN_IMAGES_PER_SECTOR images (rather than dropping individual small
    countries) so a small country's images survive by joining its
    neighbors' sector instead of being discarded.

    The manifest points at the existing kagglehub cache so the
    ~50k images aren't duplicated on disk and the cache stays untouched.
    """
    sector_map = get_sector_map()
    rows = []

    for entry in os.scandir(country_level_dir):
        if not entry.is_dir():
            continue
        if entry.name not in sector_map:
            raise KeyError(
                f"Country '{entry.name}' is missing from sectors.py mappings. "
                "Update sectors.py before preparing the manifest."
            )
        filenames = os.listdir(entry.path)
        sector = sector_map[entry.name]
        for filename in filenames:
            rows.append(
                {
                    "filepath": os.path.join(entry.path, filename),
                    "country": entry.name,
                    "sector": sector,
                }
            )

    if not rows:
        raise RuntimeError(
            f"No images found under {country_level_dir}. "
            "Dataset cache appears empty or malformed."
        )

    manifest = pd.DataFrame(rows)
    manifest["sector_image_count"] = manifest.groupby("sector")["sector"].transform("size")

    kept = manifest[manifest["sector_image_count"] >= MIN_IMAGES_PER_SECTOR]
    print(f"Sectors kept:    {kept['sector'].nunique()}")
    print(f"Sectors dropped: {manifest['sector'].nunique() - kept['sector'].nunique()}")
    return kept.reset_index(drop=True)


def assign_splits(manifest, seed=SPLIT_SEED, ratios=SPLIT_RATIOS):
    """Return manifest with a "split" column, stratified per sector.

    Splitting within each sector (rather than globally) keeps the
    smallest classes represented in val/test instead of risking a
    class with 0 images in one of those splits.
    """
    if manifest.empty:
        raise RuntimeError(
            "Manifest is empty after sector filtering. "
            "Lower MIN_IMAGES_PER_SECTOR or verify dataset contents."
        )

    train_ratio, val_ratio, _ = ratios

    shuffled = manifest.sample(frac=1, random_state=seed).reset_index(drop=True)
    group_sizes = shuffled.groupby("sector")["sector"].transform("size")
    position = shuffled.groupby("sector").cumcount()
    n_train = (group_sizes * train_ratio).astype(int)
    n_val = (group_sizes * val_ratio).astype(int)

    split = pd.Series("test", index=shuffled.index)
    split[position < n_train] = "train"
    split[(position >= n_train) & (position < n_train + n_val)] = "val"
    shuffled["split"] = split
    return shuffled


def main():
    # Reuse the cached dataset if present; otherwise download it.
    dataset_root = find_cached_download()
    if dataset_root is None:
        from download_dataset import main as download_dataset

        dataset_root = download_dataset()
    if not os.path.isdir(dataset_root):
        raise RuntimeError(
            f"Dataset root does not exist: {dataset_root}. "
            "Run download_dataset.py first."
        )

    # Locate the per-country folders, group them into sectors, and build
    # (filepath, country, sector, ...) rows for sectors that clear
    # MIN_IMAGES_PER_SECTOR.
    country_level_dir = find_country_level_dir(dataset_root)
    manifest = build_manifest(country_level_dir)
    # Stratify each sector's images into train/val/test.
    manifest = assign_splits(manifest)
    if manifest.empty:
        raise RuntimeError(
            "No rows left in manifest after splitting. "
            "Check dataset integrity and filtering thresholds."
        )

    counts = manifest["sector_image_count"]
    print(f"Images kept:       {len(manifest)}")
    print(f"Class count range: {counts.min()}-{counts.max()}")

    split_counts = manifest["split"].value_counts()
    print(f"Split sizes:       train={split_counts.get('train', 0)}, "
          f"val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)}")

    # Write the manifest itself, not the images, so the ~50k images stay
    # in the kagglehub cache instead of being duplicated on disk.
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
