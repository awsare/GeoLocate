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

from download_dataset import find_cached_download
from sectors import get_sector_map

MIN_IMAGES_PER_SECTOR = 50
MANIFEST_PATH = os.path.join("data", "manifest.csv")
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test
SPLIT_SEED = 42


def find_country_level_dir(dataset_root):
    """Return the directory holding the per-country folders.

    The country folders live under a wrapper dir (e.g. compressed_dataset/)
    whose name/depth may vary, so find it generically: it's the directory
    in the tree with the most immediate subdirectories.
    """
    return max(os.walk(dataset_root), key=lambda entry: len(entry[1]))[0]


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

    # Locate the per-country folders, group them into sectors, and build
    # (filepath, country, sector, ...) rows for sectors that clear
    # MIN_IMAGES_PER_SECTOR.
    country_level_dir = find_country_level_dir(dataset_root)
    manifest = build_manifest(country_level_dir)
    # Stratify each sector's images into train/val/test.
    manifest = assign_splits(manifest)

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
