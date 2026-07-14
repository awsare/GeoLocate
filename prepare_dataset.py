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

MIN_IMAGES_PER_COUNTRY = 100
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
    """Return a DataFrame of (filepath, country, country_image_count) rows,
    limited to countries with at least MIN_IMAGES_PER_COUNTRY images.

    The manifest points at the existing kagglehub cache so the 
    ~50k images aren't duplicated on disk and the cache stays untouched.
    """
    rows = []
    kept_countries = 0
    dropped_countries = 0

    for entry in os.scandir(country_level_dir):
        if not entry.is_dir():
            continue
        filenames = os.listdir(entry.path)
        if len(filenames) < MIN_IMAGES_PER_COUNTRY:
            dropped_countries += 1
            continue
        kept_countries += 1
        for filename in filenames:
            rows.append(
                {
                    "filepath": os.path.join(entry.path, filename),
                    "country": entry.name,
                    "country_image_count": len(filenames),
                }
            )

    print(f"Countries kept:    {kept_countries}")
    print(f"Countries dropped: {dropped_countries}")
    return pd.DataFrame(rows)


def assign_splits(manifest, seed=SPLIT_SEED, ratios=SPLIT_RATIOS):
    """Return manifest with a "split" column, stratified per country.

    Splitting within each country (rather than globally) keeps the
    smallest classes represented in val/test instead of risking a
    class with 0 images in one of those splits.
    """
    train_ratio, val_ratio, _ = ratios

    shuffled = manifest.sample(frac=1, random_state=seed).reset_index(drop=True)
    group_sizes = shuffled.groupby("country")["country"].transform("size")
    position = shuffled.groupby("country").cumcount()
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

    # Locate the per-country folders and build (filepath, country, ...)
    # rows for countries that clear MIN_IMAGES_PER_COUNTRY.
    country_level_dir = find_country_level_dir(dataset_root)
    manifest = build_manifest(country_level_dir)
    # Stratify each country's images into train/val/test.
    manifest = assign_splits(manifest)

    counts = manifest["country_image_count"]
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
