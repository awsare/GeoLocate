# GeoLocate

A machine learning project to classify images by country, using the Kaggle dataset
[`ubitquitin/geolocation-geoguessr-images-50k`](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k)
(~50k GeoGuessr Street View images across ~124 countries).

## Setup

```bash
pip install kagglehub pandas torch torchvision pillow
```

## Usage

```bash
python download_dataset.py   # download the dataset via kagglehub
python prepare_dataset.py    # build data/manifest.csv
python dataset.py            # sanity-check the PyTorch Dataset/DataLoader
```

## Data flow

`download_dataset.py` → `prepare_dataset.py` → `data/manifest.csv` → `dataset.py` → (future) training code


## Files

- **`download_dataset.py`** — Downloads the dataset via `kagglehub` into
  `~/.cache/kagglehub/...` (or `$KAGGLEHUB_CACHE` if set). `find_cached_download()`
  checks for an existing download first, so re-running is cheap and never
  re-downloads unnecessarily.

- **`prepare_dataset.py`** — Builds `data/manifest.csv`, the single source of
  truth for which images to train on and how they're split:
  - `find_country_level_dir()` locates the per-country image folders generically
    (the wrapper directory's name/depth in the Kaggle archive can vary), by
    picking the directory in the tree with the most immediate subdirectories.
  - `build_manifest()` drops any country with fewer than `MIN_IMAGES_PER_COUNTRY`
    (100) images, to reduce class imbalance.
  - `assign_splits()` stratifies each country's images into train/val/test
    (80/10/10) independently per country, so even the smallest classes stay
    represented in val/test. Uses a fixed seed for reproducibility.
  - The manifest's `filepath` column points directly into the kagglehub cache
    (absolute paths) — images are never copied. This keeps the ~50k images as
    a single on-disk copy, but also means `manifest.csv` is **not portable
    across machines/users** without re-running this script locally.

- **`dataset.py`** — Defines `GeoLocateDataset`, a `torch.utils.data.Dataset`
  that reads `data/manifest.csv`, filters to a split (train/val/test), and
  loads/transforms images (224x224, ImageNet normalization; augmented for
  train, deterministic for val/test). Country labels are encoded via a
  `country -> index` mapping built from the full manifest and persisted to
  `data/label_map.json` so indices stay stable across runs. Running the file
  directly builds all three splits and prints class/split counts plus a
  sample batch shape as a sanity check.

- **`exploration.ipynb`** — Dataset exploration notebook (per-country
  image counts, class imbalance, largest/smallest classes). Reuses
  `find_cached_download()` rather than duplicating the cache-lookup logic.

- **`data/`** — Gitignored except for `data/manifest.csv` and
  `data/label_map.json`, the only artifacts meant to be versioned. The actual
  images stay in the kagglehub cache, not in this repo.