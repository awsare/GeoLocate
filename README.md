# GeoLocate

A machine learning project to classify images by geographic sector,
using the Kaggle dataset
[`ubitquitin/geolocation-geoguessr-images-50k`](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k)
(~50k GeoGuessr Street View images across ~124 countries).

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

All tunable settings live in `config.py`.

- Edit `config.py` to change dataset source, split policy, preprocessing,
  training hyperparameters, class balancing, and smoke-test behavior.

## Usage

```bash
python download_dataset.py   # download the dataset via kagglehub
python prepare_dataset.py    # build data/manifest.csv
python dataset.py            # sanity-check the PyTorch Dataset/DataLoader
python train.py              # train the ResNet model and save a checkpoint
python evaluate.py           # load the checkpoint and report test accuracy
python smoke_test.py         # quick end-to-end pipeline check on tiny data slices
```

## Data flow

`download_dataset.py` → `prepare_dataset.py` (using `sectors.py`) → `data/manifest.csv` → `dataset.py` → `model.py` → `train.py` → `checkpoints/geolocate_net.pth` → `evaluate.py`

Config for each stage is sourced from `config.py`.

## Files

- **`download_dataset.py`** — Downloads the dataset via `kagglehub`, reusing
  an existing cached download if present (`find_cached_download()`). Uses
  `KAGGLE_DATASET` from `config.py`.

- **`config.py`** — Centralized project configuration. Contains grouped,
  documented constants for dataset source, paths/artifacts,
  manifest/split policy, image preprocessing, training hyperparameters,
  class balancing, and smoke test setup.

- **`sectors.py`** — Maps each country to a geographic sector so the model
  classifies by region instead of by individual country, keeping every
  country's images instead of dropping small ones. Two granularities are
  provided (`continent`, `subregion`); active granularity is controlled by
  `SECTOR_GRANULARITY` in `config.py`.

- **`prepare_dataset.py`** — Builds `data/manifest.csv`: groups countries
  into sectors via `sectors.py`, drops sectors with too few images
  (`MIN_IMAGES_PER_SECTOR`), and stratifies each sector into train/val/test
  splits. `filepath` points directly into the kagglehub cache, so
  `manifest.csv` isn't portable across machines without re-running this
  script, so it stays local and should be regenerated per machine.

- **`dataset.py`** — `GeoLocateDataset`, a `torch.utils.data.Dataset` that
  reads the manifest and loads/transforms images (224x224, ImageNet
  normalization). Sector labels are encoded via a mapping persisted to
  `data/label_map.json`.

- **`model.py`** — Defines `Net`, a ResNet-18 backbone with a classifier
  head sized to the active sector count. Uses ImageNet pretrained weights
  by default, with a fallback to random initialization if weights cannot be
  loaded.

- **`train.py`** — Trains `Net` with a two-phase schedule:
  phase 1 trains only the classifier head (frozen backbone), then phase 2
  fine-tunes the full network with a lower LR on the backbone and higher LR
  on the classifier head. Includes class-balancing options via weighted
  cross-entropy (`USE_CLASS_WEIGHTS`) and optional minority oversampling
  (`USE_WEIGHTED_SAMPLER`). Saves `checkpoints/geolocate_net.pth`.

- **`evaluate.py`** — Loads `checkpoints/geolocate_net.pth` and reports
  overall and per-sector test accuracy for the test split.

- **`smoke_test.py`** — Exercises the full pipeline (dataset → model →
  training step → checkpoint → eval) on a tiny data slice, to catch
  breakage quickly without a full training run.

- **`exploration.ipynb`** — Dataset exploration of per-country image
  counts and class imbalance.

- **`data/`** — Gitignored except for `data/label_map.json`. The generated
  `manifest.csv` stays local because it contains machine-specific kagglehub
  cache paths. Images stay in the kagglehub cache, not in this repo.

## Decisions

This section tracks intentional project choices and why they were made.

- **Prediction target is sector, not country**
  The model predicts a geographic sector (from `sectors.py`) rather than a
  specific country. This reduces label sparsity and keeps low-image countries
  usable by grouping them with nearby countries.

- **Model initialization is pretrained ResNet-18**
  The backbone starts from ImageNet features (default), which improves data
  efficiency versus training from scratch and helps minority sectors.

- **Training uses a two-phase fine-tuning schedule**
  Phase 1 trains only the classifier head, then phase 2 unfreezes the
  backbone for end-to-end fine-tuning with differential learning rates. This
  stabilizes optimization after swapping the classifier head.
  
- **Split strategy is sector-stratified train/val/test**
  Splits are assigned within each sector (controlled by `SPLIT_RATIOS` in
  `config.py`) so minority sectors still appear in val/test and evaluation
  remains meaningful across classes.

- **Class imbalance is handled with weighted loss (default) and optional oversampling**
  `train.py` computes class counts from the training split and applies
  inverse-frequency class weights in `CrossEntropyLoss` by default
  (`USE_CLASS_WEIGHTS` in `config.py`). Optional `WeightedRandomSampler`
  (`USE_WEIGHTED_SAMPLER` in `config.py`) can further increase minority
  exposure during training when rare sectors underperform.