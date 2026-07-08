"""
Downloads the "geolocation-geoguessr-images-50k" Kaggle dataset and prints
the local path it was downloaded to.

Requirements:
    pip install kagglehub

Auth:
    Needs a Kaggle API token. Either:
      - place kaggle.json in ~/.kaggle/kaggle.json, or
      - set env vars KAGGLE_USERNAME and KAGGLE_KEY
    Get a token at https://www.kaggle.com/settings -> API -> Create New Token
"""

import glob
import os
import kagglehub

DATASET = "ubitquitin/geolocation-geoguessr-images-50k"


def find_cached_download():
    """Return the path to an already-downloaded copy of the dataset, if any."""
    cache_root = os.environ.get("KAGGLEHUB_CACHE", os.path.expanduser("~/.cache/kagglehub"))
    pattern = os.path.join(cache_root, "datasets", *DATASET.split("/"), "versions", "*")
    versions = [p for p in glob.glob(pattern) if os.path.isdir(p) and os.listdir(p)]
    if not versions:
        return None
    return sorted(versions)[-1]  # newest version dir


def main():
    existing = find_cached_download()
    if existing:
        print(f"Dataset already downloaded at: {existing}")
        return existing

    print(f"Downloading dataset: {DATASET} ...")
    root = kagglehub.dataset_download(DATASET)
    print(f"Dataset downloaded to: {root}")
    return root


if __name__ == "__main__":
    main()
