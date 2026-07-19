"""
Downloads the "geolocation-geoguessr-images-50k" Kaggle dataset and prints
the local path it was downloaded to.

Requirements:
    pip install kagglehub

Auth:
    kagglehub handles this automatically. If it doesn't find credentials
    (env vars or ~/.kaggle/kaggle.json), it opens a browser for you to log
    into Kaggle and caches the credentials itself. No manual setup needed.
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
    if not os.path.isdir(root) or not os.listdir(root):
        raise RuntimeError(
            f"Downloaded path is invalid or empty: {root}. "
            "Check Kaggle auth/network and try again."
        )
    print(f"Dataset downloaded to: {root}")
    return root


if __name__ == "__main__":
    main()
