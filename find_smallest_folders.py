"""
Scans a directory tree and prints the 10 folders with the fewest items.

Usage:
    python find_smallest_folders.py /path/to/dataset

If no path is given, it defaults to re-downloading/locating the dataset
via download_dataset.py.
"""

import os
import sys

TOP_N = 30


def count_items_per_folder(root):
    """Return {folder_path: number_of_items_directly_inside_it}."""
    counts = {}
    for dirpath, dirnames, filenames in os.walk(root):
        counts[dirpath] = len(dirnames) + len(filenames)
    return counts


def main():
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        from download_dataset import main as download_dataset
        root = download_dataset()

    counts = count_items_per_folder(root)

    # Skip the root itself if you only want subfolders; comment out to include it.
    counts.pop(root, None)

    fewest = sorted(counts.items(), key=lambda kv: kv[1])[:TOP_N]

    print(f"Top {TOP_N} folders with the fewest items:")
    for path, n in fewest:
        rel = os.path.relpath(path, root)
        print(f"{n:>6}  {rel}")


if __name__ == "__main__":
    main()
