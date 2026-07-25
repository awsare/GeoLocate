"""Build a nearest-neighbor index of training-image embeddings.

Run this once after `prepare_dataset.py` to create `data/nn_index.joblib`
and `data/index_meta.npz` used by the web app.

Usage:
    python build_index.py
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import joblib

from dataset import GeoLocateDataset, build_transforms, validate_manifest, build_label_map
from model import Net
from config import MANIFEST_PATH, CHECKPOINT_PATH


def extract_backbone_features(net, x, device):
    # use net.backbone up to the avgpool layer (drop final fc)
    backbone = torch.nn.Sequential(*list(net.backbone.children())[:-1]).to(device)
    backbone.eval()
    with torch.no_grad():
        feat = backbone(x)
        feat = feat.view(feat.size(0), -1)
    return feat.cpu().numpy()


def main(batch_size=64, out_index_path="data/nn_index.joblib", out_meta_path="data/index_meta.npz"):
    os.makedirs(os.path.dirname(out_index_path), exist_ok=True)

    manifest = validate_manifest(MANIFEST_PATH)
    label_map = build_label_map(manifest)

    dataset = GeoLocateDataset("train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Net(len(dataset.label_map), pretrained=False).to(device)
    net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    net.eval()

    features = []
    filepaths = []
    sectors = []

    # iterate by index so we can access dataset.rows for filepaths/sectors
    for start in tqdm(range(0, len(dataset), batch_size), desc="Extracting features"):
        end = min(start + batch_size, len(dataset))
        imgs = []
        for idx in range(start, end):
            img, _ = dataset[idx]
            imgs.append(img.unsqueeze(0))
            filepaths.append(dataset.rows.iloc[idx]["filepath"])
            sectors.append(dataset.rows.iloc[idx]["sector"])

        batch = torch.cat(imgs, dim=0).to(device)
        feats = extract_backbone_features(net, batch, device)
        features.append(feats)

    X = np.vstack(features)
    print(f"Built embedding matrix: {X.shape}")

    # fit NearestNeighbors (cosine) and persist
    nn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="auto")
    nn.fit(X)

    print(f"Saving index to {out_index_path}")
    joblib.dump(nn, out_index_path)
    np.savez(out_meta_path, filepaths=filepaths, sectors=sectors)
    print(f"Saved metadata to {out_meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--index-out", default="data/nn_index.joblib")
    parser.add_argument("--meta-out", default="data/index_meta.npz")
    args = parser.parse_args()
    main(batch_size=args.batch_size, out_index_path=args.index_out, out_meta_path=args.meta_out)
