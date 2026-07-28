import io
import os
import sys
import base64
import json
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
import joblib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from dataset import validate_manifest, build_label_map, build_transforms
from model import Net
from config import CHECKPOINT_PATH, MANIFEST_PATH


def pil_to_data_uri(pil_img, fmt="JPEG", size=(160, 160)):
    pil_copy = pil_img.copy()
    pil_copy.thumbnail(size)
    buffer = io.BytesIO()
    pil_copy.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def extract_backbone_features(net, x, device):
    backbone = torch.nn.Sequential(*list(net.backbone.children())[:-1]).to(device)
    backbone.eval()
    with torch.no_grad():
        feat = backbone(x)
        feat = feat.view(feat.size(0), -1)
    return feat.cpu().numpy()


app = Flask(
    __name__,
    static_url_path="/static",
    static_folder=os.path.join(ROOT_DIR, "static"),
)

# Load manifest and label map
manifest = validate_manifest(MANIFEST_PATH)
label_map = build_label_map(manifest)
idx_to_sector = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

# Load model (try several backbone variants if checkpoint was trained with a different ResNet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = build_transforms("test")


def load_checkpoint_with_fallback(checkpoint_path, num_classes, device):
    backbones = [None, "resnet18", "resnet34", "resnet50"]
    last_exc = None
    for bk in backbones:
        try:
            if bk is None:
                candidate = Net(num_classes, pretrained=False).to(device)
            else:
                candidate = Net(num_classes, pretrained=False, backbone_name=bk).to(device)
            candidate.load_state_dict(torch.load(checkpoint_path, map_location=device))
            candidate.eval()
            print(f"Loaded checkpoint using backbone={bk or 'default'}")
            return candidate
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(
        f"Failed to load checkpoint {checkpoint_path} with tried backbones: {backbones}. Last error: {last_exc}"
    ) from last_exc


try:
    net = load_checkpoint_with_fallback(CHECKPOINT_PATH, num_classes, device)
except Exception as exc:
    raise RuntimeError(
        f"Unable to load model checkpoint. Ensure {CHECKPOINT_PATH} exists and matches a ResNet backbone (resnet18/resnet34/resnet50). Error: {exc}"
    ) from exc

# Load NN index if present
NN_INDEX_PATH = os.path.join(ROOT_DIR, "data", "nn_index.joblib")
NN_META_PATH = os.path.join(ROOT_DIR, "data", "index_meta.npz")
nn = None
meta = None
if os.path.exists(NN_INDEX_PATH) and os.path.exists(NN_META_PATH):
    nn = joblib.load(NN_INDEX_PATH)
    meta = np.load(NN_META_PATH, allow_pickle=True)
    filepaths = meta["filepaths"].tolist()
    sectors = meta["sectors"].tolist()
else:
    filepaths = []
    sectors = []


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("image")
    if not files:
        return jsonify({"error": "no files uploaded"}), 400
    if len(files) != 4:
        return jsonify({"error": "please upload exactly 4 images"}), 400

    images = []
    for f in files:
        try:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            images.append(transform(img))
        except Exception:
            return jsonify({"error": "unable to process one or more uploaded images"}), 400

    x = torch.stack(images).to(device)

    with torch.no_grad():
        logits = net(x)
        combined_logits = logits.mean(dim=0, keepdim=True)
        probs = torch.softmax(combined_logits, dim=1).squeeze(0).cpu().numpy()

    topk = int(request.form.get("topk", 3))
    inds = np.argsort(probs)[-topk:][::-1]
    predictions = [{"sector": idx_to_sector[int(i)], "score": float(probs[int(i)])} for i in inds]

    response = {"predictions": predictions}

    if nn is not None:
        feats = extract_backbone_features(net, x, device)
        query_feat = np.mean(feats, axis=0, keepdims=True)
        dists, ids = nn.kneighbors(query_feat, n_neighbors=5, return_distance=True)
        neighbors = []
        for dist, idx in zip(dists[0], ids[0]):
            fp = filepaths[int(idx)]
            sector = sectors[int(idx)]
            thumb = None
            try:
                with Image.open(fp).convert("RGB") as im:
                    thumb = pil_to_data_uri(im)
            except Exception:
                thumb = None
            neighbors.append({"filepath": fp, "sector": sector, "distance": float(dist), "thumbnail": thumb})
        response["neighbors"] = neighbors

    return jsonify(response)


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "frontend.html")


if __name__ == "__main__":
    app.run(debug=True)
