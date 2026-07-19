"""
Trains a ResNet-18 classifier for GeoLocate sector prediction using
a two-phase fine-tuning schedule:
1) train classifier head with frozen backbone,
2) unfreeze full network and fine-tune end-to-end.

Usage:
    python train.py

Requirements:
    pip install -r requirements.txt
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import (
    BACKBONE_LEARNING_RATE,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    FINETUNE_HEAD_LEARNING_RATE,
    HEAD_LEARNING_RATE,
    HEAD_WARMUP_EPOCHS,
    MANIFEST_PATH,
    MOMENTUM,
    NUM_EPOCHS,
    PRINT_EVERY,
    USE_CLASS_WEIGHTS,
    USE_WEIGHTED_SAMPLER,
    WEIGHT_DECAY,
)
from dataset import GeoLocateDataset
from model import Net


def get_device():
    """Return the best available torch device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_counts(dataset):
    """Return per-class image counts aligned to label indices."""
    counts = torch.zeros(len(dataset.label_map), dtype=torch.float32)
    sector_indices = dataset.rows["sector"].map(dataset.label_map)
    for class_idx, class_count in sector_indices.value_counts().items():
        counts[int(class_idx)] = float(class_count)
    return counts


def build_class_weights(class_counts):
    """Return normalized inverse-frequency weights for CrossEntropyLoss."""
    if torch.any(class_counts <= 0):
        raise RuntimeError("Class counts must be > 0 to compute class weights.")

    weights = 1.0 / class_counts
    # Normalize to keep average gradient scale near unweighted loss.
    weights = weights / weights.mean()
    return weights


def build_weighted_sampler(dataset, class_counts):
    """Return a sampler that oversamples minority classes."""
    sample_weights = dataset.rows["sector"].map(
        lambda sector: 1.0 / float(class_counts[dataset.label_map[sector]])
    )
    sample_weights = torch.tensor(sample_weights.to_list(), dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def run_training_epochs(net, trainloader, device, optimizer, criterion, num_epochs, epoch_offset=0):
    """Run one training phase for num_epochs."""
    for phase_epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % PRINT_EVERY == PRINT_EVERY - 1:
                print(
                    f"[{epoch_offset + phase_epoch + 1}, {i + 1:5d}] "
                    f"loss: {running_loss / PRINT_EVERY:.3f}"
                )
                running_loss = 0.0


def train(net, trainloader, device, criterion):
    """Train net with head warmup then full-network fine-tuning."""
    head_epochs = min(HEAD_WARMUP_EPOCHS, NUM_EPOCHS)
    finetune_epochs = max(NUM_EPOCHS - head_epochs, 0)

    print(f"Phase 1/2: train classifier head for {head_epochs} epoch(s)")
    net.freeze_backbone()
    head_optimizer = optim.SGD(
        net.backbone.fc.parameters(),
        lr=HEAD_LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    run_training_epochs(net, trainloader, device, head_optimizer, criterion, head_epochs)

    if finetune_epochs > 0:
        print(f"Phase 2/2: fine-tune full network for {finetune_epochs} epoch(s)")
        net.unfreeze_backbone()
        backbone_params = [
            param
            for name, param in net.backbone.named_parameters()
            if not name.startswith("fc.")
        ]
        finetune_optimizer = optim.SGD(
            [
                {"params": backbone_params, "lr": BACKBONE_LEARNING_RATE},
                {"params": net.backbone.fc.parameters(), "lr": FINETUNE_HEAD_LEARNING_RATE},
            ],
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        run_training_epochs(
            net,
            trainloader,
            device,
            finetune_optimizer,
            criterion,
            finetune_epochs,
            epoch_offset=head_epochs,
        )

    print("Finished Training")


def main():
    device = get_device()
    print(f"Using device: {device}")

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found. Run prepare_dataset.py before training."
        )

    train_dataset = GeoLocateDataset("train")
    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training split is empty. Re-run prepare_dataset.py to regenerate splits."
        )

    class_counts = compute_class_counts(train_dataset)
    class_weights = build_class_weights(class_counts)

    print(
        "Class count range (train split): "
        f"min={int(class_counts.min().item())}, max={int(class_counts.max().item())}"
    )
    print(
        "Balancing config: "
        f"USE_CLASS_WEIGHTS={USE_CLASS_WEIGHTS}, "
        f"USE_WEIGHTED_SAMPLER={USE_WEIGHTED_SAMPLER}"
    )

    sampler = build_weighted_sampler(train_dataset, class_counts) if USE_WEIGHTED_SAMPLER else None
    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
    )

    num_classes = len(train_dataset.label_map)
    if num_classes < 2:
        raise RuntimeError(
            "Training requires at least 2 classes. "
            "Adjust sectoring/filtering and rebuild the manifest."
        )

    if USE_CLASS_WEIGHTS:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    net = Net(num_classes, pretrained=True).to(device)

    train(net, trainloader, device, criterion)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    try:
        torch.save(net.state_dict(), CHECKPOINT_PATH)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write checkpoint to {CHECKPOINT_PATH}: {exc}"
        ) from exc
    print(f"Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
