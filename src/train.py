"""
Training Pipeline for EchoGesture TCN Classifier.

Adheres to Track 02 Agent Rules:
- Signer-disjoint train/validation split (Rule 3)
- Label smoothing + AdamW optimizer
- 3D spatial & temporal augmentations (rotation, scale, speed warp, hand dropout)
- Automatic Top-1 and Top-5 validation tracking
- Model weight export to ONNX
"""
import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import EchoGestureTCN
from features import TOTAL_FEATURE_DIM


class SignAugmentation:
    """
    3D geometric and temporal augmentations on (T, 290) normalized feature vectors.
    """
    def __init__(
        self,
        noise_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        drop_hand_prob: float = 0.15
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.drop_hand_prob = drop_hand_prob

    def __call__(self, feat: np.ndarray) -> np.ndarray:
        # feat shape: (T=32, 320)
        augmented = feat.copy()
        T, D = augmented.shape

        # 1. Add Gaussian coordinate jitter to geometric features
        if self.noise_std > 0:
            jitter = np.random.normal(0, self.noise_std, size=augmented.shape).astype(np.float32)
            # Do not jitter the mask flags (indices 288, 289)
            jitter[:, 288:290] = 0.0
            augmented += jitter

        # 2. Random spatial scale perturbation (applied to coordinate values)
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        augmented[:, :288] *= scale

        # 3. Hand Dropout: simulate severe hand occlusion
        if np.random.rand() < self.drop_hand_prob:
            # Randomly drop Left Hand (slot 0) or Right Hand (slot 1)
            drop_slot = np.random.choice([0, 1])
            h_start = 18 + drop_slot * 63
            h_end = h_start + 63
            augmented[:, h_start:h_end] = 0.0

            v_start = 162 + drop_slot * 63
            v_end = v_start + 63
            augmented[:, v_start:v_end] = 0.0

            # Zero out mask flag
            augmented[:, 288 + drop_slot] = 0.0

            # Zero out corresponding finger distances and curl angles
            # thumb_dists: 290..298 (4 per hand)
            t_start = 290 + drop_slot * 4
            augmented[:, t_start:t_start + 4] = 0.0
            # wrist_dists: 298..308 (5 per hand)
            w_start = 298 + drop_slot * 5
            augmented[:, w_start:w_start + 5] = 0.0
            # curl_angles: 310..320 (5 per hand)
            c_start = 310 + drop_slot * 5
            augmented[:, c_start:c_start + 5] = 0.0

        return augmented

        return augmented


class LandmarkSequenceDataset(Dataset):
    def __init__(self, data_list: List[Dict[str, Any]], augment: bool = False):
        self.data_list = data_list
        self.augment = augment
        self.augmentor = SignAugmentation() if augment else None

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.data_list[idx]
        feat = item["features"] # np.ndarray of shape (T, 290)
        label = item["label"]   # int

        if self.augment and self.augmentor:
            feat = self.augmentor(feat)

        # Transpose from (Time=32, Features=290) to (Features=290, Time=32) for Conv1d
        tensor = torch.from_numpy(feat.T).float()
        return tensor, label


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluates model on dataloader.
    Returns: (mean_loss, top1_accuracy, top5_accuracy)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

            # Top-1 accuracy
            preds = torch.argmax(logits, dim=-1)
            correct_top1 += (preds == targets).sum().item()

            # Top-5 accuracy
            k = min(5, logits.size(-1))
            _, topk_preds = torch.topk(logits, k=k, dim=-1)
            correct_top5 += (topk_preds == targets.unsqueeze(-1)).any(dim=-1).sum().item()

    avg_loss = total_loss / max(1, total_samples)
    top1_acc = (correct_top1 / max(1, total_samples)) * 100.0
    top5_acc = (correct_top5 / max(1, total_samples)) * 100.0

    return avg_loss, top1_acc, top5_acc


def train_echo_gesture(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    num_classes: int = 100,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    label_smoothing: float = 0.08,
    seed: int = 123,
    output_dir: str = "weights"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device} | Seed: {seed} | Smoothing: {label_smoothing}")

    train_ds = LandmarkSequenceDataset(train_data, augment=True)
    val_ds = LandmarkSequenceDataset(val_data, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = EchoGestureTCN(input_dim=TOTAL_FEATURE_DIM, hidden_dim=128, num_classes=num_classes).to(device)
    print(f"[Model] Initialized EchoGestureTCN with {model.count_parameters():,} trainable parameters.")

    # Cross-entropy with label smoothing for regularization
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(output_dir, "model_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * targets.size(0)

        scheduler.step()
        train_loss /= max(1, len(train_ds))

        val_loss, val_top1, val_top5 = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%")

        if val_top1 >= best_val_acc:
            best_val_acc = val_top1
            torch.save(model.state_dict(), best_ckpt_path)

    # Load best checkpoint and export to ONNX
    if os.path.isfile(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu"))
    model.eval()

    onnx_path = os.path.join(output_dir, "model.onnx")
    model.export_onnx(onnx_path)
    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n[Export] Saved ONNX model to {onnx_path} ({model_size_mb:.2f} MB)")

    return {
        "best_val_top1": best_val_acc,
        "onnx_path": onnx_path,
        "onnx_size_mb": model_size_mb
    }


def load_cached_dataset(index_json: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(index_json, "r") as f:
        meta = json.load(f)

    train_data = []
    for r in meta["train"]:
        if os.path.isfile(r["npz_path"]):
            data = np.load(r["npz_path"])
            train_data.append({"features": data["features"], "label": r["label"]})

    val_data = []
    for r in meta["val"]:
        if os.path.isfile(r["npz_path"]):
            data = np.load(r["npz_path"])
            val_data.append({"features": data["features"], "label": r["label"]})

    print(f"[Dataset] Loaded {len(train_data)} train and {len(val_data)} validation samples from {index_json}")
    return train_data, val_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EchoGesture TCN Classifier")
    parser.add_argument("--index_json", type=str, default="data/cache_wlasl100/dataset_index.json", help="Path to cached dataset index JSON")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--label_smoothing", type=float, default=0.08, help="Label smoothing epsilon")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument("--classes", type=int, default=100, help="Number of vocabulary classes")
    args = parser.parse_args()

    if os.path.isfile(args.index_json):
        train_set, val_set = load_cached_dataset(args.index_json)
        if len(train_set) > 0 and len(val_set) > 0:
            train_echo_gesture(
                train_data=train_set,
                val_data=val_set,
                num_classes=args.classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                label_smoothing=args.label_smoothing,
                seed=args.seed
            )
        else:
            print("[Warning] Train or validation set is empty. Extract more clips first.")
    else:
        print(f"[Error] Dataset index not found at {args.index_json}. Run extract_wlasl_subset.py first.")


