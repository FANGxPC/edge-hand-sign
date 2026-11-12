"""
Advanced Transfer-Learning & Pretrained Backbone Training for EchoGesture.

Leverages:
- Pretrained dilated attention backbone from WLASL-300 (2,660 clips)
- Temporal speed-warp augmentation (0.85x to 1.15x resampled trajectories)
- Stochastic hand occlusion and coordinate jitter
- Cosine Annealing with Warm Restarts
- Evaluates on both validation (165 clips) and test (100 clips) splits
"""
import os
import sys
import time
import json
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import EchoGestureTCN
from features import TOTAL_FEATURE_DIM


class AdvancedSignAugmentation:
    def __init__(self, noise_std=0.025, scale_range=(0.88, 1.12), drop_hand_prob=0.18):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.drop_hand_prob = drop_hand_prob

    def __call__(self, feat: np.ndarray) -> np.ndarray:
        # feat: (T=32, D=320)
        aug = feat.copy()
        T, D = aug.shape

        # 1. Temporal speed warp: randomly resample time dimension
        if np.random.rand() < 0.4:
            new_t = np.random.randint(26, 38)
            orig_indices = np.linspace(0, T - 1, new_t)
            resampled = np.zeros((new_t, D), dtype=np.float32)
            for d in range(D):
                resampled[:, d] = np.interp(orig_indices, np.arange(T), aug[:, d])
            # Resample back to standard 32
            back_indices = np.linspace(0, new_t - 1, T)
            for d in range(D):
                aug[:, d] = np.interp(back_indices, np.arange(new_t), resampled[:, d])

        # 2. Gaussian coordinate jitter
        if self.noise_std > 0:
            jitter = np.random.normal(0, self.noise_std, size=aug.shape).astype(np.float32)
            jitter[:, 288:290] = 0.0  # Keep presence masks binary
            aug += jitter

        # 3. Spatial scale perturbation
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        aug[:, :288] *= scale

        # 4. Hand dropout occlusion simulation
        if np.random.rand() < self.drop_hand_prob:
            drop_slot = np.random.choice([0, 1])
            # Left=0 (18..81), Right=1 (81..144)
            h_start = 18 + drop_slot * 63
            aug[:, h_start : h_start + 63] = 0.0
            v_start = 162 + drop_slot * 63
            aug[:, v_start : v_start + 63] = 0.0
            aug[:, 288 + drop_slot] = 0.0

            # Zero out associated geometric metrics
            aug[:, 290 + drop_slot * 4 : 290 + (drop_slot + 1) * 4] = 0.0
            aug[:, 298 + drop_slot * 5 : 298 + (drop_slot + 1) * 5] = 0.0
            aug[:, 310 + drop_slot * 5 : 310 + (drop_slot + 1) * 5] = 0.0

        return aug


class CachedDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
        self.aug = AdvancedSignAugmentation() if augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        feat = item["features"].copy()
        if self.augment:
            feat = self.aug(feat)
        # Transpose to (Channels=320, Time=32) for Conv1d
        x = torch.from_numpy(feat.T).float()
        y = torch.tensor(item["label"], dtype=torch.long)
        return x, y


def evaluate(model, loader, device):
    model.eval()
    correct_top1, correct_top5, total = 0, 0, 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)

            # Top 1
            preds = logits.argmax(dim=-1)
            correct_top1 += (preds == y).sum().item()

            # Top 5
            top5_preds = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
            correct_top5 += (top5_preds == y.unsqueeze(-1)).any(dim=-1).sum().item()
            total += y.size(0)

    val_loss = total_loss / max(1, total)
    top1 = (correct_top1 / max(1, total)) * 100.0
    top5 = (correct_top5 / max(1, total)) * 100.0
    return val_loss, top1, top5


def run_advanced_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Advanced Training] Using compute device: {device}")

    # 1. Load cached dataset
    index_path = "data/cache_wlasl100/dataset_index.json"
    with open(index_path, "r") as f:
        meta = json.load(f)

    train_samples = [{"features": np.load(s["npz_path"])["features"], "label": s["label"]} for s in meta["train"]]
    val_samples = [{"features": np.load(s["npz_path"])["features"], "label": s["label"]} for s in meta["val"]]
    test_samples = [{"features": np.load(s["npz_path"])["features"], "label": s["label"]} for s in meta.get("test", [])]

    print(f"[Dataset] Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    train_ds = CachedDataset(train_samples, augment=True)
    val_ds = CachedDataset(val_samples, augment=False)
    test_ds = CachedDataset(test_samples, augment=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 2. Initialize Model & Load Pretrained Backbone
    model = EchoGestureTCN(input_dim=TOTAL_FEATURE_DIM, hidden_dim=128, num_classes=100).to(device)
    pretrained_300 = "weights_wlasl300/model_300_best.pt"

    if os.path.isfile(pretrained_300):
        print(f"[Transfer Learning] Loading pretrained backbone from {pretrained_300}...")
        ckpt = torch.load(pretrained_300, map_location=device)
        # Load all layers except the final classification head (head.weight, head.bias)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict and "head" not in k and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"[Transfer Learning] Successfully transferred {len(pretrained_dict)} layer tensors into backbone!")

    # 3. Training Hyperparameters
    epochs = 40
    lr = 8e-4
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_top1 = 0.0
    best_test_top1 = 0.0
    save_path = "src/weights/model_advanced_best.pt"

    print("\n--- Starting 40 Epochs Training on RTX 2050 ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * y.size(0)

        scheduler.step()
        train_loss /= max(1, len(train_ds))
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, device)
        _, test_top1, test_top5 = evaluate(model, test_loader, device)
        elapsed = time.time() - t0

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_test_top1 = test_top1
            torch.save(model.state_dict(), save_path)
            mark = " ★ BEST"
        else:
            mark = ""

        if epoch % 5 == 0 or epoch == epochs or mark:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Loss: {train_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}% | Test Top-1: {test_top1:.2f}%{mark}")

    print(f"\n[Finished] Best Val Top-1: {best_val_top1:.2f}% | Associated Test Top-1: {best_test_top1:.2f}%")
    return best_val_top1, best_test_top1


if __name__ == "__main__":
    run_advanced_training()

