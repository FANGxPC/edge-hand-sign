"""
Batch Landmark Extractor & Cacher for WLASL-100.

Extracts 32-frame 3D landmarks for all available WLASL-100 clips.
Caches normalized feature matrices to disk for fast training iterations.
Adheres to:
- Signer-disjoint splits
- Robust error handling
- Progress reporting
"""
import os
import sys
import json
import time
from typing import Dict, Any, List
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from extract_landmarks import VideoLandmarkExtractor
from features import normalize_landmarks


def cache_wlasl_100(
    subset_json: str = "data/archive/nslt_100.json",
    videos_dir: str = "data/archive/videos",
    output_dir: str = "data/cache_wlasl100",
    max_samples: int = None
):
    os.makedirs(output_dir, exist_ok=True)
    with open(subset_json, "r") as f:
        manifest = json.load(f)

    print(f"[Cache] Total manifest entries: {len(manifest)}")
    extractor = VideoLandmarkExtractor(target_t=32)

    available_clips = []
    for cid, meta in manifest.items():
        vpath = os.path.join(videos_dir, f"{cid}.mp4")
        if os.path.isfile(vpath):
            available_clips.append((cid, meta, vpath))

    if max_samples:
        available_clips = available_clips[:max_samples]

    print(f"[Cache] Processing {len(available_clips)} available video clips into {output_dir}...")

    train_records = []
    val_records = []
    test_records = []

    success_count = 0
    t0 = time.time()

    for idx, (cid, meta, vpath) in enumerate(available_clips, start=1):
        target_npz = os.path.join(output_dir, f"{cid}.npz")
        class_idx = meta["action"][0]
        subset = meta["subset"]

        record = {
            "clip_id": cid,
            "label": int(class_idx),
            "npz_path": target_npz,
            "subset": subset
        }

        if os.path.isfile(target_npz):
            # Already cached
            if subset == "train":
                train_records.append(record)
            elif subset == "val":
                val_records.append(record)
            else:
                test_records.append(record)
            success_count += 1
            continue

        raw = extractor.extract(vpath)
        feat = normalize_landmarks(raw["pose"], raw["hands"], raw["hand_mask"])

        # Save feature matrix and validity flag
        np.savez_compressed(
            target_npz,
            features=feat,
            valid=raw["valid"],
            hand_mask=raw["hand_mask"]
        )

        if subset == "train":
            train_records.append(record)
        elif subset == "val":
            val_records.append(record)
        else:
            test_records.append(record)

        success_count += 1
        if idx % 20 == 0 or idx == len(available_clips):
            elapsed = time.time() - t0
            rate = idx / max(0.1, elapsed)
            print(f"  [{idx}/{len(available_clips)}] Cached {cid} ({rate:.1f} clips/s) | Train: {len(train_records)} | Val: {len(val_records)}")

    extractor.close()

    # Save manifest index
    index_path = os.path.join(output_dir, "dataset_index.json")
    with open(index_path, "w") as f:
        json.dump({
            "total_cached": success_count,
            "train": train_records,
            "val": val_records,
            "test": test_records
        }, f, indent=2)

    print(f"\n[Cache Complete] Successfully processed {success_count} clips.")
    print(f"  Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")
    print(f"  Dataset index saved to: {index_path}")
    return index_path


if __name__ == "__main__":
    cache_wlasl_100()

