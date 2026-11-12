"""
High-Performance Multi-Core Parallel Landmark Extractor for EchoGesture.

Utilizes all available CPU cores (up to 8 parallel workers) to extract
and cache 320-dimensional normalized 3D landmarks in minutes.
"""
import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from extract_landmarks import VideoLandmarkExtractor
from features import normalize_landmarks


def _worker_extract_clip(task: Tuple[str, Dict[str, Any], str, str]) -> Dict[str, Any]:
    """Worker function executed inside each process worker."""
    cid, meta, vpath, target_npz = task
    class_idx = meta["action"][0]
    subset = meta["subset"]

    if os.path.isfile(target_npz):
        return {
            "clip_id": cid,
            "label": int(class_idx),
            "npz_path": target_npz,
            "subset": subset,
            "success": True,
            "cached": True
        }

    try:
        # Initialize extractor inside worker process to avoid pickling locks
        extractor = VideoLandmarkExtractor(target_t=32)
        raw = extractor.extract(vpath)
        extractor.close()

        feat = normalize_landmarks(raw["pose"], raw["hands"], raw["hand_mask"])

        os.makedirs(os.path.dirname(os.path.abspath(target_npz)), exist_ok=True)
        np.savez_compressed(
            target_npz,
            features=feat,
            valid=raw["valid"],
            hand_mask=raw["hand_mask"]
        )

        return {
            "clip_id": cid,
            "label": int(class_idx),
            "npz_path": target_npz,
            "subset": subset,
            "success": True,
            "cached": False
        }
    except Exception as exc:
        return {
            "clip_id": cid,
            "label": int(class_idx),
            "npz_path": target_npz,
            "subset": subset,
            "success": False,
            "error": str(exc)
        }


def run_parallel_extraction(
    subset_json: str = "data/archive/nslt_100.json",
    videos_dir: str = "data/archive/videos",
    output_dir: str = "data/cache_wlasl100",
    max_workers: int = 8
):
    os.makedirs(output_dir, exist_ok=True)
    with open(subset_json, "r") as f:
        manifest = json.load(f)

    # Collect available video clips
    tasks = []
    for cid, meta in manifest.items():
        vpath = os.path.join(videos_dir, f"{cid}.mp4")
        if os.path.isfile(vpath):
            target_npz = os.path.join(output_dir, f"{cid}.npz")
            tasks.append((cid, meta, vpath, target_npz))

    total = len(tasks)
    print("=" * 65)
    print(f" [Parallel Extraction] Launching {max_workers} worker processes")
    print(f" Total clips to process: {total}")
    print(f" Output cache directory : {output_dir}")
    print("=" * 65)

    train_records = []
    val_records = []
    test_records = []

    t0 = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_extract_clip, task): task for task in tasks}

        for fut in as_completed(futures):
            res = fut.result()
            completed += 1

            if res.get("success"):
                rec = {
                    "clip_id": res["clip_id"],
                    "label": res["label"],
                    "npz_path": res["npz_path"],
                    "subset": res["subset"]
                }
                if res["subset"] == "train":
                    train_records.append(rec)
                elif res["subset"] == "val":
                    val_records.append(rec)
                else:
                    test_records.append(rec)

            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - t0
                speed = completed / max(0.1, elapsed)
                remaining = (total - completed) / max(0.1, speed)
                print(f"  [{completed:04d}/{total:04d}] ({speed:.1f} clips/s | ~{remaining:.0f}s left) | Train: {len(train_records)} | Val: {len(val_records)}")

    total_time = time.time() - t0
    index_path = os.path.join(output_dir, "dataset_index.json")
    with open(index_path, "w") as f:
        json.dump({
            "total_cached": len(train_records) + len(val_records) + len(test_records),
            "train": train_records,
            "val": val_records,
            "test": test_records
        }, f, indent=2)

    print("\n" + "=" * 65)
    print(f" [Parallel Extraction Complete] Finished in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")
    print(f"  Dataset index saved to: {index_path}")
    print("=" * 65)
    return index_path


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    run_parallel_extraction(max_workers=workers)

