"""
How2Sign Dataset Loader and Preprocessor for EchoGesture (Track 02).

Handles:
- Ingestion of How2Sign frontal RGB clips
- Parsing of manually re-aligned translation/gloss CSV/TSV manifests
- Extraction of high-frequency conversational sign glosses
- Generation of train/validation/test manifests matching the challenge contract
"""
import os
import sys
import csv
import glob
import tarfile
import zipfile
from typing import Dict, List, Any, Tuple, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def unpack_archives(data_dir: str = DATA_DIR):
    """Automatically extracts any .zip, .tar, or .tar.gz archives in the data directory."""
    archives = glob.glob(os.path.join(data_dir, "*.zip")) + \
               glob.glob(os.path.join(data_dir, "*.tar*")) + \
               glob.glob(os.path.join(data_dir, "*.tgz"))

    for arc in archives:
        print(f"[Archive] Extracting {os.path.basename(arc)}...")
        if arc.endswith(".zip"):
            with zipfile.ZipFile(arc, "r") as zf:
                zf.extractall(data_dir)
        elif arc.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(arc, "r:*") as tf:
                tf.extractall(data_dir)
        print(f"[Archive] Extraction complete for {os.path.basename(arc)}.")


def parse_translation_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """
    Parses How2Sign English translation CSV/TSV.
    Expected standard columns:
    SENTENCE_NAME / CLIP_ID, START, END, SENTENCE / GLOSS
    """
    if not os.path.isfile(manifest_path):
        return []

    records = []
    # Determine delimiter (comma or tab)
    with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        delimiter = "\t" if "\t" in first_line else ","

    with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            # Normalize column keys
            clean_row = {k.strip().upper(): v.strip() for k, v in row.items() if k}

            clip_id = clean_row.get("SENTENCE_NAME") or clean_row.get("CLIP_ID") or clean_row.get("ID") or clean_row.get("NAME")
            text = clean_row.get("SENTENCE") or clean_row.get("TRANSLATION") or clean_row.get("TEXT") or clean_row.get("GLOSS")

            if clip_id and text:
                records.append({
                    "clip_id": clip_id,
                    "text": text,
                    "tokens": [t.strip(".,!?;:\"").upper() for t in text.split() if t.strip(".,!?;:\"")]
                })

    print(f"[Manifest] Parsed {len(records)} records from {os.path.basename(manifest_path)}")
    return records


def find_matching_videos(data_dir: str = DATA_DIR) -> Dict[str, str]:
    """Finds all video files (.mp4, .mkv, .avi) in data_dir and maps clip_id -> full_path."""
    video_map = {}
    for ext in ("*.mp4", "*.mkv", "*.avi", "*.webm"):
        for path in glob.glob(os.path.join(data_dir, "**", ext), recursive=True):
            filename = os.path.basename(path)
            clip_id = os.path.splitext(filename)[0]
            video_map[clip_id] = path
    print(f"[Videos] Located {len(video_map)} video clips under {data_dir}")
    return video_map


def scan_and_summarize(data_dir: str = DATA_DIR):
    print("=" * 60)
    print(f" Scanning How2Sign Dataset in: {data_dir}")
    print("=" * 60)

    unpack_archives(data_dir)
    videos = find_matching_videos(data_dir)

    manifest_files = glob.glob(os.path.join(data_dir, "*.csv")) + \
                     glob.glob(os.path.join(data_dir, "*.tsv"))

    all_records = []
    for mf in manifest_files:
        all_records.extend(parse_translation_manifest(mf))

    matched = 0
    for rec in all_records:
        if rec["clip_id"] in videos:
            matched += 1

    print(f"\n[Summary]")
    print(f"  Total Video Files   : {len(videos)}")
    print(f"  Total Manifest Rows : {len(all_records)}")
    print(f"  Matched Clips       : {matched}")

    if matched > 0:
        print("[Ready] Data ready for landmark feature extraction!")
    else:
        print("[Notice] Please download/save the How2Sign validation clips into data/")
    print("=" * 60)


if __name__ == "__main__":
    scan_and_summarize()

