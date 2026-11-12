"""
Package Submission Bundle for Samsung XR Hackathon (Track 02).

Creates the exact ZIP hierarchy required by the organizers:
team_name.zip
├── src/
│   ├── main.py
│   ├── requirements.txt
│   ├── README.md
│   ├── model.py
│   ├── features.py
│   ├── extract_landmarks.py
│   ├── quantize.py
│   ├── hud_generator.py
│   └── weights/
│       ├── model_int8.onnx
│       ├── model.onnx
│       ├── hand_landmarker.task
│       ├── pose_landmarker_lite.task
│       ├── golden_vectors.json
│       └── vocab.txt
└── results/
    ├── predictions.jsonl
    └── results.csv
"""
import os
import sys
import zipfile

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_submission_zip(output_zip_path: str = "team_submission.zip"):
    src_files = [
        "src/main.py",
        "src/requirements.txt",
        "src/README.md",
        "src/Dockerfile",
        "src/model.py",
        "src/features.py",
        "src/extract_landmarks.py",
        "src/quantize.py",
        "src/validate_submission.py",
        "src/test_submission.py",
        "src/hud_generator.py",
        "src/profile_power_thermal.py",
        "src/demo_server.py",
        "src/live_webcam_test.py",
        "src/continuous_sentence_decoder.py",
        "src/weights/model_int8.onnx",
        "src/weights/model.onnx",
        "src/weights/hand_landmarker.task",
        "src/weights/pose_landmarker_lite.task",
        "src/weights/golden_vectors.json",
        "src/weights/vocab.txt",
    ]

    results_files = [
        "results/predictions.jsonl",
        "results/predictions.json",
        "results/results.csv"
    ]

    print(f"[Packaging] Creating submission zip: {output_zip_path}")
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in src_files:
            if os.path.isfile(f):
                zf.write(f, arcname=f)
                print(f"  + Added {f} ({os.path.getsize(f) / 1024:.1f} KB)")
            else:
                print(f"  ! Warning: {f} not found!")

        for f in results_files:
            if os.path.isfile(f):
                zf.write(f, arcname=f)
                print(f"  + Added {f} ({os.path.getsize(f) / 1024:.1f} KB)")
            else:
                print(f"  ! Warning: {f} not found!")

    zip_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print(f"\n[Success] Created {output_zip_path} ({zip_size_mb:.2f} MB)")
    return output_zip_path

if __name__ == "__main__":
    zip_name = sys.argv[1] if len(sys.argv) > 1 else "team_submission.zip"
    create_submission_zip(zip_name)

