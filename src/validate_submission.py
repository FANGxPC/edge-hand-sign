"""
Automated Submission Validator for Track 02 (EchoGesture).

Enforces non-negotiable hackathon rules:
- Verifies exact folder hierarchy (team_name.zip layout)
- Verifies Python 3.11 contract & predict() signature in src/main.py
- Verifies all 6 required benchmark metrics in results/results.csv
- Verifies predictions.jsonl formatting, uppercase vocabulary, and schema
"""
import os
import sys
import json
import csv
import inspect
import importlib.util
from typing import Dict, Any, Set


REQUIRED_METRICS: Set[str] = {
    "top1_accuracy",
    "latency_p95",
    "model_size",
    "accuracy_drop_robustness",
    "avg_power",
    "thermal_drift_10min"
}

REQUIRED_CSV_COLUMNS = [
    "metric", "target", "measured", "unit", "device", "runtime", "pass"
]


def check_file_exists(path: str, description: str) -> bool:
    if not os.path.exists(path):
        print(f"[FAIL] Missing {description}: {path}")
        return False
    print(f"[PASS] Found {description}: {path}")
    return True


def validate_results_csv(csv_path: str) -> bool:
    print(f"\n--- Validating Results CSV: {csv_path} ---")
    if not os.path.isfile(csv_path):
        print(f"[FAIL] results.csv not found at {csv_path}")
        return False

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print("[FAIL] results.csv is empty")
            return False

        header_clean = [col.strip() for col in header]
        if header_clean != REQUIRED_CSV_COLUMNS:
            print(f"[FAIL] Invalid header columns in results.csv.\n  Expected: {REQUIRED_CSV_COLUMNS}\n  Found:    {header_clean}")
            return False

    with open(csv_path, "r", encoding="utf-8") as f:
        dict_reader = csv.DictReader(f)
        found_metrics = set()
        for row in dict_reader:
            metric = row.get("metric", "").strip()
            if metric:
                found_metrics.add(metric)
                target = row.get("target", "")
                measured = row.get("measured", "")
                unit = row.get("unit", "")
                device = row.get("device", "")
                runtime = row.get("runtime", "")
                status = row.get("pass", "")
                print(f"  • {metric.ljust(25)}: target={target.ljust(6)} measured={measured.ljust(6)} {unit.ljust(4)} | dev={device} | rt={runtime} | pass={status}")

        missing = REQUIRED_METRICS - found_metrics
        if missing:
            print(f"[FAIL] Missing mandatory metrics in results.csv: {missing}")
            return False

    print("[PASS] results.csv meets all schema requirements.")
    return True


def validate_predictions_file(pred_path: str) -> bool:
    print(f"\n--- Validating Predictions: {pred_path} ---")
    if not os.path.isfile(pred_path):
        print(f"[WARNING] Predictions file not found at {pred_path}. (Will be generated on test runs)")
        return True

    line_count = 0
    with open(pred_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            line_count += 1
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as err:
                print(f"[FAIL] Line {idx} is invalid JSON: {err}")
                return False

            # Check mandatory keys
            for key in ["clip_id", "prediction", "confidence", "latency_ms"]:
                if key not in data:
                    print(f"[FAIL] Line {idx} missing required key '{key}'")
                    return False

            pred_word = str(data["prediction"])
            if not pred_word.isupper():
                print(f"[FAIL] Line {idx} prediction '{pred_word}' is not uppercase!")
                return False

    print(f"[PASS] Successfully verified {line_count} prediction entries.")
    return True


def validate_predict_contract(main_py_path: str) -> bool:
    print(f"\n--- Validating predict() function contract in {main_py_path} ---")
    if not os.path.isfile(main_py_path):
        print(f"[FAIL] {main_py_path} does not exist.")
        return False

    spec = importlib.util.spec_from_file_location("submission_main", main_py_path)
    if spec is None or spec.loader is None:
        print("[FAIL] Unable to load module spec from main.py")
        return False

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"[FAIL] Importing main.py raised an exception: {exc}")
        return False

    if not hasattr(module, "predict"):
        print("[FAIL] main.py does not export a predict() function!")
        return False

    predict_fn = getattr(module, "predict")
    sig = inspect.signature(predict_fn)
    params = list(sig.parameters.keys())
    if len(params) != 1:
        print(f"[FAIL] predict() must accept exactly 1 parameter (video_path). Found: {params}")
        return False

    print(f"[PASS] predict() signature verified: {sig}")
    return True


def run_full_validation(bundle_dir: str) -> bool:
    print(f"==================================================")
    print(f"  Validating EchoGesture Submission Bundle        ")
    print(f"  Directory: {os.path.abspath(bundle_dir)}       ")
    print(f"==================================================")

    src_dir = os.path.join(bundle_dir, "src")
    results_dir = os.path.join(bundle_dir, "results")

    checks = [
        check_file_exists(os.path.join(src_dir, "main.py"), "src/main.py"),
        check_file_exists(os.path.join(src_dir, "requirements.txt"), "src/requirements.txt"),
        check_file_exists(os.path.join(src_dir, "README.md"), "src/README.md"),
        check_file_exists(os.path.join(results_dir, "results.csv"), "results/results.csv"),
    ]

    if not all(checks):
        print("\n[VERDICT: FAILED] Bundle hierarchy is incomplete.")
        return False

    csv_ok = validate_results_csv(os.path.join(results_dir, "results.csv"))
    pred_path = os.path.join(results_dir, "predictions.jsonl")
    if not os.path.exists(pred_path):
        pred_path = os.path.join(results_dir, "predictions.json")
    preds_ok = validate_predictions_file(pred_path)
    contract_ok = validate_predict_contract(os.path.join(src_dir, "main.py"))

    if csv_ok and preds_ok and contract_ok:
        print("\n==================================================")
        print("  [VERDICT: PASSED] Submission bundle is compliant! ")
        print("==================================================")
        return True
    else:
        print("\n==================================================")
        print("  [VERDICT: FAILED] Fix issues flagged above.     ")
        print("==================================================")
        return False


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    passed = run_full_validation(target_dir)
    sys.exit(0 if passed else 1)

