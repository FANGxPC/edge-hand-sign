"""
Full-Integer / Dynamic INT8 Quantization & Golden Vector Export Pipeline.

Complies with Phase 4:
- Quantizes ONNX model to INT8
- Generates golden vectors (test input -> expected logits) for Android parity test
- Measures disk footprint (< 20 MB budget verification)
"""
import os
import sys
import json
import time
from typing import Dict, Any, List
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def quantize_model(
    float_onnx_path: str,
    output_int8_path: str,
    num_golden_samples: int = 20
) -> Dict[str, Any]:
    print(f"\n[Quantization] Input float model: {float_onnx_path}")
    if not os.path.isfile(float_onnx_path):
        raise FileNotFoundError(f"Source ONNX file not found: {float_onnx_path}")

    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnxruntime as ort

    os.makedirs(os.path.dirname(os.path.abspath(output_int8_path)), exist_ok=True)

    # 1. Quantize Dynamic to INT8 (Quantizing linear classification projections)
    print("[Quantization] Performing INT8 Dynamic Quantization...")
    quantize_dynamic(
        model_input=float_onnx_path,
        model_output=output_int8_path,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8
    )

    float_size_bytes = os.path.getsize(float_onnx_path)
    int8_size_bytes = os.path.getsize(output_int8_path)
    float_size_mb = float_size_bytes / (1024 * 1024)
    int8_size_mb = int8_size_bytes / (1024 * 1024)

    print(f"[Footprint] Float Model Size: {float_size_mb:.3f} MB ({float_size_bytes:,} bytes)")
    print(f"[Footprint] INT8  Model Size: {int8_size_mb:.3f} MB ({int8_size_bytes:,} bytes)")
    print(f"[Footprint] Compression ratio: {(1 - int8_size_bytes / float_size_bytes) * 100:.1f}%")
    print(f"[Footprint] Budget check (<= 20 MB): {'PASSED' if int8_size_mb <= 20.0 else 'FAILED'}")

    # 2. Generate Golden Vectors for Android Parity Unit Test
    print(f"\n[Parity] Generating {num_golden_samples} Golden Vectors for Android parity test...")
    float_session = ort.InferenceSession(float_onnx_path, providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(output_int8_path, providers=["CPUExecutionProvider"])

    input_tensor = int8_session.get_inputs()[0]
    input_name = input_tensor.name
    feat_dim = input_tensor.shape[1] if isinstance(input_tensor.shape[1], int) else 320
    time_len = input_tensor.shape[2] if isinstance(input_tensor.shape[2], int) else 32

    golden_vectors = []

    np.random.seed(42)
    for sample_idx in range(num_golden_samples):
        # Generate random normalized sequence matching model input dimensions
        test_input = np.random.uniform(-1.0, 1.0, size=(1, feat_dim, time_len)).astype(np.float32)

        float_logits = float_session.run(None, {input_name: test_input})[0][0]
        int8_logits = int8_session.run(None, {input_name: test_input})[0][0]

        # Cosine similarity between float and int8 predictions
        cos_sim = float(
            np.dot(float_logits, int8_logits) /
            (np.linalg.norm(float_logits) * np.linalg.norm(int8_logits) + 1e-7)
        )

        golden_vectors.append({
            "sample_id": sample_idx,
            "cosine_similarity": round(cos_sim, 5),
            "input_shape": list(test_input.shape),
            "input_sample_flattened": test_input.flatten()[:16].tolist(), # Snippet for sanity check
            "expected_top1_float": int(np.argmax(float_logits)),
            "expected_top1_int8": int(np.argmax(int8_logits)),
            "top5_int8_logits": [round(float(v), 4) for v in np.sort(int8_logits)[::-1][:5]]
        })

    golden_path = os.path.join(os.path.dirname(output_int8_path), "golden_vectors.json")
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden_vectors, f, indent=2)

    avg_cos_sim = np.mean([g["cosine_similarity"] for g in golden_vectors])
    print(f"[Parity] Saved golden vectors to: {golden_path}")
    print(f"[Parity] Average Float vs INT8 Cosine Similarity: {avg_cos_sim:.5f}")

    return {
        "float_size_mb": float_size_mb,
        "int8_size_mb": int8_size_mb,
        "golden_path": golden_path,
        "avg_cos_sim": float(avg_cos_sim)
    }


if __name__ == "__main__":
    float_model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CURRENT_DIR, "weights", "model.onnx")
    int8_model = sys.argv[2] if len(sys.argv) > 2 else os.path.join(CURRENT_DIR, "weights", "model_int8.onnx")
    quantize_model(float_model, int8_model)
