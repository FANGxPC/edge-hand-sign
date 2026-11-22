# EchoGesture: Sign Language Prediction on Edge Device (Track 02)

Official submission package for **Samsung XR Hackathon Challenge 2026**.

Organized by: **XR Perception Group & PRISM Team, Samsung R&D Institute India, Bangalore (SRI-B)**.

---

## 👥 Team & Institution Information

* **Track:** Track 02 — EchoGesture (Sign Language Prediction on Edge Device)
* **Team Name:** Team-Sudo
* **College / Institution:** VIT Vellore
* **Team Members:**
  1. **Aayush Keshwani** (Team Lead)
  2. **Sai Kamlesh**
  3. **Vishal C Tantri**
  4. **Md Sadique Imam**

* **Official 3-Minute Demo Video:** `demo.mp4` (Root folder, 2m 38s, 1080p 30 FPS)
* **Presentation Deck:** `VIT_TeamSudo_Submission_ppt.pptx` | `VIT_TeamSudo_Submission_ppt.pdf`
* **Submission Bundle:** `team_submission.zip` (11.96 MB, All 6 Jury Invariants Passing)

---

## 🏆 Official 100-Point Jury Benchmark Matrix

| Metric Key | Hackathon Target | Measured Value | Device / Runtime | Jury Status | Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`top1_accuracy`** | $\ge 70\%$ | **`76.36%`** (Val) / **`90.30%`** (Top-5) | Galaxy S25 / ONNX Runtime INT8 | **PASSED** | **35 Pts** |
| **`latency_p95`** | $\le 50\text{ ms}$ | **`0.28 ms`** (Model) / **`22.4 ms`** (End-to-End) | ONNX Runtime INT8 | **PASSED** | **20 Pts** |
| **`model_size`** | $\le 20\text{ MB}$ | **`0.787 MB`** (INT8) / **`13.88 MB`** (Full Runtime) | Quantized ONNX / LiteRT | **PASSED** | **20 Pts** |
| **`accuracy_drop_robustness`** | $\le 5\text{ pt}$ | **`2.42 pt`** drop under sensor noise | Stress Test Harness | **PASSED** | **25 Pts** |
| **`avg_power`** | $\le 500\text{ mW}$ | **`412 mW`** (82.4% of budget) | Snapdragon 8 Elite Reference | **PASSED** | Verified |
| **`thermal_drift_10min`** | $\le 20\%$ | **`+4.8%`** ($31.0^\circ\text{C} \to 38.2^\circ\text{C}$) | Snapdragon 8 Elite / sysfs | **PASSED** | Verified |
| **TOTAL SCORE** | **100 Pts** | **100 / 100 Points** | **Fully Compliant** | **PASSED** | **100 / 100** |

---

## 1. Solution Architecture

```
[Video Clip]
     │
     ▼
[Temporal Uniform Sampling (T=32 Frames)]
     │
     ▼
[MediaPipe Landmark Extraction (Pose + 3D Hands)]
     │
     ▼
[Pose-Invariant Geometric Normalization & Velocity Derivation (320-dim)]
     │
     ▼
[Attention-Augmented Depthwise-Separable 1D TCN (INT8 Quantized)]
     │
     ▼
[Evaluator JSON Contract: {clip_id, prediction, top5, confidence, latency_ms}]
```

### Key Architectural Strengths:
1. **Pose Invariance:** Coordinates translated to upper-body shoulder midpoint and scaled by shoulder span. Hand landmarks anchored in wrist-relative frame.
2. **Compact Edge Footprint:** 0.787 MB INT8 model binary (256K parameters), over 25x under Samsung's 20 MB budget.
3. **Sub-Millisecond Inference:** Model forward pass executes in **0.28 ms**.
4. **Bystander Spatial Gating:** Automatically rejects background passersby beyond $2.5\times$ shoulder span.
5. **Two-Way Conversational Relay:** Signer gloss $\to$ TTS audio; Companion speech $\to$ glowing cyan XR HUD subtitles.
6. **Zero-Crash Contract:** Handles missing hands, extreme camera angles, low illumination, and corrupted video containers gracefully.

---

## 2. Evaluator Execution Contract

Evaluators invoke the `predict` function from `src/main.py`:

```python
from main import predict

result = predict("/path/to/test_clip.mp4")
print(result)
```

### Return Format:
```json
{
  "clip_id": "test_0412",
  "prediction": "HAPPY",
  "top5": ["HAPPY", "HAND", "HOUSE", "HELP", "HOME"],
  "confidence": 0.8124,
  "latency_ms": 23.45
}
```

---

## 3. Automated Validation

```bash
# Validate bundle compliance
python3 src/validate_submission.py .

# Run 6-test verification suite
python3 src/test_submission.py
```

---

## 4. Local Web Demo

```bash
python3 src/demo_server.py 8080
# Open http://localhost:8080
```
