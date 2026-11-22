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

* **Official Presentation Deck:** [`VIT_TeamSudo_Submission_ppt.pptx`](file:///home/fang/Downloads/Samsung%20XR%20Hackathon/VIT_TeamSudo_Submission_ppt.pptx) | [`VIT_TeamSudo_Submission_ppt.pdf`](file:///home/fang/Downloads/Samsung%20XR%20Hackathon/VIT_TeamSudo_Submission_ppt.pdf)
* **Demo Video:** *[3-Minute Demo Video — YouTube / Google Drive Link]*
* **Demo Script & Cues:** [`DEMO_VIDEO_SCRIPT_3MIN.md`](file:///home/fang/Downloads/Samsung%20XR%20Hackathon/DEMO_VIDEO_SCRIPT_3MIN.md)
* **Submission Bundle:** `team_submission.zip` (11.96 MB, All 6 Jury Tests Passing)

---

## 🏆 Official 100-Point Jury Benchmark Matrix

All metrics rigorously measured on reference hardware (Galaxy S25 / Snapdragon 8 Elite & RTX 2050):

| Metric Key | Hackathon Target | Measured Value | Device / Runtime | Jury Status | Points |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`top1_accuracy`** | $\ge 70\%$ | **`76.36%`** (Val) / **`90.30%`** (Top-5) | Galaxy S25 / ONNX Runtime INT8 | **PASSED** | 35 Pts |
| **`latency_p95`** | $\le 50\text{ ms}$ | **`0.28 ms`** (Model) / **`22.4 ms`** (End-to-End) | ONNX Runtime INT8 | **PASSED** | 20 Pts |
| **`model_size`** | $\le 20\text{ MB}$ | **`0.787 MB`** (INT8) / **`13.88 MB`** (Full Pack) | Quantized ONNX / LiteRT | **PASSED** | 20 Pts |
| **`accuracy_drop_robustness`** | $\le 5\text{ pt}$ | **`2.42 pt`** drop under sensor noise | Stress Test Harness | **PASSED** | 25 Pts |
| **`avg_power`** | $\le 500\text{ mW}$ | **`412 mW`** (82.4% of budget) | Snapdragon 8 Elite Reference | **PASSED** | Verified |
| **`thermal_drift_10min`** | $\le 20\%$ | **`+4.8%`** ($31.0^\circ\text{C} \to 38.2^\circ\text{C}$) | Snapdragon 8 Elite / sysfs | **PASSED** | Verified |
| **TOTAL SCORE** | **100 Pts** | **100 / 100 Points** | **Fully Compliant** | **PASSED** | **100 Pts** |

---

## 1. Solution Architecture

EchoGesture addresses the **Conversation Gap** ($\le 50\text{ ms}$ latency budget) and the **On-Device Privacy Principle** (zero cloud streaming of raw video) using a two-stage edge inference pipeline:

```
                    RAW VIDEO STREAM (Frontal Exocentric View)
                                       │
                                       ▼
                [Uniform Temporal Sampling: T=32 Frames]
                                       │
                                       ▼
                [Google MediaPipe Tasks (Pose + 3D Dual-Hands)]
                                       │
                                       ▼
         [Pose-Invariant Skeletal Normalization (320-Dimensional Vector)]
    ┌──────────────────────────────────┴──────────────────────────────────┐
    │ Spatial Geometry Stream (176-dim)                                   │ Motion Dynamics Stream (144-dim)
    │ • Upper-body shoulder-span normalization (18-dim)                  │ • 1st-order velocities: Δx, Δy, Δz (144-dim)
    │ • Wrist-relative hand coordinate frame (126-dim)                   │ • Hand presence indicator masks (2-dim)
    │ • Rotation-invariant joint curl cosine angles (10-dim)             │
    │ • Fingertip-to-thumb pinch geometry (8-dim)                        │
    │ • Fingertip-to-wrist extension distances (10-dim)                  │
    │ • Inter-hand Euclidean wrist & index distance (2-dim)               │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
              [Attention-Augmented Dilated Depthwise-Separable TCN]
              • Input Projection (320 -> 128)
              • 4x Dilated Conv Blocks (d in {1, 2, 4, 8}; Receptive Field = 32)
              • Multi-Head Temporal Self-Attention (Apex Gesture Weighting)
              • Global Adaptive Pooling + Dropout Regularization (256K params)
                                       │
                                       ▼
              [INT8 Dynamically Quantized Linear Classification Head]
              • Quantized footprint: 0.787 MB (25x under Samsung's 20 MB budget)
                                       │
                                       ▼
              [Two-Way Conversational HUD Overlay & JSON Schema Contract]
              { clip_id, prediction, top5, confidence, latency_ms }
```

### Key Architectural Strengths:
1. **Pose Invariance:** Coordinates are translated to the upper-body shoulder midpoint and scaled by shoulder span. Hand landmarks are anchored in a wrist-relative coordinate frame.
2. **Compact Edge Footprint:** Classifier utilizes depthwise-separable 1D convolutions ($0.787\text{ MB}$ INT8 binary footprint), easily satisfying the hackathon's $< 20\text{ MB}$ total budget.
3. **Sub-Millisecond Inference:** Model forward pass executes in **0.28 ms**, enabling smooth 30 FPS HUD overlay with zero thermal downclocking.
4. **Bystander Spatial Gating:** Automatically rejects background passersby beyond $2.5\times$ shoulder span to prevent false detections in public spaces.
5. **Two-Way Conversational Relay:** Deaf sign translated to on-device audio (TTS); hearing companion voice transcribed to glowing cyan HUD subtitles.
6. **Zero-Crash Contract:** Handles missing hands, extreme camera angles, low illumination, and corrupted video containers without crashing.

---

## 2. Directory Hierarchy

Submissions conform strictly to the required hackathon hierarchy (Slide 17 & Slide 25):

```
team_submission.zip / TeamSudo.zip
├── src/
│   ├── main.py                  # Official evaluator entrypoint exposing predict()
│   ├── extract_landmarks.py     # MediaPipe pose & hand coordinate extraction
│   ├── features.py              # 3D normalization, wrist-relative frames, velocities
│   ├── model.py                 # Depthwise-separable 1D TCN PyTorch / ONNX definition
│   ├── quantize.py              # INT8 dynamic quantization script
│   ├── hud_generator.py         # XR conversational HUD renderer
│   ├── demo_server.py           # Interactive local web HUD demo server
│   ├── live_webcam_test.py      # Real-time single-sign webcam recognition
│   ├── profile_power_thermal.py # On-device /proc/self/stat silicon telemetry
│   ├── run_on_phone.py          # Android / Termux offline execution harness
│   ├── validate_submission.py   # Automated submission compliance verifier
│   ├── test_submission.py       # 6-test automated jury verification suite
│   ├── requirements.txt         # Pinned Python package dependencies
│   ├── Dockerfile               # Standardized container for reproducibility
│   ├── README.md                # Compilation, execution & reproducibility guide
│   └── weights/
│       ├── model_int8.onnx      # Quantized INT8 weights (0.787 MB)
│       ├── model.onnx           # FP32 weights (1.0 MB)
│       ├── hand_landmarker.task # MediaPipe hand model (7.5 MB)
│       ├── pose_landmarker_lite.task # MediaPipe pose model (5.5 MB)
│       ├── golden_vectors.json  # Cosine parity test vectors
│       └── vocab.txt            # 100 sign vocabulary
├── results/
│   ├── predictions.jsonl        # Output JSON lines for test dataset
│   ├── predictions.json         # Formatted predictions JSON
│   ├── results.csv              # Official benchmark results sheet (all 6 metrics)
│   ├── edge_thermal_power_benchmark.png # 10-minute thermal & power plot
│   └── PRESENTATION_DECK_CONTENT.md     # Presentation content reference
└── android/                     # Complete native Android Studio project
```

---

## 3. Environment & Setup

### Requirements:
- Python 3.10 / 3.11
- Standard Linux / Android ARM64 environment

### Installation:
```bash
python3 -m pip install -r src/requirements.txt
```

### Reproducible Docker Execution:
```bash
# Build the standardized container
docker build -t echogesture:latest -f src/Dockerfile src/

# Run inference inside container
docker run --rm echogesture:latest python3 -c "from main import predict; print(predict('sample.mp4'))"
```

---

## 4. Evaluator Execution Contract

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

## 5. Automated Bundle Verification

To verify that your submission bundle conforms to all jury requirements:

```bash
# Run schema and hierarchy verification
python3 src/validate_submission.py .

# Run the 6-test jury verification suite
python3 test_submission.py
```

Output:
```
============================================================
 Running EchoGesture Jury Verification Test Suite
============================================================
[Test 1/6] Testing predict() schema contract... -> PASS
[Test 2/6] Testing edge inference latency budget (<= 50 ms)... -> PASS (0.28 ms)
[Test 3/6] Testing quantized model binary footprint (< 20 MB)... -> PASS (0.787 MB)
[Test 4/6] Testing zero-crash fallback robustness... -> PASS
[Test 5/6] Testing FP32 vs INT8 numerical cosine parity... -> PASS (0.99993)
[Test 6/6] Testing on-device privacy (Zero Cloud Streaming)... -> PASS
============================================================
 [ALL 6 TESTS PASSED] SUBMISSION CODE IS 100% COMPLIANT!
============================================================
```

---

## 6. Interactive Web HUD Demonstration

Launch the local interactive console:
```bash
python3 src/demo_server.py 8080
# Open http://localhost:8080 in your browser
```

Features:
- Pre-loaded verified WLASL video clips (`YES`, `NO`, `BOOK`, `COMPUTER`, `HOW`, `SHIRT`).
- Live Webcam Single-Sign Recognition (`⚡ RECOGNIZE SIGN (1.5s)`).
- Two-Way Conversational Relay (Hearing companion speech $\to$ glowing cyan XR HUD subtitles).
- Real-time hardware telemetry dashboard (SoC frequency, power draw, temperature, memory).
- 10-Minute Thermal & Power benchmark visualization.

---

## 7. Android Native & Termux Execution (100% Offline Edge)

### Option A: Native Android App
Open the `android/` directory in **Android Studio**, build, and run directly on a Samsung Galaxy device or emulator in Airplane Mode.

### Option B: Termux Offline CLI
```bash
pkg update -y && pkg install python git -y
git clone https://github.com/FANGxPC/edge-hand-sign.git
cd edge-hand-sign
pip install numpy onnxruntime
python3 src/run_on_phone.py
```
