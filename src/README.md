# EchoGesture: Sign Language Prediction on Edge Device (Track 02)

Official submission package for **Samsung XR Hackathon Challenge 2026**.

Organized by: **XR Perception Group & PRISM Team, Samsung R&D Institute India, Bangalore (SRI-B)**.

---

## 1. Solution Architecture

EchoGesture addresses the **Conversation Gap** ($\le 50\text{ ms}$ latency budget) and the **On-Device Privacy Principle** (zero cloud streaming of raw video) using a two-stage edge inference pipeline:

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
1. **Pose Invariance:** Coordinates are translated to the upper-body shoulder midpoint and scaled by shoulder span. Hand landmarks are anchored in a wrist-relative coordinate frame.
2. **Compact Edge Footprint:** Classifier utilizes depthwise-separable 1D convolutions ($< 2\text{ MB}$ INT8 binary footprint), easily satisfying the hackathon's $< 20\text{ MB}$ total budget.
3. **Sub-50ms Latency:** Highly optimized temporal pooling and vectorized NumPy transforms ensure real-time responsiveness without thermal throttling.
4. **Zero-Crash Contract:** Handles missing hands, extreme camera angles, low illumination, and corrupted video containers without crashing.

---

## 2. Directory Hierarchy

Submissions conform strictly to the required hackathon hierarchy:

```
team_name.zip
├── src/
│   ├── main.py                  # Official evaluator entrypoint exposing predict()
│   ├── extract_landmarks.py     # MediaPipe pose & hand coordinate extraction
│   ├── features.py              # 3D normalization, wrist-relative frames, velocities
│   ├── model.py                 # Depthwise-separable 1D TCN PyTorch / ONNX definition
│   ├── validate_submission.py   # Automated submission compliance verifier
│   ├── requirements.txt         # Pinned Python package dependencies
│   └── README.md                # Compilation, execution & reproducibility guide
└── results/
    ├── predictions.jsonl        # Output JSON lines for test dataset
    └── results.csv              # Self-reported benchmark results sheet
```

---

## 3. Environment & Setup

### Requirements:
- Python 3.10 / 3.11
- Standard Linux / Android ARM64 environment

### Installation:
```bash
python3 -m pip install -r requirements.txt
```

### Reproducible Docker Execution:
```bash
# Build the standardized container
docker build -t echogesture:latest -f Dockerfile .

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
# Run from repository root
python3 src/validate_submission.py .
```

---

## 6. Interactive XR Demo & Deployment

EchoGesture includes two out-of-the-box demonstration interfaces:

### A. Zero-Dependency Local Web Dashboard
```bash
python3 src/demo_server.py 8080
# Open http://localhost:8080 in your browser
```
Features real-time telemetry counters, glowing 3D wireframe playback, candidate probability bars, and Web Speech API audio Text-To-Speech (TTS).

### B. Hugging Face Spaces / Gradio Public App
Deployable in 2 minutes to Hugging Face Spaces (Free CPU tier):
- Create a Space at https://huggingface.co/new-space (Gradio SDK)
- Push `app.py`, `src/requirements.txt`, and `src/`
- Yields a public URL: `https://huggingface.co/spaces/<user>/echogesture`

