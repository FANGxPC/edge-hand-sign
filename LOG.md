# EchoGesture (Track 02): Experiment & Benchmark Log

> **Rule 1 & 9 Compliance:** Every metric, config change, and benchmark must be recorded with exact timestamp, execution command, hardware device, and validation result.

---

## Experiment Entries

### [EXP-000] Project Initialization & Plan Formulation
- **Date & Time:** 2026-11-09T19:54:00+05:30
- **Phase:** Phase 0 (Setup & Specification)
- **Author:** Antigravity Pair-Programmer & FANGxPC
- **Change Description:**
  - Ingested hackathon slide specifications from `XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf`.
  - Created [TRACK_2_INFO.md](file:///home/fang/Downloads/Samsung%20XR%20Hackathon/TRACK_2_INFO.md) with complete contest logistics, rules, and jury rubrics.
  - Formulated [ECHOGESTURE_AGENT_PLAN.md](file:///home/fang/Downloads/Samsung%20XR%20Hackathon/ECHOGESTURE_AGENT_PLAN.md) mapping all 9 phases and non-negotiable rules from [AGENT_RULES_AND_PHASES.md](file:///home/fang/Downloads/Samsung%20XR%20Hackathon/AGENT_RULES_AND_PHASES.md).
- **Environment State:**
  - OS: Linux (Ubuntu 22.04 Jammy / Pop!_OS)
  - System Python: Python 3.10.12
- **Metrics / Status:** Project initialized, awaiting Phase 0 environment setup and dataset ingestion.


### [EXP-001] Model Architecture, INT8 Quantization & Latency Benchmarks
- **Date & Time:** 2026-11-10T10:28:00+05:30
- **Phase:** Phase 0 & Phase 4 (Quantization, Model Footprint & Latency Contract)
- **Author:** Antigravity Pair-Programmer & FANGxPC
- **Command Executed:**
  `PYTHONPATH=./.venv_libs:./src python3 src/quantize.py src/weights/model.onnx src/weights/model_int8.onnx`
- **Hardware / Device:** Local Host (x86_64, Pop!_OS Jammy / Linux 6.x)
- **Measured Numbers:**
  - `model_parameters`: 186,468 trainable weights
  - `model_size_float`: 0.714 MB (749,065 bytes)
  - `model_size_int8`: 0.683 MB (715,704 bytes) [Target: $\le 20\text{ MB}$, Status: PASSED]
  - `float_vs_int8_cos_sim`: 0.99999 across 20 golden vectors
  - `latency_p50_model_only`: 0.172 ms
  - `latency_p95_model_only`: 0.195 ms [Target: $\le 50\text{ ms}$, Status: PASSED]
- **Validation Verdict:**
  `src/validate_submission.py` executed: PASSED. Schema, folder layout, results.csv headers, and predict() contract verified.

### [EXP-002] WLASL Dataset Integration, MediaPipe Tasks & End-to-End Execution
- **Date & Time:** 2026-11-10T20:46:00+05:30
- **Phase:** Phase 1 (Landmark Extraction) & Phase 2 (TCN Baseline)
- **Author:** Antigravity Pair-Programmer & FANGxPC
- **Dataset Ingested:** WLASL (`data/archive/`)
  - Total video clips: 11,980
  - WLASL-100 available: 1,013 clips (748 train, 165 val, 100 test)
  - Vocab: 100 classes saved to `src/weights/vocab.txt`
- **Vision Pipeline:** Google MediaPipe Tasks API
  - `src/weights/hand_landmarker.task`: 7.5 MB
  - `src/weights/pose_landmarker_lite.task`: 5.6 MB
  - Total MediaPipe assets: 13.1 MB
- **Measured Metrics:**
  - 3D feature extraction verified on real WLASL videos (`00335.mp4` to `00341.mp4`).
  - RTX 2050 CUDA training loop verified with CosineAnnealingLR and AdamW.
  - Final INT8 quantized classifier: 0.686 MB (Total shipped package: 13.78 MB $\le 20\text{ MB}$ budget).
  - Evaluator `predict("00335.mp4")` verified live on real sign video with uppercase gloss output and latency profiling.

### [EXP-003] How2Sign Dataset Ingestion & Validation
- **Date & Time:** 2026-11-10T21:03:00+05:30
- **Phase:** Phase 0 & Phase 1 (Data Verification & Extraction)
- **Author:** Antigravity Pair-Programmer & FANGxPC
- **Files Ingested:**
  - `data/how2sign_realigned_train.csv`: 5.6 MB (31,217 sentence annotations)
  - `data/how2sign_realigned_val.csv`: 311 KB (1,741 annotations)
  - `data/how2sign_realigned_test.csv`: 423 KB (2,357 annotations)
  - `data/val_rgb_front_clips/raw_videos/`: 1,739 frontal video clips (99.9% match)
  - `data/test_rgb_front_clips/raw_videos/`: 2,343 frontal video clips (99.4% match)
- **Extraction Test on How2Sign:**
  - Verified clip `0zvsqf23tmw_11-2-rgb_front.mp4`:
    - Pose detected: Upper-body normalized to shoulder span.
    - Hands tracked: Left Hand (26/32 frames), Right Hand (27/32 frames).
    - Feature shape: $(32, 290)$.
  - Live `predict()` tested on How2Sign clip returning compliant jury JSON schema.
- **Disk Space Status:**
  - Available space on nvme mount: 57 GB free (plenty of room for the 31 GB train clips download).

### [EXP-004] Complete Dataset Ingestion (47,109 Videos) & Architectural Upgrades (Attention + 320-dim Skeletal Features)
- **Date & Time:** 2026-11-11T12:51:30+05:30
- **Phase:** Phase 1 (Extraction Setup) & Phase 3 (Model Upgrades)
- **Author:** Antigravity Pair-Programmer & FANGxPC
- **Dataset Inventory Verified:**
  - `data/how2sign_train_clips/raw_videos/`: 31,047 frontal video clips (99.6% match with `how2sign_realigned_train.csv`).
  - `data/val_rgb_front_clips/raw_videos/`: 1,739 frontal video clips (99.9% match).
  - `data/test_rgb_front_clips/raw_videos/`: 2,343 frontal video clips (99.4% match).
  - `data/archive/videos/`: 11,980 WLASL video clips.
  - Total video clips on disk: 47,109 clips (35,129 matched with translation manifests).
- **Architectural Enhancements Implemented:**
  1. **320-dim Feature Vector:** Extended base 290 landmarks with 30 semantic skeletal features:
     - 8 fingertip-to-thumb distances (pinch/grip geometry)
     - 10 fingertip-to-wrist distances (finger extension)
     - 2 inter-hand distances (wrist-to-wrist and index-to-index)
     - 10 finger joint curl cosine angles (rotation-invariant curvature)
  2. **Multi-Head Temporal Self-Attention:** Added a 2-head self-attention block inside `EchoGestureTCN` to weight apex gesture inflection frames.
- **Measured Metrics:**
  - Trainable parameters: 256,612
  - Float ONNX footprint: 0.994 MB (1,042,752 bytes)
  - INT8 Quantized footprint: **0.783 MB** (820,630 bytes) [Budget $\le 20\text{ MB}$, Status: PASSED]
  - Total shipped model package: 13.1 MB (MediaPipe) + 0.783 MB (TCN) = **13.88 MB** [Budget $\le 20\text{ MB}$, Status: PASSED]
  - INT8 Inference Latency (200 runs): **p50: 0.227 ms \| p95: 0.258 ms** [Budget $\le 50.0\text{ ms}$, Status: PASSED]
  - Live `predict()` tested on newly extracted train clip `02LnULLXds4_18-8-rgb_front.mp4`.

### [EXP-005] 72.73% Validation Top-1 Breakthrough, Robustness Pass & Submission Validation
- **Date & Time:** 2026-11-11T14:56:00+05:30
- **Phase:** Phase 3 (Training Optimization) & Phase 5 (Verification & Results Packaging)
- **Author:** Antigravity Pair-Programmer & FANGxPC
- **Training Configuration:**
  - Architecture: `EchoGestureTCN` (256,612 params, 320-dim input, 4 dilated blocks, 2-head self-attention)
  - Optimizer: AdamW ($lr=8e-4$, weight_decay $1e-4$)
  - Schedule: CosineAnnealingLR (60 epochs)
  - Regularization: Label Smoothing ($\epsilon=0.08$) + 3D Augmentations + Seed 123
  - Hardware: NVIDIA GeForce RTX 2050 (3.68 GB VRAM)
- **Measured Metrics:**
  - Validation Top-1 Accuracy: **72.73%** (120/165) [Target: $\ge 70.0\%$, Status: PASSED]
  - Validation Top-5 Accuracy: **88.48%** (146/165)
  - Held-Out Test Set Top-1 Accuracy: **68.00%** (Single Model) / **73.00%** (Ensemble)
  - Robustness Accuracy Drop: **2.42 pt** under 15% sensor noise and frame dropouts [Target: $\le 5.0\text{ pt}$, Status: PASSED]
  - Quantized Model Size: **0.787 MB** INT8 (Total Package: **13.88 MB** $\le 20\text{ MB}$, Status: PASSED)
  - Model Inference Latency: **p50: 0.257 ms | p95: 0.317 ms** [Target: $\le 50.0\text{ ms}$, Status: PASSED]
  - INT8 Golden Vector Parity: **0.99997** cosine similarity
- **Validation Verdict:**
  - `results/results.csv` populated with measured benchmarks (all 6 metrics PASS = TRUE).
  - `results/predictions.jsonl` generated across all 100 test manifest clips.
  - `src/validate_submission.py` executed: **[VERDICT: PASSED] Submission bundle is compliant!**

### [EXP-006] WLASL-300 Full Extraction & Vocabulary Scaling Ablation Benchmark
- **Date & Time:** 2026-11-11T15:33:00+05:30
- **Phase:** Phase 3 (Vocabulary Scalability) & Phase 5 (Ablation Documentation)
- **Author:** Antigravity Pair-Programmer & FANGxPC
- **Extraction Details:**
  - Parallel extraction completed across all **2,660 clips** of WLASL-300 using 8 CPU worker processes in 21.0 minutes.
  - Features saved to `data/cache_wlasl300/`: Train (1,897 clips), Val (446 clips), Test (317 clips).
- **Training Configuration on RTX 2050 (CUDA):**
  - Architecture: `EchoGestureTCN-300` (282,412 parameters, 320-dim input, 300 output classes)
  - Optimizer: AdamW ($lr=8e-4$, weight decay $1e-4$)
  - Schedule: CosineAnnealingLR (60 epochs in 56 seconds, ~0.9s/epoch)
  - Regularization: Label Smoothing ($\epsilon=0.08$) + 3D Augmentations + Seed 123
- **Measured Metrics:**
  - Validation Top-1 Accuracy: **52.47%** (234/446 clips across 300 classes)
  - Validation Top-5 Accuracy: **80.27%** (358/446 clips)
  - Float Model Size: 1.10 MB | INT8 Footprint: **0.842 MB**
  - Inference Latency: **0.345 ms**
- **Artifacts Generated:**
  - Preserved in `weights_wlasl300/model_300.onnx`.
  - Added vocabulary ablation benchmark to `results/PRESENTATION_DECK_CONTENT.md`.
  - Confirmed primary submission model in `src/weights/model_int8.onnx` remains isolated and untouched at 72.73% Top-1.
