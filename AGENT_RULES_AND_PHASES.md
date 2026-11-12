# EchoGesture (Track 02): Agent Rules and Implementation Plan

Paste this into your agent. Full detail, code skeletons and the acceptance matrix are in `ECHOGESTURE_AGENT_PLAN.md`.

---

## Agent rules (non-negotiable)

1. Never invent a metric. Every number in `results.csv`, the README, the deck and the video must come from a script you ran. Log the command, device and date.
2. Never fake `latency_ms`. Measure it with `time.perf_counter()`. The jury re-measures it.
3. Never tune on test data. Use signer-independent train and validation splits.
4. Label anything you didn't run as `UNTESTED`. Verify every API name and version against current docs before using it.
5. All visual processing is on-device. No cloud calls, no uploaded frames, no `INTERNET` permission in the Android manifest.
6. `predict()` must never crash. Handle unreadable videos and clips with no hands detected, and return a low-confidence guess instead.
7. Stop and ask me before changing scope: the model family, the vocabulary, or dropping a component.
8. After each phase, report what's done, the measured numbers, the open risks and what you need from me.
9. Keep a `LOG.md` with every experiment: change, config and validation result.

---

## Architecture

`video -> sample T=32 frames -> MediaPipe hand + pose landmarks -> normalized 3D features -> small TCN (depthwise-separable Conv1D) -> INT8 TFLite/ONNX -> UPPERCASE gloss + top5 + confidence`

- **Why landmarks:** they are small and fast, give pose-invariant 3D hand features, and make training cheap because you extract once and train in minutes.
- **Output contract:** `predict(video_path)` in `src/main.py` returns `{clip_id, prediction, top5, confidence, latency_ms}`.

---

## Phases

### Phase 0: Setup (tonight, Nov 9)
- In a clean Python 3.11 venv, install and import `mediapipe` and one runtime (`ai-edge-litert`, `tflite-runtime`, or `onnxruntime`). Pin the versions that work.
- Record the file sizes of the MediaPipe hand and pose landmarker models.
- Inventory which dataset clips actually load.
- Confirm the target vocabulary. If unknown, start with WLASL-100 and keep `NUM_CLASSES` configurable.

### Phase 1: Landmark extraction (Nov 10)
- For each clip, save a `(T, F)` array plus a hand-present mask.
- Pick the primary signer when several people are in frame, using the largest or most central pose.
- Make a signer-disjoint split, cache everything to disk, and report the missing-detection rate.

### Phase 2: Baseline (Nov 11-12)
- **Features:**
  - Translate to the shoulder midpoint and scale by shoulder width.
  - Add hand-local features: wrist-relative coordinates, finger angles and distances.
  - Add per-frame velocity.
  - Use zeros plus a mask flag for missing hands.
- **Training:**
  - TCN with label smoothing and AdamW.
  - Augmentations: 3D rotation, scale and shift, speed warp, jitter, hand dropout, frame drops, and handedness mirroring (test whether it helps).
- **Gate:** report validation Top-1 and Top-5. If validation Top-1 is far below 70% on the target vocabulary, tell me before spending more time.

### Phase 3: Improve (Nov 13)
- Try longer T, then two-stream fusion, then distillation from a bigger teacher, then a small ensemble.
- Stop when the gain per experiment falls below about 0.5 pt.
- Keep the classifier at 4 MB or less INT8 so the total with the landmarkers stays under 20 MB.

### Phase 4: Quantize and export (Nov 14)
- Use full-integer INT8 conversion with 200-500 real landmark sequences as the representative dataset.
- Compare INT8 against float accuracy. If the drop is more than 2 pt, try QAT or restructure the layers that quantize badly.
- Export golden vectors (input to expected logits) for the Android parity test.
- Measure the final shipped size with `ls -l`.

### Phase 5: Robustness (Nov 15)
- Perturb the validation videos: low light plus noise, camera jitter, rotation or perspective, and occlusion patches over the hands.
- Report the accuracy drop per condition against the 5-pt target.
- Fix it with matching augmentations and by training on landmarks extracted from perturbed videos.

### Phase 6: Android app (Nov 15-16)
- **Stack:** Kotlin, Compose, CameraX with `STRATEGY_KEEP_ONLY_LATEST`, MediaPipe Tasks with live-stream mode, and the LiteRT interpreter with a GPU delegate.
- **Streaming:** landmarks are computed as frames arrive, so only the classifier runs after a sign ends.
- **Segmentation and stability:** a sign window is triggered by hands raised above a threshold plus motion, and predictions are gated by a confidence threshold and 2-of-3 voting so captions don't flicker.
- **Output and UI:** large high-contrast HUD captions with a history strip and a debug panel, text-to-speech, and speech-to-text for the hearing speaker (prefer the on-device recognizer).
- **Parity test:** a unit test against the Phase 4 golden vectors. Parity bugs between Python and Android are the most common silent failure.
- **Offline check:** it must run in airplane mode.

### Phase 7: Device benchmarks (Nov 16-17)
- Run on the real phone, offline, with fixed brightness and after a warm-up.
- Run at least 200 timed runs and report p50 and p95 for three things: classifier only, per-frame landmarks, and full-clip end-to-end.
- Thermal drift: run for 10 minutes, log latency every 30 seconds, and report the percent increase.
- Power: report your measurement method and call it an approximation.
- `pass` in `results.csv` is computed by script. A truthful FALSE beats an unverifiable TRUE.

### Phase 8: Package and submit (Nov 17-18)
- **Checks:**
  - `validate_submission.py` checks the zip layout, that every clip_id is present, one JSON object per line, UPPERCASE predictions, and fields and types.
  - Do a clean-room run in a fresh Python 3.11 venv with the network off.
- **Zip layout:** `team.zip/{src/{main.py,requirements.txt,README.md}, results/{predictions.jsonl,results.csv}}`.
- **Deliverables:**
  - README with exact versions and hardware.
  - A conversion script that reproduces the quantized model.
  - A 2-3 minute demo video, in airplane mode, showing the live HUD.
  - The deck `CollegeName_TeamName_Submission_ppt.pptx` with the 5 required slide contents.
- **Timing:** submit on the 18th by early afternoon.

---

## What the agent should tell you honestly

- **Likely:** size, privacy, format and packaging, classifier latency, and the HUD and two-way UX.
- **Uncertain:** Top-1 >= 70% (depends on the vocabulary size), the <= 5-pt robustness drop, end-to-end clip latency in Python, and the power measurement definition.
