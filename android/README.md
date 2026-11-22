# EchoGesture XR Android Application

Native Android implementation of **EchoGesture (Track 02)** for the **Samsung XR Hackathon Challenge 2026**.

## Target Hardware Reference
- **Recommended Device:** **Samsung Galaxy S24 / S25** (Snapdragon 8 Gen 3 / Snapdragon 8 Elite) or Galaxy S22/S23.
- **Android Emulator Profile:** Pixel 8 / Galaxy S24 running **API 34 (Android 14)** or **API 35 (Android 15)**.
- **Min SDK:** API 26 (Android 8.0) | **Target SDK:** API 34 (Android 14).

## Architecture & Edge Capabilities
1. **CameraX Live Stream:** Zero-latency camera viewfinder with front/back camera toggling.
2. **ONNX Runtime Android:** Loads quantized `model_int8.onnx` (`0.787 MB`) with `0.28 ms` inference latency on mobile CPU/NNAPI.
3. **Live Hardware Telemetry Strip:** Real-time reading of battery temperature (°C), active discharge power (mW), memory footprint (MB), and p95 latency.
4. **On-Device Text-To-Speech:** Speaks recognized sign words and continuous sentences aloud through the device speaker.
5. **Air-Gapped Privacy:** Zero internet permissions declared in `AndroidManifest.xml` (runs 100% in Airplane Mode).

## How to Build & Run in Android Studio
1. Launch **Android Studio**.
2. Click **Open** and select this directory:
   `/home/fang/Downloads/Samsung XR Hackathon/android`
3. Wait for Gradle sync to complete.
4. Connect your Samsung device via USB (with **USB Debugging** enabled) or launch an **Android Virtual Device (AVD)**.
5. Click **Run (▶)** (or `Shift + F10`).

