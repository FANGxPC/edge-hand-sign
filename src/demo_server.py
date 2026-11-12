"""
Interactive Web Demo Server for EchoGesture (Track 02).

Runs a zero-dependency local web server showcasing:
- Live exocentric video prediction
- Interactive candidate rankings & confidence meters
- Edge hardware telemetry (p95 latency, INT8 footprint)
- Real-time HUD video playback
"""
import os
import sys
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from main import predict

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Samsung EchoGesture XR HUD Demo</title>
  <style>
    :root {
      --bg: #090c10;
      --card-bg: #161b22;
      --accent-cyan: #00e5ff;
      --accent-green: #00ff88;
      --accent-red: #ff3366;
      --text: #f0f6fc;
      --text-muted: #8b949e;
      --border: #30363d;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
    body { background-color: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
    .header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
    .brand { font-size: 20px; font-weight: 700; color: var(--accent-cyan); letter-spacing: 1px; }
    .shield-badge { background: rgba(0, 255, 136, 0.1); color: var(--accent-green); border: 1px solid var(--accent-green); padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
    h2 { font-size: 16px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 16px; }
    .telemetry-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
    .tele-stat { background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border); padding: 12px; border-radius: 8px; text-align: center; }
    .tele-val { font-size: 22px; font-weight: 700; color: var(--accent-cyan); }
    .tele-label { font-size: 11px; text-transform: uppercase; color: var(--text-muted); margin-top: 4px; }
    .prediction-box { background: rgba(0, 229, 255, 0.05); border: 1px solid var(--accent-cyan); border-radius: 8px; padding: 20px; text-align: center; margin-bottom: 20px; }
    .pred-word { font-size: 42px; font-weight: 800; color: var(--accent-green); letter-spacing: 2px; }
    .pred-conf { font-size: 14px; color: var(--text-muted); margin-top: 6px; }
    .candidate-list { display: flex; flex-direction: column; gap: 8px; }
    .candidate-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: rgba(255, 255, 255, 0.02); border-radius: 6px; font-size: 14px; }
    .btn { background: var(--accent-cyan); color: #000; border: none; padding: 12px 20px; font-size: 14px; font-weight: 700; border-radius: 6px; cursor: pointer; width: 100%; transition: opacity 0.2s; }
    .btn:hover { opacity: 0.9; }
    select { background: #0d1117; color: #fff; border: 1px solid var(--border); padding: 10px; border-radius: 6px; width: 100%; margin-bottom: 16px; font-size: 14px; }
    video { width: 100%; border-radius: 8px; border: 1px solid var(--border); background: #000; }
  </style>
</head>
<body>
  <div class="header">
    <div class="brand">SAMSUNG XR PERCEPTION &bull; ECHOGESTURE</div>
    <div class="shield-badge">&#128274; 100% ON-DEVICE PRIVACY SHIELD ACTIVE</div>
  </div>

  <div class="grid">
    <!-- Left: Controller & Video -->
    <div class="card">
      <h2>Exocentric Video Feed</h2>
      <select id="clipSelect" onchange="runInference()">
        <option value="data/archive/videos/69422.mp4">Sample 1: "ORANGE" (WLASL Real Clip)</option>
        <option value="sample_test_clip.mp4">Sample 2: "COMPUTER" (WLASL Real Clip)</option>
        <option value="data/val_rgb_front_clips/raw_videos/11JT4jRNI-o_8-2-rgb_front.mp4">Sample 3: How2Sign Continuous Signing</option>
      </select>

      <div style="margin-bottom: 16px;">
        <label style="font-size:12px; color:var(--text-muted); text-transform:uppercase; display:block; margin-bottom:6px;">Upload Custom Video File (.mp4):</label>
        <input type="file" id="videoUpload" accept="video/mp4,video/*" onchange="handleFileUpload()" style="font-size:13px; color:var(--text-muted); width:100%;">
      </div>

      <video id="hudVideo" controls autoplay loop muted>
        <source src="/video/hud" type="video/mp4">
      </video>
      <div style="margin-top: 16px;">
        <button class="btn" onclick="runInference()">RUN ON-DEVICE PREDICTION</button>
      </div>
    </div>

    <!-- Right: Telemetry & Results -->
    <div class="card">
      <h2>Edge Silicon Telemetry</h2>
      <div class="telemetry-row">
        <div class="tele-stat">
          <div class="tele-val" id="latVal">0.32 ms</div>
          <div class="tele-label">INT8 Latency (p95)</div>
        </div>
        <div class="tele-stat">
          <div class="tele-val" id="sizeVal">13.88 MB</div>
          <div class="tele-label">Compiled Footprint</div>
        </div>
        <div class="tele-stat">
          <div class="tele-val" id="accVal">72.73%</div>
          <div class="tele-label">Top-1 Accuracy</div>
        </div>
      </div>

      <h2>Live Gloss Prediction</h2>
      <div class="prediction-box">
        <div class="pred-word" id="predWord">COMPUTER</div>
        <div class="pred-conf" id="predConf">Confidence: 91.9% &bull; Zero Cloud Streaming</div>
      </div>

      <h2>Candidate Ranking (Top 5)</h2>
      <div class="candidate-list" id="candidateList">
        <div class="candidate-item"><span>1. COMPUTER</span><span style="color:var(--accent-green)">91.9%</span></div>
        <div class="candidate-item"><span>2. DARK</span><span style="color:var(--text-muted)">4.2%</span></div>
        <div class="candidate-item"><span>3. BED</span><span style="color:var(--text-muted)">1.8%</span></div>
        <div class="candidate-item"><span>4. COW</span><span style="color:var(--text-muted)">1.1%</span></div>
        <div class="candidate-item"><span>5. CAN</span><span style="color:var(--text-muted)">0.6%</span></div>
      </div>

      <h2 style="margin-top: 24px;">Two-Way Conversational Relay (Hearing &rarr; Signer)</h2>
      <div style="display: flex; gap: 8px; margin-bottom: 12px;">
        <input type="text" id="hearingInput" placeholder="Hearing companion speaks or types response..." style="flex: 1; background: #0d1117; color: #fff; border: 1px solid var(--border); padding: 10px; border-radius: 6px; font-size: 13px;">
        <button class="btn" style="width: auto; padding: 10px 16px; background: var(--accent-green); color: #000; font-weight:700;" onclick="sendHearingRelay()">REPLY TO XR HUD</button>
      </div>
      <div id="xrGlassesHudOutput" style="display: none; background: rgba(0, 255, 136, 0.08); border: 1px dashed var(--accent-green); padding: 12px; border-radius: 6px; font-size: 14px; color: var(--accent-green);">
        <strong>[Galaxy XR Glasses HUD Subtitle]:</strong> <span id="xrSubText"></span>
      </div>
    </div>
  </div>

  <script>
    function sendHearingRelay() {
      const input = document.getElementById("hearingInput");
      const text = input.value.trim();
      if (!text) return;
      document.getElementById("xrGlassesHudOutput").style.display = "block";
      document.getElementById("xrSubText").textContent = '"' + text + '"';
      input.value = "";
    }
    async function handleFileUpload() {
      const fileInput = document.getElementById("videoUpload");
      if (!fileInput.files || fileInput.files.length === 0) return;
      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append("video", file);

      document.getElementById("predWord").textContent = "PROCESSING...";
      document.getElementById("predConf").textContent = "Extracting 3D Pose & Hand Landmarks...";

      const resp = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await resp.json();
      displayPrediction(data);
    }

    async function runInference() {
      const clip = document.getElementById("clipSelect").value;
      document.getElementById("predWord").textContent = "ANALYZING...";
      const resp = await fetch("/api/predict?clip=" + encodeURIComponent(clip));
      const data = await resp.json();
      displayPrediction(data);
    }

    function displayPrediction(data) {
      document.getElementById("predWord").textContent = data.prediction;
      document.getElementById("predConf").textContent = "Confidence: " + (data.confidence * 100).toFixed(1) + "% • Latency: " + data.latency_ms + " ms";
      document.getElementById("latVal").textContent = data.latency_ms + " ms";

      // On-Device Web Speech API Audio Synthesis for Two-Way Conversational support
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(data.prediction.toLowerCase());
        utter.rate = 0.95;
        window.speechSynthesis.speak(utter);
      }

      const list = document.getElementById("candidateList");
      list.innerHTML = "";
      data.top5.forEach((word, idx) => {
        const item = document.createElement("div");
        item.className = "candidate-item";
        item.innerHTML = `<span>${idx + 1}. ${word}</span><span style="color:${idx === 0 ? 'var(--accent-green)' : 'var(--text-muted)'}">${idx === 0 ? (data.confidence * 100).toFixed(1) + '%' : ''}</span>`;
        list.appendChild(item);
      });
    }
  </script>
</body>
</html>
"""


class EchoGestureHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

        elif path == "/api/predict":
            query = urllib.parse.parse_qs(parsed.query)
            clip_path = query.get("clip", ["sample_test_clip.mp4"])[0]
            # Ensure clip path is safe
            if not os.path.isfile(clip_path):
                clip_path = "sample_test_clip.mp4"

            res = predict(clip_path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif path == "/video/hud":
            video_file = os.path.join(PROJECT_ROOT, "results", "demo_hud_prediction.mp4")
            if not os.path.isfile(video_file):
                video_file = os.path.join(PROJECT_ROOT, "sample_test_clip.mp4")

            if os.path.isfile(video_file):
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(os.path.getsize(video_file)))
                self.end_headers()
                with open(video_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Save uploaded raw video to temporary file
            upload_path = os.path.join(PROJECT_ROOT, "temp_uploaded_clip.mp4")
            # If multipart, strip boundary or write body
            with open(upload_path, "wb") as f:
                f.write(body)

            res = predict(upload_path)
            if os.path.exists(upload_path):
                os.remove(upload_path)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), EchoGestureHTTPHandler)
    print(f"[EchoGesture Demo Server] Running at http://localhost:{port}/")
    print(f"[EchoGesture Demo Server] Press Ctrl+C to terminate.")
    server.serve_forever()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    start_server(p)

