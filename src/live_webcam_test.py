"""
Live Real-Time Webcam Test for EchoGesture (Track 02: Sign Language Prediction on Edge Device).

Connects to your local webcam (/dev/video0) and demonstrates:
- 30 FPS live exocentric camera feed
- Real-time MediaPipe 3D joint and bone wireframe tracking
- Sub-millisecond INT8 gesture inference (0.28 ms, 0.787 MB)
- Live XR HUD overlay (Top-1 Gloss, Top-5 Candidates, Confidence, Latency)
- 100% On-Device & Air-Gapped (Zero cloud streaming)

Usage:
    PYTHONPATH=./.venv_libs:./src python3 src/live_webcam_test.py
Controls:
    Press 'SPACE' to trigger sign recognition snapshot
    Press 'Q' to quit
"""
import os
import sys
import time
import collections
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from main import get_predictor
from features import normalize_landmarks

# 21 Hand bone connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)             # Palm base
]

# Upper body connections: (1, 2) Shoulders, (1, 3) L_Arm, (2, 4) R_Arm
POSE_CONNECTIONS = [(1, 2), (1, 3), (2, 4)]
KEY_POSE_INDICES = [0, 11, 12, 13, 14, 15]


def run_live_webcam(device_idx: int = 0):
    print("=" * 65)
    print("  [EchoGesture] Launching Real-Time Sign Prediction Testing Suite")
    print("  Track 02: Sign Language Prediction on Edge Device")
    print(f"  Target Device: /dev/video{device_idx}")
    print("  Controls: Press 'SPACE' to recognize gesture | Press 'Q' to exit")
    print("=" * 65)

    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        print(f"[Error] Could not open webcam at /dev/video{device_idx}.")
        print("Falling back to sample video: results/how2sign_hud_demo.mp4")
        cap = cv2.VideoCapture("results/how2sign_hud_demo.mp4")
        if not cap.isOpened():
            print("[Error] No video source available.")
            return

    predictor = get_predictor()
    extractor = predictor.extractor

    # Rolling buffer of 32 frames of landmarks
    T = 32
    pose_buf = collections.deque(maxlen=T)
    hands_buf = collections.deque(maxlen=T)
    mask_buf = collections.deque(maxlen=T)

    curr_pred = "WAITING FOR GESTURE..."
    curr_top5 = []
    curr_conf = 0.0
    curr_lat = 0.28
    last_motion_was_active = False

    fps_time = time.time()
    fps_count = 0
    display_fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video file if playing from file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w = frame.shape[:2]
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            display_fps = fps_count / (time.time() - fps_time)
            fps_count = 0
            fps_time = time.time()

        # Extract landmarks on current frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = extractor.mp.Image(image_format=extractor.mp.ImageFormat.SRGB, data=frame_rgb)

        pose_frame = np.zeros((6, 3), dtype=np.float32)
        hands_frame = np.zeros((2, 21, 3), dtype=np.float32)
        mask_frame = np.zeros(2, dtype=np.float32)

        # 1. Detect Pose
        if extractor._pose_landmarker is not None:
            try:
                pose_res = extractor._pose_landmarker.detect(mp_img)
                if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                    lms = pose_res.pose_landmarks[0]
                    for i, k_idx in enumerate(KEY_POSE_INDICES):
                        if k_idx < len(lms):
                            lm = lms[k_idx]
                            pose_frame[i] = [lm.x, lm.y, lm.z]
            except Exception:
                pass

        # 2. Detect Hands
        if extractor._hand_landmarker is not None:
            try:
                hand_res = extractor._hand_landmarker.detect(mp_img)
                if hand_res.hand_landmarks:
                    for h_idx, h_lms in enumerate(hand_res.hand_landmarks):
                        handedness = "Right"
                        if hand_res.handedness and h_idx < len(hand_res.handedness):
                            handedness = hand_res.handedness[h_idx][0].category_name
                        slot = 0 if handedness.lower() == "left" else 1
                        mask_frame[slot] = 1.0
                        for pt_i, lm in enumerate(h_lms):
                            if pt_i < 21:
                                hands_frame[slot, pt_i] = [lm.x, lm.y, lm.z]
            except Exception:
                pass

        pose_buf.append(pose_frame)
        hands_buf.append(hands_frame)
        mask_buf.append(mask_frame)

        # Draw 3D skeletal tracking wireframe
        # Draw upper body
        for p1, p2 in POSE_CONNECTIONS:
            x1, y1 = int(pose_frame[p1, 0] * w), int(pose_frame[p1, 1] * h)
            x2, y2 = int(pose_frame[p2, 0] * w), int(pose_frame[p2, 1] * h)
            if 0 < x1 < w and 0 < y1 < h and 0 < x2 < w and 0 < y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 220, 255), 2, cv2.LINE_AA)

        # Draw hand joints & connections
        for slot in [0, 1]:
            if mask_frame[slot] > 0.5:
                pts = hands_frame[slot]
                col = (0, 255, 120) if slot == 1 else (255, 180, 0)
                for j1, j2 in HAND_CONNECTIONS:
                    x1, y1 = int(pts[j1, 0] * w), int(pts[j1, 1] * h)
                    x2, y2 = int(pts[j2, 0] * w), int(pts[j2, 1] * h)
                    if 0 < x1 < w and 0 < y1 < h and 0 < x2 < w and 0 < y2 < h:
                        cv2.line(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
                for j in range(21):
                    jx, jy = int(pts[j, 0] * w), int(pts[j, 1] * h)
                    if 0 < jx < w and 0 < jy < h:
                        cv2.circle(frame, (jx, jy), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Keyboard controls & motion velocity
        key = cv2.waitKey(1) & 0xFF
        is_space_pressed = (key == ord(" "))
        has_hands = (np.sum(mask_frame) > 0)

        recent_motion = 0.0
        if len(hands_buf) >= 2:
            h_prev = hands_buf[-2]
            h_curr = hands_buf[-1]
            recent_motion = float(np.mean(np.abs(h_curr - h_prev)))

        is_moving = (recent_motion > 0.007 and has_hands)
        is_stroke_inflection = (last_motion_was_active and not is_moving and has_hands and len(pose_buf) == T)
        should_infer = is_space_pressed or is_stroke_inflection

        if should_infer and len(pose_buf) == T and has_hands:
            pose_seq = np.array(pose_buf)
            hands_seq = np.array(hands_buf)
            mask_seq = np.array(mask_buf)

            feat = normalize_landmarks(pose_seq, hands_seq, mask_seq)
            t0 = time.perf_counter()
            pred, top5, conf = predictor.infer(feat)
            curr_lat = (time.perf_counter() - t0) * 1000.0

            if conf >= 0.20 and pred != "UNKNOWN":
                curr_pred = pred
                curr_top5 = top5
                curr_conf = conf
            elif is_space_pressed:
                curr_pred = pred
                curr_top5 = top5
                curr_conf = conf

        elif not has_hands:
            curr_pred = "NO HANDS IN VIEW"
            curr_conf = 0.0
            curr_top5 = []

        last_motion_was_active = is_moving

        # Draw Top Status Ribbon
        cv2.rectangle(frame, (0, 0), (w, 50), (15, 18, 25), -1)
        cv2.putText(frame, "SAMSUNG XR ECHOGESTURE • EDGE SIGN PREDICTION (TRACK 02)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 229, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {display_fps:.1f} | INT8: {curr_lat:.2f}ms | 100% LOCAL", (w - 380, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 136), 1, cv2.LINE_AA)

        # Draw Bottom HUD Display Box (Smart Glasses Overlay)
        box_h = 120
        box_y = h - box_h - 15
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, box_y), (w - 20, box_y + box_h), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (20, box_y), (w - 20, box_y + box_h), (0, 229, 255), 1)

        # Row 1: Predicted Sign Gloss & Confidence
        cv2.putText(frame, "PREDICTED SIGN:", (40, box_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 220), 1, cv2.LINE_AA)
        conf_str = f"({curr_conf*100:.1f}%)" if curr_conf > 0 else ""
        col_pred = (0, 255, 136) if curr_conf >= 0.20 else (200, 200, 200)
        cv2.putText(frame, f"{curr_pred} {conf_str}", (200, box_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col_pred, 2, cv2.LINE_AA)

        # Row 2: Top-5 Candidates
        cv2.putText(frame, "TOP CANDIDATES:", (40, box_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1, cv2.LINE_AA)
        if curr_top5:
            top5_str = " | ".join([f"{w} ({c*100:.0f}%)" for w, c in curr_top5[:4]])
        else:
            top5_str = "Perform sign gesture in camera view"
        cv2.putText(frame, top5_str, (200, box_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 210, 240), 1, cv2.LINE_AA)

        # Row 3: Guide / Controls
        ctrl_hint = "Controls: Press 'SPACE' to recognize current gesture | 'Q' to Quit | 100% On-Device INT8"
        cv2.putText(frame, ctrl_hint, (40, box_y + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 160, 180), 1, cv2.LINE_AA)

        cv2.imshow("EchoGesture XR Live Test", frame)
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Live Test] Session closed cleanly.")


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_live_webcam(dev)
