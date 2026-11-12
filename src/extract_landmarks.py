"""
MediaPipe Tasks 3D Landmark Extractor for EchoGesture.

Extracts upper body pose and 3D hand landmarks from exocentric video clips.
Features:
- Uses modern Google MediaPipe Tasks API (hand_landmarker.task & pose_landmarker_lite.task)
- Uniform temporal sampling (T=32 frames)
- Primary signer selection
- 3D coordinates extraction with left/right hand presence masks
- Zero-crash fallback handling
"""
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HAND_MODEL = os.path.join(CURRENT_DIR, "weights", "hand_landmarker.task")
DEFAULT_POSE_MODEL = os.path.join(CURRENT_DIR, "weights", "pose_landmarker_lite.task")


class VideoLandmarkExtractor:
    """
    Robust landmark extractor using Google MediaPipe Tasks API.
    Extracts 6 upper body pose landmarks + 2 x 21 3D hand landmarks.
    """
    def __init__(
        self,
        hand_model_path: str = DEFAULT_HAND_MODEL,
        pose_model_path: str = DEFAULT_POSE_MODEL,
        target_t: int = 32
    ):
        self.target_t = target_t
        self.hand_model_path = hand_model_path
        self.pose_model_path = pose_model_path

        self._hand_landmarker = None
        self._pose_landmarker = None
        self._init_models()

    def _init_models(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self.mp = mp
            if os.path.isfile(self.hand_model_path):
                hand_opts = vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=self.hand_model_path),
                    num_hands=2,
                    min_hand_detection_confidence=0.3,
                    min_hand_presence_confidence=0.3,
                    min_tracking_confidence=0.3
                )
                self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

            if os.path.isfile(self.pose_model_path):
                pose_opts = vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=self.pose_model_path),
                    min_pose_detection_confidence=0.3,
                    min_pose_presence_confidence=0.3,
                    min_tracking_confidence=0.3
                )
                self._pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

        except Exception as e:
            # Safe initialization fallback
            self._hand_landmarker = None
            self._pose_landmarker = None

    def sample_video_frames(self, video_path: str) -> Tuple[List[np.ndarray], bool]:
        if not os.path.isfile(video_path):
            return [], False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], False

        all_frames = []
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > self.target_t:
                target_indices = set(np.linspace(0, total_frames - 1, self.target_t, dtype=int))
                curr_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if curr_idx in target_indices:
                        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if len(all_frames) == self.target_t:
                            break
                    curr_idx += 1
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception:
            pass
        finally:
            cap.release()

        if len(all_frames) == 0:
            return [], False

        if len(all_frames) != self.target_t:
            indices = np.linspace(0, len(all_frames) - 1, self.target_t).astype(int)
            resampled = [all_frames[i] for i in indices]
            return resampled, True

        return all_frames, True

    def extract(self, video_path: str) -> Dict[str, np.ndarray]:
        T = self.target_t
        zero_pose = np.zeros((T, 6, 3), dtype=np.float32)
        zero_hands = np.zeros((T, 2, 21, 3), dtype=np.float32)
        zero_mask = np.zeros((T, 2), dtype=np.float32)

        frames, ok = self.sample_video_frames(video_path)
        if not ok or len(frames) != T:
            return {
                "pose": zero_pose,
                "hands": zero_hands,
                "hand_mask": zero_mask,
                "valid": False
            }

        pose_seq = np.zeros((T, 6, 3), dtype=np.float32)
        hands_seq = np.zeros((T, 2, 21, 3), dtype=np.float32)
        mask_seq = np.zeros((T, 2), dtype=np.float32)

        # Pose keypoint indices in MediaPipe Tasks Pose Landmarker:
        # 0: Nose, 11: L_Shoulder, 12: R_Shoulder, 13: L_Elbow, 14: R_Elbow, 15: L_Wrist
        KEY_POSE_INDICES = [0, 11, 12, 13, 14, 15]

        for t, frame in enumerate(frames):
            try:
                mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame)

                # 1. Pose landmarks (with Primary Signer Disambiguation)
                if self._pose_landmarker is not None:
                    pose_res = self._pose_landmarker.detect(mp_img)
                    if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                        # Exocentric Disambiguation: Pick primary signer closest to center with largest span
                        best_person_lms = pose_res.pose_landmarks[0]
                        if len(pose_res.pose_landmarks) > 1:
                            max_score = -1.0
                            for cand_lms in pose_res.pose_landmarks:
                                if len(cand_lms) > 12:
                                    span = abs(cand_lms[11].x - cand_lms[12].x)
                                    center_dist = abs((cand_lms[11].x + cand_lms[12].x) / 2.0 - 0.5)
                                    score = span / (1.0 + center_dist)
                                    if score > max_score:
                                        max_score = score
                                        best_person_lms = cand_lms
                        lms = best_person_lms
                        for i, k_idx in enumerate(KEY_POSE_INDICES):
                            if k_idx < len(lms):
                                lm = lms[k_idx]
                                pose_seq[t, i] = [lm.x, lm.y, lm.z]

                # 2. Hand landmarks (with Bystander Spatial Gating)
                if self._hand_landmarker is not None:
                    hand_res = self._hand_landmarker.detect(mp_img)
                    if hand_res.hand_landmarks:
                        # Signer reference anchor for spatial gating
                        ref_x = (pose_seq[t, 1, 0] + pose_seq[t, 2, 0]) / 2.0 if pose_seq[t, 1, 0] > 0 else 0.5
                        ref_y = (pose_seq[t, 1, 1] + pose_seq[t, 2, 1]) / 2.0 if pose_seq[t, 1, 1] > 0 else 0.5
                        shoulder_span = abs(pose_seq[t, 1, 0] - pose_seq[t, 2, 0]) if pose_seq[t, 1, 0] > 0 else 0.3
                        max_reach = max(0.45, shoulder_span * 2.5)

                        for h_idx, h_lms in enumerate(hand_res.hand_landmarks):
                            # Wrist coordinate check
                            w_x, w_y = h_lms[0].x, h_lms[0].y
                            dist_to_signer = ((w_x - ref_x) ** 2 + (w_y - ref_y) ** 2) ** 0.5

                            # Rejects background bystander hands outside signer reachable sphere
                            if dist_to_signer > max_reach:
                                continue

                            handedness = "Right"
                            if hand_res.handedness and h_idx < len(hand_res.handedness):
                                handedness = hand_res.handedness[h_idx][0].category_name

                            slot = 0 if handedness.lower() == "left" else 1
                            mask_seq[t, slot] = 1.0

                            for pt_i, lm in enumerate(h_lms):
                                if pt_i < 21:
                                    hands_seq[t, slot, pt_i] = [lm.x, lm.y, lm.z]

            except Exception:
                continue

        return {
            "pose": pose_seq,
            "hands": hands_seq,
            "hand_mask": mask_seq,
            "valid": True
        }

    def close(self):
        try:
            if self._hand_landmarker is not None:
                self._hand_landmarker.close()
            if self._pose_landmarker is not None:
                self._pose_landmarker.close()
        except Exception:
            pass
