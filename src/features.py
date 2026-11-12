"""
Pose-Invariant 3D Hand and Pose Feature Engineering for EchoGesture.

Complies with Track 02 specifications:
- Pose-invariant 3D normalization
- Resilient to distance variations (shoulder scale normalization)
- Hand-local coordinate frame (wrist-relative)
- Dynamic velocity features
- Semantic skeletal features:
  1. Fingertip-to-thumb distances (pinch & grip tracking)
  2. Fingertip-to-wrist distances (finger extension)
  3. Inter-hand distances (two-handed gesture geometry)
  4. Finger joint curl angles (curvature invariant to 3D orientation)
- Hand presence masks
"""
from typing import Dict, Any, Tuple
import numpy as np

# Feature vector dimension constants
NUM_POSE_PTS = 6        # Nose(0), L_Sh(11), R_Sh(12), L_Elb(13), R_Elb(14), L_Wr(15)
NUM_HANDS = 2           # Left, Right
NUM_HAND_PTS = 21       # 21 landmarks per hand
COORDS = 3              # (x, y, z)

POSE_FEAT_DIM = NUM_POSE_PTS * COORDS                    # 18
HAND_FEAT_DIM = NUM_HANDS * NUM_HAND_PTS * COORDS        # 126
VEL_FEAT_DIM = POSE_FEAT_DIM + HAND_FEAT_DIM             # 144
MASK_FEAT_DIM = 2                                        # 2 (Left, Right present)

# Semantic skeletal dimensions
THUMB_DIST_DIM = 8       # 4 tips x 2 hands
WRIST_DIST_DIM = 10      # 5 tips x 2 hands
INTER_HAND_DIM = 2       # Wrist-to-wrist & Index-to-index distances
CURL_ANGLE_DIM = 10      # 5 finger curl angles x 2 hands

GEOM_FEAT_DIM = THUMB_DIST_DIM + WRIST_DIST_DIM + INTER_HAND_DIM + CURL_ANGLE_DIM # 30

TOTAL_FEATURE_DIM = POSE_FEAT_DIM + HAND_FEAT_DIM + VEL_FEAT_DIM + MASK_FEAT_DIM + GEOM_FEAT_DIM # 320


def normalize_landmarks(
    pose_seq: np.ndarray,
    hands_seq: np.ndarray,
    hand_mask_seq: np.ndarray
) -> np.ndarray:
    """
    Transforms raw 3D MediaPipe coordinates into scale-, distance-, and viewpoint-normalized features.

    Args:
        pose_seq: np.ndarray of shape (T, 6, 3)
        hands_seq: np.ndarray of shape (T, 2, 21, 3) [0=Left Hand, 1=Right Hand]
        hand_mask_seq: np.ndarray of shape (T, 2) [Left, Right present]

    Returns:
        np.ndarray of shape (T, 320) containing normalized spatial, velocity, and skeletal features.
    """
    T = pose_seq.shape[0]

    # --- 1. Shoulder Midpoint and Distance Normalization ---
    l_shoulder = pose_seq[:, 1, :] # (T, 3)
    r_shoulder = pose_seq[:, 2, :] # (T, 3)

    shoulder_mid = (l_shoulder + r_shoulder) / 2.0 # (T, 3)
    shoulder_dist = np.linalg.norm(l_shoulder - r_shoulder, axis=-1, keepdims=True) # (T, 1)
    scale = np.where(shoulder_dist > 1e-4, shoulder_dist, 1.0) # (T, 1)

    norm_pose = (pose_seq - shoulder_mid[:, None, :]) / scale[:, None, :] # (T, 6, 3)

    # --- 2. Hand-Local Coordinate Normalization (Wrist-Relative) ---
    norm_hands = np.zeros_like(hands_seq)
    for h in range(NUM_HANDS):
        is_present = hand_mask_seq[:, h] > 0.5
        if np.any(is_present):
            wrist = hands_seq[:, h, 0:1, :] # (T, 1, 3)
            local_hand = hands_seq[:, h] - wrist
            norm_hands[:, h] = local_hand / scale[:, None, :]

    # --- 3. Dynamic Temporal Velocity Features ---
    pose_vel = np.diff(norm_pose, axis=0, prepend=norm_pose[0:1]) # (T, 6, 3)
    hands_vel = np.diff(norm_hands, axis=0, prepend=norm_hands[0:1]) # (T, 2, 21, 3)

    for h in range(NUM_HANDS):
        not_present = hand_mask_seq[:, h] < 0.5
        norm_hands[not_present, h] = 0.0
        hands_vel[not_present, h] = 0.0

    # --- 4. Semantic Skeletal Geometry Features (30-dim) ---
    # A. Fingertip-to-thumb distances (tips: 4, 8, 12, 16, 20)
    thumb_tips = norm_hands[:, :, 4:5, :] # (T, 2, 1, 3)
    other_tips = norm_hands[:, :, [8, 12, 16, 20], :] # (T, 2, 4, 3)
    thumb_dists = np.linalg.norm(other_tips - thumb_tips, axis=-1).reshape(T, THUMB_DIST_DIM) # (T, 8)

    # B. Fingertip-to-wrist distances
    all_tips = norm_hands[:, :, [4, 8, 12, 16, 20], :] # (T, 2, 5, 3)
    wrist_dists = np.linalg.norm(all_tips, axis=-1).reshape(T, WRIST_DIST_DIM) # (T, 10)

    # C. Inter-hand distances (Left=0, Right=1)
    both_present = (hand_mask_seq[:, 0] * hand_mask_seq[:, 1])[:, None] # (T, 1)
    wrist_dist = np.linalg.norm(norm_hands[:, 0, 0, :] - norm_hands[:, 1, 0, :], axis=-1, keepdims=True) * both_present
    index_dist = np.linalg.norm(norm_hands[:, 0, 8, :] - norm_hands[:, 1, 8, :], axis=-1, keepdims=True) * both_present
    inter_hand = np.concatenate([wrist_dist, index_dist], axis=-1) # (T, 2)

    # D. Finger joint curl angles (cosine of angle between adjacent finger segments)
    finger_joints = [
        (2, 3, 4),   # Thumb: MCP, IP, Tip
        (5, 6, 8),   # Index: MCP, PIP, Tip
        (9, 10, 12), # Middle: MCP, PIP, Tip
        (13, 14, 16),# Ring: MCP, PIP, Tip
        (17, 18, 20) # Pinky: MCP, PIP, Tip
    ]
    curl_angles = []
    for mcp, pip, tip in finger_joints:
        v1 = norm_hands[:, :, pip, :] - norm_hands[:, :, mcp, :]
        v2 = norm_hands[:, :, tip, :] - norm_hands[:, :, pip, :]
        dot = np.sum(v1 * v2, axis=-1)
        norm1 = np.linalg.norm(v1, axis=-1) + 1e-6
        norm2 = np.linalg.norm(v2, axis=-1) + 1e-6
        cos_curl = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
        curl_angles.append(cos_curl)
    flat_curl_angles = np.stack(curl_angles, axis=-1).reshape(T, CURL_ANGLE_DIM) # (T, 10)

    # --- 5. Flatten and Concatenate to 320 Dimensions ---
    flat_pose = norm_pose.reshape(T, POSE_FEAT_DIM)
    flat_hands = norm_hands.reshape(T, HAND_FEAT_DIM)
    flat_pose_vel = pose_vel.reshape(T, POSE_FEAT_DIM)
    flat_hands_vel = hands_vel.reshape(T, HAND_FEAT_DIM)
    flat_mask = hand_mask_seq.reshape(T, MASK_FEAT_DIM)

    feature_matrix = np.concatenate(
        [
            flat_pose, flat_hands, flat_pose_vel, flat_hands_vel, flat_mask,
            thumb_dists, wrist_dists, inter_hand, flat_curl_angles
        ],
        axis=-1
    ).astype(np.float32)

    return feature_matrix
