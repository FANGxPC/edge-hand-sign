"""
Utility to generate a synthetic test sign video clip for end-to-end pipeline validation.
Creates a standardized 32-frame 30fps MP4 file with moving geometric targets.
"""
import os
import cv2
import numpy as np

def generate_synthetic_video(output_path: str = "sample_test_clip.mp4", num_frames: int = 32, width: int = 640, height: int = 480):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    for i in range(num_frames):
        # Dark ambient background simulating room
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        # Upper torso & shoulders
        cv2.ellipse(frame, (width // 2, height // 2 + 100), (120, 80), 0, 0, 360, (180, 150, 120), -1)

        # Head / face
        cv2.circle(frame, (width // 2, height // 2 - 80), 50, (210, 180, 150), -1)

        # Animated hands moving up and down (signing motion)
        offset_y = int(40 * np.sin(2 * np.pi * i / num_frames))
        # Left Hand
        cv2.circle(frame, (width // 2 - 100, height // 2 + offset_y), 25, (220, 190, 160), -1)
        # Right Hand
        cv2.circle(frame, (width // 2 + 100, height // 2 - offset_y), 25, (220, 190, 160), -1)

        out.write(frame)

    out.release()
    print(f"Generated synthetic test video at: {output_path} ({os.path.getsize(output_path):,} bytes)")
    return output_path

if __name__ == "__main__":
    generate_synthetic_video()

