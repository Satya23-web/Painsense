"""
preprocessing.py

Purpose: Take raw video frames, detect+crop the face, resize, and save
processed images ready for the CNN.

DUMMY MODE: Since we don't have the real UNBC-McMaster dataset yet, this
version generates a few random fake "frames" instead of reading real
video, so we can test that the folder structure, saving, and resizing
logic all work correctly. Later, we swap generate_dummy_frames() for a
real "read frames from video file" function - the rest stays the same.
"""

import os
import numpy as np
import cv2

# Folder where processed (cropped, resized) face images will be saved
PROCESSED_DIR = "data/processed"

# Standard size we resize every face crop to before feeding the CNN
IMG_SIZE = (224, 224)


def generate_dummy_frames(num_frames=16, subject_id="subject_01"):
    """
    Creates a list of random fake 'frames' (as if they came from a video).
    Each frame is just random pixel noise shaped like a real image would be.
    This stands in for real video frames until we have the real dataset.
    """
    frames = []
    for i in range(num_frames):
        # Random image: height=300, width=300, 3 color channels (RGB)
        fake_frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        frames.append(fake_frame)
    return frames


def fake_face_crop(frame):
    """
    In the real version, this function would use MediaPipe/OpenCV to
    detect where the face is and crop just that region.

    For now (dummy mode), we just crop the center of the fake frame,
    since there's no real face to detect in random noise.
    """
    h, w, _ = frame.shape
    # Crop the center 200x200 region as a stand-in for "the face"
    center_crop = frame[h // 2 - 100:h // 2 + 100, w // 2 - 100:w // 2 + 100]
    return center_crop


def preprocess_and_save(subject_id="subject_01", num_frames=16):
    """
    Full pipeline for one 'video' (subject):
    1. Get frames (dummy: generated; real: read from video file)
    2. Crop the face from each frame
    3. Resize to fixed size
    4. Save to data/processed/<subject_id>/frame_XXX.jpg
    """
    frames = generate_dummy_frames(num_frames=num_frames, subject_id=subject_id)

    # Create a subfolder for this subject so frames stay grouped together
    subject_folder = os.path.join(PROCESSED_DIR, subject_id)
    os.makedirs(subject_folder, exist_ok=True)

    for idx, frame in enumerate(frames):
        cropped = fake_face_crop(frame)
        resized = cv2.resize(cropped, IMG_SIZE)

        save_path = os.path.join(subject_folder, f"frame_{idx:03d}.jpg")
        cv2.imwrite(save_path, resized)

    print(f"Saved {num_frames} processed frames for {subject_id} to {subject_folder}")


if __name__ == "__main__":
    # Run this file directly to test: generates dummy data for 2 fake subjects
    preprocess_and_save(subject_id="subject_01", num_frames=16)
    preprocess_and_save(subject_id="subject_02", num_frames=16)