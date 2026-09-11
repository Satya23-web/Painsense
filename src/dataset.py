"""
dataset.py

Purpose: Load the processed face-frame folders (e.g. data/processed/subject_01/)
and turn each subject's frames into one ordered sequence + one pain label,
in the format PyTorch's LSTM expects.

DUMMY MODE: Since we don't have real PSPI labels yet, we assign a random
pain label to each subject. Later, we replace the random label with the
real one read from the dataset's label file.
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

PROCESSED_DIR = "data/processed"

# The 4 pain levels we're classifying into. Index 0=none, 1=mild, etc.
PAIN_LEVELS = ["none", "mild", "moderate", "severe"]


class PainDataset(Dataset):
    """
    A PyTorch Dataset represents "how to get one example, given an index."
    PyTorch calls __getitem__ automatically during training to fetch batches.
    """

    def __init__(self, processed_dir=PROCESSED_DIR):
        # Each subfolder inside data/processed/ is one subject/sequence
        self.subject_folders = sorted([
            os.path.join(processed_dir, name)
            for name in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, name))
        ])

        # DUMMY: assign a random label to each subject.
        # Real version: read actual PSPI-derived label from a labels file.
        self.labels = {
            folder: random.randint(0, len(PAIN_LEVELS) - 1)
            for folder in self.subject_folders
        }

    def __len__(self):
        # Tells PyTorch how many total examples (subjects) exist
        return len(self.subject_folders)

    def __getitem__(self, idx):
        """
        Returns one full sequence: all frames for one subject, stacked in
        order, plus that subject's pain label.
        """
        folder = self.subject_folders[idx]

        # Get all frame filenames in order (frame_000.jpg, frame_001.jpg, ...)
        frame_files = sorted(os.listdir(folder))

        frames = []
        for fname in frame_files:
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)  # loads as (H, W, 3) BGR

            # Convert BGR (OpenCV default) to RGB, and normalize pixel
            # values from 0-255 to 0-1, since neural nets train better
            # on small, consistent number ranges.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Rearrange from (H, W, C) to (C, H, W) - PyTorch's expected
            # image tensor shape (channels first).
            img = np.transpose(img, (2, 0, 1))
            frames.append(img)

        # Stack all frames into one tensor: shape (sequence_length, C, H, W)
        sequence = torch.tensor(np.stack(frames), dtype=torch.float32)

        label = self.labels[folder]

        return sequence, label


if __name__ == "__main__":
    # Quick test: load the dataset and print the shape of one example
    dataset = PainDataset()
    print(f"Number of subjects (sequences) found: {len(dataset)}")

    sequence, label = dataset[0]
    print(f"Sequence shape: {sequence.shape}")  # (num_frames, 3, 224, 224)
    print(f"Label: {label} ({PAIN_LEVELS[label]})")