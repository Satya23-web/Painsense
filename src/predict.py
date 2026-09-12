"""
predict.py

Purpose: Take one subject's processed frames, run them through the trained
model, and print the predicted pain level. This is the "demo" script -
proof the whole pipeline works end-to-end on one real example, not just
in a training loop.

DUMMY MODE: We point this at one of our dummy subject folders. Later,
you'd point it at a brand new video, run it through preprocessing.py
first, then feed the result here.
"""

import os
import sys
import numpy as np
import cv2
import torch

from model import PainSenseModel
from dataset import PAIN_LEVELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "saved_models/painsense_model.pth"


def load_sequence_from_folder(folder_path):
    """
    Loads all frames from a processed subject folder and prepares them
    exactly the way dataset.py does, so the model sees the same format
    it was trained on.
    """
    frame_files = sorted(os.listdir(folder_path))
    frames = []

    for fname in frame_files:
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        frames.append(img)

    sequence = torch.tensor(np.stack(frames), dtype=torch.float32)

    # Model expects a batch dimension even for a single example:
    # shape (1, seq_len, 3, 224, 224)
    sequence = sequence.unsqueeze(0)
    return sequence


def predict(folder_path):
    # Load model
    model = PainSenseModel(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    sequence = load_sequence_from_folder(folder_path).to(DEVICE)

    with torch.no_grad():
        output = model(sequence)  # shape (1, 4)
        probabilities = torch.softmax(output, dim=1)  # convert scores to probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()

    print(f"Input: {folder_path}")
    print(f"Predicted pain level: {PAIN_LEVELS[predicted_class]}")
    print(f"Confidence scores: {probabilities.cpu().numpy()[0]}")


if __name__ == "__main__":
    # Default to testing on subject_01 if no argument given
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/processed/subject_01"
    predict(folder)