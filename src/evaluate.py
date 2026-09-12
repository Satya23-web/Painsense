"""
evaluate.py

Purpose: Load the trained model and measure how well it performs, using
accuracy, precision, recall, F1-score, and a confusion matrix.

DUMMY MODE: We're evaluating on the same 2 dummy subjects the model was
trained on (since that's all we have). This won't tell us anything
meaningful about real performance - it just confirms the evaluation code
itself works. With real data, you'd evaluate on a separate held-out test
set the model never saw during training.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from dataset import PainDataset, PAIN_LEVELS
from model import PainSenseModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "saved_models/painsense_model.pth"


def evaluate():
    # --- Load data ---
    dataset = PainDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # --- Load trained model ---
    model = PainSenseModel(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()  # evaluation mode: disables training-only behavior

    all_preds = []
    all_labels = []

    # torch.no_grad() tells PyTorch not to track gradients here - we're
    # not training, so this saves memory and speeds things up.
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(DEVICE)

            outputs = model(sequences)  # raw scores, shape (batch, 4)

            # Pick the class with the highest score as the prediction
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- Compute metrics ---
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(PAIN_LEVELS))))

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(PAIN_LEVELS)), PAIN_LEVELS, rotation=45)
    plt.yticks(range(len(PAIN_LEVELS)), PAIN_LEVELS)

    # Write the actual numbers inside each cell of the matrix
    for i in range(len(PAIN_LEVELS)):
        for j in range(len(PAIN_LEVELS)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig("saved_models/confusion_matrix.png")
    print("Confusion matrix saved to saved_models/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()