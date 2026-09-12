"""
train.py

Purpose: Train the CNN-LSTM model on the (currently dummy) dataset.

Since our dataset right now only has 2 dummy subjects with random labels,
this won't learn anything meaningful yet - the goal here is purely to
confirm the training loop itself runs correctly end-to-end (loads data,
forward pass, computes loss, backpropagates, updates weights) without
crashing. Once real data is in, this exact code trains for real.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PainDataset
from model import PainSenseModel

# --- Hyperparameters ---
BATCH_SIZE = 2       # how many subjects/sequences per training step
NUM_EPOCHS = 3        # how many times we loop through the whole dataset
LEARNING_RATE = 0.001  # how big a step the optimizer takes when updating weights

# Use GPU if available, otherwise fall back to CPU (dummy data is tiny,
# CPU is fine for testing)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print(f"Using device: {DEVICE}")

    # --- Load data ---
    dataset = PainDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Build model ---
    model = PainSenseModel(num_classes=4).to(DEVICE)

    # CrossEntropyLoss is the standard loss for multi-class classification.
    # It compares the model's raw output scores against the true label
    # and produces one number: how wrong the prediction was.
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer: updates the model's internal weights based on the
    # loss, using gradients computed via backpropagation.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()  # puts the model in "training mode" (affects certain layers)
        total_loss = 0.0

        for sequences, labels in dataloader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            # 1. Forward pass: get model predictions
            outputs = model(sequences)

            # 2. Compute how wrong the predictions were
            loss = criterion(outputs, labels)

            # 3. Backward pass: compute gradients (how to adjust weights
            #    to reduce the loss)
            optimizer.zero_grad()  # clear old gradients first
            loss.backward()

            # 4. Update the model's weights using those gradients
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Average Loss: {avg_loss:.4f}")

    # --- Save the trained model ---
    torch.save(model.state_dict(), "saved_models/painsense_model.pth")
    print("Model saved to saved_models/painsense_model.pth")


if __name__ == "__main__":
    train()