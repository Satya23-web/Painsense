"""
model.py

Purpose: Define the CNN-LSTM architecture.

Flow for one input sequence (shape: batch, sequence_length, 3, 224, 224):
1. Each frame in the sequence goes through a pretrained CNN (ResNet18)
   -> produces one feature vector per frame.
2. The sequence of feature vectors goes through an LSTM
   -> produces one summary vector for the whole sequence.
3. That summary vector goes through a final Linear layer
   -> produces 4 scores, one per pain level (none/mild/moderate/severe).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PainSenseModel(nn.Module):
    def __init__(self, num_classes=4, lstm_hidden_size=128, lstm_layers=1):
        super().__init__()

        # --- CNN part ---
        # Load ResNet18 pretrained on ImageNet (transfer learning: we
        # reuse its already-learned visual features instead of training
        # a CNN from zero, which we don't have enough data/time for).
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # ResNet18's last layer normally classifies into 1000 ImageNet
        # classes. We don't want that - we just want the feature vector
        # right before that layer, so we remove it (replace with Identity,
        # which just passes the input through unchanged).
        self.cnn_feature_dim = resnet.fc.in_features  # 512 for ResNet18
        resnet.fc = nn.Identity()
        self.cnn = resnet

        # --- LSTM part ---
        # Takes the sequence of per-frame CNN feature vectors (each of
        # size cnn_feature_dim) and models how they change over time.
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,  # input shape: (batch, seq_len, features)
        )

        # --- Classifier part ---
        # Takes the LSTM's final summary vector and outputs a score for
        # each of the 4 pain classes.
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, 3, 224, 224)
        """
        batch_size, seq_len, C, H, W = x.shape

        # The CNN expects (batch, 3, 224, 224) - one image at a time, not
        # a sequence. So we temporarily flatten batch and sequence together,
        # run the CNN on all frames at once (faster than looping), then
        # reshape back into sequences afterward.
        x = x.view(batch_size * seq_len, C, H, W)
        cnn_features = self.cnn(x)  # shape: (batch*seq_len, cnn_feature_dim)

        # Reshape back to (batch, seq_len, cnn_feature_dim) so the LSTM
        # can process each subject's frames as one ordered sequence again.
        cnn_features = cnn_features.view(batch_size, seq_len, self.cnn_feature_dim)

        # Run through LSTM. We only need the final hidden state (h_n),
        # which summarizes the whole sequence.
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)

        # h_n shape: (num_layers, batch, hidden_size) -> take the last layer
        final_hidden = h_n[-1]  # shape: (batch, hidden_size)

        # Final classification scores (not yet probabilities - that's
        # handled by the loss function during training).
        out = self.classifier(final_hidden)  # shape: (batch, num_classes)

        return out


if __name__ == "__main__":
    # Quick test: run one fake batch through the model to confirm shapes work
    model = PainSenseModel(num_classes=4)

    # Fake input: batch_size=2, sequence_length=16, 3 color channels, 224x224
    dummy_input = torch.randn(2, 16, 3, 224, 224)

    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # should be (2, 4)
    print(f"Output values:\n{output}")