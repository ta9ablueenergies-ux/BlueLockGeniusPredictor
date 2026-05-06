"""
Temporal Pattern Recognition Module for Form Run Analysis
"""
import torch
import torch.nn as nn
import numpy as np

class TemporalPatternRecognizer(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Temporal convolution for pattern recognition
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Pattern classifier
        self.pattern_classifier = nn.Linear(hidden_dim, 5)  # Different form states

    def forward(self, temporal_sequence):
        """
        Recognize temporal patterns in form sequences
        temporal_sequence: [batch_size, seq_len, input_dim]
        """
        # Transpose for convolution: [batch, features, seq]
        x = temporal_sequence.transpose(1, 2)
        patterns = self.temporal_conv(x)
        patterns = patterns.transpose(1, 2)  # Back to [batch, seq, features]

        # Global average pooling over sequence
        pooled = torch.mean(patterns, dim=1)
        output = self.pattern_classifier(pooled)
        return torch.softmax(output, dim=1)