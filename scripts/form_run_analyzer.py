"""
Form Run Analyzer for Temporal Pattern Recognition
"""
import torch
import torch.nn as nn
import numpy as np

class FormRunAnalyzer(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128, sequence_length=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        # Feature extraction for form analysis
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Temporal pattern recognition
        self.temporal_analyzer = nn.LSTM(
            hidden_dim // 2,
            hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Form state classification
        self.form_classifier = nn.Linear(hidden_dim // 2, 5)  # [Deep Slump, Poor Form, Neutral, Good Form, High Steam]

    def forward(self, match_sequence):
        """
        Process a sequence of matches to analyze form runs
        match_sequence: [batch_size, sequence_length, input_dim]
        """
        # Extract features from each match in sequence
        features = self.feature_extractor(match_sequence)  # [batch, seq_len, hidden_dim//2]

        # Analyze temporal patterns
        temporal_output, (hidden, cell) = self.temporal_analyzer(features)

        # Use the last output for form state
        form_state = temporal_output[:, -1, :]  # [batch, hidden_dim//2]

        # Classify form state
        form_probs = F.softmax(self.form_classifier(form_state), dim=1)

        return {
            'features': features,
            'temporal_output': temporal_output,
            'form_state': form_state,
            'form_probs': form_probs
        }

    def analyze_form_run(self, match_history):
        """
        Analyze form runs from historical match data
        """
        with torch.no_grad():
            result = self.forward(match_history)
            return result