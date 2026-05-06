"""
Attention Mechanisms for Temporal Pattern Recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # Attention weights
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Compute temporal attention over sequence data
        query, key, value: [batch_size, seq_len, d_model]
        """
        # Linear transformations
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        output = self.W_o(output)

        return output, attention_weights

class FormAttention(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.temporal_attention = TemporalAttention(d_model)
        self.form_importance = nn.Linear(d_model, 1)

    def forward(self, form_sequence):
        """
        Apply attention to form sequence to identify important matches
        form_sequence: [batch_size, seq_len, d_model]
        """
        # Apply temporal attention
        attended_output, attention_weights = self.temporal_attention(
            form_sequence, form_sequence, form_sequence
        )

        # Calculate form importance scores
        importance_scores = torch.sigmoid(self.form_importance(attended_output))

        return {
            'attended_output': attended_output,
            'attention_weights': attention_weights,
            'importance_scores': importance_scores
        }