import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalSequenceTransformer(nn.Module):
    """
    Advanced Sequence Modeling using Transformer architecture.
    Designed as a drop-in replacement for TemporalLSTM.
    """
    def __init__(self, input_dim=10, hidden_dim=128, nhead=8, num_layers=2, dropout=0.2, max_len=50):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = hidden_dim * 2 # To match bidirectional LSTM output shape
        self.d_model = 128
        
        # 1. Project input features to d_model
        self.input_proj = nn.Linear(input_dim, self.d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len)
        
        # 3. Transformer Blocks
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=self.d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 4. Attention Pooling
        self.attn_w = nn.Linear(self.d_model, 1)
        
        # 5. Output Projection to match expected downstream dimensions
        self.out_proj = nn.Sequential(
            nn.Linear(self.d_model, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, src):
        # src: [batch, seq_len, input_dim]
        
        x = self.input_proj(src)         # [batch, seq_len, d_model]
        x = self.pos_encoder(x)          # Add temporal positional context
        x = self.transformer_encoder(x)  # Deep self-attention interactions
        
        # Attention Pooling (Weighted sum over sequence)
        attn_scores = self.attn_w(x).squeeze(-1)         # [batch, seq_len]
        attn_weights = torch.softmax(attn_scores, dim=1) # [batch, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1) # [batch, d_model]
        
        # Final projection
        out = self.out_proj(context) # [batch, out_dim]
        return out