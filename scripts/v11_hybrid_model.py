import torch
import torch.nn as nn
import torch.nn.functional as F
from temporal_transformer import TemporalSequenceTransformer


class CrossMatchAttention(nn.Module):
    """Asymmetric cross-attention between home and away temporal encodings.

    Allows h_enc to attend to a_enc (and vice versa), so the model learns
    opponent-style-aware form representations rather than relying on a simple
    vector difference. Uses residual + LayerNorm to preserve the original
    encoding while adding the cross-context signal.
    """

    def __init__(self, dim: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # query, context: [batch, dim]
        q  = query.unsqueeze(1)    # [batch, 1, dim]
        kv = context.unsqueeze(1)  # [batch, 1, dim]
        out, _ = self.attn(q, kv, kv)
        out = out.squeeze(1)       # [batch, dim]
        return self.norm(query + self.drop(out))


class TemporalLSTM(nn.Module):
    """Bidirectional LSTM with attention pooling (Legacy Fallback)"""
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.attn_w = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn_w(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context  # [batch, hidden_dim * 2]

class V11Hybrid(nn.Module):
    """
    Hybrid model: V8.1 features + Neural temporal sequences
    """
    def __init__(self, num_teams, num_leagues=5, seq_len=15, hidden_dim=128, num_v8_features=20, dropout=0.3):
        super().__init__()
        self.num_v8_features = num_v8_features

        # V8.1 feature projection
        self.v8_proj = nn.Sequential(
            nn.Linear(num_v8_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Team embeddings - Upgraded to 64 dimensions for deep 25-year historic vocabulary
        emb_dim = 64
        self.team_emb = nn.Embedding(num_teams, emb_dim)

        # Temporal encoders — 20-dim per timestep matches _pad_and_encode() output
        self.h_encoder = TemporalSequenceTransformer(input_dim=20, hidden_dim=hidden_dim, nhead=8, num_layers=2, dropout=dropout, max_len=seq_len)
        self.a_encoder = TemporalSequenceTransformer(input_dim=20, hidden_dim=hidden_dim, nhead=8, num_layers=2, dropout=dropout, max_len=seq_len)

        # Cross-match attention — lets each team's encoding attend to the opponent's
        # encoding, producing opponent-aware form representations before fusion.
        enc_dim = hidden_dim * 2  # 256
        self.h_cross = CrossMatchAttention(enc_dim, nhead=8, dropout=dropout)
        self.a_cross = CrossMatchAttention(enc_dim, nhead=8, dropout=dropout)

        # Phase 2: Graph Structural Embeddings
        self.graph_proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32)
        )

        # Fusion: V8.1(64) + h_enc(hidden_dim*2) + a_enc(hidden_dim*2) + diff(hidden_dim*2) + h_emb(emb_dim) + a_emb(emb_dim) + graph(32)
        lstm_out_dim = hidden_dim * 2
        fusion_dim = 64 + (lstm_out_dim * 3) + (emb_dim * 2) + 32
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(min(0.5, dropout + 0.1)),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output heads
        self.head_1x2 = nn.Linear(256, 3)
        self.head_btts = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_o25 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_over15 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_over35 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

        # Phase 5: Yellow Card + Corners heads
        # Predicts expected total YC per match (home+away combined, softplus -> positive)
        self.head_yc = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        # Predicts expected total corners per match (home+away combined)
        self.head_corners = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # Phase 1: Dynamic Temperature Predictor
        # Predicts a calibration factor T > 1.0 based on fused intelligence
        self.temp_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, v8_feat, h_seq, a_seq, h_team_id, a_team_id, centrality):
        # Project V8.1 features
        v8_out = self.v8_proj(v8_feat)  # [batch, 64]

        # Temporal encoding
        h_enc = self.h_encoder(h_seq)    # [batch, 256]
        a_enc = self.a_encoder(a_seq)    # [batch, 256]

        # Cross-match attention: each team attends to opponent's encoded history.
        # h_cross learns "how does home team's form look given away team's style"
        h_cross = self.h_cross(h_enc, a_enc)  # [batch, 256]
        a_cross = self.a_cross(a_enc, h_enc)  # [batch, 256]
        cross_diff = h_cross - a_cross        # opponent-aware differential

        # Embeddings
        h_emb = self.team_emb(h_team_id)
        a_emb = self.team_emb(a_team_id)

        # Phase 2: Graph Projection
        # centrality: [batch, 2] (home_centrality, away_centrality)
        graph_out = self.graph_proj(centrality)

        # Fusion — cross_diff replaces naive diff for a richer matchup signal
        combined = torch.cat([v8_out, h_enc, a_enc, cross_diff, h_emb, a_emb, graph_out], dim=1)
        fused = self.fusion(combined)

        # Dynamic Temperature T
        # T = 1.0 + Softplus(temp_head) ensures T >= 1.0
        # This softens logits in high-uncertainty contexts.
        raw_temp = self.temp_head(fused)
        T = 1.0 + F.softplus(raw_temp)

        # Outputs
        logits_1x2 = self.head_1x2(fused)
        logits_1x2_scaled = logits_1x2 / T

        p_btts = torch.sigmoid(self.head_btts(fused))
        p_o25 = torch.sigmoid(self.head_o25(fused))
        p_over15 = torch.sigmoid(self.head_over15(fused))
        p_over35 = torch.sigmoid(self.head_over35(fused))

        # Phase 5: Card + Corner predictions (positive continuous via Softplus)
        yc_pred = F.softplus(self.head_yc(fused))          # expected total YC (>= 0)
        corners_pred = F.softplus(self.head_corners(fused)) # expected total corners (>= 0)

        return logits_1x2_scaled, p_btts, p_o25, p_over15, p_over35, T, yc_pred, corners_pred


# Backward-compatible name used by Phase 2 verification prompts.
V11HybridModel = V11Hybrid
