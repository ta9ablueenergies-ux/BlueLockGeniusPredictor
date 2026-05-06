"""
neural_v9_backbone.py
=====================
Antigravity V9.0 - Multi-Task Neural Backbone.
Implements:
1. Team & League Embedding Layers (Entity Embeddings).
2. Shared Hidden Backbone (Feature Extraction).
3. Specialized Heads (1X2, BTTS, Over/Under).
4. Label Smoothing Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AntigravityNeuralBackbone(nn.Module):
    def __init__(self, num_teams, num_leagues, num_numerical_feats, embedding_dim=16):
        super(AntigravityNeuralBackbone, self).__init__()
        
        # 1. Embedding Layers
        # Learns dense representations for Teams and Leagues
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        self.league_embedding = nn.Embedding(num_leagues, embedding_dim // 2)
        
        # 2. Shared Backbone
        # Input: 2 * team_embeddings + 1 * league_embedding + numerical_feats
        input_dim = (2 * embedding_dim) + (embedding_dim // 2) + num_numerical_feats
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 3. Task-Specific Heads
        # Head A: 1X2 Prediction (Multi-class)
        self.head_1x2 = nn.Linear(64, 3)
        
        # Head B: BTTS Prediction (Binary)
        self.head_btts = nn.Linear(64, 1)
        
        # Head C: Over 2.5 Goals (Binary)
        self.head_o25 = nn.Linear(64, 1)

        # Head D/E: Additional goal-total markets
        self.head_over15 = nn.Linear(64, 1)
        self.head_over35 = nn.Linear(64, 1)

    def forward(self, team_h_id, team_a_id, league_id, numerical_feats):
        # Embed categorical features
        emb_h = self.team_embedding(team_h_id)
        emb_a = self.team_embedding(team_a_id)
        emb_l = self.league_embedding(league_id)
        
        # Concatenate all inputs
        x = torch.cat([emb_h, emb_a, emb_l, numerical_feats], dim=1)
        
        # Pass through shared backbone
        features = self.shared_layers(x)
        
        # Task outputs
        out_1x2 = self.head_1x2(features) # Logits for CrossEntropy
        out_btts = torch.sigmoid(self.head_btts(features))
        out_o25 = torch.sigmoid(self.head_o25(features))
        out_over15 = torch.sigmoid(self.head_over15(features))
        out_over35 = torch.sigmoid(self.head_over35(features))
        
        return out_1x2, out_btts, out_o25, out_over15, out_over35

def label_smoothed_loss(logits, targets, smoothing=0.1):
    """
    Standard CrossEntropy but with Label Smoothing.
    Prevents the model from becoming overconfident.
    """
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Calculate negative log likelihood
    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    
    # Calculate average log probs for smoothing
    smooth_loss = -log_probs.mean(dim=-1)
    
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()
