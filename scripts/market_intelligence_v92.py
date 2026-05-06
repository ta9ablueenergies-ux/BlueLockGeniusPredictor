"""
market_intelligence_v92.py
==========================
Antigravity V9.2 - Market Dynamics & Variable Impact.
Teaches the model to understand:
1. Market Sensitivity (CLV tracking).
2. Feature Attribution (Shapley-style variable impact).
3. Dynamic Decision Weighting.
"""
import os
import torch
import numpy as np
import pandas as pd
import json
import sys

# Path setup
sys.path.append(os.path.join(os.getcwd(), 'scripts'))
from neural_v9_backbone import AntigravityNeuralBackbone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_ROOT = os.path.join(PROJECT_ROOT, 'model')

class MarketIntelligenceV92:
    def __init__(self, league='PremierLeague'):
        self.league = league
        self.model_dir = os.path.join(MODEL_ROOT, 'v9')
        self.feature_names = [
            'exp_h', 'exp_a', 'exp_total', 'exp_diff', 'form_h', 'form_a', 'form_diff',
            'momentum_h', 'momentum_a', 'momentum_diff', 'sot_h', 'sot_a', 'sot_diff',
            'sot_ratio', 'corn_h', 'corn_a', 'corn_total', 'cs_h', 'cs_a', 'ga_h',
            'ga_a', 'ga_diff', 'imp_h', 'imp_d', 'imp_a', 'edge_h', 'edge_a',
            'p_h', 'p_d', 'p_a', 'p_h_scores', 'p_a_scores', 'p_btts_model',
            'p_o25_model', 'n_games_h', 'n_games_a'
        ]

    def calculate_variable_impact(self, h_id, a_id, l_id, num_vec):
        """
        Calculates which variables most impacted the current decision.
        Uses a simplified Gradient-based Attribution.
        """
        # Load Model
        with open(os.path.join(self.model_dir, f'{self.league}_team_map.json'), 'r') as f:
            team_map = json.load(f)
            
        num_teams = len(team_map)
        model = AntigravityNeuralBackbone(num_teams, 1, 36)
        try:
            model.load_state_dict(torch.load(os.path.join(self.model_dir, f'{self.league}_backbone.pt'), map_location=torch.device('cpu')))
        except:
            return {} # Fallback
            
        model.eval()
        
        # We want gradients w.r.t the numerical features
        num_vec_tensor = torch.tensor([num_vec], dtype=torch.float32, requires_grad=True)
        h_id_tensor = torch.tensor([h_id], dtype=torch.long)
        a_id_tensor = torch.tensor([a_id], dtype=torch.long)
        l_id_tensor = torch.tensor([0], dtype=torch.long)
        
        logits, _, _ = model(h_id_tensor, a_id_tensor, l_id_tensor, num_vec_tensor)
        
        # Target the winning class
        winner = torch.argmax(logits)
        logits[0, winner].backward()
        
        # Gradients represent 'impact'
        gradients = num_vec_tensor.grad.abs().squeeze().numpy()
        
        # Normalize and map to feature names
        impact_map = {name: float(gradients[i]) for i, name in enumerate(self.feature_names)}
        sorted_impact = sorted(impact_map.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_impact[:5]) # Top 5 impactful variables

    def analyze_market_drift(self, open_odds, current_odds):
        """
        Teaches the model to understand Closing Line Value (CLV).
        """
        open_prob = 1 / np.array(open_odds)
        curr_prob = 1 / np.array(current_odds)
        
        drift = curr_prob - open_prob
        drift_magnitude = np.abs(drift).mean()
        
        status = "Stable"
        if drift_magnitude > 0.05: status = "Volatile (Smart Money Move)"
        elif drift_magnitude > 0.02: status = "Correcting"
        
        return {
            'market_drift': float(drift_magnitude),
            'market_status': status,
            'clv_signal': "Follow" if drift[np.argmax(curr_prob)] > 0.01 else "Fading"
        }

if __name__ == "__main__":
    mi = MarketIntelligenceV92('PremierLeague')
    # Dummy data for demonstration
    impact = mi.calculate_variable_impact(0, 1, 0, np.random.randn(36))
    print("Top 5 Impact Variables for Decision:")
    for k, v in impact.items():
        print(f"  {k}: {v:.4f}")
        
    market = mi.analyze_market_drift([2.0, 3.2, 3.5], [1.9, 3.3, 3.8])
    print("\nMarket Intelligence Analysis:")
    print(f"  Status: {market['market_status']}")
    print(f"  CLV Signal: {market['clv_signal']}")
