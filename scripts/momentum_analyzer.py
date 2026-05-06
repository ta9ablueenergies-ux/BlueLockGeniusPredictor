# ANTIGRAVITY MOMENTUM ANALYZER (PLAN 2 — NEURAL EDGE)
import numpy as np

def calculate_momentum_score(form_sequence):
    """
    Calculates a Neural Momentum Score based on xG performance and results.
    Recent games are weighted higher using an exponential decay.
    
    Score > 1.10: "High Steam" (Neural Upswing)
    Score < 0.90: "Deep Slump" (Neural Decay)
    """
    if not form_sequence or len(form_sequence) == 0:
        return 1.0
    
    # Weighting: Last match = 1.0, 5 matches ago = 0.5
    weights = np.linspace(0.5, 1.0, len(form_sequence))
    
    xg_deltas = []
    points = []
    
    for entry in form_sequence:
        xg = entry.get('xg', 0.0)
        xga = entry.get('xga', 0.0)
        res = entry.get('result', 'D')
        
        # xG Performance delta (Are they outperforming their expected stats?)
        # A positive delta means they are creating more than they concede
        xg_deltas.append(xg - xga)
        
        # Points mapping
        pt_map = {'W': 3, 'D': 1, 'L': 0}
        points.append(pt_map.get(res, 0))
    
    # Weighted averages
    avg_xg_delta = np.average(xg_deltas, weights=weights)
    avg_pts = np.average(points, weights=weights)
    
    # Base Momentum: Based on PPG (Points Per Game)
    # PPG 1.5 = 1.0 multiplier
    base_momentum = 0.8 + (avg_pts / 3.0) * 0.4
    
    # Neural XG Adjustment: If they are significantly outperforming/underperforming xG
    # Adjust by up to ±10%
    xg_adj = np.clip(avg_xg_delta * 0.05, -0.1, 0.1)
    
    final_score = base_momentum + xg_adj
    return round(float(final_score), 3)

def get_trend_sparkline(form_sequence):
    """Returns a string representation of the trend for UI display."""
    if not form_sequence: return "No Data"
    res_map = {'W': '📈', 'D': '➖', 'L': '📉'}
    return "".join([res_map.get(e.get('result', 'D'), '➖') for e in form_sequence])
