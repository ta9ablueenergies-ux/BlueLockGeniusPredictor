import os
import sys

# Path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_ROOT = os.path.join(PROJECT_ROOT, 'model')
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from momentum_analyzer import calculate_momentum_score, get_trend_sparkline
from scipy.stats import poisson, linregress
import torch
import torch.nn.functional as F
import xgboost as xgb
from typing import Optional

from components.ssl_utils import get_unsafe_session
from v11_hybrid_model import V11Hybrid
from persistence_manager import get_team_match_history
from pi_ratings import PiRatingSystem
from feature_schema import validate_feature_vector
from model_registry import get_active_model_path
from league_situation import (
    adjust_1x2_for_situation,
    build_v11_situation_features,
    get_match_situation_from_db,
)

try:
    from market_count_models import predict_market_context
    _market_count_available = True
except ImportError:
    _market_count_available = False

try:
    from market_router import route_market_selection
    _market_router_available = True
except ImportError:
    _market_router_available = False

# Trainable team-interaction graph layer. This is optional at runtime and only
# contributes when its per-league validation metadata enables it.
try:
    from gnn_graph_layer import get_gnn_match_signals
    _gnn_layer_available = True
except ImportError:
    _gnn_layer_available = False

# XGBoost ensemble layer — blended after GNN at fixed 15% weight when available.
_XGB_CACHE = {}
try:
    import joblib
    from xgboost_ensemble import XGBoostEnsemble
    _xgb_available = True
except ImportError:
    _xgb_available = False

# Isotonic calibration — non-parametric per-league probability correction.
try:
    from calibration import apply_isotonic_calibration
    _isotonic_available = True
except ImportError:
    _isotonic_available = False

# CLV tracker — closing line value and market movement signals.
try:
    from clv_tracker import record_snapshot as _clv_record, compute_cld_delta, compute_clv
    _clv_available = True
except ImportError:
    _clv_available = False

# Referee stats — per-referee avg cards/fouls/corners lookup.
try:
    from referee_pipeline import get_referee_stats as _get_referee_stats, init_referee_table
    _referee_available = True
except ImportError:
    _referee_available = False

# Pi Rating per-timestep — historical opp_strength lookup.
try:
    from pi_rating_timestep import get_pi_at_date as _get_pi_at_date
    _pi_timestep_available = True
except ImportError:
    _pi_timestep_available = False

# Try to import Tavily for deep research
try:
    from tavily_research import get_tavily_researcher
    _tavily_client = get_tavily_researcher()
except ImportError:
    _tavily_client = None
    print("[ML] Tavily Research module not found")

# Firecrawl Agent — primary enrichment (Tavily is now the fallback)
try:
    from firecrawl_agent import discover_league_matches, enrich_matchday, get_firecrawl_agent
    _firecrawl_pool = get_firecrawl_agent()
    _firecrawl_available = _firecrawl_pool.is_available()
    if _firecrawl_available:
        print("[Firecrawl] Agent pool connected")
except ImportError:
    _firecrawl_available = False
    print("[Firecrawl] Agent module not found")

# V11 Temporal Transformer Integration
try:
    from temporal_transformer import TemporalSequenceTransformer
    from form_run_analyzer import FormRunAnalyzer
    from sequence_memory import SequenceMemory
    from attention_model import TemporalAttention
    from temporal_pattern_recognizer import TemporalPatternRecognizer
    _temporal_transformer_available = True
    print("[V11] Temporal transformer components loaded successfully")
except ImportError:
    _temporal_transformer_available = False
    print("[V11] Temporal transformer components not found")

# ============================================================
# V9.0 HYBRID NEURAL LAYER (STAKING)
# ============================================================
_NEURAL_CACHE = {}
_CALIBRATION_CACHE = None
_V11_GATE_CACHE = None

def should_use_v11_hybrid(league_name):
    """
    Use V11 only where saved champion/challenger validation beats the base model.
    Phase 5 update: We now check model_registry.json for 'active' status as well.
    """
    global _V11_GATE_CACHE
    if _V11_GATE_CACHE is None:
        _V11_GATE_CACHE = {}
        # 1. Check the classic results gate
        report_path = os.path.join(PROJECT_ROOT, 'model', 'v11', 'v11_hybrid_results.json')
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                for row in report.get('results', []):
                    improvement = float(row.get('improvement', 0.0) or 0.0)
                    _V11_GATE_CACHE[row.get('league')] = improvement > 0
            except Exception:
                pass
        
        # 2. Check the Model Registry (Force enable if Phase 5 model is active)
        registry_path = os.path.join(PROJECT_ROOT, 'model', 'model_registry.json')
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    reg = json.load(f)
                for l_name, components in reg.get('leagues', {}).items():
                    v11_info = components.get('v11_hybrid', {})
                    if 'draww_v5' in v11_info.get('active_model_path', ''):
                        # Phase 5 rollout: all trained leagues are enabled
                        _V11_GATE_CACHE[l_name] = True
            except Exception:
                pass

    return bool(_V11_GATE_CACHE.get(league_name, False))

def load_1x2_calibration():
    """Load empirical 1X2 bias corrections from the latest calibration report."""
    global _CALIBRATION_CACHE
    if _CALIBRATION_CACHE is not None:
        return _CALIBRATION_CACHE
    _CALIBRATION_CACHE = {}
    report_path = os.path.join(PROJECT_ROOT, 'web', 'data', 'v102_deep_calibration_report.json')
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        for label, row in report.get('market_analysis_1x2', {}).items():
            predicted = max(float(row.get('predicted_rate', 0.0) or 0.0), 0.01)
            true = max(float(row.get('true_rate', predicted) or predicted), 0.01)
            _CALIBRATION_CACHE[label] = true / predicted
    except Exception:
        _CALIBRATION_CACHE = {}
    return _CALIBRATION_CACHE

def calibrate_1x2_probs(p_h, p_d, p_a, alpha=0.5):
    """
    Bias-correct 1X2 probabilities with damped empirical priors.
    alpha<1 avoids overfitting one calibration report into live inference.
    """
    factors = load_1x2_calibration()
    weights = np.array([
        factors.get('H', 1.0) ** alpha,
        factors.get('D', 1.0) ** alpha,
        factors.get('A', 1.0) ** alpha,
    ], dtype=float)
    probs = np.array([p_h, p_d, p_a], dtype=float) * weights
    probs = np.clip(probs, 0.01, 0.98)
    probs = probs / probs.sum()
    return float(probs[0]), float(probs[1]), float(probs[2])

def blend_gnn_probs(p_h, p_d, p_a, gnn_signals):
    """
    Blend validated GNN probabilities into the main 1X2 distribution.
    The cap keeps this a feature layer until per-league validation is stronger.
    """
    if not gnn_signals or not gnn_signals.get('gnn_active'):
        return p_h, p_d, p_a, 0.0
    recommended = max(0.0, float(gnn_signals.get('gnn_blend_weight', 0.0) or 0.0))
    logloss_lift = max(0.0, float(gnn_signals.get('gnn_blend_logloss_lift', 0.0) or 0.0))
    acc_lift = max(0.0, float(gnn_signals.get('gnn_blend_accuracy_lift', 0.0) or 0.0))
    weight = min(0.20, max(0.04, recommended) + min(0.06, logloss_lift * 8.0 + acc_lift))
    base = np.array([p_h, p_d, p_a], dtype=float)
    graph = np.array([
        gnn_signals.get('gnn_p_h', p_h),
        gnn_signals.get('gnn_p_d', p_d),
        gnn_signals.get('gnn_p_a', p_a),
    ], dtype=float)
    probs = (base * (1.0 - weight)) + (graph * weight)
    probs = np.clip(probs, 0.01, 0.98)
    probs = probs / probs.sum()
    return float(probs[0]), float(probs[1]), float(probs[2]), float(weight)

def implied_prob(*odds):
    inv = []
    for odd in odds:
        try:
            odd = float(odd)
            inv.append(1.0 / odd if odd > 1.01 else 0.0)
        except Exception:
            inv.append(0.0)
    total = sum(inv)
    return [x / total for x in inv] if total > 0 else [0.0 for _ in inv]

def calculate_primary_edge(match, primary_market, p_h, p_d, p_a, p_o25, p_btts):
    """
    Calculate edge against the odds relevant to the selected market.
    If market odds are unavailable, return 0 instead of reusing home-win edge.
    """
    h_o = match.get('h_course', 2.0)
    d_o = match.get('d_course', 3.2)
    a_o = match.get('a_course', 3.5)
    imp_h, imp_d, imp_a = implied_prob(h_o, d_o, a_o)

    if primary_market == 'Home Win':
        return p_h - imp_h
    if primary_market == 'Away Win':
        return p_a - imp_a
    if primary_market.startswith('1X'):
        return (p_h + p_d) - (imp_h + imp_d)
    if primary_market.startswith('X2'):
        return (p_a + p_d) - (imp_a + imp_d)

    over_odd = match.get('over25_course') or match.get('o25_course')
    btts_odd = match.get('btts_course')
    if primary_market == 'Over 2.5' and over_odd:
        return p_o25 - (1.0 / float(over_odd))
    if primary_market == 'BTTS' and btts_odd:
        return p_btts - (1.0 / float(btts_odd))
    return 0.0

def _blend_prob(base, model, weight=0.30):
    try:
        base = float(base)
        model = float(model)
        return float(np.clip((base * (1.0 - weight)) + (model * weight), 0.01, 0.99))
    except Exception:
        return base

def get_v9_neural_signals(league_name, h_team, a_team, numerical_feats):
    """Legacy V10.2 Signal Layer."""
    model_dir = os.path.join(MODEL_ROOT, 'v9')
    try:
        from neural_v9_backbone import AntigravityNeuralBackbone
        with open(os.path.join(model_dir, f'{league_name}_team_map.json'), 'r') as f:
            team_map = json.load(f)
        with open(os.path.join(model_dir, f'{league_name}_pi_ratings.json'), 'r') as f:
            pi_ratings = json.load(f)

        h_id = torch.tensor([team_map.get(h_team, 0)], dtype=torch.long)
        a_id = torch.tensor([team_map.get(a_team, 0)], dtype=torch.long)
        l_id = torch.tensor([0], dtype=torch.long)
        num_vec = torch.tensor([numerical_feats], dtype=torch.float32)

        if league_name not in _NEURAL_CACHE:
            fallback_w = os.path.join(model_dir, f'v9_{league_name}_draww_v2.pt')
            v102_w = get_active_model_path(league_name, "v9_backbone", fallback_w)
            print(f"V9: {v102_w}")
            model = AntigravityNeuralBackbone(len(team_map), 1, len(numerical_feats))
            model.load_state_dict(torch.load(v102_w, map_location=torch.device('cpu')), strict=False)
            model.eval()
            _NEURAL_CACHE[league_name] = model

        model = _NEURAL_CACHE[league_name]
        temp_map_path = os.path.join(model_dir, 'temperature_map.json')
        T = 1.5
        if os.path.exists(temp_map_path):
            with open(temp_map_path, 'r') as f: T = json.load(f).get(league_name, 1.5)

        with torch.no_grad():
            logits_1x2, p_btts, p_o25, p_o15, p_o35 = model(h_id, a_id, l_id, num_vec)
            probs_1x2 = torch.softmax(logits_1x2 / T, dim=1).numpy()[0]
            p_btts_cal = float(torch.clamp(p_btts * 1.38, 0.0, 1.0).item())
            p_o25_cal  = float(torch.clamp(p_o25  * 1.44, 0.0, 1.0).item())
            p_o15_cal  = float(torch.clamp(p_o15, 0.0, 1.0).item())
            p_o35_cal  = float(torch.clamp(p_o35, 0.0, 1.0).item())

        return {
            'v9_p_h': float(probs_1x2[0]), 'v9_p_d': float(probs_1x2[1]), 'v9_p_a': float(probs_1x2[2]),
            'v9_p_btts': p_btts_cal, 'v9_p_o25': p_o25_cal,
            'v9_p_o15': p_o15_cal, 'v9_p_o35': p_o35_cal
        }
    except: return {}

def get_v11_hybrid_signals(league_name, h_team, a_team, v8_features, h_cent, a_cent):
    """V11 Hybrid Signal Layer with Dynamic Temperature & GSN Embeddings."""
    if not should_use_v11_hybrid(league_name):
        return None
    model_dir = os.path.join(MODEL_ROOT, 'v11')
    try:
        # 1. Team Mapping
        with open(os.path.join(model_dir, f'{league_name}_team_map.json'), 'r') as f:
            team_map = json.load(f)
        h_id_raw = team_map.get(h_team)
        a_id_raw = team_map.get(a_team)
        if h_id_raw is None or a_id_raw is None: return None

        # 2. Sequential Form History
        h_hist = get_team_match_history(h_team, limit=20)
        a_hist = get_team_match_history(a_team, limit=20)
        
        def encode_seq(hist, team, seq_len=20):
            # Pad defaults match _pad_and_encode() training defaults exactly
            PAD = [0.0, 0.0, 0.33, 0.45, 0.45, 0.40, 0.40, 0.35, 0.25,
                   0.50, 0.50, 0.50, 0.50, 0.50, 0.43, 0.50, 0.38, 0.00, 0.00, 0.25]
            chron = list(reversed(hist))  # oldest → newest
            window = chron[-seq_len:]
            encoded = [PAD[:] for _ in range(seq_len - len(window))]
            for i, m in enumerate(window):
                is_h = m.get('home_team') == team
                gf = float(m.get('actual_fthg') or 0)
                ga = float(m.get('actual_ftag') or 0)
                ftr = m.get('actual_ftr')
                pts = 3 if ftr == ('H' if is_h else 'A') else (1 if ftr == 'D' else 0)
                # xG from intel_raw or sequence_history xg/xga fields
                xg_for = m.get('xg')
                xg_against = m.get('xga')
                travel = 0.0
                opp_strength = 0.5
                try:
                    intel = json.loads(m.get('intel_raw') or '{}')
                    if xg_for is None:
                        xg_for = intel.get('xg')
                    travel = float(intel.get('travel_km_a' if not is_h else 'travel_km_h') or 0) / 1000.0
                    p_opp = m.get('p_x2' if is_h else 'p_1x')
                    if p_opp is not None:
                        opp_strength = float(p_opp)
                except Exception:
                    pass
                # Use Pi Rating snapshot for opp_strength if market odds unavailable
                if opp_strength == 0.5 and _pi_timestep_available:
                    try:
                        opp_name = m.get('away_team' if is_h else 'home_team') or ''
                        m_date = str(m.get('match_date') or '')[:10]
                        if opp_name and m_date:
                            opp_strength = _get_pi_at_date(league, opp_name, m_date, is_home=not is_h)
                    except Exception:
                        pass
                try:
                    xg_diff = max(0.0, min(1.0, (float(xg_for) - float(xg_against) + 3.0) / 6.0))
                    xg_for_n  = min(1.0, float(xg_for)     / 2.0)
                    xg_agn_n  = min(1.0, float(xg_against) / 2.0)
                except Exception:
                    xg_diff = xg_for_n = xg_agn_n = 0.5
                # rest_days: gap from previous match in the window
                if i > 0:
                    try:
                        from datetime import datetime as _dt
                        d0 = _dt.strptime(window[i-1]['match_date'][:10], '%Y-%m-%d')
                        d1 = _dt.strptime(m['match_date'][:10], '%Y-%m-%d')
                        rest_days = min(1.0, abs((d1 - d0).days) / 7.0)
                    except Exception:
                        rest_days = 0.43
                else:
                    rest_days = 0.43
                was_home = 1.0 if is_h else 0.0
                try:
                    from datetime import datetime as _dt
                    yday = _dt.strptime(m['match_date'][:10], '%Y-%m-%d').timetuple().tm_yday
                    season_week = yday / 365.0
                except Exception:
                    season_week = 0.38
                encoded.append([
                    min(1.0, gf / 5), min(1.0, ga / 5), pts / 3,
                    0.45, 0.45, 0.40, 0.40, 0.35, 0.25,
                    xg_diff, xg_for_n, xg_agn_n,
                    0.50, opp_strength, rest_days, was_home,
                    season_week, min(1.0, travel), 0.00, 0.25,
                ])
            return torch.tensor([encoded], dtype=torch.float32)

        h_seq = encode_seq(h_hist, h_team)
        a_seq = encode_seq(a_hist, a_team)
        
        # 3. Model Loading
        cache_key = f"v11_{league_name}"
        if cache_key not in _NEURAL_CACHE:
            w_path = get_active_model_path(league_name, "v11_hybrid", os.path.join(model_dir, f'{league_name}_hybrid_v11_draww_v2.pt'))
            print(f"V11: {w_path}")
            model = V11Hybrid(num_teams=len(team_map), num_v8_features=28, seq_len=20)
            model.load_state_dict(torch.load(w_path, map_location='cpu'), strict=False)
            model.eval()
            _NEURAL_CACHE[cache_key] = model
            
        model = _NEURAL_CACHE[cache_key]
        
        # 4. Inference
        v8_t = torch.tensor([v8_features], dtype=torch.float32)
        h_id_t = torch.tensor([h_id_raw], dtype=torch.long)
        a_id_t = torch.tensor([a_id_raw], dtype=torch.long)
        c_t = torch.tensor([[h_cent, a_cent]], dtype=torch.float32)
        
        with torch.no_grad():
            outputs = model(v8_t, h_seq, a_seq, h_id_t, a_id_t, c_t)
            # Phase 5: Unpack 8 outputs (1x2, btts, o25, o15, o35, T, yc, corners)
            l_scaled, p_btts, p_o25, p_o15, p_o35, T, yc_pred, corn_pred = outputs
            probs = torch.softmax(l_scaled, dim=1).numpy()[0]
            
        return {
            'v11_p_h': float(probs[0]), 'v11_p_d': float(probs[1]), 'v11_p_a': float(probs[2]),
            'v11_p_btts': float(p_btts.item()), 'v11_p_o25': float(p_o25.item()),
            'v11_p_o15': float(p_o15.item()), 'v11_p_o35': float(p_o35.item()),
            'v11_expected_yc': float(yc_pred.item()),
            'v11_expected_corners': float(corn_pred.item()),
            'v11_temp': float(T.item()), 'v11_active': True
        }
    except Exception as e:
        print(f"[V11] Inference Error: {e}")
        return None

def get_xgb_signals(league, match_row, num_vec):
    """Load per-league XGBoost ensemble and return 1X2 probability dict."""
    if not _xgb_available:
        return None
    try:
        if league not in _XGB_CACHE:
            model_path = os.path.join(MODEL_ROOT, 'v11', f'xgboost_{league}')
            model_file = model_path + '_model.pkl'
            scaler_file = model_path + '_scaler.pkl'
            if not os.path.exists(model_file):
                _XGB_CACHE[league] = None
                return None
            xgb = XGBoostEnsemble(model_dir=os.path.join(MODEL_ROOT, 'v11'))
            xgb.model = joblib.load(model_file)
            xgb.scaler = joblib.load(scaler_file)
            _XGB_CACHE[league] = xgb
        xgb = _XGB_CACHE[league]
        if xgb is None:
            return None
        # Build a single-row DataFrame with the features XGBoost was trained on.
        # Map match-dict field names to the expected column names.
        row = dict(match_row) if hasattr(match_row, 'to_dict') else dict(match_row)
        row.setdefault('B365H', row.get('h_course'))
        row.setdefault('B365D', row.get('d_course'))
        row.setdefault('B365A', row.get('a_course'))
        df_row = pd.DataFrame([row])
        probs = xgb.predict_proba(df_row)[0]  # [p_home, p_draw, p_away]
        return {'xgb_p_h': float(probs[0]), 'xgb_p_d': float(probs[1]), 'xgb_p_a': float(probs[2])}
    except Exception as e:
        print(f"[XGB] Inference skipped for {league}: {e}")
        return None

# ============================================================
# CORE PIPELINE FUNCTIONS
# ============================================================

def precalculate_all_features(league):
    """Bootstrap Pi-Ratings and GSN Centrality."""
    print(f"  [Precalc] Bootstrapping cluster: {league}")
    ratings_path = os.path.join(MODEL_ROOT, 'v9', f'{league}_pi_ratings.json')
    ratings_system = PiRatingSystem()
    if os.path.exists(ratings_path): ratings_system.load_state(ratings_path)
    
    from v8_gsn_architect import V8MotifDetector
    detector = V8MotifDetector()
    detector.build_directed_topology(league)
    detector.detect_cyclic_dominance()
    detector.detect_giant_killers()
    
    return {
        'pi': ratings_system, 'motifs': detector,
        'centrality': detector.get_centrality_scores()
    }

def _congestion_games(team, match_date, window_days=21):
    """Count completed matches in the last window_days before match_date."""
    try:
        from datetime import datetime, timedelta
        from persistence_manager import get_team_match_history
        match_dt = datetime.strptime(match_date[:10], '%Y-%m-%d')
        cutoff = match_dt - timedelta(days=window_days)
        hist = get_team_match_history(team, limit=15)
        return float(sum(
            1 for h in hist
            if h.get('match_date') and
               cutoff <= datetime.strptime(str(h['match_date'])[:10], '%Y-%m-%d') < match_dt
        ))
    except Exception:
        return 2.0


def build_enhanced_features_v81(match, ratings, intel):
    """36-feature vector generation."""
    h, a = match['HomeTeam'], match['AwayTeam']
    pi = ratings['pi']
    exp_h_b = pi.get_rating(h)[0] - pi.get_rating(a)[1]
    exp_a_b = pi.get_rating(a)[0] - pi.get_rating(h)[1]
    mh, ma = ratings['motifs'].get_motif_multiplier(h, a)
    exp_h = max(0.1, (1.3 + exp_h_b) * mh)
    exp_a = max(0.1, (1.1 + exp_a_b) * ma)
    
    h_o, d_o, a_o = float(match.get('h_course', 2)), float(match.get('d_course', 3.2)), float(match.get('a_course', 3.5))
    imp_h, imp_d, imp_a = 1/h_o, 1/d_o, 1/a_o
    s = imp_h + imp_d + imp_a
    imp_h /= s; imp_d /= s; imp_a /= s
    
    p_h = sum(poisson.pmf(j, exp_h) * sum(poisson.pmf(k, exp_a) for k in range(j)) for j in range(7))
    p_a = sum(poisson.pmf(k, exp_a) * sum(poisson.pmf(j, exp_h) for j in range(k)) for k in range(7))
    p_d = max(0, 1 - p_h - p_a)
    
    mom_h, mom_a = intel.get('momentum_h', 0.5), intel.get('momentum_a', 0.5)
    
    vec = [
        exp_h, exp_a, exp_h + exp_a, exp_h - exp_a, pi.get_rating(h)[0], pi.get_rating(a)[0],
        pi.get_rating(h)[0] - pi.get_rating(a)[0], mom_h, mom_a, mom_h - mom_a,
        intel.get('sot_h', 4), intel.get('sot_a', 4), intel.get('sot_h', 4) - intel.get('sot_a', 4),
        intel.get('sot_h', 4) / max(0.1, intel.get('sot_a', 4)),
        intel.get('corn_h', 5), intel.get('corn_a', 5), intel.get('corn_h', 5) + intel.get('corn_a', 5),
        intel.get('cs_h', 0.2), intel.get('cs_a', 0.2), intel.get('ga_h', 1.2), intel.get('ga_a', 1.2),
        intel.get('ga_h', 1.2) - intel.get('ga_a', 1.2), imp_h, imp_d, imp_a,
        p_h - imp_h, p_a - imp_a, p_h, p_d, p_a,
        1 - poisson.pmf(0, exp_h), 1 - poisson.pmf(0, exp_a),
        (1 - poisson.pmf(0, exp_h)) * (1 - poisson.pmf(0, exp_a)),
        1 - sum(poisson.pmf(i, exp_h) * poisson.pmf(j, exp_a) for i in range(4) for j in range(4) if i+j <= 2),
        15, 15
    ]
    # Schedule congestion: matches played in last 21 days (replaces placeholder 15,15)
    vec[-2] = _congestion_games(h, str(match.get('Date', '')))
    vec[-1] = _congestion_games(a, str(match.get('Date', '')))
    validate_feature_vector(vec); return np.array(vec)

def predict_results(df_matches, league, ratings, intel_dicts):
    """Main Execution Loop."""
    print(f"--- V11 Hybrid Execution Engine: {league} ---")
    results = []
    from execution_engine import calculate_eqi_v2, calculate_stake, map_dominance_markets

    def normalize_intel(obj):
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return dict(obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return {}

    def intel_key(item):
        return (
            str(item.get("home_team") or item.get("HomeTeam") or item.get("Home") or "").strip().lower(),
            str(item.get("away_team") or item.get("AwayTeam") or item.get("Away") or "").strip().lower(),
            str(item.get("match_date") or item.get("Date") or "").split("T")[0],
        )

    intel_lookup = {}
    for raw_intel in intel_dicts or []:
        item = normalize_intel(raw_intel)
        key = intel_key(item)
        if key[0] and key[1]:
            intel_lookup[key] = item
    
    snap_id = os.environ.get("GRAPH_SNAPSHOT_ID", "latest")
    for idx, match in df_matches.iterrows():
        h, a = match['HomeTeam'], match['AwayTeam']
        print(f"  [Oracle] Enriching: {h} vs {a}...")
        
        # 1. Oracle News Enrichment (Option B)
        match_key = (str(h).strip().lower(), str(a).strip().lower(), str(match['Date']).split("T")[0])
        intel = dict(intel_lookup.get(match_key, {}))
        if not intel and _firecrawl_available:
            try:
                # Crawl for match intel (injuries, sentiment, tactical leaks)
                from firecrawl_agent import enrich_match
                intel_obj = enrich_match(h, a, league, match['Date'])
                if intel_obj:
                    intel = normalize_intel(intel_obj)
            except Exception as e:
                print(f"    [Oracle] Enrichment failed: {e}")

        # Calculate aggregate sentiment from whichever enrichment source was used.
        h_news = [n.get('sentiment_score') for n in intel.get('news', []) if isinstance(n, dict) and n.get('sentiment_score') is not None]
        a_news = [n.get('sentiment_score') for n in intel.get('news', []) if isinstance(n, dict) and n.get('sentiment_score') is not None]
        intel['sentiment_h'] = float(np.mean(h_news)) if h_news else float(intel.get('sentiment_h', 0.0) or 0.0)
        intel['sentiment_a'] = float(np.mean(a_news)) if a_news else float(intel.get('sentiment_a', 0.0) or 0.0)
        if intel:
            print(f"    [Oracle] Sentiment: {h} ({intel['sentiment_h']:+.2f}), {a} ({intel['sentiment_a']:+.2f})")

        situation = get_match_situation_from_db(league, h, a, match['Date'])
        intel['league_situation'] = situation
        if situation.get('match_pressure_score', 0) >= 35:
            print(
                f"    [Situation] {situation.get('label')}: "
                f"H={situation.get('home_pressure', 0):.2f} A={situation.get('away_pressure', 0):.2f}"
            )

        # Referee stats enrichment
        if _referee_available:
            try:
                referee = intel.get('referee') or str(match.get('Referee') or '')
                ref_stats = _get_referee_stats(referee or None, league)
                intel['ref_avg_yc'] = ref_stats.get('avg_yc', 3.8)
                intel['ref_avg_rc'] = ref_stats.get('avg_rc', 0.15)
                intel['ref_avg_fouls'] = ref_stats.get('avg_fouls', 22.0)
                intel['ref_avg_corners'] = ref_stats.get('avg_corners', 10.5)
                intel['ref_matches'] = ref_stats.get('matches', 0)
                if referee and ref_stats.get('matches', 0) >= 10:
                    print(f"    [Referee] {referee}: avg_yc={intel['ref_avg_yc']:.1f} fouls={intel['ref_avg_fouls']:.1f}")
            except Exception:
                pass

        # 2. Base Neural Predictions
        num_vec = build_enhanced_features_v81(match, ratings, intel)
        v9 = get_v9_neural_signals(league, h, a, num_vec)
        h_cent = ratings['centrality'].get(h, (0.05, 0.05))[0]
        a_cent = ratings['centrality'].get(a, (0.05, 0.05))[0]
        v11_vec = build_v11_situation_features(
            float(num_vec[0]),
            float(num_vec[1]),
            float(num_vec[9]),
            float(num_vec[27]),
            float(num_vec[28]),
            float(num_vec[29]),
            situation,
        )
        v11 = get_v11_hybrid_signals(league, h, a, v11_vec, h_cent, a_cent)
        
        # 3. Decision & Fusion
        if v11 and v11.get('v11_active'):
            p_h, p_d, p_a = v11['v11_p_h'], v11['v11_p_d'], v11['v11_p_a']
            p_btts, p_o25 = v11['v11_p_btts'], v11['v11_p_o25']
            p_o15, p_o35 = v11.get('v11_p_o15', 0.5), v11.get('v11_p_o35', 0.5)
            
            # 4. Oracle Sentiment Shift (Logit Level)
            # Apply news-driven tactical shifts: +1.0 sentiment = ~+15% prob lift
            s_h = intel.get('sentiment_h', 0.0)
            s_a = intel.get('sentiment_a', 0.0)
            shift = (s_h - s_a) * 0.4  # Oracle Weight Alpha
            
            # Shift 1X2 probabilities (simplified)
            p_h = np.clip(p_h + shift, 0.05, 0.95)
            p_a = np.clip(p_a - shift, 0.05, 0.95)
            s = p_h + p_d + p_a; p_h /= s; p_d /= s; p_a /= s
        else:
            p_h, p_d, p_a = v9.get('v9_p_h', 0.33), v9.get('v9_p_d', 0.34), v9.get('v9_p_a', 0.33)
            p_btts, p_o25 = v9.get('v9_p_btts', 0.5), v9.get('v9_p_o25', 0.5)
            p_o15, p_o35 = v9.get('v9_p_o15', 0.5), v9.get('v9_p_o35', 0.5)

        gnn = None
        gnn_weight = 0.0
        if _gnn_layer_available:
            try:
                gnn = get_gnn_match_signals(league, h, a, match.to_dict())
                p_h, p_d, p_a, gnn_weight = blend_gnn_probs(p_h, p_d, p_a, gnn)
                if gnn and gnn_weight > 0:
                    intel['gnn'] = {
                        'active': True,
                        'weight': round(gnn_weight, 4),
                        'p_h': round(gnn.get('gnn_p_h', 0.0), 4),
                        'p_d': round(gnn.get('gnn_p_d', 0.0), 4),
                        'p_a': round(gnn.get('gnn_p_a', 0.0), 4),
                        'lift': round(gnn.get('gnn_lift', 0.0), 4),
                        'blend_lift': round(gnn.get('gnn_blend_accuracy_lift', 0.0), 4),
                        'logloss_lift': round(gnn.get('gnn_blend_logloss_lift', 0.0), 4),
                        'validation_accuracy': round(gnn.get('gnn_validation_accuracy', 0.0), 4),
                    }
            except Exception as e:
                print(f"    [GNN] Inference skipped: {e}")

        # XGBoost ensemble blend (15% fixed weight, gated by model file existence)
        xgb_signals = get_xgb_signals(league, match, num_vec)
        if xgb_signals:
            xgb_w = 0.15
            base = np.array([p_h, p_d, p_a], dtype=float)
            xgb_p = np.array([xgb_signals['xgb_p_h'], xgb_signals['xgb_p_d'], xgb_signals['xgb_p_a']], dtype=float)
            mixed = base * (1.0 - xgb_w) + xgb_p * xgb_w
            mixed = np.clip(mixed, 0.01, 0.98)
            mixed /= mixed.sum()
            p_h, p_d, p_a = float(mixed[0]), float(mixed[1]), float(mixed[2])
            intel['xgb'] = {'active': True, 'weight': xgb_w,
                            'p_h': round(xgb_signals['xgb_p_h'], 4),
                            'p_d': round(xgb_signals['xgb_p_d'], 4),
                            'p_a': round(xgb_signals['xgb_p_a'], 4)}
            print(f"    [XGB] Blended at {xgb_w:.0%}: H={xgb_signals['xgb_p_h']:.3f} D={xgb_signals['xgb_p_d']:.3f} A={xgb_signals['xgb_p_a']:.3f}")

        market_context = {}
        if _market_count_available:
            try:
                market_context = predict_market_context(league, h, a)
                goal_model = market_context.get("goal_model", {})
                if goal_model:
                    goal_weight = 0.22 if market_context.get("model_created_at") else 0.12
                    goal_probs = np.array([
                        goal_model.get("p_home", p_h),
                        goal_model.get("p_draw", p_d),
                        goal_model.get("p_away", p_a),
                    ], dtype=float)
                    current_probs = np.array([p_h, p_d, p_a], dtype=float)
                    mixed = (current_probs * (1.0 - goal_weight)) + (goal_probs * goal_weight)
                    mixed = np.clip(mixed, 0.01, 0.98)
                    mixed = mixed / mixed.sum()
                    p_h, p_d, p_a = float(mixed[0]), float(mixed[1]), float(mixed[2])
                    p_btts = _blend_prob(p_btts, goal_model.get("p_btts", p_btts), 0.35)
                    p_o25 = _blend_prob(p_o25, goal_model.get("p_over25", p_o25), 0.35)
                    p_o15 = _blend_prob(p_o15, goal_model.get("p_over15", p_o15), 0.35)
                    p_o35 = _blend_prob(p_o35, goal_model.get("p_over35", p_o35), 0.35)
                    intel["market_model"] = market_context
            except Exception as e:
                print(f"    [MarketCount] Inference skipped: {e}")

        p_h, p_d, p_a, situation_adjustment = adjust_1x2_for_situation(p_h, p_d, p_a, situation)
        intel['league_situation_adjustment'] = situation_adjustment
        # Pass 1: empirical bias correction (fast, from calibration report JSON)
        p_h, p_d, p_a = calibrate_1x2_probs(p_h, p_d, p_a)
        # Pass 2: isotonic regression (non-parametric, per-league, from DB val set)
        if _isotonic_available:
            p_h, p_d, p_a = apply_isotonic_calibration(p_h, p_d, p_a, league)

        # CLV snapshot — record current odds + calibrated model probs for odds-movement tracking
        if _clv_available:
            try:
                _clv_record(
                    league, h, a, str(match['Date']),
                    float(match.get('h_course') or match.get('B365H') or 0.0),
                    float(match.get('d_course') or match.get('B365D') or 0.0),
                    float(match.get('a_course') or match.get('B365A') or 0.0),
                    p_h, p_d, p_a,
                )
            except Exception as _clv_err:
                print(f"    [CLV] Snapshot skipped: {_clv_err}")
            
        profile = "Balanced"
        if p_h > 0.55: profile = "Surgical"
        elif p_o25 > 0.65: profile = "Blitz"
        
        markets = map_dominance_markets(profile, p_h, p_o25, p_btts, p_a, p_d)
        if market_context:
            corners_o95 = (market_context.get("corners") or {}).get("over_9_5", 0.0)
            cards_o45 = (market_context.get("cards") or {}).get("over_4_5", 0.0)
            if corners_o95 >= 0.62:
                markets["secondary"] = "Corners Over 9.5"
            elif cards_o45 >= 0.60:
                markets["secondary"] = "Cards Over 4.5"

        market_router = {}
        if _market_router_available:
            try:
                market_router = route_market_selection(
                    match=match.to_dict() if hasattr(match, "to_dict") else dict(match),
                    league=league,
                    base_primary=markets["primary"],
                    base_secondary=markets.get("secondary", ""),
                    p_h=p_h,
                    p_d=p_d,
                    p_a=p_a,
                    p_btts=p_btts,
                    p_o15=p_o15,
                    p_o25=p_o25,
                    p_o35=p_o35,
                    market_context=market_context,
                    situation=situation,
                )
                if market_router:
                    markets["primary"] = market_router.get("selected_market") or markets["primary"]
                    markets["secondary"] = market_router.get("secondary_market") or markets.get("secondary")
                    intel["market_router"] = market_router
                    if market_router.get("promoted"):
                        print(
                            f"    [MarketRouter] {market_router.get('action')}: "
                            f"{market_router.get('base_primary')} -> {market_router.get('selected_market')} "
                            f"(H={market_router.get('entropy')}, trust={market_router.get('trust_score')})"
                        )
            except Exception as e:
                print(f"    [MarketRouter] Routing skipped: {e}")

        edge = calculate_primary_edge(match, markets['primary'], p_h, p_d, p_a, p_o25, p_btts)
        if market_router and market_router.get("selected_market") == markets["primary"]:
            edge = max(edge, float(market_router.get("edge", 0.0) or 0.0))

        cld_delta = 0.0
        clv = 0.0
        if _clv_available:
            try:
                cld_delta = compute_cld_delta(league, h, a, str(match['Date']), markets['primary'])
                clv = compute_clv(league, h, a, str(match['Date']), p_h, p_d, p_a, markets['primary'])
            except Exception:
                pass

        eqi, tier = calculate_eqi_v2(edge, 0.85, 2.5, cld_delta=cld_delta)
        stake = calculate_stake(edge, 0.85, tier)
        if market_router and market_router.get("action", "").startswith("promote"):
            eqi = max(eqi, float(market_router.get("trust_score", 0.0) or 0.0))
            if market_router.get("edge_basis") == "model_without_market_odds":
                stake = max(stake, float(market_router.get("stake_hint", 0.0) or 0.0))
        
        res_dict = {
            'Home': h, 'Away': a, 'League': league, 'Date': match['Date'], 'Time': match.get('Time', '15:00'),
            'source_url': match.get('source_url'), 'flashscore_id': match.get('flashscore_id'),
            'prediction': markets['primary'], 'primary_market': markets['primary'], 'secondary_market': markets['secondary'],
            'P(H)': p_h, 'P(D)': p_d, 'P(A)': p_a,
            'P(BTTS)': p_btts, 'P(O2.5)': p_o25, 'P(1X)': p_h + p_d, 'P(X2)': p_a + p_d, 'DNB': p_h / (p_h + p_a),
            'over15_prob': p_o15, 'under15_prob': 1.0 - p_o15, 'over35_prob': p_o35, 'under35_prob': 1.0 - p_o35,
            'Value Edge': edge, 'CLD': clv, 'CLD_delta': cld_delta, 'execution_trust': eqi, 'stake_pct': stake,
            'E(Corners)': v11.get('v11_expected_corners') if v11 else market_context.get('expected_corners_total'),
            'E(Cards)': v11.get('v11_expected_yc') if v11 else market_context.get('expected_cards_total'),
            'market_model': market_context,
            'market_router': market_router,
            'motivation_score': situation.get('match_pressure_score', 0),
            'motivation_label': situation.get('label', 'Normal table context'),
            'league_situation': situation,
            'Momentum H': str(intel.get('momentum_h', 0.5)), 'Momentum A': str(intel.get('momentum_a', 0.5)),
            'gnn_active': bool(gnn and gnn_weight > 0), 'gnn_weight': gnn_weight,
            'intel_feats': intel, 'graph_snapshot_id': snap_id
        }
        
        # FINAL RATIONALE CAPTURE (After all adjustments)
        rationale = generate_v11_rationale(p_h, p_d, p_a, p_o25, p_btts, situation, intel)
        res_dict['rationale'] = rationale
        results.append(res_dict)
        
    return pd.DataFrame(results)

def generate_v11_rationale(p_h, p_d, p_a, p_o25, p_btts, situation, intel):
    """Derive human-readable rationale from model signals."""
    drivers = []
    # Probability Drivers
    if p_h > 0.65: drivers.append("High Home Win Probability")
    elif p_h > 0.40: drivers.append("Home Win Lean")
    
    if p_a > 0.65: drivers.append("Strong Away Dominance")
    elif p_a > 0.40: drivers.append("Away Win Lean")
    
    if p_d > 0.38: drivers.append("Elevated Draw Probability")
    elif p_d > 0.25: drivers.append("Tactical Stalemate Risk")
    
    if p_o25 > 0.68: drivers.append("High Scoring Potential (O2.5)")
    elif p_o25 > 0.55: drivers.append("Attacking Overload Expected")
    
    if p_btts > 0.60: drivers.append("Both Teams Scoring Likelihood")
    
    # Situational Drivers
    press = situation.get('match_pressure_score', 0)
    if press > 65: drivers.append(f"Critical Motivation: {situation.get('label', 'High Pressure')}")
    elif press > 35: drivers.append("Significant Match Pressure")
    
    if not drivers:
        drivers.append("Standard Match Profile")
    
    # Momentum Drivers
    mom_h = float(intel.get('momentum_h', 0.5))
    mom_a = float(intel.get('momentum_a', 0.5))
    if mom_h > 0.75: drivers.append("Home Momentum Spike")
    if mom_a > 0.75: drivers.append("Away Momentum Spike")
    if abs(mom_h - mom_a) > 0.4: drivers.append("Significant Momentum Imbalance")
    
    # Market Drivers
    if "Corners" in str(intel.get('market_router', {}).get('selected_market', '')):
        drivers.append("Deep Set-Piece Signal")
    
    return {
        "top_drivers": drivers[:4],
        "primary_signal": drivers[0] if drivers else "Standard Match Profile",
        "confidence": "High" if max(p_h, p_a, p_d) > 0.6 else "Medium"
    }

def get_v9_variable_impact(league, h_team, a_team, num_vec):
    try:
        from market_intelligence_v92 import MarketIntelligenceV92
        mi = MarketIntelligenceV92(league)
        with open(os.path.join(MODEL_ROOT, 'v9', f'{league}_team_map.json'), 'r') as f:
            team_map = json.load(f)
        impact = mi.calculate_variable_impact(team_map.get(h_team, 0), team_map.get(a_team, 0), 0, num_vec)
        return impact
    except: return {}

if __name__ == "__main__":
    LEAGUES = ["PremierLeague", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]
    all_preds = []
    
    for league in LEAGUES:
        print(f"\n[Engine] Starting {league} pipeline...")
        # 1. Bootstrap
        ratings = precalculate_all_features(league)
        
        # 2. Get Matches (Using mock data for verification as requested by 'u run it')
        mock_matches = pd.DataFrame([
            {'HomeTeam': 'Arsenal', 'AwayTeam': 'Chelsea', 'Date': '2026-05-02', 'h_course': 1.85, 'd_course': 3.6, 'a_course': 4.2}
        ])
        
        # 3. Predict
        preds_df = predict_results(mock_matches, league, ratings, [])
        if not preds_df.empty:
            all_preds.append(preds_df)
            print(f"  [Engine] Generated {len(preds_df)} predictions for {league}")

    if all_preds:
        final_df = pd.concat(all_preds)
        out_path = os.path.join(PROJECT_ROOT, "web", "data", "v11_predictions.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        final_df.to_json(out_path, orient='records', indent=2)
        print(f"\n[Engine] Results exported to {out_path}")
