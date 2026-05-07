import os
import sys
import json
import zipfile
import io
import argparse
import sqlite3
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, project_root)

from scipy.stats import poisson
from v11_hybrid_model import V11Hybrid
from training_data_guard import CANONICAL_VIEW_NAME, ensure_training_data_guard
from league_situation import (
    RunningSituationBuilder,
    V11_SITUATION_FEATURE_COLUMNS,
    build_v11_situation_features,
)

# XGBoost ensemble imports
try:
    import xgboost as xgb
    from xgboost_ensemble import XGBoostEnsemble
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, continuing with neural model only")
    XGBOOST_AVAILABLE = False


def weighted_mean(loss, sample_weight=None):
    if sample_weight is None:
        return loss.mean()
    sample_weight = sample_weight.to(loss.device)
    sample_weight = sample_weight.reshape(()) if loss.dim() == 0 else sample_weight.reshape_as(loss)
    return (loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)


def weighted_bce(pred, target, pos_weight=1.0, sample_weight=None):
    w = torch.where(
        target > 0.5,
        torch.full_like(target, pos_weight),
        torch.ones_like(target),
    )
    loss = F.binary_cross_entropy(pred, target, weight=w, reduction='none')
    return weighted_mean(loss, sample_weight)

DATA_ZIP = os.path.join(project_root, 'data', 'football_data.zip')
DB_PATH = os.path.join(project_root, 'web', 'data', 'intelligence_hub.db')
V11_MODEL_DIR = os.path.join(project_root, 'model', 'v11')
os.makedirs(V11_MODEL_DIR, exist_ok=True)
_TRAINING_SOURCE_TABLE = None


def training_examples_source_table():
    global _TRAINING_SOURCE_TABLE
    if _TRAINING_SOURCE_TABLE:
        return _TRAINING_SOURCE_TABLE
    try:
        report = ensure_training_data_guard(write_report=True)
        if report.get("status") == "ready":
            _TRAINING_SOURCE_TABLE = CANONICAL_VIEW_NAME
            return _TRAINING_SOURCE_TABLE
    except Exception as exc:
        print(f"    Training data guard unavailable, using raw training_examples: {exc}")
    _TRAINING_SOURCE_TABLE = "training_examples"
    return _TRAINING_SOURCE_TABLE


MARKETS = {
    '1X2':      {'label': 'label_1x2',  'type': 'multi'},
    'BTTS':     {'label': 'label_btts', 'type': 'binary'},
    'Over2.5':  {'label': 'label_o25',  'type': 'binary'},
}

LEAGUE_PREFIXES = {
    'PremierLeague': 'PremierLeague',
    'LaLiga': 'LaLiga',
    'SerieA': 'SerieA',
    'Bundesliga': 'Bundesliga',
    'Ligue1': 'Ligue1',
}

# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_league_seasons(league_prefix):
    with zipfile.ZipFile(DATA_ZIP) as z:
        files = sorted([n for n in z.namelist() if n.startswith(league_prefix + '_')])
        seasons = []
        for fname in files:
            try:
                df = pd.read_csv(io.BytesIO(z.read(fname)))
                df.columns = [c.strip() for c in df.columns]
                req = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                if not all(c in df.columns for c in req):
                    continue
                df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
                df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
                df = df.dropna(subset=['FTHG', 'FTAG', 'FTR']).reset_index(drop=True)
                seasons.append((fname, df))
            except Exception:
                pass
    return seasons


_MIN_TRAINING_ROWS = 800  # Minimum rows required; widen window if below this

def _fetch_league_df(league, date_filter_sql):
    """Execute a single SQLite fetch for a given date range filter."""
    try:
        con = sqlite3.connect(DB_PATH)
        source_table = training_examples_source_table()
        df = pd.read_sql_query(
            f"""
            SELECT match_date AS Date,
                   home_team AS HomeTeam,
                   away_team AS AwayTeam,
                   home_goals AS FTHG,
                   away_goals AS FTAG,
                   actual_ftr AS FTR,
                   home_odds AS B365H,
                   draw_odds AS B365D,
                   away_odds AS B365A,
                   HY, AY, HR, AR, HC, AC,
                   HS, "AS", HST, AST, HF, AF, HO, AO,
                   HPoss, APoss, HXG, AXG, HBC, ABC,
                   yc_home_avg5 AS h_yc_avg5,
                   yc_away_avg5 AS a_yc_avg5,
                   corners_home_avg5 AS h_cw_avg5,
                   corners_away_avg5 AS a_cw_avg5,
                   source_confidence,
                   source
            FROM {source_table}
            WHERE league = ?
              AND actual_ftr IN ('H', 'D', 'A')
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
              {date_filter_sql}
            ORDER BY match_date ASC, id ASC
            """,
            con,
            params=(league,),
        )
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_training_examples_frame(league):
    """Load normalized SQLite examples with adaptive window fallback.

    Starts with a 1-year window. If the resulting dataset is below
    _MIN_TRAINING_ROWS, it automatically widens to 2 years, then 3 years,
    then falls back to all-time data to ensure the model has enough signal
    to converge reliably across all leagues.
    """
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    # Adaptive window cascade: 1yr → 2yr → 3yr → all-time
    data_window = os.environ.get("V11_DATA_WINDOW", "all-time").strip().lower()
    if data_window in {"all", "all-time", "full", "full-archive"}:
        windows = [("", "all-time")]
    else:
        windows = [
            ("AND match_date >= date('now', '-1 year')", "1-year"),
            ("AND match_date >= date('now', '-2 years')", "2-year"),
            ("AND match_date >= date('now', '-3 years')", "3-year"),
            ("", "all-time"),
        ]

    df = pd.DataFrame()
    for date_filter_sql, label in windows:
        candidate = _fetch_league_df(league, date_filter_sql)
        if not candidate.empty and len(candidate) >= _MIN_TRAINING_ROWS:
            df = candidate
            if label != "1-year":
                print(f"    [Adaptive Window] {league}: used {label} data ({len(candidate)} rows)")
            break
        df = candidate  # keep best available even if still thin

    if df.empty:
        return df

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
    df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
    for col in ('B365H', 'B365D', 'B365A', 'HY', 'AY', 'HR', 'AR', 'HC', 'AC',
                'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HO', 'AO',
                'HPoss', 'APoss', 'HXG', 'AXG', 'HBC', 'ABC',
                'h_yc_avg5', 'a_yc_avg5', 'h_cw_avg5', 'a_cw_avg5', 'source_confidence'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']).copy()
    df = df[df['FTR'].isin(['H', 'D', 'A'])]
    df = df.sort_values(['Date', 'HomeTeam', 'AwayTeam', 'source_confidence'], na_position='first')
    df = df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last')
    return df.reset_index(drop=True)

def compute_rolling_stats(all_matches_df, window=5):
    stats = {}
    teams = set(all_matches_df['HomeTeam'].tolist() + all_matches_df['AwayTeam'].tolist())
    for team in teams:
        h = all_matches_df[all_matches_df['HomeTeam'] == team].copy()
        a = all_matches_df[all_matches_df['AwayTeam'] == team].copy()
        def result_pts(row, side):
            return 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0) if side == 'H' else 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
        h_pts = [result_pts(r, 'H') for _, r in h.iterrows()]
        a_pts = [result_pts(r, 'A') for _, r in a.iterrows()]
        h_gf, h_ga = h['FTHG'].tolist(), h['FTAG'].tolist()
        a_gf, a_ga = a['FTAG'].tolist(), a['FTHG'].tolist()
        h_sot = h['HST'].tolist() if 'HST' in h.columns else []
        a_sot = a['AST'].tolist() if 'AST' in a.columns else []
        h_corn = h['HC'].tolist() if 'HC' in h.columns else []
        a_corn = a['AC'].tolist() if 'AC' in a.columns else []
        h_cs = [1 if ga == 0 else 0 for ga in h_ga]
        a_cs = [1 if ga == 0 else 0 for ga in a_ga]

        def tail_avg(lst, w=window): return np.mean(lst[-w:]) if lst else 0.0
        def tail_sum(lst, w=window): return sum(lst[-w:]) if lst else 0.0

        stats[team] = {
            'gf_h': tail_avg(h_gf), 'ga_h': tail_avg(h_ga),
            'gf_a': tail_avg(a_gf), 'ga_a': tail_avg(a_ga),
            'form_h': tail_avg(h_pts), 'form_a': tail_avg(a_pts),
            'momentum_h': tail_sum(h_pts, 3), 'momentum_a': tail_sum(a_pts, 3),
            'sot_h': tail_avg(h_sot) if h_sot else 4.5, 'sot_a': tail_avg(a_sot) if a_sot else 3.5,
            'corn_h': tail_avg(h_corn) if h_corn else 5.0, 'corn_a': tail_avg(a_corn) if a_corn else 4.5,
            'cs_rate_h': tail_avg(h_cs), 'cs_rate_a': tail_avg(a_cs),
            'n_games': len(h) + len(a),
        }
    return stats

# ─── Hybrid Dataset ─────────────────────────────────────────────────────────────
class HybridMatchDataset(Dataset):
    def __init__(self, matches_df, team_map, stats, centrality_scores, sequence_length=20, league=None, initial_history=None):
        self.matches_df = matches_df.reset_index(drop=True)
        self.team_map, self.stats, self.centrality = team_map, stats, centrality_scores
        self.seq_len = sequence_length
        self.league = league
        self.situation_builder = RunningSituationBuilder(league, initial_history=initial_history)
        self.reference_date = pd.to_datetime(self.matches_df.get('Date'), errors='coerce').max() if 'Date' in self.matches_df.columns else pd.NaT
        self.half_life_days = max(30.0, float(os.environ.get("V11_TIME_DECAY_HALFLIFE_DAYS", "730")))
        self.v8_features, self.h_ids, self.a_ids, self.h_seqs, self.a_seqs, self.cent_feats = [], [], [], [], [], []
        self.l_1x2, self.l_btts, self.l_o25, self.l_o15, self.l_o35 = [], [], [], [], []
        self.l_yc, self.l_corners = [], []  # Phase 5: card + corner targets
        self.sample_weights = []
        self._build_tensors()

    def _build_tensors(self):
        h_hist, a_hist = {t: [] for t in self.team_map}, {t: [] for t in self.team_map}
        h2h_hist = defaultdict(list)
        def row_float(row, names, default=-1.0):
            for name in names:
                if name in row.index and not pd.isna(row.get(name)):
                    return float(row.get(name))
            return default
        def pair_key(home, away):
            return tuple(sorted((str(home), str(away))))
        def h2h_features(home, away):
            prior = h2h_hist[pair_key(home, away)][-10:]
            if not prior:
                return {
                    "home_win_rate": 1.0 / 3.0,
                    "draw_rate": 1.0 / 3.0,
                    "away_win_rate": 1.0 / 3.0,
                    "avg_goals": 2.5,
                }
            n = float(len(prior))
            home_wins = sum(1 for item in prior if item.get("winner") == home)
            away_wins = sum(1 for item in prior if item.get("winner") == away)
            draws = sum(1 for item in prior if item.get("winner") is None)
            return {
                "home_win_rate": home_wins / n,
                "draw_rate": draws / n,
                "away_win_rate": away_wins / n,
                "avg_goals": float(np.mean([item.get("goals", 2.5) for item in prior])),
            }
        def sample_weight(row):
            try:
                row_date = pd.to_datetime(row.get('Date'), errors='coerce')
                if pd.isna(row_date) or pd.isna(self.reference_date):
                    return 1.0
                age_days = max(0.0, float((self.reference_date - row_date).days))
                recency = float(np.exp(-age_days / self.half_life_days))
                confidence = row_float(row, ['source_confidence'], default=0.9)
                return max(0.10, min(3.0, recency * max(0.25, confidence)))
            except Exception:
                return 1.0
        for _, row in self.matches_df.iterrows():
            h, a = row['HomeTeam'], row['AwayTeam']
            if h not in self.team_map or a not in self.team_map: continue
            hs, as_ = self.stats.get(h, {}), self.stats.get(a, {})
            exp_h, exp_a = hs.get('gf_h', 1.4) * as_.get('ga_a', 1.2), as_.get('gf_a', 1.2) * hs.get('ga_h', 1.4)
            p_h = sum(poisson.pmf(i, exp_h) * sum(poisson.pmf(j, exp_a) for j in range(i)) for i in range(7))
            p_a = sum(poisson.pmf(j, exp_a) * sum(poisson.pmf(i, exp_h) for i in range(j)) for j in range(7))
            p_d = 1 - p_h - p_a
            situation = self.situation_builder.situation_for_match(h, a, row.get('Date'))
            self.v8_features.append(build_v11_situation_features(
                exp_h,
                exp_a,
                hs.get('form_h', 1.5)-as_.get('form_a', 1.5),
                p_h,
                p_d,
                p_a,
                situation,
                odds_h=row_float(row, ['B365H'], default=None) or None,
                odds_d=row_float(row, ['B365D'], default=None) or None,
                odds_a=row_float(row, ['B365A'], default=None) or None,
                h2h_features=h2h_features(h, a),
            ))
            self.sample_weights.append(sample_weight(row))
            self.h_seqs.append(self._pad_and_encode(h_hist[h][-self.seq_len:]))
            self.a_seqs.append(self._pad_and_encode(a_hist[a][-self.seq_len:]))
            self.h_ids.append(self.team_map[h]); self.a_ids.append(self.team_map[a])
            self.cent_feats.append([self.centrality.get(h, (0.05, 0.05))[0], self.centrality.get(a, (0.05, 0.05))[0]])
            ftr = row['FTR']
            self.l_1x2.append(0 if ftr == 'H' else (1 if ftr == 'D' else 2))
            self.l_btts.append(1 if (row['FTHG'] > 0 and row['FTAG'] > 0) else 0)
            self.l_o25.append(1 if (row['FTHG'] + row['FTAG']) > 2.5 else 0)
            self.l_o15.append(1 if (row['FTHG'] + row['FTAG']) > 1.5 else 0)
            self.l_o35.append(1 if (row['FTHG'] + row['FTAG']) > 3.5 else 0)
            # Count-market supervision uses final match counts. Rolling features
            # are fallback only for old rows that have no stats labels.
            hyc = row_float(row, ['HY'], default=np.nan)
            ayc = row_float(row, ['AY'], default=np.nan)
            hcw = row_float(row, ['HC'], default=np.nan)
            acw = row_float(row, ['AC'], default=np.nan)
            if np.isnan(hyc) or np.isnan(ayc):
                hyc = row_float(row, ['h_yc_avg5', 'yc_home_avg5'], default=np.nan)
                ayc = row_float(row, ['a_yc_avg5', 'yc_away_avg5'], default=np.nan)
            if np.isnan(hcw) or np.isnan(acw):
                hcw = row_float(row, ['h_cw_avg5', 'corners_home_avg5'], default=np.nan)
                acw = row_float(row, ['a_cw_avg5', 'corners_away_avg5'], default=np.nan)
            self.l_yc.append(float(hyc + ayc) if not (np.isnan(hyc) or np.isnan(ayc)) else -1.0)
            self.l_corners.append(float(hcw + acw) if not (np.isnan(hcw) or np.isnan(acw)) else -1.0)
            h_hist[h].append({
                'gf': row['FTHG'], 'ga': row['FTAG'], 'ftr': ftr, 'side': 'H',
                'shots_for': row_float(row, ['HS'], default=np.nan),
                'shots_against': row_float(row, ['AS'], default=np.nan),
                'sot_for': row_float(row, ['HST'], default=np.nan),
                'sot_against': row_float(row, ['AST'], default=np.nan),
                'corners_for': row_float(row, ['HC'], default=np.nan),
                'cards_for': row_float(row, ['HY'], default=np.nan),
                'xg_for': row_float(row, ['HXG'], default=np.nan),
                'xg_against': row_float(row, ['AXG'], default=np.nan),
                'possession': row_float(row, ['HPoss'], default=50.0),
                'opponent_strength': row_float(row, ['away_rank_strength'], default=0.5),
                'rest_days': row_float(row, ['rest_days'], default=3.0),
                'was_home': 1.0,
                'season_week': row_float(row, ['season_week'], default=20.0),
                'travel_distance': row_float(row, ['travel_distance'], default=0.0),
                'referee_id': row_float(row, ['referee_id'], default=0.0),
                'crowd_size': row_float(row, ['crowd_size'], default=25000.0),
            })
            a_hist[a].append({
                'gf': row['FTAG'], 'ga': row['FTHG'], 'ftr': ftr, 'side': 'A',
                'shots_for': row_float(row, ['AS'], default=np.nan),
                'shots_against': row_float(row, ['HS'], default=np.nan),
                'sot_for': row_float(row, ['AST'], default=np.nan),
                'sot_against': row_float(row, ['HST'], default=np.nan),
                'corners_for': row_float(row, ['AC'], default=np.nan),
                'cards_for': row_float(row, ['AY'], default=np.nan),
                'xg_for': row_float(row, ['AXG'], default=np.nan),
                'xg_against': row_float(row, ['HXG'], default=np.nan),
                'possession': row_float(row, ['APoss'], default=50.0),
                'opponent_strength': row_float(row, ['home_rank_strength'], default=0.5),
                'rest_days': row_float(row, ['rest_days'], default=3.0),
                'was_home': 0.0,
                'season_week': row_float(row, ['season_week'], default=20.0),
                'travel_distance': row_float(row, ['travel_distance'], default=0.0),
                'referee_id': row_float(row, ['referee_id'], default=0.0),
                'crowd_size': row_float(row, ['crowd_size'], default=25000.0),
            })
            h2h_hist[pair_key(h, a)].append({
                "winner": h if ftr == 'H' else (a if ftr == 'A' else None),
                "goals": float(row['FTHG'] + row['FTAG']),
            })
            self.situation_builder.update_from_result(row)

    def _pad_and_encode(self, seq):
        padding = [{'gf':0, 'ga':0, 'pts':1}] * (self.seq_len - len(seq))
        full = (padding + seq)[-self.seq_len:]
        feats = []
        def norm(value, scale, default=0.0):
            try:
                value = float(value)
                if np.isnan(value) or np.isinf(value):
                    return default
                return max(0.0, min(1.0, value / scale))
            except Exception:
                return default
        for m in full:
            pts = 3 if (m.get('ftr') == ('H' if m.get('side')=='H' else 'A')) else (1 if m.get('ftr')=='D' else 0) if 'side' in m else 1
            xg_for = m.get('xg_for')
            xg_against = m.get('xg_against')
            try:
                xg_diff = float(xg_for) - float(xg_against)
                if np.isnan(xg_diff) or np.isinf(xg_diff):
                    xg_diff = 0.0
            except Exception:
                xg_diff = 0.0
            # Extended sequence features (20+ features)
            possession = m.get('possession', 50.0)  # default to 50% possession
            opponent_strength = m.get('opponent_strength', 0.5)  # default to neutral strength
            rest_days = m.get('rest_days', 3.0)  # default to 3 days rest
            was_home = m.get('was_home', 1.0 if m.get('side') == 'H' else 0.0)

            feats.append([
                norm(m.get('gf', 0), 5),
                norm(m.get('ga', 0), 5),
                pts / 3,
                norm(m.get('shots_for'), 25, default=0.45),
                norm(m.get('shots_against'), 25, default=0.45),
                norm(m.get('sot_for'), 10, default=0.40),
                norm(m.get('sot_against'), 10, default=0.40),
                norm(m.get('corners_for'), 15, default=0.35),
                norm(m.get('cards_for'), 8, default=0.25),
                max(0.0, min(1.0, (xg_diff + 3.0) / 6.0)),
                norm(xg_for, 2.0, default=0.5),  # xg_for
                norm(xg_against, 2.0, default=0.5),  # xg_against
                norm(possession, 100.0, default=0.5),  # possession %
                norm(opponent_strength, 1.0, default=0.5),  # opponent strength
                norm(rest_days, 7.0, default=0.43),  # rest days (3 days default)
                was_home,  # was home
                norm(m.get('season_week', 20.0), 52.0, default=0.38),  # season week
                norm(m.get('travel_distance', 0.0), 1000.0, default=0.0),  # travel distance
                norm(m.get('referee_id', 0.0), 10.0, default=0.0),  # referee id
                norm(m.get('crowd_size', 25000.0), 100000.0, default=0.25),  # crowd size
            ])
        return torch.tensor(feats, dtype=torch.float32)

    def __len__(self): return len(self.l_1x2)
    def __getitem__(self, idx):
        return (torch.tensor(self.v8_features[idx], dtype=torch.float32), self.h_seqs[idx], self.a_seqs[idx],
                torch.tensor(self.h_ids[idx], dtype=torch.long), torch.tensor(self.a_ids[idx], dtype=torch.long),
                torch.tensor(self.cent_feats[idx], dtype=torch.float32), torch.tensor(self.l_1x2[idx], dtype=torch.long),
                torch.tensor(self.l_btts[idx], dtype=torch.long), torch.tensor(self.l_o25[idx], dtype=torch.long),
                torch.tensor(self.l_o15[idx], dtype=torch.long), torch.tensor(self.l_o35[idx], dtype=torch.long),
                torch.tensor(self.l_yc[idx], dtype=torch.float32),
                torch.tensor(self.l_corners[idx], dtype=torch.float32),
                torch.tensor(self.sample_weights[idx], dtype=torch.float32))

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, sample_weight=None):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if sample_weight is None:
            return loss.mean()
        sample_weight = sample_weight.to(loss.device)
        return (loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

# ─── Training ─────────────────────────────────────────────────────────────────
def train_hybrid(model, train_loader, val_loader, epochs=80, lr=3e-4, wd=1e-2, league=None, warmup_epochs=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    device = next(model.parameters()).device
    best_market_acc = {}
    best_val_acc, best_quality, best_state, no_improve = 0, float("inf"), None, 0
    history = []
    aborted = False
    for epoch in range(epochs):
        # Gradual draw weight ramp — capped at 1.8 to prevent draw-collapse
        if epoch < 5:
            draw_weight = torch.tensor([1.0, 1.0, 1.0], device=device)
        elif epoch < 15:
            draw_weight = torch.tensor([1.0, 1.3, 1.0], device=device)
        else:
            draw_weight = torch.tensor([1.0, 1.8, 1.0], device=device)
        criterion_1x2 = FocalLoss(alpha=draw_weight, gamma=2.0)
        if warmup_epochs and epoch < warmup_epochs:
            warmup_lr = lr * float(epoch + 1) / float(warmup_epochs)
            for group in optimizer.param_groups:
                group['lr'] = warmup_lr
        model.train()
        train_l = train_1x2 = train_btts = train_o25 = train_o15 = train_o35 = train_temp = 0.0
        train_yc = train_corn = 0.0
        train_batches = 0
        for v8, h_s, a_s, h_id, a_id, cent, l1x2, lbtts, lo25, lo15, lo35, lyc, lcorn, sample_w in train_loader:
            optimizer.zero_grad()
            logits, pb, po, p15, p35, T, yc_pred, corn_pred = model(v8, h_s, a_s, h_id, a_id, cent)
            loss_1x2 = 1.5 * criterion_1x2(logits, l1x2, sample_w)
            loss_btts = 0.3 * weighted_mean(F.binary_cross_entropy(pb.squeeze(), lbtts.float(), reduction='none'), sample_w)
            loss_o25 = 0.3 * weighted_mean(F.binary_cross_entropy(po.squeeze(), lo25.float(), reduction='none'), sample_w)
            loss_o15 = 0.3 * weighted_bce(p15.squeeze(), lo15.float(), pos_weight=0.33, sample_weight=sample_w)
            loss_o35 = 0.3 * weighted_bce(p35.squeeze(), lo35.float(), pos_weight=2.33, sample_weight=sample_w)
            loss_temp = 0.005 * (T - 1.0).pow(2).mean()
            # Count heads are trained only where final labels exist. A zero-card
            # match is valid supervision, so missing labels are encoded as -1.
            yc_mask = (lyc >= 0).float() * sample_w
            corn_mask = (lcorn >= 0).float() * sample_w
            loss_yc = 0.1 * ((F.huber_loss(yc_pred.squeeze(), lyc.clamp_min(0), reduction='none') * yc_mask).sum() / yc_mask.sum().clamp_min(1.0))
            loss_corn = 0.1 * ((F.huber_loss(corn_pred.squeeze(), lcorn.clamp_min(0), reduction='none') * corn_mask).sum() / corn_mask.sum().clamp_min(1.0))
            loss = loss_1x2 + loss_btts + loss_o25 + loss_o15 + loss_o35 + loss_temp + loss_yc + loss_corn
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_l += loss.item()
            train_1x2 += loss_1x2.item()
            train_btts += loss_btts.item()
            train_o25 += loss_o25.item()
            train_o15 += loss_o15.item()
            train_o35 += loss_o35.item()
            train_temp += loss_temp.item()
            train_yc   += loss_yc.item()
            train_corn += loss_corn.item()
            train_batches += 1
        scheduler.step()
        model.eval()
        corr, tot = 0, 0
        val_loss = val_nll = temp_sum = 0.0
        brier_sum = 0.0
        draw_tp = draw_total = 0
        yc_abs = corn_abs = 0.0
        yc_n = corn_n = 0
        with torch.no_grad():
            btts_corr = o25_corr = o15_corr = o35_corr = 0
            for v8, h_s, a_s, h_id, a_id, cent, l1x2, lbtts, lo25, lo15, lo35, lyc, lcorn, sample_w in val_loader:
                logits, pb, po, p15, p35, T, yc_pred, corn_pred = model(v8, h_s, a_s, h_id, a_id, cent)
                val_loss += criterion_1x2(logits, l1x2).item() * l1x2.size(0)
                val_nll += F.cross_entropy(logits, l1x2, reduction='sum').item()
                probs = torch.softmax(logits, dim=1)
                one_hot = F.one_hot(l1x2, num_classes=3).float()
                brier_sum += ((probs - one_hot).pow(2).sum(dim=1)).sum().item()
                temp_sum += T.mean().item() * l1x2.size(0)
                preds = logits.argmax(1)
                corr += (preds == l1x2).sum().item(); tot += l1x2.size(0)
                draw_mask = (l1x2 == 1)
                draw_total += draw_mask.sum().item()
                draw_tp += ((preds == 1) & draw_mask).sum().item()
                btts_corr += ((pb.squeeze() >= 0.5) == lbtts.bool()).sum().item()
                o25_corr += ((po.squeeze() >= 0.5) == lo25.bool()).sum().item()
                o15_corr += ((p15.squeeze() >= 0.5) == lo15.bool()).sum().item()
                o35_corr += ((p35.squeeze() >= 0.5) == lo35.bool()).sum().item()
                yc_mask = lyc >= 0
                corn_mask = lcorn >= 0
                if yc_mask.any():
                    yc_abs += (yc_pred.squeeze()[yc_mask] - lyc[yc_mask]).abs().sum().item()
                    yc_n += int(yc_mask.sum().item())
                if corn_mask.any():
                    corn_abs += (corn_pred.squeeze()[corn_mask] - lcorn[corn_mask]).abs().sum().item()
                    corn_n += int(corn_mask.sum().item())
        val_acc = corr/tot if tot > 0 else 0
        val_loss = val_loss/tot if tot > 0 else 0
        val_log_loss = val_nll/tot if tot > 0 else 0
        val_brier_1x2 = brier_sum/tot if tot > 0 else 0
        draw_recall = draw_tp/draw_total if draw_total > 0 else 0
        temp_mean = temp_sum/tot if tot > 0 else 0
        val_market_acc = {
            "btts_accuracy": btts_corr/tot if tot > 0 else 0,
            "over25_accuracy": o25_corr/tot if tot > 0 else 0,
            "over15_accuracy": o15_corr/tot if tot > 0 else 0,
            "over35_accuracy": o35_corr/tot if tot > 0 else 0,
            "draw_recall": draw_recall,
            "val_loss": val_loss,
            "val_log_loss": val_log_loss,
            "val_brier_1x2": val_brier_1x2,
            "yc_mae": yc_abs/max(1, yc_n),
            "corners_mae": corn_abs/max(1, corn_n),
            "yc_labeled": yc_n,
            "corners_labeled": corn_n,
            "temperature_mean": temp_mean,
        }
        train_scale = max(train_batches, 1)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_l / train_scale,
            "train_1x2": train_1x2 / train_scale,
            "train_btts": train_btts / train_scale,
            "train_over25": train_o25 / train_scale,
            "train_over15": train_o15 / train_scale,
            "train_over35": train_o35 / train_scale,
            "train_temperature": train_temp / train_scale,
            "val_loss": val_loss,
            "val_log_loss": val_log_loss,
            "val_brier_1x2": val_brier_1x2,
            "val_1x2_acc": val_acc,
            "draw_recall": draw_recall,
            "yc_mae": yc_abs/max(1, yc_n),
            "corners_mae": corn_abs/max(1, corn_n),
            "yc_labeled": yc_n,
            "corners_labeled": corn_n,
            **val_market_acc,
        }
        history.append(epoch_metrics)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"    Epoch {epoch+1:03d}: train={epoch_metrics['train_loss']:.4f} "
                f"1x2={epoch_metrics['train_1x2']:.4f} btts={epoch_metrics['train_btts']:.4f} "
                f"o25={epoch_metrics['train_over25']:.4f} o15={epoch_metrics['train_over15']:.4f} "
                f"o35={epoch_metrics['train_over35']:.4f} temp={epoch_metrics['train_temperature']:.4f} "
                f"val={val_loss:.4f} acc={val_acc:.4f} dr={draw_recall:.4f} T={temp_mean:.4f}"
            )
        if val_log_loss > 1.20:
            print(f"{league} V11 diverged -- skipping")
            aborted = True
            break
        # Penalise draw_recall that strays too far from 0.28 (typical draw rate)
        draw_penalty = 0.20 * abs(draw_recall - 0.28)
        quality = val_log_loss + (0.35 * val_brier_1x2) + draw_penalty
        if quality < (best_quality - 1e-4) or (abs(quality - best_quality) <= 1e-4 and val_acc > best_val_acc):
            best_quality = quality
            best_val_acc = val_acc
            best_market_acc = val_market_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 15: break
    if best_state and not aborted:
        model.load_state_dict(best_state)
    return model, best_val_acc, best_market_acc, history, aborted, best_quality

def run_hybrid_pipeline(league, train_seasons=4, epochs=80, dropout=0.2, lr=3e-4, wd=1e-2):
    print(f"  [Pipeline] Training {league}...")
    source = "football_data_zip"
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    if os.environ.get("V11_USE_TRAINING_EXAMPLES", "1").strip().lower() not in {"0", "false", "no"}:
        examples_df = load_training_examples_frame(league)
        if len(examples_df) >= int(os.environ.get("V11_MIN_DB_EXAMPLES", "500")):
            source = "sqlite_training_examples"
            split_outer = max(1, int(len(examples_df) * 0.85))
            train_df = examples_df.iloc[:split_outer].copy()
            test_df = examples_df.iloc[split_outer:].copy()
            print(f"    Source: SQLite training_examples ({len(examples_df)} rows)")

    if train_df.empty:
        seasons = load_league_seasons(LEAGUE_PREFIXES[league])
        if len(seasons) < 2: return None
        train_df = pd.concat([s[1] for s in seasons[:train_seasons]], ignore_index=True)
        test_df = seasons[-1][1]
        print(f"    Source: zipped football-data seasons ({len(train_df)} rows)")

    if 'Date' in train_df.columns:
        train_df['Date'] = pd.to_datetime(train_df['Date'], errors='coerce')
        train_df = train_df.sort_values(['Date', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)
    if 'Date' in test_df.columns:
        test_df['Date'] = pd.to_datetime(test_df['Date'], errors='coerce')
        test_df = test_df.sort_values(['Date', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)
    all_teams = sorted(set(train_df['HomeTeam'].tolist() + test_df['HomeTeam'].tolist()))
    team_map = {t: i for i, t in enumerate(all_teams)}
    from v8_gsn_architect import V8MotifDetector
    det = V8MotifDetector(); det.graph.clear()
    for _, r in train_df.iterrows(): det.graph.add_edge(r['HomeTeam'], r['AwayTeam'], weight=1.0)
    centrality = det.get_centrality_scores()
    split = int(len(train_df) * 0.8)
    train_slice = train_df.iloc[:split].copy()
    val_slice = train_df.iloc[split:].copy()
    stats = compute_rolling_stats(train_slice)
    train_ds = HybridMatchDataset(train_slice, team_map, stats, centrality, league=league)
    val_ds = HybridMatchDataset(val_slice, team_map, stats, centrality, league=league, initial_history=train_slice)
    print(f"    Split sizes: train={len(train_ds)} val={len(val_ds)}")
    # Optimization: Balanced batch size to avoid RAM allocation errors on CPU
    tr_l = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_l = DataLoader(val_ds, batch_size=128, num_workers=0, pin_memory=True)
    model = V11Hybrid(num_teams=len(team_map), num_v8_features=28, seq_len=20, dropout=dropout)
    approach = 'scratch'
    effective_lr = lr
    warmup_epochs = 5
    trained, val_acc, market_acc, history, aborted, best_quality = train_hybrid(
        model, tr_l, val_l, epochs=epochs, lr=effective_lr, wd=wd,
        league=league, warmup_epochs=warmup_epochs,
    )
    os.makedirs(V11_MODEL_DIR, exist_ok=True)
    if aborted:
        artifact = {
            "league": league,
            "status": "skipped",
            "reason": f"{league} V11 diverged -- skipping",
            "training_approach": approach,
            "training_source": source,
            "history": history,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(V11_MODEL_DIR, f"{league}_hybrid_v11_draww_v5_metrics.json"), "w") as f:
            json.dump(artifact, f, indent=2)
        return {'league': league, 'val_acc': val_acc, 'status': 'skipped', 'history': history}
    model_path = os.path.join(V11_MODEL_DIR, f"{league}_hybrid_v11_draww_v5.pt")
    torch.save(trained.state_dict(), model_path)
    with open(os.path.join(V11_MODEL_DIR, f"{league}_team_map.json"), 'w') as f: json.dump(team_map, f)
    final_metrics = history[-1] if history else {}
    artifact = {
        "model_path": model_path,
        "metrics": {
            "val_1x2_acc": val_acc,
            "val_accuracy": val_acc,
            **market_acc,
            "training_approach": approach,
            "training_source": source,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": wd,
            "selection_quality": best_quality,
            "v11_feature_columns": V11_SITUATION_FEATURE_COLUMNS,
            "timestamp": datetime.now().isoformat(),
        },
        "history": history,
        "final_epoch_metrics": final_metrics,
    }
    with open(os.path.join(V11_MODEL_DIR, f"{league}_hybrid_v11_draww_v5_metrics.json"), "w") as f: json.dump(artifact, f, indent=2)
    with open(os.path.join(V11_MODEL_DIR, f"{league}_training_artifact.json"), "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(
        f"    Final: train_loss={final_metrics.get('train_loss', 0):.4f} "
        f"val_loss={final_metrics.get('val_loss', 0):.4f} "
        f"draw_recall={market_acc.get('draw_recall', 0):.4f}"
    )
    return {'league': league, 'val_acc': val_acc, 'status': 'trained', 'history': history, 'model_path': model_path}

def train_xgb_for_league(league: str):
    """Fit and save XGBoost ensemble for one league from training_examples."""
    if not XGBOOST_AVAILABLE:
        return None
    if not os.path.exists(DB_PATH):
        return None
    try:
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT match_date AS Date, home_team AS HomeTeam, away_team AS AwayTeam,
                   actual_ftr AS FTR,
                   home_odds AS B365H, draw_odds AS B365D, away_odds AS B365A,
                   HY, AY, HR, AR, HC, AC, HS, "AS", HST, AST, HF, AF,
                   HPoss, APoss, HXG, AXG,
                   yc_home_avg5 AS h_yc_avg5, yc_away_avg5 AS a_yc_avg5,
                   corners_home_avg5 AS h_cw_avg5, corners_away_avg5 AS a_cw_avg5
            FROM training_examples
            WHERE league = ?
              AND actual_ftr IN ('H', 'D', 'A')
              AND home_odds IS NOT NULL
            ORDER BY match_date ASC
            """,
            con,
            params=(league,),
        )
        con.close()
    except Exception as exc:
        print(f"[XGB-Train] {league}: DB read failed — {exc}")
        return None

    if len(df) < 300:
        print(f"[XGB-Train] {league}: only {len(df)} rows, need 300 — skipping")
        return None

    label_map = {"H": 0, "D": 1, "A": 2}
    df = df[df["FTR"].isin(label_map)].copy()
    y = df["FTR"].map(label_map).values
    split = int(len(df) * 0.85)
    train_df, val_df = df.iloc[:split], df.iloc[split:]
    y_train, y_val = y[:split], y[split:]

    model_dir = os.path.join(project_root, "model", "v11")
    os.makedirs(model_dir, exist_ok=True)
    xgb_model = XGBoostEnsemble(model_dir=model_dir)
    try:
        xgb_model.fit(train_df, y_train)
    except Exception as exc:
        print(f"[XGB-Train] {league}: fit failed — {exc}")
        return None

    # Save with the naming convention expected by get_xgb_signals()
    base = os.path.join(model_dir, f"xgboost_{league}")
    xgb_model.save_model(base)

    # Quick validation report
    try:
        from sklearn.metrics import accuracy_score, log_loss as sk_logloss
        val_probs = xgb_model.predict_proba(val_df)
        val_preds = val_probs.argmax(axis=1)
        acc = accuracy_score(y_val, val_preds)
        ll = sk_logloss(y_val, val_probs)
        print(f"[XGB-Train] {league}: val_acc={acc:.4f}  log_loss={ll:.4f}  ({len(val_df)} val rows)")
        return {"league": league, "val_acc": acc, "log_loss": ll, "model_path": base}
    except Exception:
        print(f"[XGB-Train] {league}: saved (validation metrics unavailable)")
        return {"league": league, "model_path": base}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--league', default=None)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--skip-neural', action='store_true',
                        help='Skip V11 training, only train XGBoost + calibration')
    parser.add_argument('--skip-xgb', action='store_true',
                        help='Skip XGBoost training after V11')
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip isotonic calibration fitting after training')
    args = parser.parse_args()

    leagues = [args.league] if args.league else ['PremierLeague', 'Bundesliga', 'SerieA', 'LaLiga', 'Ligue1']

    # ── 1. V11 Hybrid Neural Training ────────────────────────────────────────
    results = []
    if not args.skip_neural:
        for l in leagues:
            res = run_hybrid_pipeline(l, epochs=args.epochs, dropout=args.dropout, lr=args.lr, wd=args.wd)
            if res:
                results.append(res)
        print("\n=== V11 FINAL REPORT ===")
        for r in results:
            print(f"  {r['league']}: Val Acc={r['val_acc']:.4f}")

    # ── 2. XGBoost Ensemble Training ─────────────────────────────────────────
    if not args.skip_xgb and XGBOOST_AVAILABLE:
        print("\n=== XGBoost Ensemble Training ===")
        for l in leagues:
            train_xgb_for_league(l)
    elif not XGBOOST_AVAILABLE:
        print("\n[XGB] xgboost not installed — skipping (pip install xgboost scikit-learn joblib)")

    # ── 3. Isotonic Calibration Fitting ──────────────────────────────────────
    if not args.skip_calibration:
        print("\n=== Isotonic Calibration Fitting ===")
        try:
            from calibration import fit_all_leagues
            fit_all_leagues(leagues=leagues)
        except ImportError:
            print("[Cal] calibration module not found")

if __name__ == "__main__":
    main()
