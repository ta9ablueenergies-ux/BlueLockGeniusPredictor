"""Torch-only graph outcome layer for football match prediction.

This module intentionally avoids torch-geometric so the runtime can use the
existing requirements. It builds a team interaction graph from completed match
history, trains a small graph convolution model, and exposes gated inference for
the main prediction pipeline.
"""

import argparse
import glob
import json
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "gnn")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "temp_extract")
PUBLIC_REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "gnn_training_report.json")
WALK_FORWARD_REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "gnn_walk_forward_report.json")

LEAGUES = ["PremierLeague", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]
LABELS = {"H": 0, "D": 1, "A": 2}
EPS = 1e-9

_GNN_CACHE = {}


def _parse_date(value):
    if value is None or value == "":
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    try:
        parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
        return None if pd.isna(parsed) else parsed.to_pydatetime()
    except Exception:
        return None


def _safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def _implied_1x2(home_odds, draw_odds, away_odds):
    inv = []
    for odd in (home_odds, draw_odds, away_odds):
        odd = _safe_float(odd, 0.0)
        inv.append(1.0 / odd if odd > 1.01 else 0.0)
    total = sum(inv)
    if total <= 0:
        return [0.0, 0.0, 0.0]
    return [x / total for x in inv]


def _normalize_team(value):
    return str(value or "").strip()


def _csv_history_paths(league):
    pattern = os.path.join(DATA_DIR, f"{league}_*.csv")
    return sorted(glob.glob(pattern))


def _first_series(df, names, default=0.0):
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def load_training_examples_history(league=None):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='training_examples'"
        ).fetchone()
        if not table_exists:
            conn.close()
            return pd.DataFrame()
        sql = """
            SELECT league, match_date AS date, home_team, away_team,
                   home_goals AS fthg, away_goals AS ftag, actual_ftr AS ftr,
                   home_odds, draw_odds, away_odds,
                   market_prob_home, market_prob_draw, market_prob_away,
                   source, source_confidence
            FROM training_examples
            WHERE actual_ftr IN ('H', 'D', 'A')
        """
        params = None
        if league:
            sql += " AND league = ?"
            params = (league,)
        sql += " ORDER BY match_date ASC, id ASC"
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_completed_history(league=None):
    """Load completed match history from local CSVs and closed SQLite results."""
    training_examples = load_training_examples_history(league)
    if not training_examples.empty:
        training_examples["parsed_date"] = training_examples["date"].map(_parse_date)
        return training_examples.sort_values(["parsed_date", "date"], na_position="first").reset_index(drop=True)

    frames = []
    leagues = [league] if league else LEAGUES

    for lg in leagues:
        for path in _csv_history_paths(lg):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            needed = {"HomeTeam", "AwayTeam", "FTR"}
            if not needed.issubset(set(df.columns)):
                continue
            out = pd.DataFrame()
            out["league"] = lg
            out["date"] = df.get("Date")
            out["home_team"] = df["HomeTeam"].map(_normalize_team)
            out["away_team"] = df["AwayTeam"].map(_normalize_team)
            out["fthg"] = df.get("FTHG", 0).map(lambda x: int(_safe_float(x, 0)))
            out["ftag"] = df.get("FTAG", 0).map(lambda x: int(_safe_float(x, 0)))
            out["ftr"] = df["FTR"].astype(str).str.upper().str[0]
            out["home_odds"] = _first_series(df, ["h_course", "B365H", "AvgH", "PSH"]).map(_safe_float)
            out["draw_odds"] = _first_series(df, ["d_course", "B365D", "AvgD", "PSD"]).map(_safe_float)
            out["away_odds"] = _first_series(df, ["a_course", "B365A", "AvgA", "PSA"]).map(_safe_float)
            frames.append(out)

    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            sql = """
                SELECT m.league, m.match_date AS date, m.home_team, m.away_team,
                       r.actual_fthg AS fthg, r.actual_ftag AS ftag, r.actual_ftr AS ftr,
                       r.closing_home_odds AS home_odds,
                       r.closing_draw_odds AS draw_odds,
                       r.closing_away_odds AS away_odds
                FROM matches m
                JOIN match_results r ON r.id = m.id
                WHERE r.actual_ftr IN ('H', 'D', 'A')
            """
            if league:
                sql += " AND m.league = ?"
                db_df = pd.read_sql_query(sql, conn, params=(league,))
            else:
                db_df = pd.read_sql_query(sql, conn)
            conn.close()
            if not db_df.empty:
                frames.append(db_df)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    history = pd.concat(frames, ignore_index=True)
    history["home_team"] = history["home_team"].map(_normalize_team)
    history["away_team"] = history["away_team"].map(_normalize_team)
    history = history[(history["home_team"] != "") & (history["away_team"] != "")]
    history = history[history["ftr"].isin(LABELS.keys())].copy()
    history["parsed_date"] = history["date"].map(_parse_date)
    history = history.sort_values(["parsed_date", "date"], na_position="first").reset_index(drop=True)
    return history


def _team_stats(history, team_to_idx):
    n = len(team_to_idx)
    stats = np.zeros((n, 8), dtype=np.float32)

    for _, row in history.iterrows():
        h = team_to_idx[row["home_team"]]
        a = team_to_idx[row["away_team"]]
        hg = _safe_float(row.get("fthg"), 0.0)
        ag = _safe_float(row.get("ftag"), 0.0)
        ftr = row.get("ftr")

        stats[h, 0] += 1.0
        stats[a, 0] += 1.0
        stats[h, 1] += hg
        stats[h, 2] += ag
        stats[a, 1] += ag
        stats[a, 2] += hg
        stats[h, 3] += 3.0 if ftr == "H" else (1.0 if ftr == "D" else 0.0)
        stats[a, 3] += 3.0 if ftr == "A" else (1.0 if ftr == "D" else 0.0)
        stats[h, 4] += 1.0 if ftr == "H" else 0.0
        stats[a, 5] += 1.0 if ftr == "A" else 0.0
        stats[h, 6] += 1.0 if ftr == "D" else 0.0
        stats[a, 6] += 1.0 if ftr == "D" else 0.0
        stats[h, 7] += hg - ag
        stats[a, 7] += ag - hg

    games = np.maximum(stats[:, [0]], 1.0)
    features = np.concatenate(
        [
            np.log1p(stats[:, [0]]) / 4.0,
            stats[:, [1]] / games,
            stats[:, [2]] / games,
            stats[:, [3]] / (games * 3.0),
            stats[:, [4]] / games,
            stats[:, [5]] / games,
            stats[:, [6]] / games,
            stats[:, [7]] / games,
        ],
        axis=1,
    )
    return np.clip(features, -3.0, 3.0).astype(np.float32)


def _adjacency(history, team_to_idx):
    n = len(team_to_idx)
    adj = np.eye(n, dtype=np.float32)
    for _, row in history.iterrows():
        h = team_to_idx[row["home_team"]]
        a = team_to_idx[row["away_team"]]
        margin = abs(_safe_float(row.get("fthg"), 0.0) - _safe_float(row.get("ftag"), 0.0))
        w = 1.0 + min(margin, 4.0) * 0.15
        adj[h, a] += w
        adj[a, h] += w
    degree = np.maximum(adj.sum(axis=1, keepdims=True), 1.0)
    return (adj / degree).astype(np.float32)


def _graph_tensors_from_history(history, teams):
    team_to_idx = {team: i for i, team in enumerate(teams)}
    return _team_stats(history, team_to_idx), _adjacency(history, team_to_idx)


def build_graph_dataset(league):
    history = load_completed_history(league)
    if history.empty:
        return None

    teams = sorted(set(history["home_team"]).union(set(history["away_team"])))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    labels = history["ftr"].map(LABELS).astype(int).to_numpy()
    match_features = []
    match_index = []
    for _, row in history.iterrows():
        if all(pd.notna(row.get(k)) for k in ("market_prob_home", "market_prob_draw", "market_prob_away")):
            implied = [
                _safe_float(row.get("market_prob_home"), 0.0),
                _safe_float(row.get("market_prob_draw"), 0.0),
                _safe_float(row.get("market_prob_away"), 0.0),
            ]
        else:
            implied = _implied_1x2(row.get("home_odds"), row.get("draw_odds"), row.get("away_odds"))
        match_features.append(implied + [1.0 if max(implied) > 0 else 0.0])
        match_index.append([team_to_idx[row["home_team"]], team_to_idx[row["away_team"]]])

    return {
        "league": league,
        "teams": teams,
        "team_to_idx": team_to_idx,
        "history": history,
        "node_features": _team_stats(history, team_to_idx),
        "adjacency": _adjacency(history, team_to_idx),
        "match_index": np.array(match_index, dtype=np.int64),
        "match_features": np.array(match_features, dtype=np.float32),
        "labels": labels,
    }


class DenseGATLayer(nn.Module):
    """Dense Graph Attention Layer (GAT) for dynamic rivalry modeling"""
    def __init__(self, in_features, out_features, dropout=0.15, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout

    def forward(self, h, adj):
        Wh = self.W(h) # [N, out_features]
        N = Wh.size(0)
        
        Wh_i = Wh.unsqueeze(1).expand(N, N, -1)
        Wh_j = Wh.unsqueeze(0).expand(N, N, -1)
        cat_Wh = torch.cat([Wh_i, Wh_j], dim=-1) # [N, N, 2*out_features]
        
        e = self.leakyrelu(torch.matmul(cat_Wh, self.a).squeeze(-1)) # [N, N]
        
        zero_vec = -9e15 * torch.ones_like(e)
        # Mask non-existent edges, apply softmax
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)


class TeamGraphOutcomeModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=32, match_feature_dim=4):
        super().__init__()
        self.node_in = nn.Linear(node_feature_dim, hidden_dim)
        
        # Upgraded to Deep GAT Architecture
        self.gat1 = DenseGATLayer(hidden_dim, hidden_dim, dropout=0.15)
        self.gat2 = DenseGATLayer(hidden_dim, hidden_dim, dropout=0.15)
        
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + match_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 3),
        )

    def encode_nodes(self, node_features, adjacency):
        x = F.relu(self.node_in(node_features))
        
        # Multi-layer Graph Attention
        x = self.gat1(x, adjacency)
        x = self.gat2(x, adjacency)
        
        return x

    def forward(self, node_features, adjacency, match_index, match_features):
        emb = self.encode_nodes(node_features, adjacency)
        h = emb[match_index[:, 0]]
        a = emb[match_index[:, 1]]
        pair = torch.cat([h, a, h - a, h * a, match_features], dim=1)
        return self.pair_mlp(pair)


def _chronological_split(n, val_fraction=0.2):
    split = max(1, min(n - 1, int(n * (1.0 - val_fraction))))
    train_idx = np.arange(0, split, dtype=np.int64)
    val_idx = np.arange(split, n, dtype=np.int64)
    return train_idx, val_idx


def _log_loss_from_probs(probs, labels):
    selected = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(selected, EPS, 1.0)))) if len(labels) else 0.0


def _brier_from_probs(probs, labels):
    if len(labels) == 0:
        return 0.0
    target = np.zeros_like(probs)
    target[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def _fit_temperature(logits, labels):
    """Grid-search a validation temperature without adding sklearn dependency."""
    labels_np = labels.detach().cpu().numpy()
    best_t = 1.0
    best_loss = float("inf")
    for t in np.linspace(0.7, 3.0, 47):
        probs = torch.softmax(logits / float(t), dim=1).detach().cpu().numpy()
        loss = _log_loss_from_probs(probs, labels_np)
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t, best_loss


def _best_market_graph_blend(market_probs, graph_probs, labels):
    if len(labels) == 0 or len(market_probs) == 0:
        return {"weight": 0.0, "accuracy": 0.0, "log_loss": 0.0, "brier": 0.0}
    best = None
    for weight in np.linspace(0.0, 0.35, 36):
        probs = (market_probs * (1.0 - weight)) + (graph_probs * weight)
        probs = np.clip(probs, EPS, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        acc = float((probs.argmax(axis=1) == labels).mean())
        log_loss = _log_loss_from_probs(probs, labels)
        brier = _brier_from_probs(probs, labels)
        candidate = {
            "weight": float(weight),
            "accuracy": acc,
            "log_loss": log_loss,
            "brier": brier,
        }
        if best is None or (candidate["log_loss"], -candidate["accuracy"]) < (best["log_loss"], -best["accuracy"]):
            best = candidate
    return best


def train_gnn_league(league, epochs=160, learning_rate=0.01, min_matches=120):
    dataset = build_graph_dataset(league)
    if dataset is None or len(dataset["labels"]) < min_matches:
        return {
            "league": league,
            "enabled": False,
            "reason": "insufficient_history",
            "matches": 0 if dataset is None else int(len(dataset["labels"])),
        }

    torch.manual_seed(42)
    n = len(dataset["labels"])
    train_idx, val_idx = _chronological_split(n)

    train_history = dataset["history"].iloc[train_idx].copy()
    train_node_features, train_adjacency = _graph_tensors_from_history(train_history, dataset["teams"])
    full_node_features, full_adjacency = _graph_tensors_from_history(dataset["history"], dataset["teams"])

    node_features = torch.tensor(train_node_features, dtype=torch.float32)
    adjacency = torch.tensor(train_adjacency, dtype=torch.float32)
    match_index = torch.tensor(dataset["match_index"], dtype=torch.long)
    match_features = torch.tensor(dataset["match_features"], dtype=torch.float32)
    labels = torch.tensor(dataset["labels"], dtype=torch.long)

    model = TeamGraphOutcomeModel(
        node_feature_dim=node_features.shape[1],
        match_feature_dim=match_features.shape[1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(node_features, adjacency, match_index[train_idx], match_features[train_idx])
        loss = F.cross_entropy(logits, labels[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(node_features, adjacency, match_index[val_idx], match_features[val_idx])
        temperature, val_log_loss = _fit_temperature(val_logits, labels[val_idx])
        val_probs = torch.softmax(val_logits / temperature, dim=1)
        val_pred = val_probs.argmax(dim=1)
        val_acc = float((val_pred == labels[val_idx]).float().mean().item())
        val_probs_np = val_probs.detach().cpu().numpy()

    val_implied = dataset["match_features"][val_idx, :3]
    has_market = val_implied.sum(axis=1) > 0
    market_coverage = float(has_market.mean()) if len(has_market) else 0.0
    val_brier = _brier_from_probs(val_probs_np, dataset["labels"][val_idx])
    blend_metrics = {"weight": 0.0, "accuracy": val_acc, "log_loss": val_log_loss, "brier": val_brier}
    if has_market.any():
        market_pred = val_implied[has_market].argmax(axis=1)
        labels_market = dataset["labels"][val_idx][has_market]
        market_acc = float((market_pred == labels_market).mean())
        market_log_loss = _log_loss_from_probs(val_implied[has_market], labels_market)
        market_brier = _brier_from_probs(val_implied[has_market], labels_market)
        blend_metrics = _best_market_graph_blend(val_implied[has_market], val_probs_np[has_market], labels_market)
    else:
        market_acc = 0.0
        market_log_loss = 0.0
        market_brier = 0.0

    class_counts = np.bincount(dataset["labels"][train_idx], minlength=3)
    majority_acc = float(class_counts.max() / max(class_counts.sum(), 1))
    baseline_acc = max(market_acc, majority_acc)
    lift = val_acc - baseline_acc
    blend_acc_lift = blend_metrics["accuracy"] - baseline_acc
    blend_logloss_lift = market_log_loss - blend_metrics["log_loss"] if has_market.any() else 0.0
    min_lift = _safe_float(os.environ.get("GNN_MIN_ENABLE_LIFT"), 0.02)
    min_logloss_lift = _safe_float(os.environ.get("GNN_MIN_LOGLOSS_LIFT"), 0.002)
    enabled = bool(
        len(val_idx) >= 30
        and market_coverage >= 0.8
        and blend_metrics["weight"] > 0
        and (blend_acc_lift >= min_lift or blend_logloss_lift >= min_logloss_lift)
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, f"{league}_team_gnn.pt")
    meta_path = os.path.join(MODEL_DIR, f"{league}_team_gnn_meta.json")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "teams": dataset["teams"],
            "node_features": torch.tensor(full_node_features, dtype=torch.float32),
            "adjacency": torch.tensor(full_adjacency, dtype=torch.float32),
            "node_feature_dim": int(node_features.shape[1]),
            "match_feature_dim": int(match_features.shape[1]),
            "temperature": float(temperature),
            "recommended_blend_weight": float(blend_metrics["weight"]),
        },
        checkpoint_path,
    )
    meta = {
        "league": league,
        "enabled": enabled,
        "matches": int(n),
        "teams": int(len(dataset["teams"])),
        "validation_matches": int(len(val_idx)),
        "validation_accuracy": round(val_acc, 4),
        "validation_log_loss": round(val_log_loss, 4),
        "validation_brier": round(val_brier, 4),
        "market_accuracy": round(market_acc, 4),
        "market_coverage": round(market_coverage, 4),
        "market_log_loss": round(market_log_loss, 4),
        "market_brier": round(market_brier, 4),
        "majority_accuracy": round(majority_acc, 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "lift": round(lift, 4),
        "blend_weight": round(blend_metrics["weight"], 4),
        "blend_accuracy": round(blend_metrics["accuracy"], 4),
        "blend_log_loss": round(blend_metrics["log_loss"], 4),
        "blend_brier": round(blend_metrics["brier"], 4),
        "blend_accuracy_lift": round(blend_acc_lift, 4),
        "blend_logloss_lift": round(blend_logloss_lift, 4),
        "min_enable_lift": round(min_lift, 4),
        "min_logloss_lift": round(min_logloss_lift, 4),
        "temperature": round(float(temperature), 4),
        "checkpoint": checkpoint_path,
        "trained_at": datetime.utcnow().isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_gnn_bundle(league):
    if league in _GNN_CACHE:
        return _GNN_CACHE[league]

    meta_path = os.path.join(MODEL_DIR, f"{league}_team_gnn_meta.json")
    checkpoint_path = os.path.join(MODEL_DIR, f"{league}_team_gnn.pt")
    if not os.path.exists(meta_path) or not os.path.exists(checkpoint_path):
        _GNN_CACHE[league] = None
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not meta.get("enabled"):
            _GNN_CACHE[league] = None
            return None
        if os.path.exists(WALK_FORWARD_REPORT_PATH):
            with open(WALK_FORWARD_REPORT_PATH, "r", encoding="utf-8") as f:
                wf_report = json.load(f)
            wf_row = next((r for r in wf_report.get("results", []) if r.get("league") == league), None)
            if wf_row and not wf_row.get("enabled"):
                _GNN_CACHE[league] = None
                return None
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = TeamGraphOutcomeModel(
            node_feature_dim=int(checkpoint["node_feature_dim"]),
            match_feature_dim=int(checkpoint["match_feature_dim"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        bundle = {
            "meta": meta,
            "model": model,
            "teams": checkpoint["teams"],
            "team_to_idx": {team: i for i, team in enumerate(checkpoint["teams"])},
            "node_features": checkpoint["node_features"].detach().clone().float(),
            "adjacency": checkpoint["adjacency"].detach().clone().float(),
            "temperature": float(checkpoint.get("temperature", meta.get("temperature", 1.0)) or 1.0),
            "recommended_blend_weight": float(checkpoint.get("recommended_blend_weight", meta.get("blend_weight", 0.0)) or 0.0),
        }
        _GNN_CACHE[league] = bundle
        return bundle
    except Exception as exc:
        print(f"[GNN] Load failed for {league}: {exc}")
        _GNN_CACHE[league] = None
        return None


def get_gnn_match_signals(league, home_team, away_team, match=None):
    """Return gated GNN probabilities for a live fixture, or None when inactive."""
    bundle = load_gnn_bundle(league)
    if not bundle:
        return None

    team_to_idx = bundle["team_to_idx"]
    if home_team not in team_to_idx or away_team not in team_to_idx:
        return None

    match = match or {}
    implied = _implied_1x2(match.get("h_course"), match.get("d_course"), match.get("a_course"))
    match_features = torch.tensor([implied + [1.0 if max(implied) > 0 else 0.0]], dtype=torch.float32)
    match_index = torch.tensor([[team_to_idx[home_team], team_to_idx[away_team]]], dtype=torch.long)

    with torch.no_grad():
        logits = bundle["model"](
            bundle["node_features"],
            bundle["adjacency"],
            match_index,
            match_features,
        )
        probs = torch.softmax(logits / bundle["temperature"], dim=1).numpy()[0]

    meta = bundle["meta"]
    lift = max(0.0, float(meta.get("lift", 0.0) or 0.0))
    confidence = min(1.0, 0.5 + lift * 4.0)
    return {
        "gnn_p_h": float(probs[0]),
        "gnn_p_d": float(probs[1]),
        "gnn_p_a": float(probs[2]),
        "gnn_confidence": confidence,
        "gnn_lift": float(meta.get("lift", 0.0) or 0.0),
        "gnn_blend_weight": float(bundle.get("recommended_blend_weight", 0.0) or 0.0),
        "gnn_blend_accuracy_lift": float(meta.get("blend_accuracy_lift", 0.0) or 0.0),
        "gnn_blend_logloss_lift": float(meta.get("blend_logloss_lift", 0.0) or 0.0),
        "gnn_validation_accuracy": float(meta.get("validation_accuracy", 0.0) or 0.0),
        "gnn_active": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Train or inspect the football GNN layer.")
    parser.add_argument("--league", default="ALL")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--min-matches", type=int, default=120)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    leagues = LEAGUES if args.league == "ALL" else [args.league]
    results = []
    for league in leagues:
        if args.build_only:
            dataset = build_graph_dataset(league)
            results.append({
                "league": league,
                "matches": 0 if dataset is None else int(len(dataset["labels"])),
                "teams": 0 if dataset is None else int(len(dataset["teams"])),
            })
        else:
            results.append(train_gnn_league(league, epochs=args.epochs, min_matches=args.min_matches))

    os.makedirs(MODEL_DIR, exist_ok=True)
    report_path = os.path.join(MODEL_DIR, "gnn_training_report.json")
    report = {"created_at": datetime.utcnow().isoformat(), "results": results}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    os.makedirs(os.path.dirname(PUBLIC_REPORT_PATH), exist_ok=True)
    with open(PUBLIC_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"report": report_path, "results": results}, indent=2))


if __name__ == "__main__":
    main()
