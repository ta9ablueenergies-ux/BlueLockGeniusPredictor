"""Probability-quality report for closed predictions.

Accuracy tells us the picked label. This report measures whether the probability
distribution was honest: log loss, Brier score, ECE, draw recall, and market
baseline quality from devigged odds where available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")
REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "probability_quality_report.json")


LABELS = {"H": 0, "D": 1, "A": 2}


def _clip_probs(probs: Iterable[float]) -> np.ndarray:
    arr = np.array([float(x or 0.0) for x in probs], dtype=float)
    arr = np.clip(arr, 1e-6, 1.0)
    total = float(arr.sum())
    if total <= 0:
        arr = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    else:
        arr = arr / total
    return arr


def _derive_model_probs(row: sqlite3.Row) -> Optional[np.ndarray]:
    try:
        p_1x = float(row["p_1x"])
        p_x2 = float(row["p_x2"])
        dnb = float(row["p_dnb"]) if row["p_dnb"] is not None else None
    except Exception:
        return None
    p_d = p_1x + p_x2 - 1.0
    p_h = p_1x - p_d
    p_a = p_x2 - p_d
    if dnb is not None and (p_h <= 0 or p_a <= 0):
        nondraw = max(0.01, 1.0 - max(0.0, p_d))
        p_h = nondraw * dnb
        p_a = nondraw * (1.0 - dnb)
    return _clip_probs([p_h, p_d, p_a])


def _market_probs(row: sqlite3.Row) -> Optional[np.ndarray]:
    odds = [row["home_odds"], row["draw_odds"], row["away_odds"]]
    inv = []
    for odd in odds:
        try:
            odd = float(odd)
            inv.append(1.0 / odd if odd > 1.01 else 0.0)
        except Exception:
            inv.append(0.0)
    if sum(inv) <= 0:
        return None
    return _clip_probs(inv)


def _ece(prob_vectors: List[np.ndarray], y_true: List[int], bins: int = 10) -> float:
    if not prob_vectors:
        return 0.0
    conf = np.array([float(p.max()) for p in prob_vectors])
    pred = np.array([int(p.argmax()) for p in prob_vectors])
    y = np.array(y_true)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.any():
            acc = float((pred[mask] == y[mask]).mean())
            avg_conf = float(conf[mask].mean())
            ece += float(mask.mean()) * abs(avg_conf - acc)
    return round(ece, 5)


def _metrics(prob_vectors: List[np.ndarray], y_true: List[int]) -> Dict[str, float]:
    if not prob_vectors:
        return {"n": 0, "accuracy": 0.0, "log_loss": 0.0, "brier": 0.0, "ece": 0.0, "draw_recall": 0.0}
    probs = np.vstack(prob_vectors)
    y = np.array(y_true, dtype=int)
    picked = probs.argmax(axis=1)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y)), y] = 1.0
    log_loss = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1.0))))
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    draw_mask = y == LABELS["D"]
    draw_recall = float(((picked == LABELS["D"]) & draw_mask).sum() / max(1, draw_mask.sum()))
    return {
        "n": int(len(y)),
        "accuracy": round(float((picked == y).mean()), 5),
        "log_loss": round(log_loss, 5),
        "brier": round(brier, 5),
        "ece": _ece(prob_vectors, y_true),
        "draw_recall": round(draw_recall, 5),
    }


def build_probability_quality_report(league: Optional[str] = None, limit: int = 2000) -> Dict[str, object]:
    if not os.path.exists(DB_PATH):
        return {"created_at": datetime.utcnow().isoformat(), "error": "database_missing"}
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    params: List[object] = []
    league_clause = ""
    if league:
        league_clause = "AND m.league = ?"
        params.append(league)
    params.append(limit)
    rows = con.execute(
        f"""
        SELECT m.league, m.match_date, m.home_team, m.away_team,
               m.p_1x, m.p_x2, m.p_dnb,
               r.actual_ftr,
               te.home_odds, te.draw_odds, te.away_odds
        FROM matches m
        JOIN match_results r ON r.id = m.id
        LEFT JOIN training_examples te ON te.id = m.id
        WHERE m.is_mock = 0
          AND r.actual_ftr IN ('H', 'D', 'A')
          {league_clause}
        ORDER BY m.match_date DESC, m.last_updated DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    con.close()

    model_by_league: Dict[str, List[np.ndarray]] = defaultdict(list)
    market_by_league: Dict[str, List[np.ndarray]] = defaultdict(list)
    y_by_league: Dict[str, List[int]] = defaultdict(list)
    y_market_by_league: Dict[str, List[int]] = defaultdict(list)
    all_model: List[np.ndarray] = []
    all_market: List[np.ndarray] = []
    all_y: List[int] = []
    all_y_market: List[int] = []

    for row in rows:
        label = LABELS.get(row["actual_ftr"])
        if label is None:
            continue
        probs = _derive_model_probs(row)
        if probs is not None:
            model_by_league[row["league"]].append(probs)
            y_by_league[row["league"]].append(label)
            all_model.append(probs)
            all_y.append(label)
        m_probs = _market_probs(row)
        if m_probs is not None:
            market_by_league[row["league"]].append(m_probs)
            y_market_by_league[row["league"]].append(label)
            all_market.append(m_probs)
            all_y_market.append(label)

    leagues = {}
    for lg in sorted(set(list(model_by_league.keys()) + list(market_by_league.keys()))):
        leagues[lg] = {
            "model": _metrics(model_by_league.get(lg, []), y_by_league.get(lg, [])),
            "market_baseline": _metrics(market_by_league.get(lg, []), y_market_by_league.get(lg, [])),
        }

    report = {
        "created_at": datetime.utcnow().isoformat(),
        "league_filter": league,
        "rows_scanned": len(rows),
        "overall": {
            "model": _metrics(all_model, all_y),
            "market_baseline": _metrics(all_market, all_y_market),
        },
        "by_league": leagues,
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default=None)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(build_probability_quality_report(args.league, args.limit), indent=2))


if __name__ == "__main__":
    main()
