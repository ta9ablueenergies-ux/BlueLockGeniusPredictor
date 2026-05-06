"""Probabilistic goal, corner, and card market models.

The V11 classifier remains the main neural layer. This module adds a separate
count-model layer for markets that are naturally counts: goals, corners, and
cards. It trains from finalized `training_examples`, writes calibration/eval
reports, and can enrich pending fixtures with expected counts and line odds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from league_market_profiles import get_league_profile, write_profiles_report
from training_data_guard import CANONICAL_VIEW_NAME, ensure_training_data_guard

DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "market_counts")
REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "market_count_model_report.json")
_MODEL_CACHE: Dict[str, Dict[str, object]] = {}
_TRAINING_SOURCE_TABLE: Optional[str] = None

CORE_LINES = {
    "goals": [1.5, 2.5, 3.5],
    "corners": [8.5, 9.5, 10.5],
    "cards": [3.5, 4.5, 5.5],
}


def _training_examples_source_table() -> str:
    global _TRAINING_SOURCE_TABLE
    if _TRAINING_SOURCE_TABLE:
        return _TRAINING_SOURCE_TABLE
    try:
        report = ensure_training_data_guard(write_report=True)
        if report.get("status") == "ready":
            _TRAINING_SOURCE_TABLE = CANONICAL_VIEW_NAME
            return _TRAINING_SOURCE_TABLE
    except Exception as exc:
        print(f"[market_count_models] training data guard unavailable, using raw table: {exc}")
    _TRAINING_SOURCE_TABLE = "training_examples"
    return _TRAINING_SOURCE_TABLE


def _as_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _mean(values: Iterable[object], default: float) -> float:
    vals = [_as_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float(default)


def _fit_dispersion(total_counts: Iterable[object]) -> Optional[float]:
    vals = [_as_float(v) for v in total_counts]
    vals = np.array([v for v in vals if v is not None and v >= 0], dtype=float)
    if len(vals) < 30:
        return None
    mu = float(vals.mean())
    var = float(vals.var(ddof=1)) if len(vals) > 1 else mu
    if mu <= 0 or var <= mu:
        return None
    r = (mu * mu) / max(1e-6, var - mu)
    return float(_clip(r, 1.2, 80.0))


def _over_prob_count(mu: float, line: float, dispersion: Optional[float]) -> float:
    mu = max(0.01, float(mu))
    threshold = int(math.floor(line))
    if dispersion is None:
        return float(poisson.sf(threshold, mu))
    r = max(0.1, float(dispersion))
    p = r / (r + mu)
    return float(nbinom.sf(threshold, r, p))


def _score_distribution(home_mu: float, away_mu: float, max_goals: int = 10) -> Dict[str, float]:
    home_mu = max(0.05, float(home_mu))
    away_mu = max(0.05, float(away_mu))
    p_home = p_draw = p_away = p_btts = 0.0
    over = {line: 0.0 for line in CORE_LINES["goals"]}
    for hg in range(max_goals + 1):
        ph = float(poisson.pmf(hg, home_mu))
        for ag in range(max_goals + 1):
            p = ph * float(poisson.pmf(ag, away_mu))
            if hg > ag:
                p_home += p
            elif hg == ag:
                p_draw += p
            else:
                p_away += p
            if hg > 0 and ag > 0:
                p_btts += p
            for line in CORE_LINES["goals"]:
                if hg + ag > line:
                    over[line] += p
    total = max(1e-9, p_home + p_draw + p_away)
    return {
        "p_home": p_home / total,
        "p_draw": p_draw / total,
        "p_away": p_away / total,
        "p_btts": _clip(p_btts, 0.0, 1.0),
        "p_over15": _clip(over[1.5], 0.0, 1.0),
        "p_over25": _clip(over[2.5], 0.0, 1.0),
        "p_over35": _clip(over[3.5], 0.0, 1.0),
    }


def _metric_line(probs: List[float], actuals: List[int]) -> Dict[str, float]:
    if not probs:
        return {"n": 0, "brier": 0.0, "log_loss": 0.0, "ece": 0.0}
    p = np.clip(np.array(probs, dtype=float), 1e-6, 1 - 1e-6)
    y = np.array(actuals, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {"n": int(len(probs)), "brier": round(brier, 5), "log_loss": round(log_loss, 5), "ece": round(ece, 5)}


def _count_metrics(preds: List[float], actuals: List[float]) -> Dict[str, float]:
    if not preds:
        return {"n": 0, "mae": 0.0, "rmse": 0.0}
    p = np.array(preds, dtype=float)
    y = np.array(actuals, dtype=float)
    return {
        "n": int(len(preds)),
        "mae": round(float(np.mean(np.abs(p - y))), 4),
        "rmse": round(float(np.sqrt(np.mean((p - y) ** 2))), 4),
    }


def _load_examples(league: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    source_table = _training_examples_source_table()
    con = sqlite3.connect(DB_PATH)
    where = "WHERE actual_ftr IN ('H', 'D', 'A') AND home_goals IS NOT NULL AND away_goals IS NOT NULL"
    params: Tuple[object, ...] = ()
    if league:
        where += " AND league = ?"
        params = (league,)
    df = pd.read_sql_query(
        f"""
        SELECT id, league, match_date, home_team, away_team,
               home_goals, away_goals, actual_ftr,
               HY, AY, HR, AR, HC, AC, source_confidence, source
        FROM {source_table}
        {where}
        ORDER BY league ASC, match_date ASC, source_confidence ASC, id ASC
        """,
        con,
        params=params,
    )
    con.close()
    if df.empty:
        return df
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    numeric_cols = ["home_goals", "away_goals", "HY", "AY", "HR", "AR", "HC", "AC", "source_confidence"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["league", "match_date", "home_team", "away_team"]).copy()
    df = df.sort_values(["league", "match_date", "home_team", "away_team", "source_confidence"], na_position="first")
    df = df.drop_duplicates(subset=["league", "match_date", "home_team", "away_team"], keep="last")
    return df.reset_index(drop=True)


def _side_stats(df: pd.DataFrame, team: str, side: str, field: str, against: bool, default: float) -> float:
    if side == "home":
        sub = df[df["home_team"] == team]
        col = field[0] if not against else field[1]
    else:
        sub = df[df["away_team"] == team]
        col = field[1] if not against else field[0]
    if sub.empty or col not in sub.columns:
        return default
    return _mean(sub[col].tolist(), default)


def _build_team_stats(train: pd.DataFrame, league_means: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    teams = sorted(set(train["home_team"].dropna().tolist() + train["away_team"].dropna().tolist()))
    stats: Dict[str, Dict[str, float]] = {}
    fields = {
        "goals": ("home_goals", "away_goals"),
        "corners": ("HC", "AC"),
        "cards": ("HY", "AY"),
    }
    for team in teams:
        row: Dict[str, float] = {}
        for market, cols in fields.items():
            h_default = league_means[f"home_{market}"]
            a_default = league_means[f"away_{market}"]
            row[f"home_{market}_for"] = _side_stats(train, team, "home", cols, False, h_default)
            row[f"home_{market}_against"] = _side_stats(train, team, "home", cols, True, a_default)
            row[f"away_{market}_for"] = _side_stats(train, team, "away", cols, False, a_default)
            row[f"away_{market}_against"] = _side_stats(train, team, "away", cols, True, h_default)
        row["home_count"] = int((train["home_team"] == team).sum())
        row["away_count"] = int((train["away_team"] == team).sum())
        stats[team] = row
    return stats


def _fit_model(league: str, train: pd.DataFrame) -> Dict[str, object]:
    profile = get_league_profile(league)
    league_means = {
        "home_goals": _mean(train["home_goals"].tolist(), 1.35),
        "away_goals": _mean(train["away_goals"].tolist(), 1.15),
        "home_corners": _mean(train["HC"].tolist(), float(profile["corner_total_prior"]) / 2.0),
        "away_corners": _mean(train["AC"].tolist(), float(profile["corner_total_prior"]) / 2.0),
        "home_cards": _mean(train["HY"].tolist(), float(profile["card_total_prior"]) / 2.0),
        "away_cards": _mean(train["AY"].tolist(), float(profile["card_total_prior"]) / 2.0),
    }
    model = {
        "league": league,
        "created_at": datetime.utcnow().isoformat(),
        "n_train": int(len(train)),
        "profile": profile,
        "league_means": league_means,
        "dispersion": {
            "goals": _fit_dispersion((train["home_goals"] + train["away_goals"]).tolist()),
            "corners": _fit_dispersion((train["HC"] + train["AC"]).dropna().tolist()),
            "cards": _fit_dispersion((train["HY"] + train["AY"]).dropna().tolist()),
        },
        "team_stats": _build_team_stats(train, league_means),
    }
    return model


def _get_team(model: Dict[str, object], team: str) -> Dict[str, float]:
    return dict(model.get("team_stats", {}).get(team, {}))


def _adjust_total_with_profile(home_mu: float, away_mu: float, profile: Dict[str, object], key: str, strength: float) -> Tuple[float, float]:
    total = max(0.01, home_mu + away_mu)
    target = float(profile.get(key, total) or total)
    factor = _clip(1.0 + strength * ((target / total) - 1.0), 0.85, 1.15)
    return home_mu * factor, away_mu * factor


def _fallback_model(league: str) -> Dict[str, object]:
    profile = get_league_profile(league)
    half_goals = float(profile["goal_total_prior"]) / 2.0
    half_corners = float(profile["corner_total_prior"]) / 2.0
    half_cards = float(profile["card_total_prior"]) / 2.0
    return {
        "league": league,
        "profile": profile,
        "league_means": {
            "home_goals": half_goals,
            "away_goals": half_goals,
            "home_corners": half_corners,
            "away_corners": half_corners,
            "home_cards": half_cards,
            "away_cards": half_cards,
        },
        "dispersion": {"goals": None, "corners": None, "cards": None},
        "team_stats": {},
    }


def _predict_from_model_data(model: Dict[str, object], home_team: str, away_team: str) -> Dict[str, object]:
    league = str(model.get("league") or "default")
    means = model["league_means"]
    profile = model.get("profile") or get_league_profile(league)
    h = _get_team(model, home_team)
    a = _get_team(model, away_team)

    def team_val(stats: Dict[str, float], key: str, fallback: float) -> float:
        return float(stats.get(key, fallback) if stats.get(key) is not None else fallback)

    hg = (
        0.50 * team_val(h, "home_goals_for", means["home_goals"])
        + 0.30 * team_val(a, "away_goals_against", means["home_goals"])
        + 0.20 * means["home_goals"]
    )
    ag = (
        0.50 * team_val(a, "away_goals_for", means["away_goals"])
        + 0.30 * team_val(h, "home_goals_against", means["away_goals"])
        + 0.20 * means["away_goals"]
    )
    hc = (
        0.52 * team_val(h, "home_corners_for", means["home_corners"])
        + 0.28 * team_val(a, "away_corners_against", means["home_corners"])
        + 0.20 * means["home_corners"]
    )
    ac = (
        0.52 * team_val(a, "away_corners_for", means["away_corners"])
        + 0.28 * team_val(h, "home_corners_against", means["away_corners"])
        + 0.20 * means["away_corners"]
    )
    hy = (
        0.45 * team_val(h, "home_cards_for", means["home_cards"])
        + 0.35 * team_val(a, "away_cards_against", means["home_cards"])
        + 0.20 * means["home_cards"]
    )
    ay = (
        0.45 * team_val(a, "away_cards_for", means["away_cards"])
        + 0.35 * team_val(h, "home_cards_against", means["away_cards"])
        + 0.20 * means["away_cards"]
    )

    hg, ag = _adjust_total_with_profile(hg, ag, profile, "goal_total_prior", 0.08)
    hc, ac = _adjust_total_with_profile(hc, ac, profile, "corner_total_prior", 0.10)
    hy, ay = _adjust_total_with_profile(hy, ay, profile, "card_total_prior", 0.12)

    score = _score_distribution(hg, ag)
    total_corners = max(0.1, hc + ac)
    total_cards = max(0.1, hy + ay)
    dispersion = model.get("dispersion", {})
    context = {
        "league": league,
        "model_created_at": model.get("created_at"),
        "style_label": profile.get("style_label"),
        "profile": {
            "tempo": profile.get("tempo"),
            "directness": profile.get("directness"),
            "pressing": profile.get("pressing"),
            "set_piece_weight": profile.get("set_piece_weight"),
            "card_pressure": profile.get("card_pressure"),
        },
        "expected_goals_home": round(float(hg), 3),
        "expected_goals_away": round(float(ag), 3),
        "expected_corners_home": round(float(hc), 3),
        "expected_corners_away": round(float(ac), 3),
        "expected_cards_home": round(float(hy), 3),
        "expected_cards_away": round(float(ay), 3),
        "expected_corners_total": round(float(total_corners), 3),
        "expected_cards_total": round(float(total_cards), 3),
        "goal_model": {k: round(float(v), 4) for k, v in score.items()},
        "corners": {
            f"over_{str(line).replace('.', '_')}": round(_over_prob_count(total_corners, line, dispersion.get("corners")), 4)
            for line in CORE_LINES["corners"]
        },
        "cards": {
            f"over_{str(line).replace('.', '_')}": round(_over_prob_count(total_cards, line, dispersion.get("cards")), 4)
            for line in CORE_LINES["cards"]
        },
    }
    return context


def _evaluate_model(model: Dict[str, object], val: pd.DataFrame) -> Dict[str, object]:
    preds = defaultdict(list)
    actuals = defaultdict(list)
    line_probs = defaultdict(list)
    line_actuals = defaultdict(list)

    for _, row in val.iterrows():
        ctx = _predict_from_model_data(model, str(row["home_team"]), str(row["away_team"]))
        total_goals = _as_float(row["home_goals"], 0.0) + _as_float(row["away_goals"], 0.0)
        preds["goals_total"].append(ctx["expected_goals_home"] + ctx["expected_goals_away"])
        actuals["goals_total"].append(total_goals)
        for line in CORE_LINES["goals"]:
            key = f"goals_over_{line}"
            prob_key = f"p_over{str(line).replace('.', '')}"
            line_probs[key].append(float(ctx["goal_model"][prob_key]))
            line_actuals[key].append(1 if total_goals > line else 0)
        line_probs["btts"].append(float(ctx["goal_model"]["p_btts"]))
        line_actuals["btts"].append(1 if row["home_goals"] > 0 and row["away_goals"] > 0 else 0)

        if pd.notna(row.get("HC")) and pd.notna(row.get("AC")):
            total_corners = float(row["HC"] + row["AC"])
            preds["corners_total"].append(float(ctx["expected_corners_total"]))
            actuals["corners_total"].append(total_corners)
            for line in CORE_LINES["corners"]:
                key = f"corners_over_{line}"
                prob_key = f"over_{str(line).replace('.', '_')}"
                line_probs[key].append(float(ctx["corners"][prob_key]))
                line_actuals[key].append(1 if total_corners > line else 0)

        if pd.notna(row.get("HY")) and pd.notna(row.get("AY")):
            total_cards = float(row["HY"] + row["AY"])
            preds["cards_total"].append(float(ctx["expected_cards_total"]))
            actuals["cards_total"].append(total_cards)
            for line in CORE_LINES["cards"]:
                key = f"cards_over_{line}"
                prob_key = f"over_{str(line).replace('.', '_')}"
                line_probs[key].append(float(ctx["cards"][prob_key]))
                line_actuals[key].append(1 if total_cards > line else 0)

    return {
        "count_metrics": {
            key: _count_metrics(preds[key], actuals[key])
            for key in sorted(preds.keys())
        },
        "line_metrics": {
            key: _metric_line(line_probs[key], line_actuals[key])
            for key in sorted(line_probs.keys())
        },
    }


def predict_market_context(league: str, home_team: str, away_team: str) -> Dict[str, object]:
    """Predict market-count context for one fixture from the saved league model."""
    model_path = os.path.join(MODEL_DIR, f"{league}_market_count_model.json")
    if league in _MODEL_CACHE:
        model = _MODEL_CACHE[league]
    elif os.path.exists(model_path):
        with open(model_path, "r", encoding="utf-8") as handle:
            model = json.load(handle)
        _MODEL_CACHE[league] = model
    else:
        model = _fallback_model(league)
        _MODEL_CACHE[league] = model
    return _predict_from_model_data(model, home_team, away_team)


def train_league_market_model(
    league: str,
    min_rows: int = 200,
    examples_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    if examples_df is None:
        df = _load_examples(league)
    else:
        df = examples_df[examples_df["league"] == league].copy()
    if len(df) < min_rows:
        return {
            "league": league,
            "status": "skipped",
            "reason": f"not enough completed examples ({len(df)} < {min_rows})",
            "n_examples": int(len(df)),
        }
    df = df.sort_values(["match_date", "id"]).reset_index(drop=True)
    split = max(1, int(len(df) * 0.80))
    train = df.iloc[:split].copy()
    val = df.iloc[split:].copy()
    model = _fit_model(league, train)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{league}_market_count_model.json")
    with open(model_path, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2)
    _MODEL_CACHE[league] = model
    metrics = _evaluate_model(model, val)
    report = {
        "league": league,
        "status": "trained",
        "model_path": model_path,
        "n_examples": int(len(df)),
        "n_train": int(len(train)),
        "n_validation": int(len(val)),
        "stats_coverage": {
            "corners": round(float(df[["HC", "AC"]].notna().all(axis=1).mean()), 4),
            "cards": round(float(df[["HY", "AY"]].notna().all(axis=1).mean()), 4),
        },
        "league_means": model["league_means"],
        "dispersion": model["dispersion"],
        **metrics,
    }
    return report


def ensure_market_columns(con: sqlite3.Connection) -> None:
    existing = [row[1] for row in con.execute("PRAGMA table_info(matches)").fetchall()]
    additions = [
        ("market_model_json", "TEXT"),
        ("referee", "TEXT"),
        ("venue", "TEXT"),
    ]
    for col, typ in additions:
        if col not in existing:
            con.execute(f"ALTER TABLE matches ADD COLUMN {col} {typ}")
    existing_te = [row[1] for row in con.execute("PRAGMA table_info(training_examples)").fetchall()]
    for col, typ in [("referee", "TEXT"), ("venue", "TEXT")]:
        if col not in existing_te:
            con.execute(f"ALTER TABLE training_examples ADD COLUMN {col} {typ}")
    con.commit()


def enrich_existing_matches(limit: Optional[int] = None) -> Dict[str, object]:
    if not os.path.exists(DB_PATH):
        return {"updated": 0, "error": "database_missing"}
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    ensure_market_columns(con)
    sql = """
        SELECT id, league, home_team, away_team
        FROM matches
        WHERE is_mock = 0
        ORDER BY match_date DESC, match_time DESC
    """
    if limit:
        sql += " LIMIT ?"
        rows = con.execute(sql, (int(limit),)).fetchall()
    else:
        rows = con.execute(sql).fetchall()
    updated = 0
    for row in rows:
        ctx = predict_market_context(row["league"], row["home_team"], row["away_team"])
        con.execute(
            """
            UPDATE matches
            SET corners_exp = ?,
                cards_exp = ?,
                market_model_json = ?
            WHERE id = ?
            """,
            (
                ctx.get("expected_corners_total"),
                ctx.get("expected_cards_total"),
                json.dumps(ctx),
                row["id"],
            ),
        )
        updated += 1
    con.commit()
    con.close()
    return {"updated": updated}


def run_market_count_pipeline(leagues: Optional[List[str]] = None, enrich_limit: Optional[int] = None) -> Dict[str, object]:
    write_profiles_report()
    df = _load_examples()
    if df.empty:
        report = {"created_at": datetime.utcnow().isoformat(), "error": "no_training_examples", "leagues": {}}
    else:
        target_leagues = leagues or sorted(df["league"].dropna().unique().tolist())
        league_reports = {}
        for league in target_leagues:
            league_reports[league] = train_league_market_model(league, examples_df=df)
        report = {
            "created_at": datetime.utcnow().isoformat(),
            "model_family": "goals_corners_cards_count_v1",
            "leagues": league_reports,
        }
    enrich_report = enrich_existing_matches(limit=enrich_limit)
    report["match_enrichment"] = enrich_report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default=None)
    parser.add_argument("--enrich-limit", type=int, default=None)
    args = parser.parse_args()
    leagues = [args.league] if args.league else None
    print(json.dumps(run_market_count_pipeline(leagues=leagues, enrich_limit=args.enrich_limit), indent=2))


if __name__ == "__main__":
    main()
