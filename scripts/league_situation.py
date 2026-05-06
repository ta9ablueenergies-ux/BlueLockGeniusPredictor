"""League table and motivation features for match prediction.

This layer turns completed results into standings pressure signals. It is
deliberately conservative: relegation pressure is treated as context for the
model and as a small calibrated probability nudge, not as a rule that overrides
team strength.
"""

import os
import sqlite3
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")


LEAGUE_RULES = {
    "PremierLeague": {"season_games": 38, "relegation_slots": 3},
    "LaLiga": {"season_games": 38, "relegation_slots": 3},
    "SerieA": {"season_games": 38, "relegation_slots": 3},
    "Bundesliga": {"season_games": 34, "relegation_slots": 3},
    "Ligue1": {"season_games": 34, "relegation_slots": 3},
    "Championship": {"season_games": 46, "relegation_slots": 3},
    "ScottishPremiership": {"season_games": 38, "relegation_slots": 1},
    "Eredivisie": {"season_games": 34, "relegation_slots": 3},
    "LigaNOS": {"season_games": 34, "relegation_slots": 2},
    "BelgianProLeague": {"season_games": 30, "relegation_slots": 2},
    "SuperLig": {"season_games": 38, "relegation_slots": 4},
    "2Bundesliga": {"season_games": 34, "relegation_slots": 3},
    "Ligue2": {"season_games": 34, "relegation_slots": 3},
    "LaLiga2": {"season_games": 42, "relegation_slots": 4},
    "SerieB": {"season_games": 38, "relegation_slots": 4},
}

V11_SITUATION_FEATURE_COLUMNS = [
    # Core Poisson & form signals
    "exp_home_goals",
    "exp_away_goals",
    "expected_goal_diff",
    "form_diff",
    "home_rank_strength",
    "away_rank_strength",
    "home_relegation_pressure",
    "away_relegation_pressure",
    "poisson_home_away_edge",
    "poisson_draw_probability",
    "relegation_pressure_diff",
    "max_relegation_pressure",
    # Bookmaker market signals (8 new high-signal features)
    "impl_prob_h",       # Implied home win probability from odds
    "impl_prob_d",       # Implied draw probability from odds
    "impl_prob_a",       # Implied away win probability from odds
    "market_overround",  # Bookmaker margin (lower = more efficient market)
    "value_edge_h",      # Model p_h minus implied p_h
    "value_edge_d",      # Model p_d minus implied p_d
    "value_edge_a",      # Model p_a minus implied p_a
    "odds_confidence",   # 1 / overround as market confidence score
    "odds_h_raw",        # Capped raw home odds
    "odds_d_raw",        # Capped raw draw odds
    "odds_a_raw",        # Capped raw away odds
    "market_favorite_gap",
    # Leakage-free head-to-head features from prior meetings
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_away_win_rate",
    "h2h_avg_goals",
]


def clamp(value, low=0.0, high=1.0):
    try:
        value = float(value)
    except Exception:
        value = low
    return max(low, min(high, value))


def parse_match_date(value):
    if value is None or value == "":
        return None
    if isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
        parsed = pd.to_datetime(value, errors="coerce")
    else:
        text = str(value).strip()
        if len(text) >= 10 and text[:4].isdigit() and text[4] in {"-", "/"}:
            parsed = pd.to_datetime(text, errors="coerce", yearfirst=True)
        else:
            parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def season_key(value):
    dt = parse_match_date(value)
    if dt is None:
        return "unknown"
    start = dt.year if dt.month >= 7 else dt.year - 1
    return f"{start}-{start + 1}"


def result_points(ftr, side):
    ftr = str(ftr or "").upper()[:1]
    if ftr == "D":
        return 1
    if ftr == side:
        return 3
    return 0


def normalize_completed_frame(df):
    """Return a standard completed-results frame."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "season_key"])
    rename = {
        "match_date": "Date",
        "home_team": "HomeTeam",
        "away_team": "AwayTeam",
        "home_goals": "FTHG",
        "away_goals": "FTAG",
        "actual_ftr": "FTR",
        "actual_fthg": "FTHG",
        "actual_ftag": "FTAG",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    for col in required:
        if col not in out.columns:
            out[col] = np.nan
    out["Date"] = pd.to_datetime(out["Date"].apply(parse_match_date), errors="coerce")
    out["FTHG"] = pd.to_numeric(out["FTHG"], errors="coerce")
    out["FTAG"] = pd.to_numeric(out["FTAG"], errors="coerce")
    out["FTR"] = out["FTR"].astype(str).str.upper().str[:1]
    out = out.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = out[out["FTR"].isin(["H", "D", "A"])].copy()
    out["HomeTeam"] = out["HomeTeam"].astype(str).str.strip()
    out["AwayTeam"] = out["AwayTeam"].astype(str).str.strip()
    out["season_key"] = out["Date"].apply(season_key)
    out = out.sort_values(["Date", "HomeTeam", "AwayTeam"])
    out = out.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
    return out.reset_index(drop=True)


def empty_team_row(team):
    return {
        "team": team,
        "played": 0,
        "points": 0,
        "gf": 0,
        "ga": 0,
        "gd": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "position": None,
        "teams_count": 0,
    }


def build_standings(completed_df):
    rows = normalize_completed_frame(completed_df)
    table = defaultdict(lambda: empty_team_row(""))
    for _, row in rows.iterrows():
        home = str(row["HomeTeam"]).strip()
        away = str(row["AwayTeam"]).strip()
        if not table[home]["team"]:
            table[home] = empty_team_row(home)
        if not table[away]["team"]:
            table[away] = empty_team_row(away)

        h_goals = int(row["FTHG"])
        a_goals = int(row["FTAG"])
        ftr = str(row["FTR"])
        h_pts = result_points(ftr, "H")
        a_pts = result_points(ftr, "A")
        table[home]["played"] += 1
        table[away]["played"] += 1
        table[home]["points"] += h_pts
        table[away]["points"] += a_pts
        table[home]["gf"] += h_goals
        table[home]["ga"] += a_goals
        table[away]["gf"] += a_goals
        table[away]["ga"] += h_goals
        if ftr == "H":
            table[home]["wins"] += 1
            table[away]["losses"] += 1
        elif ftr == "A":
            table[away]["wins"] += 1
            table[home]["losses"] += 1
        else:
            table[home]["draws"] += 1
            table[away]["draws"] += 1

    ranked = []
    for item in table.values():
        item["gd"] = item["gf"] - item["ga"]
        ranked.append(dict(item))
    ranked.sort(key=lambda r: (-r["points"], -r["gd"], -r["gf"], r["team"]))
    teams_count = len(ranked)
    for idx, item in enumerate(ranked, start=1):
        item["position"] = idx
        item["teams_count"] = teams_count
    return ranked


def _points_at_position(table, position):
    for row in table:
        if row.get("position") == position:
            return int(row.get("points", 0))
    return None


def _league_rule(league):
    return LEAGUE_RULES.get(str(league or ""), {"season_games": 38, "relegation_slots": 3})


def team_situation(table, team, league):
    team = str(team or "").strip()
    teams_count = len(table)
    rule = _league_rule(league)
    relegation_slots = min(int(rule.get("relegation_slots", 3)), max(0, teams_count - 1))
    season_games = int(rule.get("season_games", 38))

    row = next((r for r in table if r.get("team") == team), None)
    if not row or teams_count <= 1:
        return {
            "team": team,
            "position": None,
            "teams_count": teams_count,
            "points": 0,
            "played": 0,
            "games_left": season_games,
            "rank_strength": 0.5,
            "late_season_factor": 0.0,
            "relegation_zone": False,
            "near_relegation_zone": False,
            "points_to_safety": None,
            "points_above_drop": None,
            "relegation_pressure": 0.0,
            "top_pressure": 0.0,
            "must_win_survival": False,
            "label": "No table signal",
        }

    position = int(row["position"])
    points = int(row["points"])
    played = int(row["played"])
    games_left = max(0, season_games - played)
    rank_strength = 1.0 - ((position - 1) / max(1, teams_count - 1))
    late_factor = clamp((played / max(1, season_games) - 0.55) / 0.35)

    safety_pos = max(1, teams_count - relegation_slots)
    drop_entry_pos = min(teams_count, safety_pos + 1)
    safety_points = _points_at_position(table, safety_pos)
    drop_points = _points_at_position(table, drop_entry_pos)
    in_relegation = relegation_slots > 0 and position > safety_pos

    points_to_safety = None
    points_above_drop = None
    if in_relegation and safety_points is not None:
        points_to_safety = max(0, safety_points - points + 1)
        danger_raw = 1.0
    elif drop_points is not None:
        points_above_drop = max(0, points - drop_points)
        danger_raw = clamp((7.0 - points_above_drop) / 7.0)
    else:
        danger_raw = 0.0

    near_zone = bool(in_relegation or danger_raw >= 0.15)
    relegation_pressure = clamp(danger_raw * (0.35 + 0.65 * late_factor))
    if in_relegation and games_left <= 3:
        relegation_pressure = max(relegation_pressure, 0.85)
    must_win = bool(relegation_pressure >= 0.72 and games_left <= 3)

    leader_points = table[0].get("points", points) if table else points
    top_pressure = 0.0
    if position <= 4:
        top_gap = max(0, int(leader_points) - points)
        top_pressure = clamp((5.0 - top_gap) / 5.0) * (0.25 + 0.75 * late_factor)

    if must_win:
        label = "Survival must-win"
    elif relegation_pressure >= 0.65:
        label = "High survival pressure"
    elif relegation_pressure >= 0.35:
        label = "Relegation pressure"
    elif top_pressure >= 0.45:
        label = "Top-table pressure"
    else:
        label = "Normal table context"

    return {
        "team": team,
        "position": position,
        "teams_count": teams_count,
        "points": points,
        "played": played,
        "games_left": games_left,
        "rank_strength": round(float(rank_strength), 4),
        "late_season_factor": round(float(late_factor), 4),
        "relegation_zone": bool(in_relegation),
        "near_relegation_zone": bool(near_zone),
        "points_to_safety": points_to_safety,
        "points_above_drop": points_above_drop,
        "relegation_pressure": round(float(relegation_pressure), 4),
        "top_pressure": round(float(top_pressure), 4),
        "must_win_survival": must_win,
        "label": label,
    }


def match_situation_from_table(table, league, home_team, away_team):
    home = team_situation(table, home_team, league)
    away = team_situation(table, away_team, league)
    h_pressure = float(home.get("relegation_pressure", 0.0) or 0.0)
    a_pressure = float(away.get("relegation_pressure", 0.0) or 0.0)
    pressure_score = max(h_pressure, a_pressure)
    if h_pressure >= 0.35 and a_pressure < 0.35:
        label = home["label"]
    elif a_pressure >= 0.35 and h_pressure < 0.35:
        label = away["label"]
    elif h_pressure >= 0.35 and a_pressure >= 0.35:
        label = "Both teams under pressure"
    else:
        label = "Normal table context"
    return {
        "home": home,
        "away": away,
        "home_pressure": round(h_pressure, 4),
        "away_pressure": round(a_pressure, 4),
        "pressure_diff": round(h_pressure - a_pressure, 4),
        "match_pressure_score": round(pressure_score * 100.0, 1),
        "label": label,
    }


def match_situation_from_history(history_df, league, home_team, away_team, match_date=None):
    rows = normalize_completed_frame(history_df)
    if rows.empty:
        return match_situation_from_table([], league, home_team, away_team)

    date_obj = parse_match_date(match_date)
    table_source = "current_season"
    data_quality = "direct"
    if date_obj is not None:
        target_season = season_key(date_obj)
        current_rows = rows[(rows["season_key"] == target_season) & (rows["Date"] < pd.Timestamp(date_obj))]
        if len(current_rows) >= 40:
            rows = current_rows
        else:
            prior_counts = rows[rows["season_key"] != target_season].groupby("season_key").size()
            prior_counts = prior_counts[prior_counts >= 40]
            if not prior_counts.empty:
                proxy_season = prior_counts.index.sort_values()[-1]
                rows = rows[rows["season_key"] == proxy_season]
                table_source = "previous_season_proxy"
                data_quality = "proxy_rank_only"
            else:
                rows = current_rows
        if rows.empty:
            return match_situation_from_table([], league, home_team, away_team)
    else:
        latest_season = rows.iloc[-1]["season_key"]
        rows = rows[rows["season_key"] == latest_season]

    table = build_standings(rows)
    situation = match_situation_from_table(table, league, home_team, away_team)
    situation["season_key"] = season_key(date_obj) if date_obj else (rows.iloc[-1]["season_key"] if not rows.empty else None)
    situation["table_source"] = table_source
    situation["data_quality"] = data_quality
    situation["table_matches"] = int(len(rows))
    if data_quality == "proxy_rank_only":
        for side in ("home", "away"):
            team = situation.get(side) or {}
            team["relegation_pressure"] = 0.0
            team["top_pressure"] = 0.0
            team["must_win_survival"] = False
            team["label"] = "Previous-season rank proxy"
        situation["home_pressure"] = 0.0
        situation["away_pressure"] = 0.0
        situation["pressure_diff"] = 0.0
        situation["match_pressure_score"] = 0.0
        situation["label"] = "Previous-season rank proxy"
    return situation


def load_completed_examples_from_db(league, before_date=None, db_path=DB_PATH):
    if not os.path.exists(db_path):
        return pd.DataFrame()
    params = [league]
    sql = """
        SELECT match_date AS Date,
               home_team AS HomeTeam,
               away_team AS AwayTeam,
               home_goals AS FTHG,
               away_goals AS FTAG,
               actual_ftr AS FTR
        FROM training_examples
        WHERE league = ?
          AND actual_ftr IN ('H', 'D', 'A')
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
    """
    if before_date:
        sql += " AND match_date < ?"
        params.append(str(before_date).split("T")[0])
    sql += " ORDER BY match_date ASC, id ASC"
    try:
        con = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql, con, params=params)
        con.close()
        return normalize_completed_frame(df)
    except Exception:
        return pd.DataFrame()


def get_match_situation_from_db(league, home_team, away_team, match_date, db_path=DB_PATH):
    history = load_completed_examples_from_db(league, before_date=match_date, db_path=db_path)
    return match_situation_from_history(history, league, home_team, away_team, match_date)


def _safe_implied_probs(odds_h, odds_d, odds_a):
    """Convert decimal odds to normalized implied probabilities."""
    try:
        raw_h = 1.0 / float(odds_h) if odds_h and float(odds_h) > 0 else None
        raw_d = 1.0 / float(odds_d) if odds_d and float(odds_d) > 0 else None
        raw_a = 1.0 / float(odds_a) if odds_a and float(odds_a) > 0 else None
        if raw_h is None or raw_d is None or raw_a is None:
            return None, None, None, None
        overround = raw_h + raw_d + raw_a
        if overround <= 0:
            return None, None, None, None
        return raw_h / overround, raw_d / overround, raw_a / overround, overround
    except Exception:
        return None, None, None, None


def _safe_raw_odds(value):
    try:
        value = float(value)
        if value <= 0:
            return 0.0
        return min(value, 20.0)
    except Exception:
        return 0.0


def build_v11_situation_features(exp_h, exp_a, form_diff, p_h, p_d, p_a, situation,
                                  odds_h=None, odds_d=None, odds_a=None,
                                  h2h_features=None):
    """Map model inputs + table context + market/H2H signals into the fixed V11 vector."""
    situation = situation or {}
    h2h_features = h2h_features or {}
    home = situation.get("home") or {}
    away = situation.get("away") or {}
    h_rank = float(home.get("rank_strength", 0.5) or 0.5)
    a_rank = float(away.get("rank_strength", 0.5) or 0.5)
    h_press = float(home.get("relegation_pressure", 0.0) or 0.0)
    a_press = float(away.get("relegation_pressure", 0.0) or 0.0)

    # Bookmaker market signal computation
    impl_h, impl_d, impl_a, overround = _safe_implied_probs(odds_h, odds_d, odds_a)
    has_odds = impl_h is not None

    # Fall back to Poisson probs when odds are missing (training rows without odds)
    impl_h    = impl_h    if has_odds else float(p_h)
    impl_d    = impl_d    if has_odds else float(p_d)
    impl_a    = impl_a    if has_odds else float(p_a)
    overround = overround if has_odds else 1.0  # neutral overround
    confidence = 1.0 / overround if overround > 0 else 0.0

    # Value edge: how much our Poisson model disagrees with the market
    v_edge_h = float(p_h) - impl_h
    v_edge_d = float(p_d) - impl_d
    v_edge_a = float(p_a) - impl_a
    odds_h_raw = _safe_raw_odds(odds_h) if has_odds else 0.0
    odds_d_raw = _safe_raw_odds(odds_d) if has_odds else 0.0
    odds_a_raw = _safe_raw_odds(odds_a) if has_odds else 0.0
    ordered_market = sorted([impl_h, impl_d, impl_a], reverse=True)
    market_favorite_gap = ordered_market[0] - ordered_market[1] if len(ordered_market) >= 2 else 0.0

    return [
        # Core 12 features (unchanged)
        float(exp_h),
        float(exp_a),
        float(exp_h) - float(exp_a),
        float(form_diff),
        h_rank,
        a_rank,
        h_press,
        a_press,
        float(p_h) - float(p_a),
        float(p_d),
        h_press - a_press,
        max(h_press, a_press),
        # 8 new bookmaker market features
        impl_h,
        impl_d,
        impl_a,
        min(overround, 1.5),  # cap to prevent outliers
        v_edge_h,
        v_edge_d,
        v_edge_a,
        confidence,
        odds_h_raw,
        odds_d_raw,
        odds_a_raw,
        market_favorite_gap,
        float(h2h_features.get("home_win_rate", 1.0 / 3.0)),
        float(h2h_features.get("draw_rate", 1.0 / 3.0)),
        float(h2h_features.get("away_win_rate", 1.0 / 3.0)),
        float(h2h_features.get("avg_goals", float(exp_h) + float(exp_a))),
    ]


def adjust_1x2_for_situation(p_h, p_d, p_a, situation, max_shift=0.065):
    """Apply a small pressure-aware adjustment and return the audit payload."""
    situation = situation or {}
    home = situation.get("home") or {}
    away = situation.get("away") or {}
    h_press = float(home.get("relegation_pressure", 0.0) or 0.0)
    a_press = float(away.get("relegation_pressure", 0.0) or 0.0)
    h_rank = float(home.get("rank_strength", 0.5) or 0.5)
    a_rank = float(away.get("rank_strength", 0.5) or 0.5)

    shift = (h_press - a_press) * 0.045
    if home.get("must_win_survival") and not away.get("must_win_survival"):
        shift += 0.014
    if away.get("must_win_survival") and not home.get("must_win_survival"):
        shift -= 0.014
    if h_press >= 0.65 and (a_rank - h_rank) >= 0.30:
        shift += 0.010
    if a_press >= 0.65 and (h_rank - a_rank) >= 0.30:
        shift -= 0.010

    shift = clamp(shift, -max_shift, max_shift)
    draw_shift = 0.0
    if h_press >= 0.45 and a_press >= 0.45:
        draw_shift = 0.010

    probs = np.array([p_h + shift, p_d + draw_shift, p_a - shift], dtype=float)
    probs = np.clip(probs, 0.03, 0.94)
    probs = probs / probs.sum()
    adjustment = {
        "applied": bool(abs(shift) >= 0.001 or draw_shift > 0),
        "home_shift": round(float(probs[0] - p_h), 4),
        "draw_shift": round(float(probs[1] - p_d), 4),
        "away_shift": round(float(probs[2] - p_a), 4),
        "home_pressure": round(h_press, 4),
        "away_pressure": round(a_press, 4),
        "label": situation.get("label", "Normal table context"),
    }
    return float(probs[0]), float(probs[1]), float(probs[2]), adjustment


class RunningSituationBuilder:
    """Chronological standings state for leakage-free training features."""

    def __init__(self, league, initial_history=None):
        self.league = league
        self.rows_by_season = defaultdict(list)
        seed = normalize_completed_frame(initial_history)
        for _, row in seed.iterrows():
            self._append_row(row)

    def _append_row(self, row):
        sk = row.get("season_key") or season_key(row.get("Date"))
        self.rows_by_season[sk].append(
            {
                "Date": row.get("Date"),
                "HomeTeam": row.get("HomeTeam"),
                "AwayTeam": row.get("AwayTeam"),
                "FTHG": row.get("FTHG"),
                "FTAG": row.get("FTAG"),
                "FTR": row.get("FTR"),
            }
        )

    def situation_for_match(self, home_team, away_team, match_date):
        sk = season_key(match_date)
        rows = pd.DataFrame(self.rows_by_season.get(sk, []))
        return match_situation_from_history(rows, self.league, home_team, away_team, match_date)

    def update_from_result(self, row):
        frame = normalize_completed_frame(pd.DataFrame([dict(row)]))
        if not frame.empty:
            self._append_row(frame.iloc[0])
