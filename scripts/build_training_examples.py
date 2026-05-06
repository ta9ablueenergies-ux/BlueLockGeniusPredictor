"""Backfill normalized training examples for neural and GNN models."""

import argparse
import glob
import json
import os
import sqlite3
import sys
from datetime import datetime

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from persistence_manager import DB_PATH, WHY_STAT_COLUMNS

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "temp_extract")
REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "training_examples_report.json")

LEAGUES = ["PremierLeague", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]
LABELS = {"H", "D", "A"}
COUNT_STAT_COLUMNS = ["HY", "AY", "HR", "AR", "HC", "AC"]
ALL_STAT_COLUMNS = COUNT_STAT_COLUMNS + WHY_STAT_COLUMNS


def qcol(name):
    return f'"{name}"'


def parse_date(value):
    if value is None or value == "":
        return ""
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def first_value(row, names, default=None):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def normalize_team(value):
    return str(value or "").strip()


def example_id(league, match_date, home_team, away_team, source="hist"):
    raw = f"{source}_{league}_{match_date}_{home_team}_{away_team}"
    return raw.replace("/", "-").replace("\\", "-")


def normalized_market_probs(home_odds, draw_odds, away_odds):
    inv = []
    for odd in (home_odds, draw_odds, away_odds):
        odd = safe_float(odd, 0.0)
        inv.append(1.0 / odd if odd > 1.01 else 0.0)
    total = sum(inv)
    if total <= 0:
        return None, None, None
    return tuple(x / total for x in inv)


def bulk_upsert_training_examples(examples):
    if not examples:
        return 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for example in examples:
        home_odds = example.get("home_odds")
        draw_odds = example.get("draw_odds")
        away_odds = example.get("away_odds")
        p_h, p_d, p_a = normalized_market_probs(home_odds, draw_odds, away_odds)
        stat_values = [example.get(col) for col in ALL_STAT_COLUMNS]
        rows.append((
            example.get("id"), example.get("league"), example.get("match_date"),
            example.get("home_team"), example.get("away_team"),
            example.get("home_goals"), example.get("away_goals"), example.get("actual_ftr"),
            home_odds, draw_odds, away_odds, p_h, p_d, p_a,
            example.get("closing_home_odds"), example.get("closing_draw_odds"), example.get("closing_away_odds"),
            example.get("source", "historical_csv"), example.get("source_confidence", 0.90),
            json.dumps(example.get("features", {})),
            *stat_values,
            example.get("flashscore_id"), example.get("source_url"), example.get("stats_scraped_at"),
            example.get("created_at", now), now,
        ))

    conn = sqlite3.connect(DB_PATH)
    stat_cols_sql = ", ".join(qcol(col) for col in ALL_STAT_COLUMNS)
    stat_updates_sql = ",\n            ".join(
        f'{qcol(col)}=COALESCE(excluded.{qcol(col)}, training_examples.{qcol(col)})'
        for col in ALL_STAT_COLUMNS
    )
    placeholders = ", ".join(["?"] * (25 + len(ALL_STAT_COLUMNS)))
    conn.executemany(f'''
        INSERT INTO training_examples (
            id, league, match_date, home_team, away_team,
            home_goals, away_goals, actual_ftr,
            home_odds, draw_odds, away_odds,
            market_prob_home, market_prob_draw, market_prob_away,
            closing_home_odds, closing_draw_odds, closing_away_odds,
            source, source_confidence, features_json,
            {stat_cols_sql}, flashscore_id, source_url, stats_scraped_at,
            created_at, updated_at
        ) VALUES ({placeholders})
        ON CONFLICT(id) DO UPDATE SET
            league=excluded.league,
            match_date=excluded.match_date,
            home_team=excluded.home_team,
            away_team=excluded.away_team,
            home_goals=excluded.home_goals,
            away_goals=excluded.away_goals,
            actual_ftr=excluded.actual_ftr,
            home_odds=excluded.home_odds,
            draw_odds=excluded.draw_odds,
            away_odds=excluded.away_odds,
            market_prob_home=excluded.market_prob_home,
            market_prob_draw=excluded.market_prob_draw,
            market_prob_away=excluded.market_prob_away,
            closing_home_odds=excluded.closing_home_odds,
            closing_draw_odds=excluded.closing_draw_odds,
            closing_away_odds=excluded.closing_away_odds,
            source=excluded.source,
            source_confidence=excluded.source_confidence,
            features_json=excluded.features_json,
            {stat_updates_sql},
            flashscore_id=COALESCE(excluded.flashscore_id, training_examples.flashscore_id),
            source_url=COALESCE(excluded.source_url, training_examples.source_url),
            stats_scraped_at=COALESCE(excluded.stats_scraped_at, training_examples.stats_scraped_at),
            updated_at=excluded.updated_at
    ''', rows)
    conn.commit()
    conn.close()
    return len(rows)


def feature_payload(row):
    keys = [
        "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR",
        "h_pts_avg3", "a_pts_avg3", "h_pts_avg5", "a_pts_avg5",
        "h_gz_avg5", "a_gz_avg5", "h_gs_avg5", "a_gs_avg5",
        "h_sh_od_avg5", "a_sh_od_avg5",
    ]
    features = {}
    for key in keys:
        value = row.get(key)
        if value is not None and pd.notna(value):
            features[key] = safe_float(value, 0.0)
    return features


def stat_payload(row):
    values = {}
    aliases = {
        "HY": ["HY", "HomeYellow", "HomeYellowCards"],
        "AY": ["AY", "AwayYellow", "AwayYellowCards"],
        "HR": ["HR", "HomeRed", "HomeRedCards"],
        "AR": ["AR", "AwayRed", "AwayRedCards"],
        "HC": ["HC", "HomeCorners", "HomeCorner", "HomeCornersTotal"],
        "AC": ["AC", "AwayCorners", "AwayCorner", "AwayCornersTotal"],
        "HS": ["HS", "HomeShots", "HomeShotsTotal"],
        "AS": ["AS", "AwayShots", "AwayShotsTotal"],
        "HST": ["HST", "HomeTarget", "HomeShotsOnTarget", "HomeSOT"],
        "AST": ["AST", "AwayTarget", "AwayShotsOnTarget", "AwaySOT"],
        "HF": ["HF", "HomeFouls"],
        "AF": ["AF", "AwayFouls"],
        "HO": ["HO", "HomeOffsides"],
        "AO": ["AO", "AwayOffsides"],
        "HPoss": ["HPoss", "HomePossession"],
        "APoss": ["APoss", "AwayPossession"],
        "HXG": ["HXG", "HomeXG", "HomexG"],
        "AXG": ["AXG", "AwayXG", "AwayxG"],
        "HBC": ["HBC", "HomeBigChances"],
        "ABC": ["ABC", "AwayBigChances"],
    }
    for target, names in aliases.items():
        values[target] = safe_float(first_value(row, names))
    return values


def backfill_from_csv(league):
    pattern = os.path.join(DATA_DIR, f"{league}_*.csv")
    count = 0
    skipped = 0
    examples = []
    for path in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(path)
        except Exception:
            skipped += 1
            continue
        required = {"Date", "HomeTeam", "AwayTeam", "FTR"}
        if not required.issubset(df.columns):
            skipped += len(df)
            continue

        for _, row in df.iterrows():
            actual_ftr = str(row.get("FTR", "")).upper()[:1]
            home_team = normalize_team(row.get("HomeTeam"))
            away_team = normalize_team(row.get("AwayTeam"))
            match_date = parse_date(row.get("Date"))
            if actual_ftr not in LABELS or not home_team or not away_team or not match_date:
                skipped += 1
                continue

            home_odds = safe_float(first_value(row, ["h_course", "B365H", "AvgH", "PSH"]))
            draw_odds = safe_float(first_value(row, ["d_course", "B365D", "AvgD", "PSD"]))
            away_odds = safe_float(first_value(row, ["a_course", "B365A", "AvgA", "PSA"]))
            examples.append({
                "id": example_id(league, match_date, home_team, away_team),
                "league": league,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": safe_int(row.get("FTHG")),
                "away_goals": safe_int(row.get("FTAG")),
                "actual_ftr": actual_ftr,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "closing_home_odds": home_odds,
                "closing_draw_odds": draw_odds,
                "closing_away_odds": away_odds,
                "source": "historical_csv",
                "source_confidence": 0.90,
                "features": feature_payload(row),
                **stat_payload(row),
            })
            count += 1
    written = bulk_upsert_training_examples(examples)
    return {"league": league, "csv_examples": count, "csv_written": written, "csv_skipped": skipped}


def backfill_from_closed_predictions(league=None):
    if not os.path.exists(DB_PATH):
        return {"closed_examples": 0}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT
            m.id, m.league, m.match_date, m.home_team, m.away_team,
            m.source_confidence, m.intel_raw,
            r.actual_fthg, r.actual_ftag, r.actual_ftr,
            r.closing_home_odds, r.closing_draw_odds, r.closing_away_odds,
            m.flashscore_id, m.source_url, m.HY, m.AY, m.HR, m.AR, m.HC, m.AC,
            m.HS, m."AS" AS "AS", m.HST, m.AST, m.HF, m.AF, m.HO, m.AO,
            m.HPoss, m.APoss, m.HXG, m.AXG, m.HBC, m.ABC,
            m.stats_scraped_at
        FROM matches m
        JOIN match_results r ON r.id = m.id
        WHERE r.actual_ftr IN ('H', 'D', 'A')
    """
    params = ()
    if league:
        sql += " AND m.league = ?"
        params = (league,)
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    count = 0
    examples = []
    for row in rows:
        features = {}
        try:
            features = json.loads(row["intel_raw"] or "{}")
        except Exception:
            features = {}
        examples.append({
            "id": f"closed_{row['id']}",
            "league": row["league"],
            "match_date": parse_date(row["match_date"]),
            "home_team": normalize_team(row["home_team"]),
            "away_team": normalize_team(row["away_team"]),
            "home_goals": row["actual_fthg"],
            "away_goals": row["actual_ftag"],
            "actual_ftr": row["actual_ftr"],
            "home_odds": row["closing_home_odds"],
            "draw_odds": row["closing_draw_odds"],
            "away_odds": row["closing_away_odds"],
            "closing_home_odds": row["closing_home_odds"],
            "closing_draw_odds": row["closing_draw_odds"],
            "closing_away_odds": row["closing_away_odds"],
            "source": "closed_prediction",
            "source_confidence": row["source_confidence"] or 0.75,
            "flashscore_id": row["flashscore_id"],
            "source_url": row["source_url"],
            "HY": row["HY"],
            "AY": row["AY"],
            "HR": row["HR"],
            "AR": row["AR"],
            "HC": row["HC"],
            "AC": row["AC"],
            **{col: row[col] for col in WHY_STAT_COLUMNS},
            "stats_scraped_at": row["stats_scraped_at"],
            "features": features,
        })
        count += 1
    written = bulk_upsert_training_examples(examples)
    return {"closed_examples": count, "closed_written": written}


def summarize_counts():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT league, COUNT(*) AS examples,
               SUM(CASE WHEN market_prob_home IS NOT NULL THEN 1 ELSE 0 END) AS with_market
        FROM training_examples
        GROUP BY league
        ORDER BY league
    """).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="ALL")
    args = parser.parse_args()

    leagues = LEAGUES if args.league == "ALL" else [args.league]
    results = [backfill_from_csv(league) for league in leagues]
    closed = backfill_from_closed_predictions(None if args.league == "ALL" else args.league)
    report = {
        "created_at": datetime.utcnow().isoformat(),
        "results": results,
        "closed_predictions": closed,
        "counts": summarize_counts(),
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
