"""Import xgabora/Club-Football-Match-Data-2000-2025 into training_examples."""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from build_training_examples import bulk_upsert_training_examples, stat_payload, summarize_counts


DATA_DIR = os.path.join(PROJECT_ROOT, "external_data", "xgabora")
MATCHES_PATH = os.path.join(DATA_DIR, "Matches.csv")
REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "xgabora_import_report.json")

DIVISION_MAP = {
    "E0": "PremierLeague",
    "SP1": "LaLiga",
    "I1": "SerieA",
    "D1": "Bundesliga",
    "F1": "Ligue1",
}

FEATURE_COLUMNS = [
    "Division", "MatchTime",
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home", "Form3Away", "Form5Away",
    "HTHome", "HTAway", "HTResult",
    "HomeShots", "AwayShots", "HomeTarget", "AwayTarget",
    "HomeFouls", "AwayFouls", "HomeCorners", "AwayCorners",
    "HomeYellow", "AwayYellow", "HomeRed", "AwayRed",
    "MaxHome", "MaxDraw", "MaxAway",
    "Over25", "Under25", "MaxOver25", "MaxUnder25",
    "HandiSize", "HandiHome", "HandiAway",
    "C_LTH", "C_LTA", "C_VHD", "C_VAD", "C_HTB", "C_PHB",
]


def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value) or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if value is None or pd.isna(value) or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def normalize_date(value):
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def normalize_team(value):
    return str(value or "").strip()


def league_for_division(division, include_extra=False):
    division = str(division or "").strip()
    if division in DIVISION_MAP:
        return DIVISION_MAP[division]
    if include_extra and division:
        return f"xgabora_{division}"
    return None


def feature_payload(row):
    features = {}
    for col in FEATURE_COLUMNS:
        if col not in row or pd.isna(row[col]):
            continue
        value = row[col]
        if isinstance(value, str):
            features[col] = value
        else:
            features[col] = safe_float(value, 0.0)
    if "HomeElo" in features and "AwayElo" in features:
        features["EloDifference"] = round(features["HomeElo"] - features["AwayElo"], 4)
    if "Form5Home" in features and "Form5Away" in features:
        features["Form5Difference"] = features["Form5Home"] - features["Form5Away"]
    return features


def example_id(league, match_date, home_team, away_team):
    raw = f"hist_{league}_{match_date}_{home_team}_{away_team}"
    return raw.replace("/", "-").replace("\\", "-")


def import_matches(matches_path, include_extra=False, chunksize=25000):
    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"Missing xgabora Matches.csv: {matches_path}")

    total_seen = 0
    total_considered = 0
    total_written = 0
    skipped = 0
    per_league = {}
    columns_seen = None

    target_divisions = set(DIVISION_MAP.keys())

    for chunk in pd.read_csv(matches_path, chunksize=chunksize, low_memory=False):
        total_seen += len(chunk)
        if columns_seen is None:
            columns_seen = list(chunk.columns)
        if not include_extra:
            chunk = chunk[chunk["Division"].isin(target_divisions)].copy()
            if chunk.empty:
                continue
        examples = []
        for row in chunk.to_dict("records"):
            total_considered += 1
            league = league_for_division(row.get("Division"), include_extra=include_extra)
            match_date = normalize_date(row.get("MatchDate"))
            home_team = normalize_team(row.get("HomeTeam"))
            away_team = normalize_team(row.get("AwayTeam"))
            actual_ftr = str(row.get("FTResult", "")).upper()[:1]
            if not league or not match_date or not home_team or not away_team or actual_ftr not in {"H", "D", "A"}:
                skipped += 1
                continue

            home_odds = safe_float(row.get("OddHome"))
            draw_odds = safe_float(row.get("OddDraw"))
            away_odds = safe_float(row.get("OddAway"))
            examples.append({
                "id": example_id(league, match_date, home_team, away_team),
                "league": league,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": safe_int(row.get("FTHome")),
                "away_goals": safe_int(row.get("FTAway")),
                "actual_ftr": actual_ftr,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "closing_home_odds": home_odds,
                "closing_draw_odds": draw_odds,
                "closing_away_odds": away_odds,
                "source": "xgabora_matches",
                "source_confidence": 0.92,
                "features": feature_payload(row),
                **stat_payload(row),
            })
            per_league[league] = per_league.get(league, 0) + 1

        total_written += bulk_upsert_training_examples(examples)

    return {
        "matches_path": matches_path,
        "rows_seen": total_seen,
        "rows_considered": total_considered,
        "rows_written": total_written,
        "rows_skipped": skipped,
        "per_league": dict(sorted(per_league.items())),
        "columns_seen": columns_seen or [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches-path", default=MATCHES_PATH)
    parser.add_argument("--include-extra-leagues", action="store_true")
    parser.add_argument("--chunksize", type=int, default=25000)
    args = parser.parse_args()

    result = import_matches(args.matches_path, include_extra=args.include_extra_leagues, chunksize=args.chunksize)
    report = {
        "created_at": datetime.utcnow().isoformat(),
        "source": "https://github.com/xgabora/Club-Football-Match-Data-2000-2025",
        "result": result,
        "training_example_counts": summarize_counts(),
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def run_import(include_extra_leagues=False):
    """Programmatic entry point for pipeline integration."""
    result = import_matches(MATCHES_PATH, include_extra=include_extra_leagues)
    report = {
        "created_at": datetime.utcnow().isoformat(),
        "source": "https://github.com/xgabora/Club-Football-Match-Data-2000-2025",
        "result": result,
        "training_example_counts": summarize_counts(),
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main()
