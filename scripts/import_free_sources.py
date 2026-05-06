from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from typing import List

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from free_football_source import FreeMatch, discover_public_sources, load_from_local_csv, load_from_zip
from persistence_manager import DB_PATH, init_db


REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "free_source_import_report.json")


def match_to_example(item: FreeMatch) -> dict:
    return {
        "id": f"free_{item.league}_{item.match_date}_{item.home_team}_{item.away_team}".replace("/", "-"),
        "league": item.league,
        "match_date": item.match_date,
        "home_team": item.home_team,
        "away_team": item.away_team,
        "home_goals": item.home_goals,
        "away_goals": item.away_goals,
        "actual_ftr": item.actual_ftr,
        "home_odds": item.home_odds,
        "draw_odds": item.draw_odds,
        "away_odds": item.away_odds,
        "closing_home_odds": item.home_odds,
        "closing_draw_odds": item.draw_odds,
        "closing_away_odds": item.away_odds,
        "source": item.source,
        "source_confidence": 0.85 if item.source == "openfootball" else 0.75,
    }


def import_matches(matches: List[FreeMatch]) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    written = 0
    try:
        for item in matches:
            example = match_to_example(item)
            cur.execute(
                """
                INSERT OR IGNORE INTO training_examples (
                    id, league, match_date, home_team, away_team,
                    home_goals, away_goals, actual_ftr,
                    home_odds, draw_odds, away_odds,
                    market_prob_home, market_prob_draw, market_prob_away,
                    closing_home_odds, closing_draw_odds, closing_away_odds,
                    source, source_confidence, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    NULL, NULL, NULL,
                    ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    example["id"],
                    example["league"],
                    example["match_date"],
                    example["home_team"],
                    example["away_team"],
                    example["home_goals"],
                    example["away_goals"],
                    example["actual_ftr"],
                    example["home_odds"],
                    example["draw_odds"],
                    example["away_odds"],
                    example["closing_home_odds"],
                    example["closing_draw_odds"],
                    example["closing_away_odds"],
                    example["source"],
                    example["source_confidence"],
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            written += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return written


def main():
    parser = argparse.ArgumentParser(description="Import free football data sources")
    parser.add_argument("--zip-path", default=os.path.join(PROJECT_ROOT, "data", "football_data.zip"))
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--league", default=None, help="Optional league name when importing a single CSV")
    parser.add_argument("--show-sources", action="store_true")
    args = parser.parse_args()

    if args.show_sources:
        print(json.dumps(discover_public_sources(), indent=2))
        return

    init_db()

    matches: List[FreeMatch] = []
    if args.csv_path and args.league:
        matches.extend(load_from_local_csv(args.csv_path, args.league))
    else:
        matches.extend(load_from_zip(args.zip_path))

    written = import_matches(matches)
    report = {
        "created_at": datetime.utcnow().isoformat(),
        "source": "local_free_datasets",
        "rows_loaded": len(matches),
        "rows_written": written,
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


def run_import(zip_path=None, csv_path=None, league=None):
    """Programmatic entry point for pipeline integration."""
    init_db()
    if csv_path and league:
        matches = load_from_local_csv(csv_path, league)
    else:
        matches = load_from_zip(zip_path or os.path.join(PROJECT_ROOT, "data", "football_data.zip"))
    written = import_matches(matches)
    report = {
        "created_at": datetime.utcnow().isoformat(),
        "source": "local_free_datasets",
        "rows_loaded": len(matches),
        "rows_written": written,
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    main()
