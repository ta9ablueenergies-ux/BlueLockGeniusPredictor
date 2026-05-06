"""Refresh predictions for already stored fixtures with table-pressure context."""

import argparse
import os
import sqlite3
import sys

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from main_script import precalculate_all_features, predict_results
from persistence_manager import DB_PATH, get_league_json_from_sqlite, upsert_match_sqlite
from platform_orchestrator import build_global_manifest, build_global_tickets, export_all_league_json


def load_fixtures(target_date):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT league, match_date, match_time, home_team, away_team,
               data_source, source_url, flashscore_id
        FROM matches
        WHERE match_date = ?
          AND is_mock = 0
          AND home_team IS NOT NULL
          AND away_team IS NOT NULL
        ORDER BY league ASC, match_time ASC, home_team ASC
        """,
        (target_date,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def refresh(target_date):
    fixtures = load_fixtures(target_date)
    if not fixtures:
        print(f"[SituationRefresh] No stored real fixtures found for {target_date}")
        return {"target_date": target_date, "fixtures": 0, "updated": 0}

    updated = 0
    by_league = {}
    for row in fixtures:
        by_league.setdefault(row["league"], []).append(row)

    for league, rows in by_league.items():
        print(f"[SituationRefresh] Refreshing {league}: {len(rows)} fixtures")
        ratings = precalculate_all_features(league)
        frame_rows = []
        intel_rows = []
        for item in rows:
            frame_rows.append(
                {
                    "HomeTeam": item["home_team"],
                    "AwayTeam": item["away_team"],
                    "Time": item.get("match_time") or "15:00",
                    "Date": item["match_date"],
                    "h_course": 2.0,
                    "d_course": 3.0,
                    "a_course": 3.5,
                    "source_url": item.get("source_url"),
                    "flashscore_id": item.get("flashscore_id"),
                }
            )
            intel_rows.append(
                {
                    "home_team": item["home_team"],
                    "away_team": item["away_team"],
                    "match_date": item["match_date"],
                    "kickoff_utc": item.get("match_time") or "15:00",
                    "odds_home": 2.0,
                    "odds_draw": 3.0,
                    "odds_away": 3.5,
                }
            )
        predictions = predict_results(pd.DataFrame(frame_rows), league, ratings, intel_rows)
        for _, pred in predictions.iterrows():
            source = next(
                (
                    item.get("data_source") or "browser"
                    for item in rows
                    if item["home_team"] == pred["Home"] and item["away_team"] == pred["Away"]
                ),
                "browser",
            )
            upsert_match_sqlite(
                pred.to_dict(),
                data_source=source,
                run_id=f"situation-refresh-{target_date}",
            )
            updated += 1

    os.environ.setdefault("PIPELINE_START_DATE", target_date)
    os.environ.setdefault("PIPELINE_END_DATE", target_date)
    export_all_league_json()
    build_global_tickets()
    build_global_manifest()
    print(f"[SituationRefresh] Updated {updated} fixtures and rebuilt public JSON")
    return {"target_date": target_date, "fixtures": len(fixtures), "updated": updated}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format")
    args = parser.parse_args()
    refresh(args.date)


if __name__ == "__main__":
    main()

