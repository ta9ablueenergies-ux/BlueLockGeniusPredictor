"""Run open-source/open CSV dataset ingestion for model learning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "open_data_pipeline_report.json")
sys.path.insert(0, SCRIPT_DIR)

from build_training_examples import backfill_from_closed_predictions, backfill_from_csv, summarize_counts  # noqa: E402
from import_xgabora_dataset import MATCHES_PATH, import_matches  # noqa: E402
from open_event_data_importer import run_event_data_import  # noqa: E402
from training_data_guard import ensure_training_data_guard  # noqa: E402
from why_learning_report import build_why_learning_report  # noqa: E402


OPEN_DATA_SOURCES = {
    "xgabora": {
        "url": "https://github.com/xgabora/Club-Football-Match-Data-2000-2025",
        "local_path": MATCHES_PATH,
        "role": "core historical match outcomes, odds, Elo, form, shots, shots on target, fouls, corners, cards",
        "license": "MIT per upstream repository",
        "imported": True,
    },
    "football_data_local_zip": {
        "url": "https://www.football-data.co.uk/data.php",
        "local_path": os.path.join(PROJECT_ROOT, "data", "football_data.zip"),
        "role": "local Football-Data style CSV extracts already present in data/temp_extract",
        "license": "free CSV data; verify commercial-use terms before redistribution",
        "imported": True,
    },
    "statsbomb_open_data": {
        "url": "https://github.com/statsbomb/open-data",
        "local_path": os.path.join(PROJECT_ROOT, "external_data", "statsbomb-open-data"),
        "role": "optional event/xG/lineup learning source for selected competitions",
        "license": "StatsBomb open-data terms; research/non-commercial style use",
        "imported": False,
    },
    "wyscout_public_events": {
        "url": "https://figshare.com/collections/Soccer_match_event_dataset/4415000",
        "local_path": os.path.join(PROJECT_ROOT, "external_data", "wyscout"),
        "role": "optional event-level pattern recognition source",
        "license": "CC BY 4.0",
        "imported": False,
    },
}


def run_open_data_pipeline(
    include_extra_xgabora_leagues: bool = False,
    refresh_market_counts: bool = False,
    include_event_sources: bool = False,
    download_event_sources: bool = False,
    event_max_matches_per_source: Optional[int] = None,
) -> Dict[str, object]:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report: Dict[str, object] = {
        "created_at": datetime.utcnow().isoformat(),
        "sources": OPEN_DATA_SOURCES,
        "steps": {},
    }

    if os.path.exists(MATCHES_PATH):
        report["steps"]["xgabora"] = import_matches(
            MATCHES_PATH,
            include_extra=include_extra_xgabora_leagues,
        )
    else:
        report["steps"]["xgabora"] = {"error": f"missing {MATCHES_PATH}"}

    core_leagues = ["PremierLeague", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]
    report["steps"]["football_data_temp_extract"] = [
        backfill_from_csv(league) for league in core_leagues
    ]
    report["steps"]["closed_predictions"] = backfill_from_closed_predictions()

    if include_event_sources:
        report["steps"]["open_event_data"] = run_event_data_import(
            sources=("statsbomb", "wyscout"),
            download=download_event_sources,
            leagues=core_leagues,
            max_matches_per_source=event_max_matches_per_source,
        )

    report["training_example_counts"] = summarize_counts()
    report["training_data_guard"] = ensure_training_data_guard(write_report=True)
    report["why_learning"] = build_why_learning_report(limit=5000)

    if refresh_market_counts:
        from market_count_models import run_market_count_pipeline

        report["steps"]["market_count_refresh"] = run_market_count_pipeline()

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-extra-xgabora-leagues", action="store_true")
    parser.add_argument("--refresh-market-counts", action="store_true")
    parser.add_argument("--include-event-sources", action="store_true")
    parser.add_argument("--download-event-sources", action="store_true")
    parser.add_argument("--event-max-matches-per-source", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(
        run_open_data_pipeline(
            include_extra_xgabora_leagues=args.include_extra_xgabora_leagues,
            refresh_market_counts=args.refresh_market_counts,
            include_event_sources=args.include_event_sources,
            download_event_sources=args.download_event_sources,
            event_max_matches_per_source=args.event_max_matches_per_source,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
