from __future__ import annotations

import json
import os
from datetime import datetime

from build_training_examples import backfill_from_csv
from import_free_sources import import_matches as import_free_dataset_matches, run_import as run_free_dataset_import
from import_xgabora_dataset import run_import as run_xgabora_import
from free_football_source import (
    build_source_registry,
    discover_local_repos,
    load_football_datasets_repo,
    load_openfootball_repo,
)
from training_data_guard import ensure_training_data_guard

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "free_source_pipeline_report.json")


def run_free_source_pipeline(include_extra_leagues=False, import_temp_extract=True):
    openfootball_matches = load_openfootball_repo()
    football_datasets_matches = load_football_datasets_repo()
    openfootball_written = import_free_dataset_matches(openfootball_matches)
    football_datasets_written = import_free_dataset_matches(football_datasets_matches)
    results = {
        "xgabora": run_xgabora_import(include_extra_leagues=include_extra_leagues),
        "free_datasets": run_free_dataset_import(),
        "openfootball_repo": {
            "loaded": len(openfootball_matches),
            "written": openfootball_written,
        },
        "football_datasets_repo": {
            "loaded": len(football_datasets_matches),
            "written": football_datasets_written,
        },
    }
    if import_temp_extract:
        temp_results = []
        for league in ["PremierLeague", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]:
            temp_results.append(backfill_from_csv(league))
        results["temp_extract"] = temp_results
    report = {
        "created_at": datetime.utcnow().isoformat(),
        "source_registry": build_source_registry(),
        "local_repos": discover_local_repos(),
        "results": results,
        "training_data_guard": ensure_training_data_guard(write_report=True),
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    print(json.dumps(run_free_source_pipeline(), indent=2))
