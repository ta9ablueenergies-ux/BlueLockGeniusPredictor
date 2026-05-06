from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from persistence_manager import DB_PATH, WHY_STAT_COLUMNS  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "web" / "data" / "training_data_health_report.json"
CANONICAL_VIEW_NAME = "training_examples_canonical"
DUPLICATE_VIEW_NAME = "training_examples_duplicate_groups"

COUNT_STAT_COLUMNS = ["HY", "AY", "HR", "AR", "HC", "AC"]
MODEL_RELEVANT_COLUMNS = COUNT_STAT_COLUMNS + list(WHY_STAT_COLUMNS)


def _qcol(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _acol(alias: str, name: str) -> str:
    return f"{alias}.{_qcol(name)}"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({_qcol(table)})").fetchall()]


def _score_expr(columns: Iterable[str], alias: str = "te") -> str:
    parts = [
        f"CASE WHEN {_acol(alias, col)} IS NOT NULL THEN 1 ELSE 0 END"
        for col in columns
    ]
    return " + ".join(parts) if parts else "0"


def _source_priority_expr(alias: str = "te") -> str:
    src = f"LOWER(COALESCE({_acol(alias, 'source')}, ''))"
    return f"""
        CASE
            WHEN {src} = 'closed_prediction' THEN 100
            WHEN {src} LIKE '%flashscore%' THEN 98
            WHEN {src} LIKE '%statsbomb%' THEN 95
            WHEN {src} LIKE '%wyscout%' THEN 94
            WHEN {src} LIKE '%xgabora%' THEN 92
            WHEN {src} = 'historical_csv' THEN 90
            WHEN {src} = 'football_datasets_repo' THEN 86
            WHEN {src} = 'openfootball_repo' THEN 84
            WHEN {src} = 'free_dataset' THEN 82
            ELSE 50
        END
    """


def create_canonical_training_views(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Create non-destructive views that expose one best row per fixture."""
    if not _table_exists(conn, "training_examples"):
        return {"status": "missing_training_examples"}

    columns = _table_columns(conn, "training_examples")
    if not columns:
        return {"status": "empty_schema"}

    existing_stats = [col for col in MODEL_RELEVANT_COLUMNS if col in columns]
    completeness = _score_expr(existing_stats)
    priority = _source_priority_expr()
    select_cols = ", ".join(f"ranked.{_qcol(col)}" for col in columns)

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_training_examples_fixture_key
        ON training_examples(league, match_date, home_team, away_team)
        """
    )
    conn.execute(f"DROP VIEW IF EXISTS {_qcol(CANONICAL_VIEW_NAME)}")
    conn.execute(f"DROP VIEW IF EXISTS {_qcol(DUPLICATE_VIEW_NAME)}")

    conn.execute(
        f"""
        CREATE VIEW {_qcol(CANONICAL_VIEW_NAME)} AS
        WITH ranked AS (
            SELECT
                te.*,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        LOWER(TRIM(te.league)),
                        te.match_date,
                        LOWER(TRIM(te.home_team)),
                        LOWER(TRIM(te.away_team))
                    ORDER BY
                        ({completeness}) DESC,
                        COALESCE(te.source_confidence, 0) DESC,
                        ({priority}) DESC,
                        CASE WHEN te.actual_ftr IN ('H', 'D', 'A') THEN 1 ELSE 0 END DESC,
                        COALESCE(te.stats_scraped_at, te.updated_at, te.created_at, '') DESC,
                        te.id DESC
                ) AS _canonical_rank
            FROM training_examples te
            WHERE te.league IS NOT NULL
              AND te.match_date IS NOT NULL
              AND te.home_team IS NOT NULL
              AND te.away_team IS NOT NULL
        )
        SELECT {select_cols}
        FROM ranked
        WHERE _canonical_rank = 1
        """
    )

    conn.execute(
        f"""
        CREATE VIEW {_qcol(DUPLICATE_VIEW_NAME)} AS
        SELECT
            league,
            match_date,
            LOWER(TRIM(home_team)) AS home_key,
            LOWER(TRIM(away_team)) AS away_key,
            COUNT(*) AS duplicate_count,
            GROUP_CONCAT(DISTINCT source) AS sources
        FROM training_examples
        WHERE league IS NOT NULL
          AND match_date IS NOT NULL
          AND home_team IS NOT NULL
          AND away_team IS NOT NULL
        GROUP BY league, match_date, home_key, away_key
        HAVING COUNT(*) > 1
        """
    )
    return {"status": "ready", "canonical_view": CANONICAL_VIEW_NAME, "duplicate_view": DUPLICATE_VIEW_NAME}


def _fetch_dicts(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _fetch_one(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return None if row is None else row[0]


LOCAL_TZ = datetime.now().astimezone().tzinfo


def _to_utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _local_naive_to_utc(value: datetime) -> datetime:
    if LOCAL_TZ is None:
        return value
    return value.replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc).replace(tzinfo=None)


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return _local_naive_to_utc(datetime.strptime(text, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        pass
    try:
        return _to_utc_naive(datetime.fromisoformat(text))
    except ValueError:
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _model_timestamp(path: Path, nested_metrics: bool) -> Optional[str]:
    data = _read_json(path)
    if nested_metrics:
        metrics = data.get("metrics") or {}
        for key in ("timestamp", "trained_at", "created_at", "generated_at"):
            if metrics.get(key):
                return str(metrics[key])
    for key in ("timestamp", "trained_at", "created_at", "generated_at"):
        if data.get(key):
            return str(data[key])
    return None


def _model_staleness(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    latest_rows = _fetch_dicts(
        conn,
        """
        SELECT league, MAX(COALESCE(updated_at, created_at, stats_scraped_at, '')) AS latest_training_update
        FROM training_examples
        GROUP BY league
        ORDER BY league
        """,
    )
    output: Dict[str, Dict[str, Any]] = {}
    for row in latest_rows:
        league = str(row.get("league") or "")
        latest = str(row.get("latest_training_update") or "")
        latest_dt = _parse_dt(latest)

        v11_path = PROJECT_ROOT / "model" / "v11" / f"{league}_hybrid_v11_draww_v5_metrics.json"
        v11_ts = _model_timestamp(v11_path, nested_metrics=True)
        v11_dt = _parse_dt(v11_ts)

        market_path = PROJECT_ROOT / "model" / "market_counts" / f"{league}_market_count_model.json"
        market_ts = _model_timestamp(market_path, nested_metrics=False)
        market_dt = _parse_dt(market_ts)

        output[league] = {
            "latest_training_update": latest or None,
            "v11_metrics_path": str(v11_path) if v11_path.exists() else None,
            "v11_model_timestamp": v11_ts,
            "v11_stale": bool(latest_dt and (v11_dt is None or latest_dt > v11_dt)),
            "market_count_model_path": str(market_path) if market_path.exists() else None,
            "market_count_model_timestamp": market_ts,
            "market_count_stale": bool(latest_dt and (market_dt is None or latest_dt > market_dt)),
        }
    return output


def build_training_data_health_report(conn: sqlite3.Connection) -> Dict[str, Any]:
    raw_rows = int(_fetch_one(conn, "SELECT COUNT(*) FROM training_examples") or 0)
    canonical_rows = int(_fetch_one(conn, f"SELECT COUNT(*) FROM {_qcol(CANONICAL_VIEW_NAME)}") or 0)

    by_league = _fetch_dicts(
        conn,
        """
        WITH grouped AS (
            SELECT
                league,
                match_date,
                LOWER(TRIM(home_team)) AS home_key,
                LOWER(TRIM(away_team)) AS away_key,
                COUNT(*) AS n
            FROM training_examples
            WHERE league IS NOT NULL
              AND match_date IS NOT NULL
              AND home_team IS NOT NULL
              AND away_team IS NOT NULL
            GROUP BY league, match_date, home_key, away_key
        )
        SELECT
            league,
            SUM(n) AS raw_rows,
            COUNT(*) AS canonical_rows,
            SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END) AS duplicate_groups,
            SUM(CASE WHEN n > 1 THEN n - 1 ELSE 0 END) AS duplicate_extra_rows,
            MAX(n) AS max_dupes
        FROM grouped
        GROUP BY league
        ORDER BY duplicate_extra_rows DESC, raw_rows DESC
        """,
    )

    canonical_coverage = _fetch_dicts(
        conn,
        f"""
        SELECT
            league,
            COUNT(*) AS examples,
            SUM(CASE WHEN market_prob_home IS NOT NULL THEN 1 ELSE 0 END) AS with_market,
            SUM(CASE WHEN HC IS NOT NULL AND AC IS NOT NULL THEN 1 ELSE 0 END) AS with_corners,
            SUM(CASE WHEN HY IS NOT NULL AND AY IS NOT NULL THEN 1 ELSE 0 END) AS with_cards,
            SUM(CASE WHEN HS IS NOT NULL AND "AS" IS NOT NULL THEN 1 ELSE 0 END) AS with_shots,
            SUM(CASE WHEN HXG IS NOT NULL AND AXG IS NOT NULL THEN 1 ELSE 0 END) AS with_xg
        FROM {_qcol(CANONICAL_VIEW_NAME)}
        GROUP BY league
        ORDER BY examples DESC
        """,
    )

    top_sources = _fetch_dicts(
        conn,
        """
        SELECT COALESCE(source, 'unknown') AS source, COUNT(*) AS rows
        FROM training_examples
        GROUP BY COALESCE(source, 'unknown')
        ORDER BY rows DESC
        LIMIT 20
        """,
    )

    stale = _model_staleness(conn)
    stale_v11 = sum(1 for row in stale.values() if row.get("v11_stale"))
    stale_market = sum(1 for row in stale.values() if row.get("market_count_stale"))

    duplicate_extra = raw_rows - canonical_rows
    duplicate_groups = int(_fetch_one(conn, f"SELECT COUNT(*) FROM {_qcol(DUPLICATE_VIEW_NAME)}") or 0)
    report = {
        "created_at": datetime.utcnow().isoformat(),
        "db_path": DB_PATH,
        "report_path": str(REPORT_PATH),
        "views": {
            "canonical": CANONICAL_VIEW_NAME,
            "duplicates": DUPLICATE_VIEW_NAME,
        },
        "summary": {
            "raw_rows": raw_rows,
            "canonical_rows": canonical_rows,
            "duplicate_extra_rows": duplicate_extra,
            "duplicate_groups": duplicate_groups,
            "v11_stale_leagues": stale_v11,
            "market_count_stale_leagues": stale_market,
        },
        "by_league": by_league,
        "canonical_coverage": canonical_coverage,
        "top_sources": top_sources,
        "model_staleness": stale,
    }
    return report


def ensure_training_data_guard(write_report: bool = True) -> Dict[str, Any]:
    if not os.path.exists(DB_PATH):
        report = {
            "created_at": datetime.utcnow().isoformat(),
            "db_path": DB_PATH,
            "status": "missing_db",
            "summary": {},
            "report_path": str(REPORT_PATH),
        }
        if write_report:
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    conn = sqlite3.connect(DB_PATH)
    try:
        view_status = create_canonical_training_views(conn)
        conn.commit()
        if view_status.get("status") != "ready":
            report = {
                "created_at": datetime.utcnow().isoformat(),
                "db_path": DB_PATH,
                "status": view_status.get("status"),
                "summary": {},
                "report_path": str(REPORT_PATH),
            }
        else:
            report = build_training_data_health_report(conn)
            report["status"] = "ready"
        report["view_status"] = view_status
    finally:
        conn.close()

    if write_report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    print(json.dumps(ensure_training_data_guard(write_report=not args.no_write_report), indent=2))


if __name__ == "__main__":
    main()
