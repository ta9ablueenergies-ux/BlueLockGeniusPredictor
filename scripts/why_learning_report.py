"""Coverage report for scraped why-signals and post-match learning patterns."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")
REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "why_learning_report.json")

CORE_WHY_COLUMNS = ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY"]
ADVANCED_WHY_COLUMNS = ["HF", "AF", "HO", "AO", "HPoss", "APoss", "HXG", "AXG", "HBC", "ABC"]


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _row_coverage(row: sqlite3.Row, cols: List[str]) -> float:
    return _rate(sum(1 for col in cols if row[col] is not None), len(cols))


def build_why_learning_report(league: Optional[str] = None, limit: int = 3000) -> Dict[str, object]:
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
        SELECT m.id, m.league, m.match_date,
               m.HS, m."AS" AS "AS", m.HST, m.AST, m.HC, m.AC, m.HY, m.AY,
               m.HF, m.AF, m.HO, m.AO, m.HPoss, m.APoss, m.HXG, m.AXG, m.HBC, m.ABC,
               r.actual_ftr
        FROM matches m
        LEFT JOIN match_results r ON r.id = m.id
        WHERE m.is_mock = 0
          {league_clause}
        ORDER BY m.match_date DESC, m.last_updated DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()

    evidence_rows = con.execute(
        """
        SELECT league, evidence_type, COUNT(*) AS n, AVG(data_quality) AS avg_quality
        FROM raw_match_evidence
        GROUP BY league, evidence_type
        """
    ).fetchall()
    pattern_rows = con.execute(
        """
        SELECT pattern_key, COUNT(*) AS n, AVG(confidence) AS avg_confidence
        FROM match_learning_patterns
        GROUP BY pattern_key
        ORDER BY n DESC, pattern_key ASC
        """
    ).fetchall()
    training_rows = con.execute(
        """
        SELECT league,
               COUNT(*) AS examples,
               SUM(CASE WHEN HS IS NOT NULL AND "AS" IS NOT NULL
                         AND HST IS NOT NULL AND AST IS NOT NULL
                         AND HC IS NOT NULL AND AC IS NOT NULL
                         AND HY IS NOT NULL AND AY IS NOT NULL
                        THEN 1 ELSE 0 END) AS full_core_rows,
               SUM(CASE WHEN HF IS NOT NULL AND AF IS NOT NULL THEN 1 ELSE 0 END) AS foul_rows,
               SUM(CASE WHEN HXG IS NOT NULL AND AXG IS NOT NULL THEN 1 ELSE 0 END) AS xg_rows
        FROM training_examples
        GROUP BY league
        ORDER BY league
        """
    ).fetchall()
    con.close()

    by_league = defaultdict(lambda: {
        "matches": 0,
        "resolved": 0,
        "core_sum": 0.0,
        "advanced_sum": 0.0,
        "full_core": 0,
    })
    for row in rows:
        item = by_league[row["league"]]
        item["matches"] += 1
        if row["actual_ftr"] in {"H", "D", "A"}:
            item["resolved"] += 1
        core_cov = _row_coverage(row, CORE_WHY_COLUMNS)
        adv_cov = _row_coverage(row, ADVANCED_WHY_COLUMNS)
        item["core_sum"] += core_cov
        item["advanced_sum"] += adv_cov
        if core_cov >= 1.0:
            item["full_core"] += 1

    league_report = {}
    for lg, item in sorted(by_league.items()):
        n = item["matches"]
        league_report[lg] = {
            "matches": n,
            "resolved": item["resolved"],
            "core_why_coverage": round(item["core_sum"] / n, 4) if n else 0.0,
            "advanced_why_coverage": round(item["advanced_sum"] / n, 4) if n else 0.0,
            "full_core_rows": item["full_core"],
        }

    evidence = [
        {
            "league": row["league"],
            "evidence_type": row["evidence_type"],
            "n": row["n"],
            "avg_quality": round(float(row["avg_quality"] or 0.0), 4),
        }
        for row in evidence_rows
    ]
    patterns = [
        {
            "pattern_key": row["pattern_key"],
            "n": row["n"],
            "avg_confidence": round(float(row["avg_confidence"] or 0.0), 4),
        }
        for row in pattern_rows
    ]
    training_coverage = {
        row["league"]: {
            "examples": row["examples"],
            "full_core_why_rows": row["full_core_rows"] or 0,
            "foul_rows": row["foul_rows"] or 0,
            "xg_rows": row["xg_rows"] or 0,
            "full_core_why_rate": _rate(row["full_core_rows"] or 0, row["examples"] or 0),
        }
        for row in training_rows
    }

    report = {
        "created_at": datetime.utcnow().isoformat(),
        "league_filter": league,
        "rows_scanned": len(rows),
        "coverage_columns": {
            "core": CORE_WHY_COLUMNS,
            "advanced": ADVANCED_WHY_COLUMNS,
        },
        "by_league": league_report,
        "training_examples_coverage": training_coverage,
        "evidence": evidence,
        "patterns": patterns,
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default=None)
    parser.add_argument("--limit", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(build_why_learning_report(league=args.league, limit=args.limit), indent=2))


if __name__ == "__main__":
    main()
