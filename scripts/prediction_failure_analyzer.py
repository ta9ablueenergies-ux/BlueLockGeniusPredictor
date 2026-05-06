"""Post-result failure analysis for the prediction feedback loop."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")
REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "prediction_failure_report.json")


def _market_hit_set(prediction: str) -> Set[str]:
    text = str(prediction or "").lower()
    if "home" in text:
        return {"H"}
    if "away" in text:
        return {"A"}
    if text == "draw" or " draw" in text:
        return {"D"}
    if "1x" in text:
        return {"H", "D"}
    if "x2" in text:
        return {"D", "A"}
    return set()


def _prediction_hit(row: sqlite3.Row) -> bool:
    prediction = str(row["primary_market"] or row["prediction"] or "").lower()
    fthg = row["actual_fthg"] or 0
    ftag = row["actual_ftag"] or 0
    total_goals = fthg + ftag
    if "over 1.5" in prediction:
        return total_goals > 1.5
    if "over 2.5" in prediction:
        return total_goals > 2.5
    if "over 3.5" in prediction:
        return total_goals > 3.5
    if "btts" in prediction or "both teams" in prediction:
        return fthg > 0 and ftag > 0
    hit_set = _market_hit_set(prediction)
    return bool(hit_set and row["actual_ftr"] in hit_set)


def _reason_tags(row: sqlite3.Row, hit: bool) -> List[str]:
    if hit:
        return ["hit"]
    tags: List[str] = []
    actual = row["actual_ftr"]
    pred = str(row["primary_market"] or row["prediction"] or "")
    pred_lower = pred.lower()
    hit_set = _market_hit_set(pred)
    if "over " in pred_lower or "btts" in pred_lower or "both teams" in pred_lower:
        tags.append("goals_market_miss")
    if actual == "D" and "D" not in hit_set:
        tags.append("missed_draw")
    if actual == "A" and "A" not in hit_set:
        tags.append("away_upset_or_home_bias")
    if actual == "H" and "H" not in hit_set:
        tags.append("home_win_underestimated")
    if (row["source_confidence"] or 0) < 0.80:
        tags.append("low_source_confidence")
    if row["HY"] is None or row["HC"] is None:
        tags.append("missing_cards_corners")
    else:
        try:
            market_model = json.loads(row["market_model_json"] or "{}")
        except Exception:
            market_model = {}
        actual_corners = float(row["HC"] or 0) + float(row["AC"] or 0)
        actual_cards = float(row["HY"] or 0) + float(row["AY"] or 0)
        exp_corners = market_model.get("expected_corners_total")
        exp_cards = market_model.get("expected_cards_total")
        if exp_corners is not None:
            if actual_corners >= float(exp_corners) + 3.0:
                tags.append("corners_underestimation")
            elif actual_corners <= float(exp_corners) - 3.0:
                tags.append("corners_overestimation")
        if exp_cards is not None:
            if actual_cards >= float(exp_cards) + 2.0:
                tags.append("card_spike_or_referee_distortion")
            elif actual_cards <= float(exp_cards) - 2.0:
                tags.append("cards_overestimation")
    if (row["value_edge"] or 0) <= 0:
        tags.append("no_positive_edge")
    if (row["eqi_score"] or 0) >= 70:
        tags.append("high_confidence_miss")
    return tags or ["unclassified_miss"]


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


def analyze_failures(league: Optional[str] = None, limit: int = 500) -> Dict[str, object]:
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
        SELECT m.id, m.league, m.match_date, m.home_team, m.away_team,
               m.prediction, m.primary_market, m.eqi_score, m.value_edge,
               m.source_confidence, m.data_source, m.HY, m.AY, m.HC, m.AC, m.market_model_json,
               r.actual_fthg, r.actual_ftag, r.actual_ftr
        FROM matches m
        JOIN match_results r ON r.id = m.id
        WHERE m.is_mock = 0
          AND r.actual_ftr IN ('H', 'D', 'A')
          {league_clause}
        ORDER BY m.match_date DESC, m.last_updated DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    con.close()

    league_stats = defaultdict(lambda: {"total": 0, "hits": 0, "misses": 0})
    reason_counts: Counter[str] = Counter()
    market_counts = defaultdict(lambda: {"total": 0, "hits": 0, "misses": 0})
    high_confidence_misses = []
    count_errors = {
        "corners": {"n": 0, "abs_error": 0.0},
        "cards": {"n": 0, "abs_error": 0.0},
    }

    for row in rows:
        hit = _prediction_hit(row)
        league_row = league_stats[row["league"]]
        league_row["total"] += 1
        league_row["hits" if hit else "misses"] += 1

        market = row["primary_market"] or row["prediction"] or "unknown"
        market_row = market_counts[market]
        market_row["total"] += 1
        market_row["hits" if hit else "misses"] += 1

        tags = _reason_tags(row, hit)
        reason_counts.update(tags)
        if not hit and (row["eqi_score"] or 0) >= 70:
            high_confidence_misses.append({
                "id": row["id"],
                "league": row["league"],
                "date": row["match_date"],
                "home": row["home_team"],
                "away": row["away_team"],
                "prediction": market,
                "actual_ftr": row["actual_ftr"],
                "score": f"{row['actual_fthg']}-{row['actual_ftag']}",
                "eqi": row["eqi_score"],
                "value_edge": row["value_edge"],
                "reason_tags": tags,
            })
        try:
            market_model = json.loads(row["market_model_json"] or "{}")
        except Exception:
            market_model = {}
        if row["HC"] is not None and row["AC"] is not None and market_model.get("expected_corners_total") is not None:
            count_errors["corners"]["n"] += 1
            count_errors["corners"]["abs_error"] += abs(float(row["HC"] + row["AC"]) - float(market_model["expected_corners_total"]))
        if row["HY"] is not None and row["AY"] is not None and market_model.get("expected_cards_total") is not None:
            count_errors["cards"]["n"] += 1
            count_errors["cards"]["abs_error"] += abs(float(row["HY"] + row["AY"]) - float(market_model["expected_cards_total"]))

    for item in league_stats.values():
        item["hit_rate"] = _rate(item["hits"], item["total"])
    for item in market_counts.values():
        item["hit_rate"] = _rate(item["hits"], item["total"])

    report = {
        "created_at": datetime.utcnow().isoformat(),
        "league_filter": league,
        "resolved_predictions": len(rows),
        "overall_hit_rate": _rate(sum(v["hits"] for v in league_stats.values()), len(rows)),
        "by_league": dict(sorted(league_stats.items())),
        "by_primary_market": dict(sorted(market_counts.items())),
        "reason_counts": dict(reason_counts.most_common()),
        "count_market_mae": {
            key: round(val["abs_error"] / val["n"], 4) if val["n"] else 0.0
            for key, val in count_errors.items()
        },
        "high_confidence_misses": high_confidence_misses[:50],
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default=None)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(analyze_failures(league=args.league, limit=args.limit), indent=2))


if __name__ == "__main__":
    main()
