from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")
REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "flashscore_stats_backfill_report.json")

sys.path.insert(0, SCRIPT_DIR)

from browser_scraper import BrowserRenderer  # noqa: E402
from persistence_manager import (  # noqa: E402
    init_db,
    upsert_match_learning_patterns,
    upsert_raw_match_evidence,
)
from why_signal_extractor import derive_post_match_patterns  # noqa: E402


STAT_COLUMNS = [
    "HY", "AY", "HR", "AR", "HC", "AC",
    "HS", "AS", "HST", "AST", "HF", "AF", "HO", "AO",
    "HPoss", "APoss", "HXG", "AXG", "HBC", "ABC",
]

STAT_LABELS = {
    "Corner kicks": ("HC", "AC", True),
    "Corners": ("HC", "AC", True),
    "Yellow cards": ("HY", "AY", True),
    "Red cards": ("HR", "AR", True),
    "Goal attempts": ("HS", "AS", True),
    "Total shots": ("HS", "AS", True),
    "Shots total": ("HS", "AS", True),
    "Shots on goal": ("HST", "AST", True),
    "Shots on target": ("HST", "AST", True),
    "Fouls": ("HF", "AF", True),
    "Fouls committed": ("HF", "AF", True),
    "Offsides": ("HO", "AO", True),
    "Ball possession": ("HPoss", "APoss", False),
    "Possession": ("HPoss", "APoss", False),
    "Expected Goals (xG)": ("HXG", "AXG", False),
    "Expected goals": ("HXG", "AXG", False),
    "xG": ("HXG", "AXG", False),
    "Big chances": ("HBC", "ABC", True),
}


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "--"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return int(float(match.group(0)))
    except Exception:
        return None


def _safe_number(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text or text in {"-", "--"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _extract_mid(source_url: Optional[str], flashscore_id: Optional[str]) -> Optional[str]:
    if flashscore_id:
        return str(flashscore_id).strip()
    if not source_url:
        return None
    try:
        parsed = urlparse(source_url)
        mid = parse_qs(parsed.query).get("mid", [None])[0]
        if mid:
            return str(mid).strip()
    except Exception:
        pass
    match = re.search(r"mid=([A-Za-z0-9]+)", str(source_url))
    return match.group(1) if match else None


def _canonical_match_urls(source_url: str, flashscore_id: Optional[str]) -> Dict[str, str]:
    parsed = urlparse(source_url)
    mid = _extract_mid(source_url, flashscore_id)
    query = f"mid={mid}" if mid else parsed.query
    base_path = parsed.path
    if "/summary/" in base_path:
        base_path = base_path.split("/summary/", 1)[0]
    base_path = base_path.rstrip("/") + "/"
    summary = urlunparse((parsed.scheme, parsed.netloc, base_path, "", query, ""))
    stats = urlunparse((parsed.scheme, parsed.netloc, base_path + "summary/stats/overall/", "", query, ""))
    return {"summary": summary, "stats": stats}


def _parse_stat_pair(lines: List[str], label: str, integer: bool = True) -> tuple[Optional[float], Optional[float]]:
    target = _normalize_label(label)
    for idx, line in enumerate(lines):
        if _normalize_label(line) != target:
            continue
        parser = _safe_int if integer else _safe_number
        home = parser(lines[idx - 1]) if idx > 0 else None
        away = parser(lines[idx + 1]) if idx + 1 < len(lines) else None
        if home is not None or away is not None:
            return home, away
    return None, None


def parse_flashscore_stats_text(text: str) -> Dict[str, Optional[float]]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    stats: Dict[str, Optional[float]] = {key: None for key in STAT_COLUMNS}
    for label, (home_key, away_key, integer) in STAT_LABELS.items():
        home, away = _parse_stat_pair(lines, label, integer=integer)
        if home is not None and stats.get(home_key) is None:
            stats[home_key] = float(home)
        if away is not None and stats.get(away_key) is None:
            stats[away_key] = float(away)
    return stats


def _extract_incident_cards(renderer: BrowserRenderer) -> Dict[str, int]:
    js = """
    () => Array.from(document.querySelectorAll('.smv__participantRow')).reduce((acc, row) => {
        const side = row.classList.contains('smv__homeParticipant') ? 'home' :
            (row.classList.contains('smv__awayParticipant') ? 'away' : '');
        if (!side) return acc;
        const yellow = row.querySelector('.yellowCard-ico') ? 1 : 0;
        const red = (row.querySelector('.redCard-ico') || row.querySelector('.secondYellowCard-ico')) ? 1 : 0;
        if (yellow) acc[side + '_yellow'] += 1;
        if (red) acc[side + '_red'] += 1;
        return acc;
    }, {home_yellow: 0, away_yellow: 0, home_red: 0, away_red: 0})
    """
    try:
        if getattr(renderer, "_page", None) is not None:
            payload = renderer._page.evaluate(js) or {}
        elif getattr(renderer, "_sb", None) is not None:
            payload = renderer._sb.driver.execute_script(f"return ({js})()") or {}
        else:
            payload = {}
    except Exception:
        payload = {}
    return {
        "HY": int(payload.get("home_yellow") or 0),
        "AY": int(payload.get("away_yellow") or 0),
        "HR": int(payload.get("home_red") or 0),
        "AR": int(payload.get("away_red") or 0),
    }


def _extract_incidents(renderer: BrowserRenderer) -> List[Dict[str, object]]:
    js = """
    () => Array.from(document.querySelectorAll('.smv__participantRow, .smv__incident, [class*="incident"]')).slice(0, 80).map((row) => {
        const cls = row.className || '';
        const text = (row.innerText || row.textContent || '').trim();
        const minute = (text.match(/\\b\\d{1,3}(?:\\+\\d+)?'/) || [''])[0];
        const side = cls.includes('home') || cls.includes('Home') ? 'home' :
            (cls.includes('away') || cls.includes('Away') ? 'away' : null);
        let type = 'event';
        if (row.querySelector('[class*="yellowCard"], .yellowCard-ico')) type = 'yellow_card';
        if (row.querySelector('[class*="redCard"], .redCard-ico, .secondYellowCard-ico')) type = 'red_card';
        if (row.querySelector('[class*="soccerBall"], [class*="football"]')) type = 'goal';
        if (row.querySelector('[class*="substitution"]')) type = 'substitution';
        return {minute, side, type, text};
    }).filter((item) => item.text)
    """
    try:
        if getattr(renderer, "_page", None) is not None:
            return renderer._page.evaluate(js) or []
        if getattr(renderer, "_sb", None) is not None:
            return renderer._sb.driver.execute_script(f"return ({js})()") or []
    except Exception:
        return []
    return []


def scrape_flashscore_match_stats(renderer: BrowserRenderer, source_url: str, flashscore_id: Optional[str] = None) -> Dict[str, object]:
    urls = _canonical_match_urls(source_url, flashscore_id)
    stats_payload = renderer.fetch(urls["stats"], wait_selector=None, use_cache=False)
    stats = parse_flashscore_stats_text(str(stats_payload.get("text") or ""))

    summary_payload = renderer.fetch(urls["summary"], wait_selector=None, use_cache=False)
    incidents = _extract_incidents(renderer)
    if stats.get("HY") is None or stats.get("AY") is None:
        cards = _extract_incident_cards(renderer)
        for key in ("HY", "AY", "HR", "AR"):
            if stats.get(key) is None:
                stats[key] = cards.get(key)
    if stats.get("HR") is None:
        stats["HR"] = 0
    if stats.get("AR") is None:
        stats["AR"] = 0
    stats["_evidence"] = {
        "urls": urls,
        "stats_text": str(stats_payload.get("text") or ""),
        "summary_text": str(summary_payload.get("text") or ""),
        "incidents": incidents,
    }

    return stats


def _where_clause(date: Optional[str], leagues: Optional[Iterable[str]], only_missing: bool) -> tuple[str, List[object]]:
    clauses = ["(source_url LIKE '%flashscore.com/match/%' OR flashscore_id IS NOT NULL)"]
    params: List[object] = []
    if date:
        clauses.append("match_date = ?")
        params.append(date)
    if leagues:
        league_list = list(leagues)
        placeholders = ",".join("?" * len(league_list))
        clauses.append(f"league IN ({placeholders})")
        params.extend(league_list)
    if only_missing:
        missing_clause = " OR ".join([f'"{col}" IS NULL' for col in STAT_COLUMNS])
        clauses.append(f"({missing_clause})")
    return " AND ".join(clauses), params


def load_pending_matches(con: sqlite3.Connection, date: Optional[str], leagues: Optional[List[str]], limit: int, only_missing: bool) -> List[sqlite3.Row]:
    where_sql, params = _where_clause(date, leagues, only_missing)
    query = f"""
        SELECT m.id, m.league, m.match_date, m.home_team, m.away_team,
               m.source_url, m.flashscore_id,
               r.actual_fthg AS home_goals,
               r.actual_ftag AS away_goals,
               r.actual_ftr AS actual_ftr
        FROM matches m
        LEFT JOIN match_results r ON r.id = m.id
        WHERE {where_sql}
        ORDER BY m.match_date ASC, m.league ASC, m.home_team ASC
        LIMIT ?
    """
    params.append(limit)
    con.row_factory = sqlite3.Row
    return list(con.execute(query, tuple(params)).fetchall())


def _quality_score(stats: Dict[str, object]) -> float:
    core = ["HC", "AC", "HY", "AY", "HS", "AS", "HST", "AST"]
    present = sum(1 for key in core if stats.get(key) is not None)
    return round(max(0.20, min(1.0, present / len(core))), 3)


def _clean_stats(stats: Dict[str, object]) -> Dict[str, object]:
    return {key: stats.get(key) for key in STAT_COLUMNS if stats.get(key) is not None}


def update_stats(con: sqlite3.Connection, row: sqlite3.Row, stats: Dict[str, object]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    values = [stats.get(key) for key in STAT_COLUMNS]
    any_value = any(value is not None for value in values)
    if not any_value:
        return
    set_clause = ",\n            ".join([f'"{col}" = COALESCE(?, "{col}")' for col in STAT_COLUMNS])

    con.execute(
        f"""
        UPDATE matches
        SET {set_clause},
            source_url = COALESCE(source_url, ?),
            flashscore_id = COALESCE(flashscore_id, ?),
            stats_scraped_at = ?
        WHERE id = ?
        """,
        (*values, row["source_url"], row["flashscore_id"], now, row["id"]),
    )

    con.execute(
        f"""
        UPDATE training_examples
        SET {set_clause},
            source_url = COALESCE(source_url, ?),
            flashscore_id = COALESCE(flashscore_id, ?),
            stats_scraped_at = ?
        WHERE (flashscore_id IS NOT NULL AND flashscore_id = ?)
           OR (
                league = ?
            AND match_date = ?
            AND lower(home_team) = lower(?)
            AND lower(away_team) = lower(?)
           )
        """,
        (
            *values,
            row["source_url"],
            row["flashscore_id"],
            now,
            row["flashscore_id"],
            row["league"],
            row["match_date"],
            row["home_team"],
            row["away_team"],
        ),
    )

    con.commit()
    evidence = stats.get("_evidence", {}) if isinstance(stats.get("_evidence"), dict) else {}
    extracted = _clean_stats(stats)
    upsert_raw_match_evidence({
        "match_id": row["id"],
        "league": row["league"],
        "match_date": row["match_date"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "source": "flashscore",
        "source_url": row["source_url"],
        "evidence_type": "post_match_stats_events",
        "raw": evidence,
        "extracted": extracted,
        "data_quality": _quality_score(stats),
        "scraped_at": now,
    })
    pattern_row = {
        "home_goals": row["home_goals"],
        "away_goals": row["away_goals"],
        "actual_ftr": row["actual_ftr"],
    }
    patterns = derive_post_match_patterns(stats, pattern_row)
    upsert_match_learning_patterns(row["id"], patterns, source="flashscore_stats_backfill")


def run_backfill(
    date: Optional[str] = None,
    leagues: Optional[List[str]] = None,
    limit: int = 100,
    only_missing: bool = True,
    engine: Optional[str] = None,
    sleep_seconds: float = 0.5,
) -> Dict[str, object]:
    init_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = load_pending_matches(con, date=date, leagues=leagues, limit=limit, only_missing=only_missing)

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date": date,
        "leagues": leagues,
        "requested": len(rows),
        "updated": 0,
        "failed": 0,
        "items": [],
    }
    if not rows:
        with open(REPORT_PATH, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        con.close()
        return report

    with BrowserRenderer(engine=engine, headless=True, timeout_ms=int(os.environ.get("BROWSER_TIMEOUT_MS", "30000"))) as renderer:
        for row in rows:
            try:
                source_url = row["source_url"]
                if not source_url:
                    raise ValueError("missing source_url")
                stats = scrape_flashscore_match_stats(renderer, source_url, row["flashscore_id"])
                update_stats(con, row, stats)
                con.commit()
                report["updated"] += 1
                report["items"].append({
                    "id": row["id"],
                    "league": row["league"],
                    "home": row["home_team"],
                    "away": row["away_team"],
                    "stats": _clean_stats(stats),
                })
            except Exception as exc:
                report["failed"] += 1
                report["items"].append({
                    "id": row["id"],
                    "league": row["league"],
                    "home": row["home_team"],
                    "away": row["away_team"],
                    "error": str(exc),
                })
            if sleep_seconds:
                time.sleep(sleep_seconds)

    con.close()
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill cards/corners from Flashscore match-detail pages.")
    parser.add_argument("--date", default=os.environ.get("PIPELINE_START_DATE"))
    parser.add_argument("--leagues", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--all", action="store_true", help="Scrape rows even if stats are already populated.")
    parser.add_argument("--engine", default=os.environ.get("BROWSER_ENGINE"))
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    date = args.date.split("T", 1)[0] if args.date else None
    report = run_backfill(
        date=date,
        leagues=args.leagues,
        limit=args.limit,
        only_missing=not args.all,
        engine=args.engine,
        sleep_seconds=args.sleep,
    )
    print(json.dumps({k: report[k] for k in ("requested", "updated", "failed")}, indent=2))


if __name__ == "__main__":
    main()
