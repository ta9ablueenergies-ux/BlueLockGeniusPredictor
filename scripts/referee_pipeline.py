"""Referee data pipeline: scrape → aggregate → lookup.

Extracts referee names from Flashscore match pages, builds a per-referee
stats table (avg YC, RC, fouls per game), and provides a fast lookup for
inference-time enrichment of the intel dict.

Usage:
    python referee_pipeline.py                # backfill referee from existing URLs
    python referee_pipeline.py --limit 200    # process up to 200 rows
    python referee_pipeline.py --recompute    # recompute stats only (no scraping)
    python referee_pipeline.py --lookup "M. Oliver"  # quick CLI lookup
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")

sys.path.insert(0, SCRIPT_DIR)

# ── DB schema ─────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS referee_stats (
    referee     TEXT PRIMARY KEY,
    matches     INTEGER DEFAULT 0,
    avg_yc      REAL DEFAULT 0.0,
    avg_rc      REAL DEFAULT 0.0,
    avg_fouls   REAL DEFAULT 0.0,
    avg_corners REAL DEFAULT 0.0,
    avg_cards   REAL DEFAULT 0.0,
    league_primary TEXT,
    updated_at  TEXT
);
"""


def init_referee_table(con: sqlite3.Connection) -> None:
    con.executescript(DDL)
    con.commit()


# ── Referee extraction from Flashscore ───────────────────────────────────────

_JS_EXTRACT_REFEREE = """
() => {
    // Try structured match-info items first (.mi__item layout)
    const items = Array.from(document.querySelectorAll('.mi__item, [class*="matchInfo"] li, [class*="match-info"] li'));
    for (const item of items) {
        const label = item.querySelector('.mi__item__name, .mi__item__title, [class*="label"]');
        const val   = item.querySelector('.mi__item__val, .mi__item__value, [class*="value"]');
        if (label && val && /referee/i.test(label.textContent)) {
            const name = val.textContent.trim();
            if (name && name.length > 2) return name;
        }
    }
    // Fallback: grep visible text for "Referee: Name" pattern
    const text = document.body.innerText || document.body.textContent || '';
    const m = text.match(/Referee\\s*:\\s*([^\\n\\r,;]{3,50})/i);
    if (m) return m[1].trim();
    return null;
}
"""


def _js_run(renderer, js: str):
    """Execute JS via whatever browser backend is active."""
    page = getattr(renderer, "_page", None)
    if page is not None:
        return page.evaluate(js)
    sb = getattr(renderer, "_sb", None)
    if sb is not None:
        return sb.driver.execute_script(f"return ({js})()")
    return None


def _build_summary_url(source_url: str, fs_id: Optional[str]) -> str:
    """Construct the match summary URL (referee shown there)."""
    parsed = urlparse(source_url)
    mid = fs_id or parse_qs(parsed.query).get("mid", [None])[0]
    query = f"mid={mid}" if mid else parsed.query
    base_path = parsed.path
    if "/summary/" in base_path:
        base_path = base_path.split("/summary/", 1)[0]
    base_path = base_path.rstrip("/") + "/"
    return urlunparse((parsed.scheme, parsed.netloc, base_path, "", query, ""))


def scrape_referee(renderer, source_url: str, fs_id: Optional[str] = None) -> Optional[str]:
    """Navigate to match summary page and extract referee name."""
    summary_url = _build_summary_url(source_url, fs_id)
    try:
        renderer.fetch(summary_url, wait_selector=None, use_cache=False)
        name = _js_run(renderer, _JS_EXTRACT_REFEREE)
        if name and isinstance(name, str) and len(name.strip()) > 2:
            return name.strip()
    except Exception:
        pass
    return None


# ── Backfill: scrape referee for training_examples with source_url ────────────

def run_backfill(
    con: sqlite3.Connection,
    renderer,
    limit: int = 200,
    sleep_s: float = 1.0,
) -> Tuple[int, int]:
    rows = con.execute("""
        SELECT id, league, match_date, home_team, away_team, source_url, flashscore_id,
               HY, AY, HR, AR, HF, AF, HC, AC
        FROM training_examples
        WHERE (source_url IS NOT NULL OR flashscore_id IS NOT NULL)
          AND (referee IS NULL OR referee = '')
          AND actual_ftr IN ('H','D','A')
        ORDER BY match_date DESC
        LIMIT ?
    """, (limit,)).fetchall()

    updated = failed = 0
    for row in rows:
        (rid, league, match_date, home, away, source_url, fs_id,
         hy, ay, hr, ar, hf, af, hc, ac) = row
        try:
            referee = scrape_referee(renderer, source_url, fs_id)
            if referee:
                con.execute(
                    "UPDATE training_examples SET referee = ? WHERE id = ?",
                    (referee, rid)
                )
                con.commit()
                updated += 1
                print(f"  [{league} {match_date}] {home} v {away}: referee={referee}")
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            print(f"  [{league} {match_date}] {home} v {away}: ERROR {exc}")
        if sleep_s:
            time.sleep(sleep_s)
    return updated, failed


# ── Stats aggregation ─────────────────────────────────────────────────────────

def recompute_stats(con: sqlite3.Connection) -> int:
    """Aggregate per-referee stats from training_examples and upsert referee_stats."""
    rows = con.execute("""
        SELECT
            referee,
            COUNT(*) AS matches,
            AVG(COALESCE(HY, 0) + COALESCE(AY, 0)) AS avg_yc,
            AVG(COALESCE(HR, 0) + COALESCE(AR, 0)) AS avg_rc,
            AVG(COALESCE(HF, 0) + COALESCE(AF, 0)) AS avg_fouls,
            AVG(COALESCE(HC, 0) + COALESCE(AC, 0)) AS avg_corners,
            AVG(COALESCE(HY, 0) + COALESCE(AY, 0) + COALESCE(HR, 0) + COALESCE(AR, 0)) AS avg_cards
        FROM training_examples
        WHERE referee IS NOT NULL AND referee != ''
          AND actual_ftr IN ('H','D','A')
        GROUP BY referee
        HAVING COUNT(*) >= 3
        ORDER BY matches DESC
    """).fetchall()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    written = 0
    for row in rows:
        referee, matches, avg_yc, avg_rc, avg_fouls, avg_corners, avg_cards = row

        # Determine primary league
        league_row = con.execute("""
            SELECT league, COUNT(*) n FROM training_examples
            WHERE referee = ? GROUP BY league ORDER BY n DESC LIMIT 1
        """, (referee,)).fetchone()
        league_primary = league_row[0] if league_row else None

        con.execute("""
            INSERT INTO referee_stats (referee, matches, avg_yc, avg_rc, avg_fouls, avg_corners, avg_cards, league_primary, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(referee) DO UPDATE SET
                matches = excluded.matches,
                avg_yc = excluded.avg_yc,
                avg_rc = excluded.avg_rc,
                avg_fouls = excluded.avg_fouls,
                avg_corners = excluded.avg_corners,
                avg_cards = excluded.avg_cards,
                league_primary = excluded.league_primary,
                updated_at = excluded.updated_at
        """, (referee, matches, avg_yc or 0, avg_rc or 0, avg_fouls or 0, avg_corners or 0, avg_cards or 0, league_primary, now))
        written += 1

    con.commit()
    return written


# ── Lookup cache ──────────────────────────────────────────────────────────────

_REFEREE_CACHE: Dict[str, Dict] = {}
_LEAGUE_DEFAULTS: Dict[str, Dict] = {}
_GLOBAL_DEFAULT: Dict = {}


def _load_cache(con: sqlite3.Connection) -> None:
    global _REFEREE_CACHE, _LEAGUE_DEFAULTS, _GLOBAL_DEFAULT
    rows = con.execute("""
        SELECT referee, matches, avg_yc, avg_rc, avg_fouls, avg_corners, avg_cards, league_primary
        FROM referee_stats
        ORDER BY matches DESC
    """).fetchall()

    all_yc, all_rc, all_fouls, all_corners, count = 0.0, 0.0, 0.0, 0.0, 0
    league_buckets: Dict[str, list] = {}

    for ref, matches, avg_yc, avg_rc, avg_fouls, avg_corners, avg_cards, league in rows:
        d = {"matches": matches, "avg_yc": avg_yc, "avg_rc": avg_rc,
             "avg_fouls": avg_fouls, "avg_corners": avg_corners, "avg_cards": avg_cards}
        _REFEREE_CACHE[ref.lower()] = d
        if league:
            league_buckets.setdefault(league, []).append(d)
        all_yc += avg_yc * matches
        all_rc += avg_rc * matches
        all_fouls += avg_fouls * matches
        all_corners += avg_corners * matches
        count += matches

    # Per-league defaults (weighted avg)
    for league, items in league_buckets.items():
        total = sum(i["matches"] for i in items)
        if total:
            _LEAGUE_DEFAULTS[league] = {
                "avg_yc": sum(i["avg_yc"] * i["matches"] for i in items) / total,
                "avg_rc": sum(i["avg_rc"] * i["matches"] for i in items) / total,
                "avg_fouls": sum(i["avg_fouls"] * i["matches"] for i in items) / total,
                "avg_corners": sum(i["avg_corners"] * i["matches"] for i in items) / total,
                "avg_cards": sum(i["avg_cards"] * i["matches"] for i in items) / total,
            }

    _GLOBAL_DEFAULT = {
        "avg_yc": all_yc / count if count else 3.8,
        "avg_rc": all_rc / count if count else 0.15,
        "avg_fouls": all_fouls / count if count else 22.0,
        "avg_corners": all_corners / count if count else 10.5,
        "avg_cards": (all_yc + all_rc) / count if count else 3.9,
    }


_cache_loaded = False


def get_referee_stats(referee: Optional[str], league: Optional[str] = None) -> Dict:
    """Return referee stats dict. Falls back to league avg then global avg.

    Keys: avg_yc, avg_rc, avg_fouls, avg_corners, avg_cards (all per-game totals).
    """
    global _cache_loaded
    if not _cache_loaded:
        try:
            con = sqlite3.connect(DB_PATH)
            init_referee_table(con)
            _load_cache(con)
            con.close()
        except Exception:
            pass
        _cache_loaded = True

    if referee:
        key = referee.strip().lower()
        if key in _REFEREE_CACHE:
            return dict(_REFEREE_CACHE[key])
        # Try last-name match
        last = key.split()[-1] if " " in key else key
        for k, v in _REFEREE_CACHE.items():
            if k.endswith(last):
                return dict(v)

    if league and league in _LEAGUE_DEFAULTS:
        return dict(_LEAGUE_DEFAULTS[league])

    return dict(_GLOBAL_DEFAULT) if _GLOBAL_DEFAULT else {
        "avg_yc": 3.8, "avg_rc": 0.15, "avg_fouls": 22.0, "avg_corners": 10.5, "avg_cards": 3.9
    }


def invalidate_cache() -> None:
    global _cache_loaded
    _cache_loaded = False
    _REFEREE_CACHE.clear()
    _LEAGUE_DEFAULTS.clear()
    _GLOBAL_DEFAULT.clear()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Referee stats pipeline.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--recompute", action="store_true", help="Recompute stats only, no scraping")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--lookup", default=None, help="Look up stats for a referee name")
    args = parser.parse_args()

    con = sqlite3.connect(DB_PATH)
    init_referee_table(con)

    if args.lookup:
        _load_cache(con)
        stats = get_referee_stats(args.lookup)
        print(f"Referee: {args.lookup}")
        for k, v in stats.items():
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
        con.close()
        return

    if not args.recompute:
        sys.path.insert(0, SCRIPT_DIR)
        from browser_scraper import BrowserRenderer  # noqa: E402

        pending = con.execute("""
            SELECT count(*) FROM training_examples
            WHERE (source_url IS NOT NULL OR flashscore_id IS NOT NULL)
              AND (referee IS NULL OR referee = '')
              AND actual_ftr IN ('H','D','A')
        """).fetchone()[0]
        print(f"[Referee] {pending} rows with URL but no referee name")

        if pending > 0:
            with BrowserRenderer(engine=args.engine, headless=True) as renderer:
                updated, failed = run_backfill(con, renderer, limit=args.limit, sleep_s=args.sleep)
            print(f"[Referee] Backfill done: updated={updated}, failed={failed}")

    count = recompute_stats(con)
    print(f"[Referee] Stats table: {count} referees written")

    total = con.execute("SELECT count(*) FROM training_examples WHERE referee IS NOT NULL AND referee != ''").fetchone()[0]
    print(f"[Referee] training_examples with referee: {total}")
    top = con.execute("SELECT referee, matches FROM referee_stats ORDER BY matches DESC LIMIT 5").fetchall()
    for ref, n in top:
        print(f"  {ref}: {n} matches")

    con.close()


if __name__ == "__main__":
    main()
