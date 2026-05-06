"""Flashscore URL discovery + xG stats backfill for historical training_examples.

Two-phase operation:
  Phase 1 — scrape stats for rows that already have source_url/flashscore_id.
  Phase 2 — discover URLs for rows without them by scraping Flashscore league
             results pages grouped by (league, date), then scrape stats.

Run:
    python flashscore_url_discovery.py                      # Big 5, most recent first
    python flashscore_url_discovery.py --leagues PremierLeague Bundesliga
    python flashscore_url_discovery.py --start 2023-01-01 --end 2025-12-31
    python flashscore_url_discovery.py --phase1-only        # only already-have-URL rows
    python flashscore_url_discovery.py --limit 500          # process N (league,date) batches
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
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, parse_qs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "flashscore_url_discovery_checkpoint.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "web", "data", "browser_sources.json")

sys.path.insert(0, SCRIPT_DIR)
from browser_scraper import BrowserRenderer, _extract_flashscore_matches_from_dom, _clean_flashscore_team_name  # noqa: E402
from team_name_normalizer import canonical_team_name  # noqa: E402

# Priority order for leagues (Big 5 first, then others)
LEAGUE_PRIORITY = [
    "PremierLeague", "LaLiga", "SerieA", "Bundesliga", "Ligue1",
    "Championship", "LaLiga2", "SerieB", "Ligue2", "2Bundesliga",
    "Eredivisie", "LigaNOS", "BelgianProLeague", "ScottishPremiership",
    "ScottishPrem", "SuperLig", "ChampionsLeague",
]

STAT_COLUMNS = [
    "HY", "AY", "HR", "AR", "HC", "AC",
    "HS", "AS", "HST", "AST", "HF", "AF", "HO", "AO",
    "HPoss", "APoss", "HXG", "AXG", "HBC", "ABC",
]


# ── URL template loading ──────────────────────────────────────────────────────

def _load_url_templates() -> Dict[str, str]:
    """Return {league: url_template} from browser_sources.json."""
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    sources = data.get("sources", data) if isinstance(data, dict) else data
    out: Dict[str, str] = {}
    for s in (sources if isinstance(sources, list) else []):
        league = s.get("league")
        templates = s.get("url_templates", [])
        if league and templates:
            out[league] = templates[0]
    return out


def _build_url(template: str, match_date: str) -> str:
    try:
        dt = datetime.strptime(match_date[:10], "%Y-%m-%d")
        ddmmyyyy = dt.strftime("%d%m%Y")
        return template.replace("{date_ddmmyyyy}", ddmmyyyy)
    except Exception:
        return template


# ── Team name fuzzy matching ──────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _best_match(fs_name: str, candidates: List[str], threshold: float = 0.72) -> Optional[str]:
    """Find best fuzzy match for a Flashscore team name among DB candidates."""
    if not fs_name or not candidates:
        return None
    fs_norm = re.sub(r"[^a-z0-9 ]", "", fs_name.lower()).strip()
    best_score = 0.0
    best = None
    for c in candidates:
        c_norm = re.sub(r"[^a-z0-9 ]", "", c.lower()).strip()
        score = _similarity(fs_norm, c_norm)
        if score > best_score:
            best_score = score
            best = c
    return best if best_score >= threshold else None


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> Set[str]:
    """Return set of already-processed 'league|date' combos."""
    if not os.path.exists(CHECKPOINT_PATH):
        return set()
    try:
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("done", []))
    except Exception:
        return set()


def _save_checkpoint(done: Set[str]) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"done": sorted(done), "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)


# ── Phase 1: scrape stats for rows with existing source_url ──────────────────

def _phase1_scrape(con: sqlite3.Connection, renderer: BrowserRenderer, limit: int, sleep_s: float) -> Tuple[int, int]:
    """Scrape xG stats for training_examples rows that already have a source_url."""
    rows = con.execute("""
        SELECT id, league, match_date, home_team, away_team, source_url, flashscore_id
        FROM training_examples
        WHERE HXG IS NULL
          AND actual_ftr IN ('H','D','A')
          AND (source_url IS NOT NULL OR flashscore_id IS NOT NULL)
        ORDER BY match_date DESC
        LIMIT ?
    """, (limit,)).fetchall()

    updated = failed = 0
    for row in rows:
        row_id, league, match_date, home, away, source_url, fs_id = row
        try:
            from flashscore_stats_backfill import scrape_flashscore_match_stats, parse_flashscore_stats_text  # noqa
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(source_url)
            mid = fs_id or parse_qs(parsed.query).get("mid", [None])[0]
            query = f"mid={mid}" if mid else parsed.query
            base_path = parsed.path
            if "/summary/" in base_path:
                base_path = base_path.split("/summary/", 1)[0]
            base_path = base_path.rstrip("/") + "/"
            stats_url = urlunparse((parsed.scheme, parsed.netloc, base_path + "summary/stats/overall/", "", query, ""))

            payload = renderer.fetch(stats_url, wait_selector=None, use_cache=False)
            stats = parse_flashscore_stats_text(str(payload.get("text") or ""))
            hxg = stats.get("HXG")
            axg = stats.get("AXG")
            if hxg is not None or axg is not None:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                set_parts = []
                vals = []
                for col in STAT_COLUMNS:
                    v = stats.get(col)
                    if v is not None:
                        set_parts.append(f'"{col}" = COALESCE(?, "{col}")')
                        vals.append(v)
                if set_parts:
                    set_parts.append("stats_scraped_at = ?")
                    vals.append(now)
                    vals.append(row_id)
                    con.execute(f"UPDATE training_examples SET {', '.join(set_parts)} WHERE id = ?", vals)
                    con.commit()
                    updated += 1
                    print(f"    [P1] {league} {match_date} {home} v {away}: xG={hxg}/{axg}")
            else:
                failed += 1
                print(f"    [P1] {league} {match_date} {home} v {away}: no xG found")
        except Exception as exc:
            failed += 1
            print(f"    [P1] {league} {match_date} {home} v {away}: ERROR {exc}")
        if sleep_s:
            time.sleep(sleep_s)
    return updated, failed


# ── Phase 2: URL discovery + stats scrape ────────────────────────────────────

def _get_pending_batches(
    con: sqlite3.Connection,
    leagues: Optional[List[str]],
    start_date: str,
    end_date: str,
    done: Set[str],
) -> List[Tuple[str, str]]:
    """Return ordered (league, date) tuples that still have rows missing xG."""
    where_leagues = ""
    params: list = [start_date, end_date]
    if leagues:
        ph = ",".join("?" * len(leagues))
        where_leagues = f"AND league IN ({ph})"
        params.extend(leagues)

    rows = con.execute(f"""
        SELECT DISTINCT league, match_date
        FROM training_examples
        WHERE HXG IS NULL
          AND actual_ftr IN ('H','D','A')
          AND source_url IS NULL
          AND flashscore_id IS NULL
          AND match_date >= ?
          AND match_date <= ?
          {where_leagues}
        ORDER BY match_date DESC
    """, params).fetchall()

    # Sort by league priority, then by date desc (already sorted by date)
    def priority(league: str) -> int:
        try:
            return LEAGUE_PRIORITY.index(league)
        except ValueError:
            return 999

    batches = [(league, date) for league, date in rows if f"{league}|{date}" not in done]
    batches.sort(key=lambda x: (priority(x[0]), x[1]), reverse=False)
    # Re-sort: by priority asc, date desc within same priority
    batches.sort(key=lambda x: (priority(x[0]), x[1] if x[1] else ""), reverse=False)
    # Re-apply date desc within same league
    from itertools import groupby
    result: List[Tuple[str, str]] = []
    for league_key, group in groupby(batches, key=lambda x: x[0]):
        result.extend(sorted(group, key=lambda x: x[1], reverse=True))
    return result


def _get_te_rows_for_date(con: sqlite3.Connection, league: str, match_date: str) -> List[dict]:
    """Get training_examples rows for a specific league+date needing URL discovery."""
    rows = con.execute("""
        SELECT id, home_team, away_team
        FROM training_examples
        WHERE league = ? AND match_date = ?
          AND HXG IS NULL AND actual_ftr IN ('H','D','A')
          AND source_url IS NULL AND flashscore_id IS NULL
    """, (league, match_date)).fetchall()
    return [{"id": r[0], "home_team": r[1], "away_team": r[2]} for r in rows]


def _match_fs_to_te(
    fs_matches: List[Dict],
    te_rows: List[dict],
    league: str,
) -> List[Tuple[dict, dict]]:
    """Match Flashscore extracted matches to training_examples rows."""
    matched: List[Tuple[dict, dict]] = []
    used_te: Set[int] = set()

    for fs in fs_matches:
        fs_home = str(fs.get("home_team") or "").strip()
        fs_away = str(fs.get("away_team") or "").strip()
        if not fs_home or not fs_away:
            continue

        best_te = None
        best_score = 0.0
        for te in te_rows:
            if te["id"] in used_te:
                continue
            te_home = str(te.get("home_team") or "").strip()
            te_away = str(te.get("away_team") or "").strip()
            score = (_similarity(fs_home, te_home) + _similarity(fs_away, te_away)) / 2
            if score > best_score:
                best_score = score
                best_te = te

        if best_te is not None and best_score >= 0.65:
            matched.append((fs, best_te))
            used_te.add(best_te["id"])

    return matched


def _scrape_stats_for_url(renderer: BrowserRenderer, source_url: str, fs_id: Optional[str]) -> Dict:
    """Fetch xG and other stats from a Flashscore match stats page."""
    from flashscore_stats_backfill import parse_flashscore_stats_text  # noqa
    from urllib.parse import urlunparse

    parsed = urlparse(source_url)
    mid = fs_id or parse_qs(parsed.query).get("mid", [None])[0]
    query = f"mid={mid}" if mid else parsed.query
    base_path = parsed.path
    if "/summary/" in base_path:
        base_path = base_path.split("/summary/", 1)[0]
    base_path = base_path.rstrip("/") + "/"
    stats_url = urlunparse((parsed.scheme, parsed.netloc, base_path + "summary/stats/overall/", "", query, ""))

    payload = renderer.fetch(stats_url, wait_selector=None, use_cache=False)
    return parse_flashscore_stats_text(str(payload.get("text") or ""))


def _update_te_stats(con: sqlite3.Connection, te_id: int, source_url: str, fs_id: Optional[str], stats: Dict) -> bool:
    """Update training_examples row with URL + stats. Returns True if any stat written."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    set_parts = [
        "source_url = COALESCE(source_url, ?)",
        "flashscore_id = COALESCE(flashscore_id, ?)",
    ]
    vals: list = [source_url, fs_id]

    for col in STAT_COLUMNS:
        v = stats.get(col)
        if v is not None:
            set_parts.append(f'"{col}" = COALESCE(?, "{col}")')
            vals.append(v)

    has_stats = any(stats.get(c) is not None for c in STAT_COLUMNS)
    if has_stats:
        set_parts.append("stats_scraped_at = ?")
        vals.append(now)

    vals.append(te_id)
    con.execute(f"UPDATE training_examples SET {', '.join(set_parts)} WHERE id = ?", vals)
    con.commit()
    return has_stats


def _phase2_discover(
    con: sqlite3.Connection,
    renderer: BrowserRenderer,
    url_templates: Dict[str, str],
    batches: List[Tuple[str, str]],
    limit: int,
    sleep_s: float,
    done: Set[str],
) -> Tuple[int, int, int]:
    """Discover URLs and scrape stats for (league, date) batches."""
    urls_found = stats_updated = failed = 0
    processed = 0

    for league, match_date in batches:
        if processed >= limit:
            break
        batch_key = f"{league}|{match_date}"

        template = url_templates.get(league)
        if not template:
            print(f"    [P2] No URL template for {league}, skipping")
            done.add(batch_key)
            continue

        te_rows = _get_te_rows_for_date(con, league, match_date)
        if not te_rows:
            done.add(batch_key)
            continue

        page_url = _build_url(template, match_date)
        print(f"  [{league} {match_date}] {len(te_rows)} rows → {page_url[:70]}")

        try:
            renderer.fetch(page_url, wait_selector=None, use_cache=False)
            fs_matches = _extract_flashscore_matches_from_dom(renderer, match_date, league, f"discovery-{league}")
        except Exception as exc:
            print(f"    [P2] Page fetch failed: {exc}")
            failed += len(te_rows)
            done.add(batch_key)
            processed += 1
            if sleep_s:
                time.sleep(sleep_s * 2)
            continue

        if not fs_matches:
            print(f"    [P2] No matches extracted from DOM")
            done.add(batch_key)
            processed += 1
            if sleep_s:
                time.sleep(sleep_s)
            continue

        matched_pairs = _match_fs_to_te(fs_matches, te_rows, league)
        print(f"    [P2] DOM extracted {len(fs_matches)}, matched {len(matched_pairs)}/{len(te_rows)}")

        for fs, te in matched_pairs:
            source_url = str(fs.get("source_url") or "").strip()
            fs_id = str(fs.get("source_id") or "").strip() or None
            if not source_url and not fs_id:
                continue
            urls_found += 1

            try:
                stats = _scrape_stats_for_url(renderer, source_url, fs_id) if source_url else {}
                hxg = stats.get("HXG")
                axg = stats.get("AXG")
                has_stats = _update_te_stats(con, te["id"], source_url, fs_id, stats)
                if has_stats:
                    stats_updated += 1
                print(f"      {te['home_team']} v {te['away_team']}: xG={hxg}/{axg} url={'ok' if source_url else 'none'}")
            except Exception as exc:
                print(f"      {te['home_team']} v {te['away_team']}: stats ERROR {exc}")
                # Still save the URL even if stats scrape failed
                try:
                    if source_url or fs_id:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        con.execute(
                            "UPDATE training_examples SET source_url=COALESCE(source_url,?), flashscore_id=COALESCE(flashscore_id,?) WHERE id=?",
                            (source_url or None, fs_id, te["id"])
                        )
                        con.commit()
                except Exception:
                    pass

            if sleep_s:
                time.sleep(sleep_s * 0.5)

        unmatched = len(te_rows) - len(matched_pairs)
        if unmatched:
            failed += unmatched

        done.add(batch_key)
        processed += 1
        _save_checkpoint(done)

        if sleep_s:
            time.sleep(sleep_s)

    return urls_found, stats_updated, failed


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Discover Flashscore URLs and backfill xG for historical training_examples.")
    parser.add_argument("--leagues", nargs="+", default=None, help="Filter to specific leagues")
    parser.add_argument("--start", default="2019-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=200, help="Max (league, date) batches to process in phase 2")
    parser.add_argument("--phase1-only", action="store_true", help="Only scrape stats for rows with existing URLs")
    parser.add_argument("--phase2-only", action="store_true", help="Skip phase 1, go straight to URL discovery")
    parser.add_argument("--sleep", type=float, default=1.5, help="Sleep seconds between requests")
    parser.add_argument("--engine", default=None, help="Browser engine (playwright/seleniumbase)")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Clear discovery checkpoint and restart")
    args = parser.parse_args()

    if args.reset_checkpoint and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("[Discovery] Checkpoint cleared.")

    con = sqlite3.connect(DB_PATH)
    url_templates = _load_url_templates()
    done = _load_checkpoint()

    print(f"[Discovery] URL templates loaded for {len(url_templates)} leagues")
    print(f"[Discovery] Checkpoint: {len(done)} (league,date) combos already processed")

    total_p1_updated = total_p1_failed = 0
    total_urls = total_stats = total_failed = 0

    with BrowserRenderer(engine=args.engine, headless=True) as renderer:

        # Phase 1 — existing URLs
        if not args.phase2_only:
            p1_count = con.execute("""
                SELECT count(*) FROM training_examples
                WHERE HXG IS NULL AND actual_ftr IN ('H','D','A')
                  AND (source_url IS NOT NULL OR flashscore_id IS NOT NULL)
            """).fetchone()[0]
            if p1_count > 0:
                print(f"\n[Phase 1] {p1_count} rows with existing URLs, missing xG")
                u, f = _phase1_scrape(con, renderer, limit=min(p1_count, 500), sleep_s=args.sleep)
                total_p1_updated, total_p1_failed = u, f
                print(f"[Phase 1] Done: updated={u}, failed={f}")
            else:
                print("[Phase 1] No rows with existing URLs need stats scraping.")

        # Phase 2 — URL discovery
        if not args.phase1_only:
            batches = _get_pending_batches(con, args.leagues, args.start, args.end, done)
            print(f"\n[Phase 2] {len(batches)} (league, date) batches pending URL discovery (limit={args.limit})")

            if batches:
                urls, stats, fail = _phase2_discover(
                    con, renderer, url_templates, batches,
                    limit=args.limit, sleep_s=args.sleep, done=done,
                )
                total_urls, total_stats, total_failed = urls, stats, fail
                _save_checkpoint(done)
                print(f"\n[Phase 2] Done: urls_found={urls}, stats_updated={stats}, failed={fail}")
            else:
                print("[Phase 2] All batches already processed. Run with --reset-checkpoint to restart.")

    con.close()
    print(f"\n[Summary] Phase1: updated={total_p1_updated} failed={total_p1_failed} | "
          f"Phase2: urls={total_urls} stats={total_stats} failed={total_failed}")


if __name__ == "__main__":
    main()
