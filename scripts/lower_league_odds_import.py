"""Import historical odds for lower leagues from football-data.co.uk CSV archive.

football-data.co.uk publishes free season CSV files with B365H/B365D/B365A
odds for dozens of leagues. This script downloads them and backfills
training_examples.home_odds / draw_odds / away_odds where currently NULL.

Usage:
    python lower_league_odds_import.py                           # all lower leagues, all seasons
    python lower_league_odds_import.py --leagues Championship    # single league
    python lower_league_odds_import.py --seasons 2425 2324 2223  # specific seasons
    python lower_league_odds_import.py --dry-run                 # count matches without writing
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")

# Local CSV drop folder — user can manually download from football-data.co.uk
# and place files as: data/fdco_csvs/E1_2425.csv, D2_2324.csv, etc.
LOCAL_CSV_DIR = os.path.join(PROJECT_ROOT, "data", "fdco_csvs")

# football-data.co.uk CSV URL template
FDCO_BASE = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

# Lower leagues only — Big 5 already have odds from FDO API feed
LEAGUE_CODES: Dict[str, str] = {
    "Championship":      "E1",
    "2Bundesliga":       "D2",
    "Ligue2":            "F2",
    "SerieB":            "I2",
    "LaLiga2":           "SP2",
    "BelgianProLeague":  "B1",
    "Eredivisie":        "N1",
    "LigaNOS":           "P1",
    "SuperLig":          "T1",
    "ScottishPremiership": "SC0",
    "ScottishPrem":      "SC0",
}

# Seasons to try (SSYY format, most recent first)
ALL_SEASONS = [
    "2526", "2425", "2324", "2223", "2122", "2021",
    "1920", "1819", "1718", "1617", "1516",
]


def _season_year_range(season: str) -> Tuple[int, int]:
    """'2425' → (2024, 2025)"""
    try:
        yy1, yy2 = int(season[:2]), int(season[2:])
        base = 2000 if yy1 >= 10 else 1900
        return base + yy1, base + yy2
    except Exception:
        return 2000, 2001


def _parse_date(value: str) -> Optional[str]:
    """Parse DD/MM/YY or DD/MM/YYYY → YYYY-MM-DD."""
    text = str(value or "").strip()
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _best_team_match(name: str, candidates: List[str]) -> Optional[str]:
    if not name or not candidates:
        return None
    best_score, best = 0.0, None
    n = re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()
    for c in candidates:
        c_n = re.sub(r"[^a-z0-9 ]", "", c.lower()).strip()
        score = _similarity(n, c_n)
        if score > best_score:
            best_score, best = score, c
    return best if best_score >= 0.70 else None


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.football-data.co.uk/englandm.php",
}


def _load_local_csv(code: str, season: str) -> Optional[str]:
    """Check LOCAL_CSV_DIR for a pre-downloaded file named {CODE}_{season}.csv."""
    if not os.path.isdir(LOCAL_CSV_DIR):
        return None
    candidates = [
        os.path.join(LOCAL_CSV_DIR, f"{code}_{season}.csv"),
        os.path.join(LOCAL_CSV_DIR, f"{code.lower()}_{season}.csv"),
        os.path.join(LOCAL_CSV_DIR, f"{code}.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
    return None


def download_csv(league: str, season: str) -> Optional[str]:
    """Return CSV text for league+season. Checks local drop folder first, then HTTP."""
    code = LEAGUE_CODES.get(league)
    if not code:
        return None

    # 1. Local drop folder (user manually downloaded CSVs)
    local = _load_local_csv(code, season)
    if local:
        return local

    # 2. HTTP download (may be blocked by Cloudflare on some IPs)
    url = FDCO_BASE.format(season=season, code=code)
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(url, timeout=15, verify=False, headers=_HEADERS)
        if resp.status_code == 200 and len(resp.content) > 500:
            first_line = resp.text.split("\n")[0]
            if "Div" in first_line or "HomeTeam" in first_line or "Date" in first_line:
                # Cache locally for future use
                os.makedirs(LOCAL_CSV_DIR, exist_ok=True)
                cache_path = os.path.join(LOCAL_CSV_DIR, f"{code}_{season}.csv")
                with open(cache_path, "w", encoding="utf-8") as cf:
                    cf.write(resp.text)
                return resp.text
        elif resp.status_code == 403:
            print(f"    [Odds] 403 blocked for {league} {season}. "
                  f"Download manually from https://www.football-data.co.uk/mmz4281/{season}/{code}.csv "
                  f"and place in {LOCAL_CSV_DIR}/{code}_{season}.csv")
        return None
    except Exception:
        return None


def parse_csv(text: str, league: str) -> List[Dict]:
    """Parse football-data.co.uk CSV, return list of match dicts."""
    import csv
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        date_str = _parse_date(row.get("Date") or row.get("date") or "")
        home = str(row.get("HomeTeam") or row.get("Home") or "").strip()
        away = str(row.get("AwayTeam") or row.get("Away") or "").strip()
        if not date_str or not home or not away:
            continue

        # Prefer B365 odds, fallback to Bet365/PS (Pinnacle)
        def _odd(keys):
            for k in keys:
                v = row.get(k, "").strip()
                if v:
                    try:
                        f = float(v)
                        if f > 1.0:
                            return f
                    except Exception:
                        pass
            return None

        h_odd = _odd(["B365H", "BbAvH", "PSH", "WHH"])
        d_odd = _odd(["B365D", "BbAvD", "PSD", "WHD"])
        a_odd = _odd(["B365A", "BbAvA", "PSA", "WHA"])

        if h_odd is None and d_odd is None and a_odd is None:
            continue

        rows.append({
            "league": league,
            "match_date": date_str,
            "home_team": home,
            "away_team": away,
            "home_odds": h_odd,
            "draw_odds": d_odd,
            "away_odds": a_odd,
        })
    return rows


def _get_te_candidates(con: sqlite3.Connection, league: str, date: str) -> List[Dict]:
    rows = con.execute("""
        SELECT id, home_team, away_team, home_odds
        FROM training_examples
        WHERE league = ? AND match_date = ?
          AND actual_ftr IN ('H','D','A')
    """, (league, date)).fetchall()
    return [{"id": r[0], "home_team": r[1], "away_team": r[2], "has_odds": r[3] is not None} for r in rows]


def import_season(
    con: sqlite3.Connection,
    league: str,
    season: str,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Download + import one season. Returns (matched, updated)."""
    text = download_csv(league, season)
    if not text:
        return 0, 0

    csv_rows = parse_csv(text, league)
    if not csv_rows:
        return 0, 0

    # Group by date for efficient DB lookups
    by_date: Dict[str, List[Dict]] = {}
    for r in csv_rows:
        by_date.setdefault(r["match_date"], []).append(r)

    matched = updated = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for date, csv_day in by_date.items():
        te_candidates = _get_te_candidates(con, league, date)
        if not te_candidates:
            continue

        te_home_names = [c["home_team"] for c in te_candidates]

        for csv_match in csv_day:
            # Fuzzy match home team
            best_home = _best_team_match(csv_match["home_team"], te_home_names)
            if not best_home:
                continue
            te_row = next((c for c in te_candidates if c["home_team"] == best_home), None)
            if not te_row:
                continue

            # Verify away team similarity
            if _similarity(
                re.sub(r"[^a-z0-9 ]", "", csv_match["away_team"].lower()),
                re.sub(r"[^a-z0-9 ]", "", te_row["away_team"].lower()),
            ) < 0.60:
                continue

            matched += 1
            if te_row["has_odds"]:
                continue  # already have odds, skip

            if not dry_run:
                con.execute("""
                    UPDATE training_examples
                    SET home_odds  = COALESCE(home_odds,  ?),
                        draw_odds  = COALESCE(draw_odds,  ?),
                        away_odds  = COALESCE(away_odds,  ?),
                        market_prob_home = COALESCE(market_prob_home, ?),
                        market_prob_draw = COALESCE(market_prob_draw, ?),
                        market_prob_away = COALESCE(market_prob_away, ?),
                        updated_at = ?
                    WHERE id = ?
                """, (
                    csv_match["home_odds"], csv_match["draw_odds"], csv_match["away_odds"],
                    # Implied probs (normalized)
                    *_implied_probs(csv_match["home_odds"], csv_match["draw_odds"], csv_match["away_odds"]),
                    now, te_row["id"],
                ))
                updated += 1

    if not dry_run:
        con.commit()
    return matched, updated


def _implied_probs(h: Optional[float], d: Optional[float], a: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        ih = 1.0 / h if h and h > 1 else None
        id_ = 1.0 / d if d and d > 1 else None
        ia = 1.0 / a if a and a > 1 else None
        total = (ih or 0) + (id_ or 0) + (ia or 0)
        if total > 0:
            return (
                round(ih / total, 4) if ih else None,
                round(id_ / total, 4) if id_ else None,
                round(ia / total, 4) if ia else None,
            )
    except Exception:
        pass
    return None, None, None


def run_import(
    leagues: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
    dry_run: bool = False,
    sleep_s: float = 0.3,
) -> Dict:
    leagues = leagues or list(LEAGUE_CODES.keys())
    seasons = seasons or ALL_SEASONS
    # Deduplicate (ScottishPrem/ScottishPremiership both map to SC0)
    seen_codes: set = set()
    deduped_leagues = []
    for l in leagues:
        code = LEAGUE_CODES.get(l)
        if code and code not in seen_codes:
            seen_codes.add(code)
            deduped_leagues.append(l)
    leagues = deduped_leagues

    con = sqlite3.connect(DB_PATH)
    report = {"leagues": {}, "total_matched": 0, "total_updated": 0, "dry_run": dry_run}

    for league in leagues:
        report["leagues"][league] = {"seasons": {}, "matched": 0, "updated": 0}
        for season in seasons:
            y1, y2 = _season_year_range(season)
            # Check if we already have odds for this league in this year range
            existing = con.execute("""
                SELECT COUNT(*) FROM training_examples
                WHERE league = ? AND match_date BETWEEN ? AND ?
                  AND home_odds IS NOT NULL AND home_odds > 1
            """, (league, f"{y1}-01-01", f"{y2}-12-31")).fetchone()[0]
            total = con.execute("""
                SELECT COUNT(*) FROM training_examples
                WHERE league = ? AND match_date BETWEEN ? AND ?
                  AND actual_ftr IN ('H','D','A')
            """, (league, f"{y1}-01-01", f"{y2}-12-31")).fetchone()[0]

            if existing > 0 and existing >= total * 0.9:
                print(f"  [{league} {season}] skipping — {existing}/{total} already have odds")
                continue

            matched, updated = import_season(con, league, season, dry_run=dry_run)
            report["leagues"][league]["seasons"][season] = {"matched": matched, "updated": updated}
            report["leagues"][league]["matched"] += matched
            report["leagues"][league]["updated"] += updated
            report["total_matched"] += matched
            report["total_updated"] += updated

            if matched or updated:
                print(f"  [{league} {season}] matched={matched} updated={updated}")
            if sleep_s:
                time.sleep(sleep_s)

    con.close()
    return report


def print_download_urls(leagues: Optional[List[str]] = None, seasons: Optional[List[str]] = None) -> None:
    """Print all CSV download URLs to paste in browser."""
    leagues = leagues or list(LEAGUE_CODES.keys())
    seasons = seasons or ALL_SEASONS[:6]  # Last 6 seasons by default
    print(f"\nDownload these CSVs from your browser and place in:")
    print(f"  {LOCAL_CSV_DIR}\n")
    seen_codes: set = set()
    for league in leagues:
        code = LEAGUE_CODES.get(league)
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)
        for season in seasons:
            url = FDCO_BASE.format(season=season, code=code)
            dest = os.path.join(LOCAL_CSV_DIR, f"{code}_{season}.csv")
            if os.path.isfile(dest):
                print(f"  [OK]  {dest}")
            else:
                print(f"  {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import lower league odds from football-data.co.uk.")
    parser.add_argument("--leagues", nargs="+", default=None)
    parser.add_argument("--seasons", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--show-urls", action="store_true", help="Print all CSV download URLs")
    args = parser.parse_args()

    if args.show_urls:
        print_download_urls(args.leagues, args.seasons)
        return

    print(f"[Odds Import] Starting {'DRY RUN' if args.dry_run else 'LIVE'}")
    report = run_import(
        leagues=args.leagues,
        seasons=args.seasons,
        dry_run=args.dry_run,
        sleep_s=args.sleep,
    )
    print(f"\n[Odds Import] Done: matched={report['total_matched']} updated={report['total_updated']}")

    # Show per-league summary
    for league, data in report["leagues"].items():
        if data["matched"] or data["updated"]:
            print(f"  {league}: matched={data['matched']} updated={data['updated']}")


if __name__ == "__main__":
    main()
