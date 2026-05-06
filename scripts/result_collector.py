# ANTIGRAVITY RESULT COLLECTOR V1.0 (U5)
# Fetches actual match results and writes them to the match_results table.
# Also closes the shadow_log.json PENDING loop and writes CLV tracking records.
import os
import sys
import sqlite3
import json
import hashlib
import argparse
import re
from collections import defaultdict
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'web', 'data', 'intelligence_hub.db')
SHADOW_LOG = os.path.join(PROJECT_ROOT, 'web', 'data', 'shadow_log.json')

# Add scripts to path
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))


def fetch_actual_result_from_fdo(league, match_date, home_team, away_team):
    """
    Try to fetch actual result from FDO (historical file in data/football_data.zip).
    Returns (fthg, ftag, ftr) or None.
    """
    import zipfile
    import io
    import pandas as pd

    league_prefix_map = {
        'PremierLeague': 'PremierLeague', 'LaLiga': 'LaLiga', 'SerieA': 'SerieA',
        'Bundesliga': 'Bundesliga', 'Ligue1': 'Ligue1'
    }
    prefix = league_prefix_map.get(league, league)
    zip_path = os.path.join(PROJECT_ROOT, 'data', 'football_data.zip')

    if not os.path.exists(zip_path):
        return None

    try:
        with zipfile.ZipFile(zip_path) as z:
            files = sorted([n for n in z.namelist() if n.startswith(prefix + '_')])
            for fname in reversed(files):  # most recent first
                with z.open(fname) as f:
                    df = pd.read_csv(f)
                    df.columns = [c.strip() for c in df.columns]

                # Try multiple date formats
                for fmt in ('%d/%m/%Y', '%Y-%m-%d'):
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format=fmt)
                        target_date = datetime.strptime(match_date, '%Y-%m-%d')
                        break
                    except Exception:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        break

                # Fuzzy team match
                home_lower = home_team.lower()
                away_lower = away_team.lower()
                for _, row in df.iterrows():
                    ht = str(row.get('HomeTeam', '')).lower()
                    at = str(row.get('AwayTeam', '')).lower()
                    if (home_lower in ht or ht in home_lower) and (away_lower in at or at in away_lower):
                        # Check date proximity (within 2 days)
                        try:
                            row_date = row.get('Date')
                            if pd.isna(row_date):
                                continue
                            match_dt = datetime.strptime(match_date, '%Y-%m-%d')
                            if abs((row_date - match_dt).days) <= 2:
                                fthg = int(row['FTHG']) if pd.notna(row['FTHG']) else None
                                ftag = int(row['FTAG']) if pd.notna(row['FTAG']) else None
                                ftr = row.get('FTR', '')
                                if fthg is not None and ftag is not None:
                                    return fthg, ftag, ftr
                        except Exception:
                            continue

        return None
    except Exception as e:
        print(f"  [ResultCollect] FDO lookup failed: {e}")
        return None


def fetch_closing_odds_from_shadow(event_id):
    """
    Get closing odds from shadow_log entry for a prediction.
    """
    if not os.path.exists(SHADOW_LOG):
        return None, None, None

    try:
        with open(SHADOW_LOG, 'r') as f:
            entries = json.load(f)
        for e in entries:
            if e.get('id') == event_id:
                return (
                    e.get('closing_odds'),
                    e.get('open_odds'),
                    e.get('odds')
                )
    except Exception:
        pass
    return None, None, None


def write_result_to_sqlite(m_id, fthg, ftag, ftr, closing_home=None, closing_draw=None, closing_away=None):
    """Write result to match_results table via persistence_manager."""
    from persistence_manager import write_match_result
    write_match_result(
        m_id=m_id,
        actual_fthg=fthg,
        actual_ftag=ftag,
        actual_ftr=ftr,
        closing_home=closing_home,
        closing_draw=closing_draw,
        closing_away=closing_away
    )


def _norm_team(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _upsert_closed_training_example(row, fthg, ftag, ftr, source="flashscore_closed_prediction"):
    """Persist a resolved prediction as supervised training data."""
    from persistence_manager import upsert_training_example

    try:
        features = json.loads(row["intel_raw"] or "{}")
    except Exception:
        features = {}

    example = {
        "id": f"closed_{row['id']}",
        "league": row["league"],
        "match_date": row["match_date"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "home_goals": fthg,
        "away_goals": ftag,
        "actual_ftr": ftr,
        "source": source,
        "source_confidence": row["source_confidence"] or 0.78,
        "features": features,
        "flashscore_id": row["flashscore_id"],
        "source_url": row["source_url"],
        "HY": row["HY"],
        "AY": row["AY"],
        "HR": row["HR"],
        "AR": row["AR"],
        "HC": row["HC"],
        "AC": row["AC"],
        "HS": row["HS"] if "HS" in row.keys() else None,
        "AS": row["AS"] if "AS" in row.keys() else None,
        "HST": row["HST"] if "HST" in row.keys() else None,
        "AST": row["AST"] if "AST" in row.keys() else None,
        "HF": row["HF"] if "HF" in row.keys() else None,
        "AF": row["AF"] if "AF" in row.keys() else None,
        "HO": row["HO"] if "HO" in row.keys() else None,
        "AO": row["AO"] if "AO" in row.keys() else None,
        "HPoss": row["HPoss"] if "HPoss" in row.keys() else None,
        "APoss": row["APoss"] if "APoss" in row.keys() else None,
        "HXG": row["HXG"] if "HXG" in row.keys() else None,
        "AXG": row["AXG"] if "AXG" in row.keys() else None,
        "HBC": row["HBC"] if "HBC" in row.keys() else None,
        "ABC": row["ABC"] if "ABC" in row.keys() else None,
        "stats_scraped_at": row["stats_scraped_at"],
    }
    upsert_training_example(example)


def collect_flashscore_results(league=None, days_back=14, target_date=None):
    """
    Resolve completed Flashscore-backed matches from the browser scraper.

    This closes the new browser-scraping feedback loop: league-page scraping
    finds completed full-time scores, then the result is written to
    match_results and training_examples.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    params = []
    date_clause = ""
    if target_date:
        date_clause = "AND m.match_date = ?"
        params.append(str(target_date).split("T")[0])
    else:
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')
        date_clause = "AND m.match_date >= ? AND m.match_date < ?"
        params.extend([cutoff, today])

    league_clause = ""
    if league:
        league_clause = "AND m.league = ?"
        params.append(league)

    rows = cursor.execute(f"""
        SELECT m.*
        FROM matches m
        LEFT JOIN match_results r ON r.id = m.id
        WHERE r.id IS NULL
          AND m.is_mock = 0
          AND (
              m.data_source = 'browser'
              OR m.flashscore_id IS NOT NULL
              OR m.source_url LIKE '%flashscore.com/match/%'
          )
          {date_clause}
          {league_clause}
        ORDER BY m.match_date ASC, m.league ASC, m.home_team ASC
    """, tuple(params)).fetchall()
    conn.close()

    if not rows:
        print("[ResultCollect] Flashscore checked 0, resolved 0")
        return 0

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["match_date"], row["league"])].append(row)

    resolved = 0
    checked = 0
    for (match_date, lg), group_rows in grouped.items():
        try:
            from browser_scraper import discover_browser_fixtures_for_date

            fixtures = discover_browser_fixtures_for_date(match_date, leagues=[lg])
        except Exception as exc:
            print(f"  [ResultCollect] Flashscore discovery failed for {lg} {match_date}: {exc}")
            continue

        by_flashscore_id = {}
        by_team = {}
        for item in fixtures:
            if not item.actual_ftr or item.home_goals is None or item.away_goals is None:
                continue
            if item.source_id:
                by_flashscore_id[str(item.source_id)] = item
            by_team[(_norm_team(item.home_team), _norm_team(item.away_team))] = item

        for row in group_rows:
            checked += 1
            item = None
            if row["flashscore_id"]:
                item = by_flashscore_id.get(str(row["flashscore_id"]))
            if item is None:
                item = by_team.get((_norm_team(row["home_team"]), _norm_team(row["away_team"])))
            if item is None:
                continue
            fthg = int(item.home_goals)
            ftag = int(item.away_goals)
            ftr = str(item.actual_ftr).upper()[:1]
            if ftr not in {"H", "D", "A"}:
                continue
            write_result_to_sqlite(row["id"], fthg, ftag, ftr)
            _upsert_closed_training_example(row, fthg, ftag, ftr)
            resolved += 1

    print(f"[ResultCollect] Flashscore checked {checked}, resolved {resolved}")
    return resolved


def write_clv_tracking(entry, hit, pnl, kelly_fraction=0.25):
    """Write a CLV tracking record after result is resolved."""
    from persistence_manager import write_clv_record

    pred_id = entry.get('id', '')
    match_date = entry.get('date', '')
    league = entry.get('league', '')
    home = entry.get('home', '')
    away = entry.get('away', '')

    closing_odds = entry.get('closing_odds')
    open_odds = entry.get('open_odds')
    predicted_prob = entry.get('prob')
    stake_pct = entry.get('stake_pct', 0)
    edge = entry.get('edge', 0)
    cld = entry.get('cld', 0)
    actual_ftr = entry.get('result', '')

    # Compute closing_prob from closing_odds
    closing_prob = 1 / closing_odds if closing_odds else None
    open_prob = 1 / open_odds if open_odds else None

    from persistence_manager import write_clv_record as _write
    _write(
        prediction_id=pred_id,
        match_date=match_date,
        league=league,
        home_team=home,
        away_team=away,
        predicted_prob=predicted_prob,
        closing_prob=closing_prob,
        open_prob=open_prob,
        cld=cld,
        actual_ftr=actual_ftr,
        hit=hit,
        edge=edge,
        stake_pct=stake_pct,
        kelly_fraction=kelly_fraction,
        pnl=pnl
    )


def resolve_single_prediction(entry):
    """
    Resolve one PENDING prediction: fetch result, write to both tables.
    Returns: True if resolved, False if still pending or failed.
    """
    event_id = entry.get('id')
    home = entry.get('home', '')
    away = entry.get('away', '')
    date = entry.get('date', '')
    league = entry.get('league', '')
    prediction = entry.get('prediction', '')
    closing_odds = entry.get('closing_odds')
    stake_pct = entry.get('stake_pct', 0)

    m_id = f"{date}_{home}_{away}".replace("/", "-")

    # Try FDO for actual result
    result = fetch_actual_result_from_fdo(league, date, home, away)
    if not result:
        return False

    fthg, ftag, actual_ftr = result

    # Determine hit
    pred_map = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}
    hit = 1 if pred_map.get(prediction) == actual_ftr else 0

    # Compute P&L
    if closing_odds and hit:
        pnl = stake_pct * (closing_odds - 1)
    elif closing_odds:
        pnl = -stake_pct
    else:
        pnl = 0

    # Store the available closing odd in the matching 1X2 bucket.
    closing_home = closing_odds if prediction == 'Home Win' else None
    closing_draw = closing_odds if prediction == 'Draw' else None
    closing_away = closing_odds if prediction == 'Away Win' else None
    write_result_to_sqlite(
        m_id, fthg, ftag, actual_ftr,
        closing_home=closing_home,
        closing_draw=closing_draw,
        closing_away=closing_away,
    )

    # Write CLV tracking
    write_clv_tracking(entry, hit, pnl)

    return True, actual_ftr, hit, pnl


def collect_all_pending(league=None):
    """
    Main entry point: resolve all PENDING predictions in shadow_log.json.
    Optional league filter.
    """
    if not os.path.exists(SHADOW_LOG):
        print("[ResultCollect] No shadow_log found")
        return 0

    with open(SHADOW_LOG, 'r') as f:
        entries = json.load(f)

    resolved = 0
    checked = 0
    today = datetime.now().strftime('%Y-%m-%d')
    max_checks = int(os.environ.get('RESULT_COLLECTOR_MAX_CHECKS', '200'))
    for entry in entries:
        if entry.get('result') != 'PENDING':
            continue
        if league and entry.get('league') != league:
            continue
        entry_date = str(entry.get('date', '')).split('T')[0]
        if entry_date >= today:
            continue
        if checked >= max_checks:
            break
        checked += 1

        try:
            resolved_info = resolve_single_prediction(entry)
            if resolved_info:
                _, actual_ftr, hit, pnl = resolved_info
                resolved += 1
                entry['actual_ftr'] = actual_ftr
                entry['hit'] = hit
                entry['pnl'] = pnl
                entry['result'] = 'HIT' if hit else 'MISS'
                entry['resolved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"  [ResultCollect] Failed {entry.get('match')}: {e}")

    with open(SHADOW_LOG, 'w') as f:
        json.dump(entries, f, indent=4)

    print(f"[ResultCollect] Checked {checked}, resolved {resolved} predictions")
    return resolved


def collect_persisted_results(league=None, days_back=14):
    """
    Resolve completed persisted matches from the matches table, even when no
    shadow_log entry exists. This closes the DB feedback loop for V11 history.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')

    params = [cutoff, today]
    league_clause = ''
    if league:
        league_clause = 'AND m.league = ?'
        params.append(league)

    cursor.execute(f'''
        SELECT m.id, m.league, m.match_date, m.home_team, m.away_team
        FROM matches m
        LEFT JOIN match_results r ON r.id = m.id
        WHERE r.id IS NULL
          AND m.is_mock = 0
          AND m.match_date >= ?
          AND m.match_date < ?
          {league_clause}
        ORDER BY m.match_date DESC
    ''', params)
    rows = cursor.fetchall()
    conn.close()

    resolved = 0
    for row in rows:
        result = fetch_actual_result_from_fdo(
            row['league'],
            row['match_date'],
            row['home_team'],
            row['away_team'],
        )
        if not result:
            continue
        fthg, ftag, ftr = result
        write_result_to_sqlite(row['id'], fthg, ftag, ftr)
        resolved += 1

    print(f"[ResultCollect] Resolved {resolved} persisted matches")
    return resolved


def backfill_from_sqlite(league=None, days_back=30):
    """
    For past predictions that already have results in match_results table,
    retroactively update shadow_log.json and write CLV records.
    """
    from persistence_manager import get_all_pending_predictions

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cutoff = datetime.now() - timedelta(days=days_back)
    cutoff_str = cutoff.strftime('%Y-%m-%d')

    if league:
        cursor.execute('''
            SELECT id, actual_ftr, closing_prob_home, closing_prob_draw, closing_prob_away
            FROM match_results mr
            JOIN matches m ON mr.id = m.id
            WHERE m.league = ? AND m.match_date >= ?
        ''', (league, cutoff_str))
    else:
        cursor.execute('''
            SELECT id, actual_ftr, closing_prob_home, closing_prob_draw, closing_prob_away
            FROM match_results mr
            JOIN matches m ON mr.id = m.id
            WHERE m.match_date >= ?
        ''', (cutoff_str,))

    rows = cursor.fetchall()
    conn.close()

    if not os.path.exists(SHADOW_LOG):
        return 0

    with open(SHADOW_LOG, 'r') as f:
        entries = json.load(f)

    resolved = 0
    for row in rows:
        m_id, actual_ftr, cph, cpd, cpa = row
        for entry in entries:
            # Match by date+home+away in ID
            if entry.get('result') != 'PENDING':
                continue
            entry_id = f"{entry.get('date')}_{entry.get('home')}_{entry.get('away')}".replace("/", "-")
            if entry_id == m_id:
                entry['result'] = actual_ftr
                entry['resolved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                resolved += 1
                break

    with open(SHADOW_LOG, 'w') as f:
        json.dump(entries, f, indent=4)

    print(f"[ResultCollect] Backfilled {resolved} from SQLite")
    return resolved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Result Collector for Antigravity')
    parser.add_argument('--league', help='Filter by league')
    parser.add_argument('--backfill', action='store_true', help='Backfill from SQLite instead of FDO')
    parser.add_argument('--persisted', action='store_true', help='Resolve persisted matches from DB')
    parser.add_argument('--flashscore', action='store_true', help='Resolve Flashscore-backed persisted matches')
    parser.add_argument('--date', help='Resolve one YYYY-MM-DD date')
    parser.add_argument('--days', type=int, default=30, help='Days to backfill (default 30)')
    args = parser.parse_args()

    if args.flashscore:
        collect_flashscore_results(league=args.league, days_back=args.days, target_date=args.date)
    elif args.persisted:
        collect_persisted_results(league=args.league, days_back=args.days)
    elif args.backfill:
        backfill_from_sqlite(league=args.league, days_back=args.days)
    else:
        collect_all_pending(league=args.league)
