"""Pi Rating per-timestep snapshot builder.

Replays all training_examples chronologically through PiRatingSystem per
league, capturing each team's ratings BEFORE the match is updated. Stores
snapshots in `pi_snapshots` table for use in encode_seq as opp_strength.

This replaces the hardcoded 0.5 default with actual historical strength
estimates — no encoder architecture changes or retraining required.

Usage:
    python pi_rating_timestep.py                     # build all leagues
    python pi_rating_timestep.py --leagues PremierLeague Bundesliga
    python pi_rating_timestep.py --lookup Arsenal 2024-01-01
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")

sys.path.insert(0, SCRIPT_DIR)
from pi_ratings import PiRatingSystem  # noqa: E402

DDL = """
CREATE TABLE IF NOT EXISTS pi_snapshots (
    league      TEXT NOT NULL,
    team        TEXT NOT NULL,
    match_date  TEXT NOT NULL,
    pi_h        REAL NOT NULL DEFAULT 0.0,
    pi_a        REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (league, team, match_date)
);
CREATE INDEX IF NOT EXISTS pi_snap_team ON pi_snapshots(team, match_date);
"""

# Normalize strength to [0,1] for use as opp_strength feature.
# Pi ratings range roughly from -3 to +3 for top leagues.
_PI_SCALE = 3.0


def _pi_to_strength(pi_h: float, pi_a: float, is_home: bool) -> float:
    """Convert Pi rating component to a [0,1] opp_strength-compatible value."""
    raw = pi_h if is_home else pi_a
    return max(0.0, min(1.0, (raw + _PI_SCALE) / (2.0 * _PI_SCALE)))


def init_tables(con: sqlite3.Connection) -> None:
    con.executescript(DDL)
    con.commit()


# ── Replay ────────────────────────────────────────────────────────────────────

def _load_matches_for_league(con: sqlite3.Connection, league: str) -> List[Tuple]:
    """Return (match_date, home_team, away_team, home_goals, away_goals) sorted."""
    rows = con.execute("""
        SELECT match_date, home_team, away_team, home_goals, away_goals
        FROM training_examples
        WHERE league = ?
          AND actual_ftr IN ('H','D','A')
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
        ORDER BY match_date ASC, home_team ASC
    """, (league,)).fetchall()
    return rows


def replay_league(
    con: sqlite3.Connection,
    league: str,
    batch_size: int = 500,
) -> int:
    """Replay matches for one league, insert pi_snapshots. Returns rows inserted."""
    matches = _load_matches_for_league(con, league)
    if not matches:
        return 0

    pi = PiRatingSystem()
    rows_to_insert: List[Tuple] = []

    for match_date, home, away, hg, ag in matches:
        # Snapshot BEFORE the update (pre-match rating)
        h_pi = pi.get_rating(home)
        a_pi = pi.get_rating(away)

        rows_to_insert.append((league, home, match_date, h_pi[0], h_pi[1]))
        rows_to_insert.append((league, away, match_date, a_pi[0], a_pi[1]))

        try:
            pi.update(home, away, int(hg), int(ag))
        except Exception:
            pass

        if len(rows_to_insert) >= batch_size * 2:
            con.executemany("""
                INSERT OR REPLACE INTO pi_snapshots (league, team, match_date, pi_h, pi_a)
                VALUES (?, ?, ?, ?, ?)
            """, rows_to_insert)
            con.commit()
            rows_to_insert = []

    if rows_to_insert:
        con.executemany("""
            INSERT OR REPLACE INTO pi_snapshots (league, team, match_date, pi_h, pi_a)
            VALUES (?, ?, ?, ?, ?)
        """, rows_to_insert)
        con.commit()

    return len(matches) * 2


def rebuild_all(con: sqlite3.Connection, leagues: Optional[List[str]] = None) -> None:
    if leagues is None:
        rows = con.execute(
            "SELECT DISTINCT league FROM training_examples ORDER BY league"
        ).fetchall()
        leagues = [r[0] for r in rows]

    total = 0
    for league in leagues:
        count = replay_league(con, league)
        total += count
        print(f"  [{league}] {count} snapshots written")

    print(f"\n[Pi] Total snapshots: {total}")


# ── Lookup cache ──────────────────────────────────────────────────────────────

_PI_CACHE: Dict[Tuple[str, str], List[Tuple[str, float, float]]] = {}
_PI_LOADED_LEAGUES: set = set()


def _load_league_cache(con: sqlite3.Connection, league: str) -> None:
    if league in _PI_LOADED_LEAGUES:
        return
    rows = con.execute("""
        SELECT team, match_date, pi_h, pi_a
        FROM pi_snapshots
        WHERE league = ?
        ORDER BY team, match_date ASC
    """, (league,)).fetchall()
    bucket: Dict[str, List[Tuple[str, float, float]]] = {}
    for team, match_date, pi_h, pi_a in rows:
        bucket.setdefault(team, []).append((match_date, pi_h, pi_a))
    for team, entries in bucket.items():
        _PI_CACHE[(league, team)] = entries
    _PI_LOADED_LEAGUES.add(league)


_con_cache: Optional[sqlite3.Connection] = None


def _get_con() -> sqlite3.Connection:
    global _con_cache
    if _con_cache is None:
        _con_cache = sqlite3.connect(DB_PATH)
    return _con_cache


def get_pi_at_date(
    league: str,
    team: str,
    match_date: str,
    is_home: bool = True,
) -> float:
    """Return [0,1] strength estimate for team at match_date.

    Looks up the most recent pi_snapshot BEFORE match_date.
    Falls back to 0.5 if no data.
    """
    try:
        con = _get_con()
        _load_league_cache(con, league)
        key = (league, team)
        entries = _PI_CACHE.get(key)
        if not entries:
            # Try alias matching (last-word partial)
            target_last = team.strip().lower().split()[-1] if team.strip() else ""
            for (l, t), v in _PI_CACHE.items():
                if l == league and t.lower().split()[-1] == target_last:
                    entries = v
                    break
        if not entries:
            return 0.5

        # Binary-search for latest entry before match_date
        date_str = match_date[:10]
        pi_h, pi_a = 0.0, 0.0
        for md, ph, pa in entries:
            if md <= date_str:
                pi_h, pi_a = ph, pa
            else:
                break
        return _pi_to_strength(pi_h, pi_a, is_home)
    except Exception:
        return 0.5


def invalidate_pi_cache() -> None:
    _PI_CACHE.clear()
    _PI_LOADED_LEAGUES.clear()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pi Rating per-timestep snapshot builder.")
    parser.add_argument("--leagues", nargs="+", default=None)
    parser.add_argument("--lookup", nargs=2, metavar=("TEAM", "DATE"), default=None,
                        help="Look up team strength at a date. E.g. --lookup Arsenal 2024-01-01")
    parser.add_argument("--lookup-league", default="PremierLeague")
    parser.add_argument("--clear", action="store_true", help="Drop and rebuild snapshots table")
    args = parser.parse_args()

    con = sqlite3.connect(DB_PATH)
    init_tables(con)

    if args.clear:
        con.execute("DELETE FROM pi_snapshots")
        con.commit()
        print("[Pi] Table cleared.")

    if args.lookup:
        team, date = args.lookup
        _load_league_cache(con, args.lookup_league)
        strength = get_pi_at_date(args.lookup_league, team, date)
        # Also show raw pi
        key = (args.lookup_league, team)
        entries = _PI_CACHE.get(key, [])
        raw = None
        for md, ph, pa in entries:
            if md <= date[:10]:
                raw = (ph, pa)
        print(f"[Pi] {team} @ {date} ({args.lookup_league}): strength={strength:.3f} raw_pi={raw}")
        con.close()
        return

    existing = con.execute("SELECT COUNT(*) FROM pi_snapshots").fetchone()[0]
    print(f"[Pi] Existing snapshots: {existing}")

    rebuild_all(con, args.leagues)

    final = con.execute("SELECT COUNT(*) FROM pi_snapshots").fetchone()[0]
    print(f"[Pi] Final snapshots: {final}")

    # Top teams by current rating in each Big 5 league
    for league in ["PremierLeague", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]:
        top = con.execute("""
            SELECT team, pi_h, pi_a, match_date FROM pi_snapshots
            WHERE league = ?
            AND match_date = (SELECT MAX(match_date) FROM pi_snapshots WHERE league = ? AND team = pi_snapshots.team)
            ORDER BY pi_h DESC LIMIT 3
        """, (league, league)).fetchall()
        if top:
            print(f"\n  [{league}] Top teams:")
            for team, ph, pa, md in top:
                print(f"    {team}: pi_h={ph:.3f}, pi_a={pa:.3f} (at {md})")

    con.close()


if __name__ == "__main__":
    main()
