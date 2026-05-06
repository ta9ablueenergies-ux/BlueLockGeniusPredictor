"""Closing Line Value (CLV) tracker.

Records per-match odds snapshots and computes:
  CLD delta: opening_implied_p - current_implied_p for the primary outcome.
             Positive = market shortened (sharp money confirmation).
             Negative = market drifted (value evaporated / trap warning).
  CLV:       model_p - current_implied_p for the primary outcome.
             Positive = model is above the market (genuine edge).

Both signals feed into calculate_eqi_v2() and are stored in the result row
for post-match analysis and long-run edge tracking.

DB table: odds_snapshots in intelligence_hub.db
  - 'opening' snapshot: first snapshot ever recorded for this match
  - 'current' snapshot: always overwritten to the latest odds
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")

_SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS odds_snapshots (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        match_key       TEXT    NOT NULL,
        league          TEXT    NOT NULL,
        home_team       TEXT    NOT NULL,
        away_team       TEXT    NOT NULL,
        match_date      TEXT    NOT NULL,
        snapshot_at     TEXT    NOT NULL,
        snapshot_type   TEXT    NOT NULL DEFAULT 'current',
        h_odds          REAL,
        d_odds          REAL,
        a_odds          REAL,
        model_p_h       REAL,
        model_p_d       REAL,
        model_p_a       REAL
    )""",
    """CREATE UNIQUE INDEX IF NOT EXISTS uix_odds_snap
        ON odds_snapshots (match_key, snapshot_type)""",
    """CREATE INDEX IF NOT EXISTS ix_odds_snap_key
        ON odds_snapshots (match_key, snapshot_at)""",
]

_HOME_MARKETS = {"home win", "1", "h"}
_AWAY_MARKETS = {"away win", "2", "a"}


def _ensure_schema(con: sqlite3.Connection) -> None:
    for stmt in _SCHEMA_STATEMENTS:
        con.execute(stmt)
    con.commit()


def _implied(odds: float) -> float:
    if not odds or odds <= 1.0:
        return 0.0
    return max(0.01, min(0.99, 1.0 / odds))


def make_match_key(league: str, home: str, away: str, date: str) -> str:
    return f"{league}|{home}|{away}|{date[:10]}"


def record_snapshot(
    league: str,
    home: str,
    away: str,
    date: str,
    h_odds: float,
    d_odds: float,
    a_odds: float,
    model_p_h: float = 0.0,
    model_p_d: float = 0.0,
    model_p_a: float = 0.0,
) -> None:
    """Upsert odds snapshot. First call → saved as 'opening'. Every call → 'current' overwritten."""
    if not os.path.exists(DB_PATH):
        return
    match_key = make_match_key(league, home, away, date)
    now = datetime.utcnow().isoformat(timespec="seconds")
    try:
        con = sqlite3.connect(DB_PATH, timeout=10)
        _ensure_schema(con)

        has_opening = con.execute(
            "SELECT 1 FROM odds_snapshots WHERE match_key=? AND snapshot_type='opening'",
            (match_key,),
        ).fetchone()

        if not has_opening:
            con.execute(
                """INSERT OR IGNORE INTO odds_snapshots
                   (match_key, league, home_team, away_team, match_date,
                    snapshot_at, snapshot_type, h_odds, d_odds, a_odds,
                    model_p_h, model_p_d, model_p_a)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (match_key, league, home, away, date[:10],
                 now, "opening", h_odds, d_odds, a_odds,
                 model_p_h, model_p_d, model_p_a),
            )

        # Upsert current (overwrites previous current row for this match)
        con.execute(
            """INSERT OR REPLACE INTO odds_snapshots
               (match_key, league, home_team, away_team, match_date,
                snapshot_at, snapshot_type, h_odds, d_odds, a_odds,
                model_p_h, model_p_d, model_p_a)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (match_key, league, home, away, date[:10],
             now, "current", h_odds, d_odds, a_odds,
             model_p_h, model_p_d, model_p_a),
        )
        con.commit()
        con.close()
    except Exception as exc:
        print(f"[CLV] Snapshot write failed ({match_key}): {exc}")


def _odds_for_market(h_odds: float, d_odds: float, a_odds: float, market: str) -> float:
    m = market.strip().lower()
    if m in _HOME_MARKETS:
        return h_odds
    if m in _AWAY_MARKETS:
        return a_odds
    return 0.0  # non-1X2 market — CLD not computable from match odds


def compute_cld_delta(
    league: str,
    home: str,
    away: str,
    date: str,
    primary_market: str,
) -> float:
    """opening_implied_p - current_implied_p. Zero if no opening snapshot or non-1X2 market."""
    if not os.path.exists(DB_PATH):
        return 0.0
    match_key = make_match_key(league, home, away, date)
    try:
        con = sqlite3.connect(DB_PATH, timeout=10)
        _ensure_schema(con)
        rows = {
            r[0]: (r[1], r[2], r[3])
            for r in con.execute(
                "SELECT snapshot_type, h_odds, d_odds, a_odds FROM odds_snapshots "
                "WHERE match_key=? AND snapshot_type IN ('opening','current')",
                (match_key,),
            ).fetchall()
        }
        con.close()
        if "opening" not in rows or "current" not in rows:
            return 0.0
        open_o = _odds_for_market(*rows["opening"], primary_market)
        cur_o  = _odds_for_market(*rows["current"],  primary_market)
        if not open_o or not cur_o:
            return 0.0
        return _implied(open_o) - _implied(cur_o)
    except Exception as exc:
        print(f"[CLV] CLD delta failed ({match_key}): {exc}")
        return 0.0


def compute_clv(
    league: str,
    home: str,
    away: str,
    date: str,
    model_p_h: float,
    model_p_d: float,
    model_p_a: float,
    primary_market: str,
) -> float:
    """model_p - current_implied_p for primary outcome. Positive = genuine model edge."""
    if not os.path.exists(DB_PATH):
        return 0.0
    match_key = make_match_key(league, home, away, date)
    try:
        con = sqlite3.connect(DB_PATH, timeout=10)
        _ensure_schema(con)
        row = con.execute(
            "SELECT h_odds, d_odds, a_odds FROM odds_snapshots "
            "WHERE match_key=? AND snapshot_type='current' "
            "ORDER BY snapshot_at DESC LIMIT 1",
            (match_key,),
        ).fetchone()
        con.close()
        if not row:
            return 0.0
        m = primary_market.strip().lower()
        if m in _HOME_MARKETS:
            return model_p_h - _implied(row[0])
        if m in _AWAY_MARKETS:
            return model_p_a - _implied(row[2])
        return 0.0  # BTTS/goals markets — CLV not meaningful from 1X2 odds
    except Exception as exc:
        print(f"[CLV] CLV compute failed ({match_key}): {exc}")
        return 0.0
