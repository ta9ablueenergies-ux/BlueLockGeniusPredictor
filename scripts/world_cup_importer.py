"""World Cup historical dataset importer.

Downloads international match results from martj42/international-football
(free, no auth) and filters for FIFA World Cup matches 1930-2022.
Imports into training_examples in intelligence_hub.db.

Usage:
    python world_cup_importer.py                # download + import
    python world_cup_importer.py --local results.csv  # use local CSV
    python world_cup_importer.py --from-year 1986     # limit years
    python world_cup_importer.py --dry-run            # preview only
"""

import argparse
import hashlib
import io
import os
import sqlite3
import sys
from datetime import datetime

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/shootouts.csv"
)

# Normalise common name variants to our canonical forms
TEAM_ALIASES = {
    "Germany DR":           "Germany",
    "West Germany":         "Germany",
    "Soviet Union":         "Russia",
    "Czechoslovakia":       "Czech Republic",
    "Yugoslavia":           "Serbia",
    "Republic of Ireland":  "Ireland",
    "Northern Ireland":     "Northern Ireland",
    "China PR":             "China",
    "Chinese Taipei":       "Taiwan",
    "Korea Republic":       "South Korea",
    "Korea DPR":            "North Korea",
    "United States":        "USA",
    "Trinidad and Tobago":  "Trinidad & Tobago",
    "Bosnia-Herzegovina":   "Bosnia",
    "North Macedonia":      "Macedonia",
    "Ivory Coast":          "Ivory Coast",
    "Cape Verde":           "Cape Verde Islands",
}


def _norm(name: str) -> str:
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def _ftr(home_score, away_score) -> str | None:
    try:
        h, a = int(home_score), int(away_score)
        if h > a:
            return "H"
        if a > h:
            return "A"
        return "D"
    except Exception:
        return None


def _example_id(match_date, home_team, away_team) -> str:
    raw = f"wc_{match_date}_{home_team}_{away_team}"
    return hashlib.md5(raw.encode()).hexdigest()[:24]


def download_csv(url: str) -> pd.DataFrame:
    print(f"  Downloading {url}...", flush=True)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def load_results(local_path: str | None = None) -> pd.DataFrame:
    if local_path and os.path.exists(local_path):
        print(f"  Reading local file: {local_path}", flush=True)
        return pd.read_csv(local_path)
    return download_csv(RESULTS_URL)


def load_shootouts() -> pd.DataFrame:
    try:
        return download_csv(SHOOTOUTS_URL)
    except Exception:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "winner"])


def build_examples(df: pd.DataFrame, from_year: int) -> list[dict]:
    examples = []
    for _, row in df.iterrows():
        date_str = str(row.get("date", "")).strip()
        try:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        except Exception:
            continue
        if dt.year < from_year:
            continue

        home = _norm(row.get("home_team", ""))
        away = _norm(row.get("away_team", ""))
        if not home or not away:
            continue

        h_score = row.get("home_score")
        a_score = row.get("away_score")
        ftr = _ftr(h_score, a_score)

        try:
            h_goals = int(h_score)
            a_goals = int(a_score)
        except Exception:
            h_goals = a_goals = None

        # Stage metadata from tournament name (e.g. "FIFA World Cup" for group
        # stage, separate stage column not available in this dataset)
        tournament = str(row.get("tournament", "")).strip()

        examples.append({
            "id":              _example_id(date_str[:10], home, away),
            "league":          "WorldCup",
            "match_date":      date_str[:10],
            "home_team":       home,
            "away_team":       away,
            "home_goals":      h_goals,
            "away_goals":      a_goals,
            "actual_ftr":      ftr,
            "home_odds":       None,
            "draw_odds":       None,
            "away_odds":       None,
            "source":          "martj42_intl_football",
            "source_confidence": 0.80,
            "features":        {"neutral": int(bool(row.get("neutral", False))),
                                "tournament": tournament},
        })
    return examples


def upsert(examples: list[dict]) -> int:
    if not examples:
        return 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Resolve stat column list from DB schema
    con = sqlite3.connect(DB_PATH)
    cols_info = con.execute("PRAGMA table_info(training_examples)").fetchall()
    all_cols = [r[1] for r in cols_info]
    con.close()

    CORE = [
        "id", "league", "match_date", "home_team", "away_team",
        "home_goals", "away_goals", "actual_ftr",
        "home_odds", "draw_odds", "away_odds",
        "source", "source_confidence", "features_json",
        "created_at", "updated_at",
    ]
    # Add optional columns that may exist in the schema
    OPTIONAL = ["market_prob_home", "market_prob_draw", "market_prob_away"]
    insert_cols = CORE + [c for c in OPTIONAL if c in all_cols]

    con = sqlite3.connect(DB_PATH)
    inserted = 0
    for ex in examples:
        features_json = __import__("json").dumps(ex.get("features", {}))
        vals = {
            "id":                ex["id"],
            "league":            ex["league"],
            "match_date":        ex["match_date"],
            "home_team":         ex["home_team"],
            "away_team":         ex["away_team"],
            "home_goals":        ex.get("home_goals"),
            "away_goals":        ex.get("away_goals"),
            "actual_ftr":        ex.get("actual_ftr"),
            "home_odds":         ex.get("home_odds"),
            "draw_odds":         ex.get("draw_odds"),
            "away_odds":         ex.get("away_odds"),
            "source":            ex.get("source", "wc_import"),
            "source_confidence": ex.get("source_confidence", 0.80),
            "features_json":     features_json,
            "market_prob_home":  None,
            "market_prob_draw":  None,
            "market_prob_away":  None,
            "created_at":        now,
            "updated_at":        now,
        }
        row = [vals[c] for c in insert_cols]
        placeholders = ", ".join(["?"] * len(insert_cols))
        col_sql = ", ".join(insert_cols)
        update_sql = ", ".join(
            f"{c}=excluded.{c}"
            for c in insert_cols
            if c not in ("id", "created_at")
        )
        try:
            con.execute(
                f"INSERT INTO training_examples ({col_sql}) VALUES ({placeholders})"
                f" ON CONFLICT(id) DO UPDATE SET {update_sql}",
                row,
            )
            inserted += 1
        except Exception as e:
            print(f"  [WARN] row failed: {e}", flush=True)
    con.commit()
    con.close()
    return inserted


def print_summary(examples: list[dict]) -> None:
    if not examples:
        print("  No examples to import.")
        return
    by_year: dict[int, dict] = {}
    for ex in examples:
        yr = int(ex["match_date"][:4])
        bucket = by_year.setdefault(yr, {"total": 0, "with_result": 0})
        bucket["total"] += 1
        if ex.get("actual_ftr"):
            bucket["with_result"] += 1
    print(f"\n  {'Year':<8} {'Matches':<10} {'With Result'}")
    print(f"  {'-'*30}")
    for yr in sorted(by_year):
        b = by_year[yr]
        print(f"  {yr:<8} {b['total']:<10} {b['with_result']}")
    total = sum(b["total"] for b in by_year.values())
    with_res = sum(b["with_result"] for b in by_year.values())
    print(f"  {'-'*30}")
    print(f"  {'TOTAL':<8} {total:<10} {with_res}")


def main():
    parser = argparse.ArgumentParser(description="Import historical World Cup data")
    parser.add_argument("--local", metavar="CSV_PATH", default=None,
                        help="Path to local results.csv (skip download)")
    parser.add_argument("--from-year", type=int, default=1954,
                        help="Only import matches from this year onwards (default: 1954)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and preview without writing to DB")
    args = parser.parse_args()

    print(f"\n[WC Import] Loading international results...", flush=True)
    df_all = load_results(args.local)
    print(f"  Total rows: {len(df_all)}", flush=True)

    wc = df_all[df_all["tournament"].str.contains("FIFA World Cup", na=False)].copy()
    print(f"  World Cup rows: {len(wc)}", flush=True)

    examples = build_examples(wc, from_year=args.from_year)
    print(f"  Valid examples (from {args.from_year}): {len(examples)}", flush=True)

    print_summary(examples)

    if args.dry_run:
        print("\n  [Dry run] No DB writes performed.")
        return

    if not os.path.exists(DB_PATH):
        print(f"\n  [ERROR] DB not found at {DB_PATH}")
        sys.exit(1)

    print(f"\n[WC Import] Writing to {DB_PATH}...", flush=True)
    n = upsert(examples)
    print(f"[WC Import] Done — {n} rows upserted.", flush=True)

    # Quick verification
    con = sqlite3.connect(DB_PATH)
    count = con.execute(
        "SELECT COUNT(*) FROM training_examples WHERE league='WorldCup'"
    ).fetchone()[0]
    con.close()
    print(f"[WC Import] training_examples WorldCup total: {count}", flush=True)


if __name__ == "__main__":
    main()
