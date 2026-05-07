"""Retrain all V11 models sequentially (22-dim with streak+form features).

Run after current retrains finish:
    python retrain_all.py
    python retrain_all.py --leagues PremierLeague LaLiga SerieA
    python retrain_all.py --epochs 30  # faster but less convergence
"""
import argparse
import subprocess
import sys
import os
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)

DEFAULT_LEAGUES = ["PremierLeague", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "WorldCup"]


def main():
    parser = argparse.ArgumentParser(description="Retrain all V11 22-dim models")
    parser.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    print(f"[retrain_all] Starting {len(args.leagues)} leagues × {args.epochs} epochs", flush=True)
    print(f"[retrain_all] Leagues: {', '.join(args.leagues)}", flush=True)

    results = {}
    for league in args.leagues:
        log = os.path.join(ROOT, f"retrain_{league.lower()}_22dim.log")
        err = os.path.join(ROOT, f"retrain_{league.lower()}_22dim.log.err")
        t0 = datetime.now()
        print(f"\n[{t0:%H:%M}] Starting {league}...", flush=True)
        with open(log, "w") as out, open(err, "w") as errf:
            r = subprocess.run(
                [sys.executable, "-u", "run_v11_hybrid.py",
                 "--league", league, "--epochs", str(args.epochs)],
                stdout=out, stderr=errf, cwd=SCRIPTS_DIR,
            )
        elapsed = (datetime.now() - t0).seconds // 60
        results[league] = r.returncode
        status = "OK" if r.returncode == 0 else f"FAIL(exit={r.returncode})"
        print(f"[retrain_all] {league}: {status} ({elapsed}min) -> {log}", flush=True)

    print(f"\n[retrain_all] Complete: {sum(v == 0 for v in results.values())}/{len(results)} succeeded")
    for league, rc in results.items():
        mark = "✓" if rc == 0 else "✗"
        print(f"  {mark} {league}")


if __name__ == "__main__":
    main()
