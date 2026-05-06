"""Isotonic regression calibration for V11 1X2 probabilities.

Fitted on held-out validation examples from intelligence_hub.db.
One calibrator per (league, outcome). Stored as pickle under model/calibration/.
Applied at inference as the final probability correction step, on top of the
existing empirical bias correction — giving two calibration passes:
  1. Empirical bias correction (load_1x2_calibration in main_script)
  2. Isotonic regression (this module) — non-parametric, per-league
"""

import os
import pickle
import sqlite3

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CAL_DIR = os.path.join(PROJECT_ROOT, "model", "calibration")
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")

_CAL_CACHE: dict = {}


def _cal_path(league: str) -> str:
    return os.path.join(CAL_DIR, f"{league}_isotonic.pkl")


def fit_league_calibrators(league: str, min_samples: int = 200):
    """Fit per-outcome isotonic calibrators from training_examples in the DB.

    Uses the most recent 3000 completed matches with known market probabilities.
    Saves calibrators to model/calibration/{league}_isotonic.pkl.
    Returns the calibrator dict or None if insufficient data.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        print("[Cal] sklearn not available — pip install scikit-learn")
        return None

    if not os.path.exists(DB_PATH):
        print(f"[Cal] DB not found at {DB_PATH}")
        return None

    try:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            """
            SELECT actual_ftr,
                   market_prob_home,
                   market_prob_draw,
                   market_prob_away
            FROM training_examples
            WHERE league = ?
              AND actual_ftr IN ('H', 'D', 'A')
              AND market_prob_home IS NOT NULL
              AND market_prob_draw IS NOT NULL
              AND market_prob_away IS NOT NULL
            ORDER BY match_date DESC
            LIMIT 3000
            """,
            (league,),
        ).fetchall()
        con.close()
    except Exception as exc:
        print(f"[Cal] DB read failed for {league}: {exc}")
        return None

    if len(rows) < min_samples:
        print(f"[Cal] {league}: {len(rows)} samples < {min_samples} minimum — skipping")
        return None

    outcome_idx = {"H": 0, "D": 1, "A": 2}
    probs = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
    labels = np.array([outcome_idx[r[0]] for r in rows], dtype=int)

    calibrators = {}
    for i, outcome in enumerate(["H", "D", "A"]):
        y_prob = probs[:, i]
        y_true = (labels == i).astype(float)
        valid = ~(np.isnan(y_prob) | np.isinf(y_prob))
        if valid.sum() < min_samples:
            print(f"[Cal] {league}/{outcome}: insufficient valid rows")
            return None
        cal = IsotonicRegression(out_of_bounds="clip", increasing=True)
        cal.fit(y_prob[valid], y_true[valid])
        calibrators[outcome] = cal

    os.makedirs(CAL_DIR, exist_ok=True)
    with open(_cal_path(league), "wb") as fh:
        pickle.dump(calibrators, fh)

    print(f"[Cal] {league}: calibrators fitted on {len(rows)} rows -> saved")
    _CAL_CACHE[league] = calibrators
    return calibrators


def load_league_calibrators(league: str):
    """Load fitted isotonic calibrators (memory-cached after first load)."""
    if league in _CAL_CACHE:
        return _CAL_CACHE[league]
    path = _cal_path(league)
    if not os.path.exists(path):
        _CAL_CACHE[league] = None
        return None
    try:
        with open(path, "rb") as fh:
            cal = pickle.load(fh)
        _CAL_CACHE[league] = cal
        return cal
    except Exception as exc:
        print(f"[Cal] Load failed for {league}: {exc}")
        _CAL_CACHE[league] = None
        return None


def apply_isotonic_calibration(
    p_h: float, p_d: float, p_a: float, league: str, alpha: float = 0.70
) -> tuple:
    """Apply per-league isotonic calibration with damping alpha.

    alpha=1.0  → full isotonic correction
    alpha=0.70 → 70% isotonic + 30% raw (guards against overfitting small sets)

    Falls back to uncorrected probs if no calibrator exists for the league.
    """
    cal = load_league_calibrators(league)
    if cal is None:
        return p_h, p_d, p_a
    try:
        raw = np.array([p_h, p_d, p_a], dtype=float)
        corrected = np.array(
            [
                float(cal["H"].predict([p_h])[0]),
                float(cal["D"].predict([p_d])[0]),
                float(cal["A"].predict([p_a])[0]),
            ],
            dtype=float,
        )
        blended = raw * (1.0 - alpha) + corrected * alpha
        blended = np.clip(blended, 0.01, 0.98)
        blended /= blended.sum()
        return float(blended[0]), float(blended[1]), float(blended[2])
    except Exception:
        return p_h, p_d, p_a


def fit_all_leagues(leagues=None, min_samples: int = 200):
    """Fit calibrators for all given leagues and print a summary report."""
    if leagues is None:
        leagues = [
            "PremierLeague", "LaLiga", "SerieA", "Bundesliga", "Ligue1",
            "ChampionsLeague", "Championship", "ScottishPremiership",
            "Eredivisie", "LigaNOS", "BelgianProLeague", "SuperLig",
            "2Bundesliga", "Ligue2", "LaLiga2", "SerieB",
        ]
    results = {}
    for league in leagues:
        cal = fit_league_calibrators(league, min_samples=min_samples)
        results[league] = "fitted" if cal else "skipped"
    print("\n[Cal] Summary:")
    for league, status in results.items():
        print(f"  {league}: {status}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="ALL")
    parser.add_argument("--min-samples", type=int, default=200)
    args = parser.parse_args()
    if args.league == "ALL":
        fit_all_leagues(min_samples=args.min_samples)
    else:
        fit_league_calibrators(args.league, min_samples=args.min_samples)
