"""League-level tactical priors used by the market-count models.

These are deliberately light-touch priors. Historical data should dominate when
coverage is good; the profile only stabilizes sparse leagues and gives the UI a
plain-language explanation for why markets behave differently by competition.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "league_market_profiles.json")


DEFAULT_PROFILE = {
    "tempo": 0.50,
    "directness": 0.50,
    "pressing": 0.50,
    "set_piece_weight": 0.50,
    "card_pressure": 0.50,
    "goal_total_prior": 2.60,
    "corner_total_prior": 9.60,
    "card_total_prior": 4.20,
    "style_label": "balanced",
    "modeling_note": "Use the learned league baseline; no strong cultural adjustment.",
}


LEAGUE_PROFILES: Dict[str, Dict[str, object]] = {
    "PremierLeague": {
        "tempo": 0.70,
        "directness": 0.72,
        "pressing": 0.66,
        "set_piece_weight": 0.72,
        "card_pressure": 0.54,
        "goal_total_prior": 2.78,
        "corner_total_prior": 10.10,
        "card_total_prior": 4.05,
        "style_label": "direct transition / set-piece heavy",
        "modeling_note": "Corners and transition goals deserve more weight; cards are less referee-heavy than southern leagues.",
    },
    "Bundesliga": {
        "tempo": 0.76,
        "directness": 0.78,
        "pressing": 0.78,
        "set_piece_weight": 0.66,
        "card_pressure": 0.50,
        "goal_total_prior": 2.95,
        "corner_total_prior": 10.00,
        "card_total_prior": 3.95,
        "style_label": "vertical / high-intensity",
        "modeling_note": "High tempo supports goals and corners; draw calibration needs extra care because open games swing late.",
    },
    "SerieA": {
        "tempo": 0.44,
        "directness": 0.42,
        "pressing": 0.48,
        "set_piece_weight": 0.56,
        "card_pressure": 0.68,
        "goal_total_prior": 2.48,
        "corner_total_prior": 9.20,
        "card_total_prior": 4.65,
        "style_label": "structured / tactical control",
        "modeling_note": "Cards and game-state pressure matter more; pure tempo features should be damped.",
    },
    "LaLiga": {
        "tempo": 0.46,
        "directness": 0.45,
        "pressing": 0.58,
        "set_piece_weight": 0.55,
        "card_pressure": 0.72,
        "goal_total_prior": 2.42,
        "corner_total_prior": 9.15,
        "card_total_prior": 4.85,
        "style_label": "structured build-up / higher discipline variance",
        "modeling_note": "Lower scoring and higher card sensitivity; referee and draw handling should be explicit.",
    },
    "Ligue1": {
        "tempo": 0.63,
        "directness": 0.60,
        "pressing": 0.70,
        "set_piece_weight": 0.56,
        "card_pressure": 0.58,
        "goal_total_prior": 2.68,
        "corner_total_prior": 9.55,
        "card_total_prior": 4.30,
        "style_label": "pressing / transition trend",
        "modeling_note": "Recent attacking shift supports totals; keep season drift visible in validation.",
    },
    "Championship": {
        "tempo": 0.68,
        "directness": 0.72,
        "pressing": 0.57,
        "set_piece_weight": 0.72,
        "card_pressure": 0.58,
        "goal_total_prior": 2.55,
        "corner_total_prior": 10.20,
        "card_total_prior": 4.35,
        "style_label": "direct / physical",
        "modeling_note": "Set pieces and cards should stay visible; avoid copying Premier League priors blindly.",
    },
}


def get_league_profile(league: str | None) -> Dict[str, object]:
    profile = dict(DEFAULT_PROFILE)
    if league and league in LEAGUE_PROFILES:
        profile.update(LEAGUE_PROFILES[league])
    profile["league"] = league or "default"
    return profile


def profile_feature_vector(league: str | None) -> list[float]:
    profile = get_league_profile(league)
    return [
        float(profile["tempo"]),
        float(profile["directness"]),
        float(profile["pressing"]),
        float(profile["set_piece_weight"]),
        float(profile["card_pressure"]),
    ]


def write_profiles_report(path: str = REPORT_PATH) -> Dict[str, object]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "created_at": datetime.utcnow().isoformat(),
        "default": DEFAULT_PROFILE,
        "profiles": LEAGUE_PROFILES,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    print(json.dumps(write_profiles_report(), indent=2))
