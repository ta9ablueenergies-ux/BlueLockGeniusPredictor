"""Derive post-match learning patterns from scraped match evidence."""

from __future__ import annotations

import math
from typing import Dict, List, Optional


def _f(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _pattern(key: str, confidence: float, **details: object) -> Dict[str, object]:
    return {
        "pattern_key": key,
        "confidence": round(max(0.05, min(1.0, float(confidence))), 3),
        "details": details,
    }


def derive_post_match_patterns(stats: Dict[str, object], row: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
    """Convert scraped match stats into stable tags for later autopsy/training."""
    row = row or {}
    patterns: List[Dict[str, object]] = []

    hg = _f(row.get("home_goals"))
    ag = _f(row.get("away_goals"))
    hs, ass = _f(stats.get("HS")), _f(stats.get("AS"))
    hst, ast = _f(stats.get("HST")), _f(stats.get("AST"))
    hc, ac = _f(stats.get("HC")), _f(stats.get("AC"))
    hy, ay = _f(stats.get("HY")), _f(stats.get("AY"))
    hr, ar = _f(stats.get("HR"), 0.0), _f(stats.get("AR"), 0.0)
    hf, af = _f(stats.get("HF")), _f(stats.get("AF"))
    hposs, aposs = _f(stats.get("HPoss")), _f(stats.get("APoss"))
    hxg, axg = _f(stats.get("HXG")), _f(stats.get("AXG"))
    hbc, abc = _f(stats.get("HBC")), _f(stats.get("ABC"))

    missing = [
        key for key in ("HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY")
        if stats.get(key) is None
    ]
    if len(missing) >= 4:
        patterns.append(_pattern("low_event_data_quality", 0.55, missing=missing))

    if (hr or 0) + (ar or 0) > 0:
        patterns.append(_pattern("red_card_distortion", 0.92, home_red=hr, away_red=ar))

    if hy is not None and ay is not None:
        total_cards = hy + ay + 2.0 * ((hr or 0) + (ar or 0))
        if total_cards >= 6:
            patterns.append(_pattern("cards_spike", 0.82, total_cards=total_cards, home_cards=hy, away_cards=ay))
        if abs(hy - ay) >= 3:
            side = "home" if hy > ay else "away"
            patterns.append(_pattern("discipline_imbalance", 0.72, side=side, home_cards=hy, away_cards=ay))

    if hc is not None and ac is not None:
        corner_diff = hc - ac
        if abs(corner_diff) >= 4:
            side = "home" if corner_diff > 0 else "away"
            patterns.append(_pattern("set_piece_pressure", 0.78, side=side, home_corners=hc, away_corners=ac))
        if hc + ac >= 12:
            patterns.append(_pattern("high_corner_environment", 0.70, total_corners=hc + ac))

    if hst is not None and ast is not None:
        sot_diff = hst - ast
        if abs(sot_diff) >= 3:
            side = "home" if sot_diff > 0 else "away"
            patterns.append(_pattern("shot_on_target_dominance", 0.80, side=side, home_sot=hst, away_sot=ast))
        if hg is not None and ag is not None:
            if hg > ag and hst <= ast:
                patterns.append(_pattern("home_finishing_outperformed_pressure", 0.66, score=f"{int(hg)}-{int(ag)}", home_sot=hst, away_sot=ast))
            if ag > hg and ast <= hst:
                patterns.append(_pattern("away_finishing_outperformed_pressure", 0.66, score=f"{int(hg)}-{int(ag)}", home_sot=hst, away_sot=ast))

    if hs is not None and ass is not None:
        shot_diff = hs - ass
        if abs(shot_diff) >= 7:
            side = "home" if shot_diff > 0 else "away"
            patterns.append(_pattern("territory_shot_volume_dominance", 0.68, side=side, home_shots=hs, away_shots=ass))
        total_shots = hs + ass
        total_sot = (hst or 0) + (ast or 0)
        if total_shots >= 22 and total_sot <= 6:
            patterns.append(_pattern("low_shot_quality", 0.70, total_shots=total_shots, total_sot=total_sot))

    if hposs is not None and aposs is not None and abs(hposs - aposs) >= 18:
        side = "home" if hposs > aposs else "away"
        patterns.append(_pattern("possession_control", 0.62, side=side, home_possession=hposs, away_possession=aposs))

    if hxg is not None and axg is not None and hg is not None and ag is not None:
        if hg >= hxg + 1.2:
            patterns.append(_pattern("home_finishing_variance", 0.76, goals=hg, xg=hxg))
        if ag >= axg + 1.2:
            patterns.append(_pattern("away_finishing_variance", 0.76, goals=ag, xg=axg))
        if hxg >= axg + 0.8 and hg <= ag:
            patterns.append(_pattern("home_xg_underperformance", 0.74, home_xg=hxg, away_xg=axg, score=f"{int(hg)}-{int(ag)}"))
        if axg >= hxg + 0.8 and ag <= hg:
            patterns.append(_pattern("away_xg_underperformance", 0.74, home_xg=hxg, away_xg=axg, score=f"{int(hg)}-{int(ag)}"))

    if hbc is not None and abc is not None and abs(hbc - abc) >= 2:
        side = "home" if hbc > abc else "away"
        patterns.append(_pattern("big_chance_gap", 0.74, side=side, home_big_chances=hbc, away_big_chances=abc))

    if hf is not None and af is not None and abs(hf - af) >= 7:
        side = "home" if hf > af else "away"
        patterns.append(_pattern("foul_pressure_imbalance", 0.60, side=side, home_fouls=hf, away_fouls=af))

    if not patterns:
        patterns.append(_pattern("standard_match_profile", 0.35))
    return patterns
