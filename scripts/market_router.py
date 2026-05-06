"""Market router for high-competitiveness football fixtures.

The 1X2 neural layer can be accurate on average while still producing fixtures
where the winner market is too flat to use. This module detects those cases and
routes the prediction toward calibrated count markets when the validation report
supports it.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "market_count_model_report.json")

_REPORT_CACHE: Optional[Dict[str, object]] = None


MARKET_METRICS = {
    "Over 1.5": ("goals", "goals_over_1.5", 0.66),
    "Over 2.5": ("goals", "goals_over_2.5", 0.60),
    "Over 3.5": ("goals", "goals_over_3.5", 0.58),
    "Under 2.5": ("goals", "goals_over_2.5", 0.60),
    "Under 3.5": ("goals", "goals_over_3.5", 0.64),
    "BTTS": ("goals", "btts", 0.60),
    "Corners Over 8.5": ("corners", "corners_over_8.5", 0.61),
    "Corners Over 9.5": ("corners", "corners_over_9.5", 0.60),
    "Corners Over 10.5": ("corners", "corners_over_10.5", 0.58),
    "Cards Over 3.5": ("cards", "cards_over_3.5", 0.59),
    "Cards Over 4.5": ("cards", "cards_over_4.5", 0.58),
    "Cards Over 5.5": ("cards", "cards_over_5.5", 0.55),
}


def _load_report() -> Dict[str, object]:
    global _REPORT_CACHE
    if _REPORT_CACHE is not None:
        return _REPORT_CACHE
    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as handle:
            _REPORT_CACHE = json.load(handle)
    except Exception:
        _REPORT_CACHE = {}
    return _REPORT_CACHE


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalized_entropy(probs: Iterable[float]) -> float:
    vals = [_clip(_safe_float(p), 1e-9, 1.0) for p in probs]
    total = sum(vals)
    if total <= 0:
        return 1.0
    vals = [p / total for p in vals]
    entropy = -sum(p * math.log(p) for p in vals)
    return float(entropy / math.log(len(vals)))


def _margin(probs: Iterable[float]) -> float:
    vals = sorted((_safe_float(p) for p in probs), reverse=True)
    if len(vals) < 2:
        return 0.0
    return float(vals[0] - vals[1])


def _league_report(league: str) -> Dict[str, object]:
    report = _load_report()
    leagues = report.get("leagues", {}) if isinstance(report, dict) else {}
    row = leagues.get(league, {}) if isinstance(leagues, dict) else {}
    return row if isinstance(row, dict) else {}


def _line_metric(league_row: Dict[str, object], metric_key: str) -> Dict[str, object]:
    line_metrics = league_row.get("line_metrics", {}) if isinstance(league_row, dict) else {}
    row = line_metrics.get(metric_key, {}) if isinstance(line_metrics, dict) else {}
    return row if isinstance(row, dict) else {}


def _coverage(league_row: Dict[str, object], family: str) -> float:
    if family == "goals":
        return 1.0
    cov = league_row.get("stats_coverage", {}) if isinstance(league_row, dict) else {}
    if not isinstance(cov, dict):
        return 0.0
    return _clip(_safe_float(cov.get(family), 0.0), 0.0, 1.0)


def _quality_from_metric(metric: Dict[str, object]) -> float:
    brier = _safe_float(metric.get("brier"), 0.25)
    ece = _safe_float(metric.get("ece"), 0.12)
    brier_quality = _clip((0.30 - brier) / 0.16, 0.0, 1.0)
    calibration_quality = _clip(1.0 - (ece / 0.16), 0.0, 1.0)
    return _clip((brier_quality * 0.65) + (calibration_quality * 0.35), 0.0, 1.0)


def _context_bonus(family: str, market_context: Dict[str, object], situation: Dict[str, object], high_entropy: bool) -> float:
    profile = market_context.get("profile", {}) if isinstance(market_context, dict) else {}
    if not isinstance(profile, dict):
        profile = {}
    bonus = 0.0
    if high_entropy and family in {"goals", "corners", "cards"}:
        bonus += 0.08
    if family == "corners":
        bonus += max(0.0, _safe_float(profile.get("set_piece_weight"), 0.5) - 0.55) * 0.20
        bonus += max(0.0, _safe_float(profile.get("directness"), 0.5) - 0.60) * 0.08
    if family == "cards":
        bonus += max(0.0, _safe_float(profile.get("card_pressure"), 0.5) - 0.55) * 0.25
        pressure = _safe_float(situation.get("match_pressure_score"), 0.0) if isinstance(situation, dict) else 0.0
        bonus += min(0.06, pressure / 1000.0)
    if family == "goals":
        bonus += max(0.0, _safe_float(profile.get("tempo"), 0.5) - 0.62) * 0.10
    return _clip(bonus, 0.0, 0.16)


def _odds_for_market(match: Dict[str, object], label: str) -> Optional[float]:
    normalized = label.lower().replace(" ", "_").replace(".", "_")
    explicit_keys = [
        f"{normalized}_odds",
        f"{normalized}_course",
        f"odds_{normalized}",
    ]
    aliases = {
        "Over 2.5": ["over25_course", "o25_course", "over_2_5_odds"],
        "Under 2.5": ["under25_course", "u25_course", "under_2_5_odds"],
        "BTTS": ["btts_course", "btts_odds"],
    }
    for key in explicit_keys + aliases.get(label, []):
        odd = _safe_float(match.get(key), 0.0)
        if odd > 1.01:
            return odd
    return None


def _candidate(
    label: str,
    probability: float,
    league_row: Dict[str, object],
    market_context: Dict[str, object],
    situation: Dict[str, object],
    match: Dict[str, object],
    high_entropy: bool,
) -> Dict[str, object]:
    family, metric_key, min_probability = MARKET_METRICS[label]
    metric = _line_metric(league_row, metric_key)
    coverage = _coverage(league_row, family)
    quality = _quality_from_metric(metric)
    probability = _clip(_safe_float(probability), 0.01, 0.99)
    confidence_gap = max(0.0, probability - min_probability)
    context = _context_bonus(family, market_context, situation, high_entropy)
    ece = _safe_float(metric.get("ece"), 0.12)
    brier = _safe_float(metric.get("brier"), 0.25)
    calibration = _clip(1.0 - (ece / 0.16), 0.0, 1.0)
    odds = _odds_for_market(match, label)
    if odds:
        edge = probability - (1.0 / odds)
        edge_basis = "market_odds"
    else:
        edge = confidence_gap * quality * coverage * 0.75
        edge_basis = "model_without_market_odds"
    score = 100.0 * (
        probability * 0.42
        + quality * 0.20
        + coverage * 0.14
        + calibration * 0.10
        + min(0.25, confidence_gap * 1.6) * 0.08
        + context
    )
    eligible = (
        probability >= min_probability
        and coverage >= (0.45 if family in {"corners", "cards"} else 0.0)
        and quality >= 0.22
        and ece <= 0.14
        and brier <= 0.285
    )
    return {
        "market": label,
        "family": family,
        "probability": round(probability, 4),
        "min_probability": min_probability,
        "score": round(_clip(score, 1.0, 92.0), 1),
        "eligible": bool(eligible),
        "edge": round(max(0.0, edge), 4),
        "edge_basis": edge_basis,
        "odds": round(odds, 3) if odds else None,
        "coverage": round(coverage, 4),
        "brier": round(brier, 5),
        "ece": round(ece, 5),
        "quality": round(quality, 4),
    }


def _probabilities_from_context(
    p_btts: float,
    p_o15: float,
    p_o25: float,
    p_o35: float,
    market_context: Dict[str, object],
) -> Dict[str, float]:
    corners = market_context.get("corners", {}) if isinstance(market_context, dict) else {}
    cards = market_context.get("cards", {}) if isinstance(market_context, dict) else {}
    return {
        "Over 1.5": p_o15,
        "Over 2.5": p_o25,
        "Over 3.5": p_o35,
        "Under 2.5": 1.0 - p_o25,
        "Under 3.5": 1.0 - p_o35,
        "BTTS": p_btts,
        "Corners Over 8.5": _safe_float(corners.get("over_8_5"), 0.0),
        "Corners Over 9.5": _safe_float(corners.get("over_9_5"), 0.0),
        "Corners Over 10.5": _safe_float(corners.get("over_10_5"), 0.0),
        "Cards Over 3.5": _safe_float(cards.get("over_3_5"), 0.0),
        "Cards Over 4.5": _safe_float(cards.get("over_4_5"), 0.0),
        "Cards Over 5.5": _safe_float(cards.get("over_5_5"), 0.0),
    }


def route_market_selection(
    match: Dict[str, object],
    league: str,
    base_primary: str,
    base_secondary: str,
    p_h: float,
    p_d: float,
    p_a: float,
    p_btts: float,
    p_o15: float,
    p_o25: float,
    p_o35: float,
    market_context: Optional[Dict[str, object]] = None,
    situation: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Select the market that should be exposed as primary for a fixture."""
    market_context = market_context or {}
    situation = situation or {}
    probs_1x2 = [_safe_float(p_h), _safe_float(p_d), _safe_float(p_a)]
    entropy = _normalized_entropy(probs_1x2)
    margin = _margin(probs_1x2)
    high_competitiveness = entropy >= 0.94 or margin <= 0.085
    league_row = _league_report(league)

    probabilities = _probabilities_from_context(p_btts, p_o15, p_o25, p_o35, market_context)
    candidates = [
        _candidate(label, probability, league_row, market_context, situation, match, high_competitiveness)
        for label, probability in probabilities.items()
        if probability > 0
    ]
    eligible = [c for c in candidates if c["eligible"]]
    ranked = sorted(candidates, key=lambda c: (c["eligible"], c["score"], c["probability"]), reverse=True)
    event_ranked = [c for c in ranked if c["family"] in {"goals", "corners", "cards"}]
    event_eligible = [c for c in eligible if c["family"] in {"goals", "corners", "cards"}]
    event_eligible = sorted(event_eligible, key=lambda c: (c["score"], c["probability"]), reverse=True)

    selected = None
    action = "keep_base"
    reason = "1X2 not flagged as high-competitiveness."
    if high_competitiveness:
        if event_eligible and event_eligible[0]["score"] >= 58.0:
            selected = event_eligible[0]
            action = "promote_event_market"
            reason = "1X2 entropy is high; selected strongest calibrated event market."
        else:
            action = "keep_base_no_event_edge"
            reason = "1X2 entropy is high, but no event market passed calibration and coverage gates."
    elif event_eligible and event_eligible[0]["score"] >= 74.0:
        selected = event_eligible[0]
        action = "promote_strong_event_market"
        reason = "Event market score materially exceeds the default routing threshold."

    promoted = bool(selected and selected["market"] != base_primary)
    selected_market = selected["market"] if selected else base_primary
    selected_probability = selected["probability"] if selected else None
    selected_edge = selected["edge"] if selected else 0.0
    edge_basis = selected["edge_basis"] if selected else "base_market"
    trust_score = selected["score"] if selected else 0.0
    if selected and edge_basis == "model_without_market_odds":
        trust_score = min(trust_score, 72.0)

    secondary_pool = [c for c in event_ranked if c["market"] != selected_market]
    secondary_market = secondary_pool[0]["market"] if secondary_pool else base_secondary
    stake_hint = 0.0
    if selected and selected_edge > 0:
        stake_hint = round(_clip((trust_score - 55.0) / 18.0, 0.0, 2.0), 1)

    return {
        "version": "market_router_v1",
        "action": action,
        "reason": reason,
        "high_competitiveness": bool(high_competitiveness),
        "entropy": round(entropy, 4),
        "margin": round(margin, 4),
        "base_primary": base_primary,
        "base_secondary": base_secondary,
        "selected_market": selected_market,
        "selected_probability": selected_probability,
        "selected_family": selected["family"] if selected else None,
        "secondary_market": secondary_market,
        "promoted": promoted,
        "trust_score": round(trust_score, 1),
        "edge": selected_edge,
        "edge_basis": edge_basis,
        "stake_hint": stake_hint,
        "top_candidates": ranked[:5],
    }
