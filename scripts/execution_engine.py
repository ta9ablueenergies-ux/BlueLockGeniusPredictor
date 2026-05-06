# ANTIGRAVITY V5.4 SHARP EXECUTION ENGINE
import pandas as pd
import numpy as np

def calculate_eqi_v2(value_edge, trust_score, volatility, cld_delta=0.0):
    """
    EQI V2 = (Value Edge * Trust * Stability * CLD_Factor) / Volatility
    Includes Market Intelligence (Closing Line Delta).
    """
    if value_edge <= 0: return 0.0, "NO EDGE"
    
    stability = max(0.1, 10 - volatility)
    
    # Market Intelligence Factor (CLD)
    cld_factor = 1.0
    if cld_delta > 0.01: cld_factor = 1.20   # Sharp money confirmation
    elif cld_delta < -0.01: cld_factor = 0.80 # Market drift / Trap warning
    
    raw_eqi = (value_edge * 100 * trust_score * stability * cld_factor) / max(1, volatility * 2)
    
    # Tiering V2 (Tighter Thresholds for V5.9)
    tier = "NO EDGE"
    if raw_eqi > 120: tier = "🔒 SHARP EDGE"
    elif raw_eqi > 80: tier = "✅ PLAYABLE"
    elif raw_eqi > 50: tier = "⚠️ WEAK EDGE"
    
    return round(raw_eqi, 1), tier

def calculate_stake(value_edge, trust_score, tier):
    """
    Kelly-Lite Capital Allocation.
    Ensures bankroll protection.
    """
    if tier == "NO EDGE" or value_edge <= 0: return 0
    
    # Scale value_edge from decimal (0.05) to integer (5%)
    edge_pct = value_edge * 100
    base_stake = (edge_pct * trust_score) / 100
    
    # Cap based on Tier
    if tier == "🔒 SHARP EDGE": return round(max(0, min(10, base_stake)), 1)
    if tier == "✅ PLAYABLE": return round(max(0, min(5, base_stake)), 1)
    if tier == "⚠️ WEAK EDGE": return round(max(0, min(2, base_stake)), 1)
    
    return 0

def map_dominance_markets(profile, p_h, p_o25, p_btts, p_a, p_d=0.0):
    """
    Market Priority Matrix (V10.2) - Calibrated Dominance Logic.
    
    KEY UPGRADES vs V6.2:
    - Draw is NEVER a primary market (24.8% accuracy — unplayable)
    - Calibrated thresholds derived from deep study (929 matches)
    - BTTS/O2.5 only offered when calibrated prob >= 0.62 (true ~75% events)
    - Home Win bias corrected: lower dominance threshold from 2.2x to 1.6x
    """
    # ── RULE 1: Outright Home Dominance (calibrated threshold) ───────────────
    # Old: p_h > p_a * 2.2  |  New: p_h > p_a * 1.6 (corrects home underestimation)
    if p_h > (p_a * 1.6) and p_h > 0.42:
        return {"primary": "Home Win", "secondary": "Over 1.5"}

    # ── RULE 2: Outright Away Dominance ──────────────────────────────────────
    if p_a > (p_h * 1.8) and p_a > 0.42:
        return {"primary": "Away Win", "secondary": "Away +1.5 Team Goals"}

    # ── RULE 3: High-Confidence Goals (calibrated — model 53% = true ~76%) ──
    if p_o25 >= 0.62:
        return {"primary": "Over 2.5", "secondary": "BTTS"}

    # ── RULE 4: BTTS Edge ────────────────────────────────────────────────────
    if p_btts >= 0.62 and p_o25 >= 0.55:
        return {"primary": "BTTS", "secondary": "Over 1.5"}

    # ── RULE 5: Defensive Profile ────────────────────────────────────────────
    if profile == "Defensive":
        return {"primary": "Under 2.5", "secondary": "No BTTS"}

    # ── RULE 6: Draw Guard — NEVER offer Draw as primary ─────────────────────
    # Draw ECE=0.2193 makes it unplayable. Redirect to safer markets.
    if p_d > 0.35:
        # High draw probability match — safest play is Double Chance
        if p_h >= p_a:
            return {"primary": "1X (Double Chance)", "secondary": "Over 1.5"}
        else:
            return {"primary": "X2 (Double Chance)", "secondary": "Over 1.5"}

    # ── RULE 7: Slight Home Edge ─────────────────────────────────────────────
    if p_h > p_a and p_h > 0.38:
        return {"primary": "Home Win", "secondary": "Over 1.5"}

    # ── RULE 8: Slight Away Edge ─────────────────────────────────────────────
    if p_a > p_h and p_a > 0.38:
        return {"primary": "X2 (Double Chance)", "secondary": "Over 1.5"}

    # ── RULE 9: Profile-Based Defaults (Draw fully removed) ──────────────────
    matrix = {
        "Surgical":   {"primary": "Home Win",        "secondary": "Over 1.5"},
        "Blitz":      {"primary": "BTTS",             "secondary": "Over 2.5"},
        "Trap":       {"primary": "X2 (Double Chance)","secondary": "Away +1.5"},
        "Stagnant":   {"primary": "Under 2.5",        "secondary": "No BTTS"},
        "High Tempo": {"primary": "Over 2.5",         "secondary": "BTTS"},
        "Balanced":   {"primary": "Over 1.5",         "secondary": "1X (Double Chance)"},
    }
    return matrix.get(profile, {"primary": "Over 1.5", "secondary": "1X (Double Chance)"})

def _safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def trust_score_from_eqi(value):
    """Translate the legacy EQI value into a user-facing 1-100 Trust Score."""
    score = _safe_float(value, 0.0)
    return round(max(1.0, min(100.0, score)), 1)


def _market_probability(match):
    primary = str(match.get("primary_market") or match.get("best_market") or match.get("prediction") or "").lower()
    router = match.get("market_router") if isinstance(match.get("market_router"), dict) else {}
    if router:
        selected = str(router.get("selected_market") or "").lower()
        selected_probability = _safe_float(router.get("selected_probability"), 0.0)
        if selected and selected == primary and selected_probability > 0:
            return selected_probability

    market_model = match.get("market_model") if isinstance(match.get("market_model"), dict) else {}
    corners = market_model.get("corners", {}) if isinstance(market_model, dict) else {}
    cards = market_model.get("cards", {}) if isinstance(market_model, dict) else {}
    count_mapping = [
        ("corners over 8.5", corners.get("over_8_5") if isinstance(corners, dict) else None),
        ("corners over 9.5", corners.get("over_9_5") if isinstance(corners, dict) else None),
        ("corners over 10.5", corners.get("over_10_5") if isinstance(corners, dict) else None),
        ("cards over 3.5", cards.get("over_3_5") if isinstance(cards, dict) else None),
        ("cards over 4.5", cards.get("over_4_5") if isinstance(cards, dict) else None),
        ("cards over 5.5", cards.get("over_5_5") if isinstance(cards, dict) else None),
        ("under 1.5", match.get("under15_prob")),
        ("under 2.5", 1.0 - _safe_float(match.get("P(O2.5)"), 0.0)),
        ("under 3.5", match.get("under35_prob")),
    ]
    for marker, probability in count_mapping:
        if marker in primary:
            return _safe_float(probability, 0.0)

    mapping = [
        ("btts", "P(BTTS)"),
        ("over 2.5", "P(O2.5)"),
        ("over 1.5", "over15_prob"),
        ("over 3.5", "over35_prob"),
        ("1x", "P(1X)"),
        ("x2", "P(X2)"),
        ("dnb", "DNB"),
    ]
    for marker, key in mapping:
        if marker in primary:
            return _safe_float(match.get(key), 0.0)
    if "home" in primary:
        return _safe_float(match.get("P(1X)"), 0.0) * 0.72
    if "away" in primary:
        return _safe_float(match.get("P(X2)"), 0.0) * 0.72
    return max(
        _safe_float(match.get("P(BTTS)"), 0.0),
        _safe_float(match.get("P(O2.5)"), 0.0),
        _safe_float(match.get("over15_prob"), 0.0),
        _safe_float(match.get("P(1X)"), 0.0),
        _safe_float(match.get("P(X2)"), 0.0),
    )


def _prepare_ticket_candidate(match):
    home = match.get("Home") or match.get("home_team")
    away = match.get("Away") or match.get("away_team")
    if not home or not away:
        return None

    raw_eqi = (
        match.get("trust_score")
        if match.get("trust_score") is not None
        else match.get("Trust Score")
        if match.get("Trust Score") is not None
        else match.get("execution_trust")
        if match.get("execution_trust") is not None
        else match.get("eqi")
    )
    trust_score = trust_score_from_eqi(raw_eqi)
    market_prob = _market_probability(match)
    source_conf = _safe_float(match.get("source_confidence") or match.get("Source Confidence"), 0.5)
    value_edge = max(0.0, _safe_float(match.get("Value Edge") or match.get("value_edge"), 0.0))
    stake_pct = _safe_float(match.get("stake_pct"), 0.0)

    ticket_score = (
        trust_score * 0.45
        + min(1.0, market_prob) * 35.0
        + min(1.0, source_conf) * 15.0
        + min(0.5, value_edge) * 20.0
        + min(10.0, stake_pct) * 0.5
    )

    candidate = dict(match)
    candidate["Home"] = home
    candidate["Away"] = away
    candidate["League"] = match.get("League") or match.get("league") or "GLOBAL"
    candidate["Date"] = match.get("Date") or match.get("match_date") or match.get("date") or ""
    candidate["best_market"] = match.get("primary_market") or match.get("best_market") or match.get("prediction") or "Best Market"
    candidate["prediction"] = match.get("prediction") or candidate["best_market"]
    candidate["trust_score"] = trust_score
    candidate["eqi"] = trust_score
    candidate["ticket_score"] = round(ticket_score, 2)
    candidate["market_probability"] = round(market_prob, 4)
    candidate["stake_pct"] = round(stake_pct, 1)
    return candidate


def _select_legs(candidates, max_legs=3, unique_leagues=False):
    ticket = []
    leagues_used = set()
    teams_used = set()
    for item in candidates:
        league = item.get("League", "GLOBAL")
        teams = {item.get("Home"), item.get("Away")}
        if unique_leagues and league in leagues_used:
            continue
        if teams_used.intersection(teams):
            continue
        ticket.append(item)
        leagues_used.add(league)
        teams_used.update(teams)
        if len(ticket) >= max_legs:
            break
    return ticket


def construct_quantum_ticket_bundle(matches, max_legs=3):
    """
    Build global/league ticket bundles from current 1-100 Trust Scores.

    The old engine only emitted tickets when legacy EQI crossed 80/120, which
    left valid fixture pages empty after the scoring scale was normalized. This
    version always emits the best available ticket when real matches exist.
    """
    candidates = []
    for match in matches or []:
        prepared = _prepare_ticket_candidate(match)
        if prepared:
            candidates.append(prepared)

    empty = {
        "TYPE_A_ULTRA_SAFE": [],
        "TYPE_B_BALANCED": [],
        "TYPE_C_VALUE": [],
    }
    if not candidates:
        return empty

    unique_leagues = len({item.get("League") for item in candidates}) > 1
    by_safety = sorted(
        candidates,
        key=lambda item: (item["ticket_score"], item["trust_score"], item["market_probability"]),
        reverse=True,
    )
    by_balance = sorted(
        candidates,
        key=lambda item: (item["market_probability"], item["trust_score"], item["ticket_score"]),
        reverse=True,
    )
    by_value = sorted(
        candidates,
        key=lambda item: (_safe_float(item.get("Value Edge") or item.get("value_edge"), 0.0), item["ticket_score"]),
        reverse=True,
    )

    return {
        "TYPE_A_ULTRA_SAFE": _select_legs(by_safety, max_legs=max_legs, unique_leagues=unique_leagues),
        "TYPE_B_BALANCED": _select_legs(by_balance, max_legs=max_legs, unique_leagues=unique_leagues),
        "TYPE_C_VALUE": _select_legs(by_value, max_legs=max_legs, unique_leagues=unique_leagues),
    }


def construct_quantum_accumulator_v54(matches):
    """Backward-compatible accumulator: return the safest ticket legs."""
    return construct_quantum_ticket_bundle(matches).get("TYPE_A_ULTRA_SAFE", [])


def construct_tickets(matches):
    """Legacy wrapper for construct_quantum_accumulator_v54."""
    return construct_quantum_accumulator_v54(matches)
