"""Import open event datasets into why-signal evidence and training stats.

Supported local layouts:
- StatsBomb open-data repository layout under external_data/statsbomb-open-data/data
- Wyscout public event dataset files under external_data/wyscout

The importer aggregates event JSON into match-level signals that the current
pipeline can learn from: shots, shots on target, xG, corners, cards, fouls,
possession share, and derived post-match pattern tags.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXTERNAL_DATA_DIR = os.path.join(PROJECT_ROOT, "external_data")
STATSBOMB_DIR = os.path.join(EXTERNAL_DATA_DIR, "statsbomb-open-data")
STATSBOMB_DATA_DIR = os.path.join(STATSBOMB_DIR, "data")
WYSCOUT_DIR = os.path.join(EXTERNAL_DATA_DIR, "wyscout")
REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "event_data_import_report.json")
sys.path.insert(0, SCRIPT_DIR)

from persistence_manager import (  # noqa: E402
    DB_PATH,
    WHY_STAT_COLUMNS,
    init_db,
    resolve_team_name,
    upsert_match_learning_patterns,
    upsert_raw_match_evidence,
)
from why_signal_extractor import derive_post_match_patterns  # noqa: E402


COUNT_STAT_COLUMNS = ["HY", "AY", "HR", "AR", "HC", "AC"]
STAT_COLUMNS = COUNT_STAT_COLUMNS + WHY_STAT_COLUMNS
RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

STATSBOMB_LEAGUES = {
    "Premier League": "PremierLeague",
    "La Liga": "LaLiga",
    "Serie A": "SerieA",
    "1. Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue1",
}

WYSCOUT_COUNTRIES = {
    "England": "PremierLeague",
    "Spain": "LaLiga",
    "Italy": "SerieA",
    "Germany": "Bundesliga",
    "France": "Ligue1",
}

WYSCOUT_DOWNLOADS = {
    "events.zip": "https://figshare.com/ndownloader/files/14464685/events.zip",
    "matches.zip": "https://figshare.com/ndownloader/files/14464622/matches.zip",
    "teams.json": "https://figshare.com/ndownloader/files/15073697/teams.json",
    "players.json": "https://figshare.com/ndownloader/files/15073721/players.json",
}


def _safe_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def _download(url: str, path: str, overwrite: bool = False) -> bool:
    if os.path.exists(path) and not overwrite:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "football-predictions-research-importer/1.0",
            "Accept": "application/json, application/zip, */*",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp_path, "wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(tmp_path, path)
    return True


def download_wyscout(overwrite: bool = False) -> Dict[str, object]:
    os.makedirs(WYSCOUT_DIR, exist_ok=True)
    downloaded = []
    reused = []
    for filename, url in WYSCOUT_DOWNLOADS.items():
        path = os.path.join(WYSCOUT_DIR, filename)
        changed = _download(url, path, overwrite=overwrite)
        (downloaded if changed else reused).append(filename)
    return {"downloaded": downloaded, "reused": reused}


def download_statsbomb(
    leagues: Optional[Sequence[str]] = None,
    max_matches: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, object]:
    leagues = set(leagues or STATSBOMB_LEAGUES.values())
    competitions_path = os.path.join(STATSBOMB_DATA_DIR, "competitions.json")
    _download(f"{RAW_BASE}/competitions.json", competitions_path, overwrite=overwrite)
    competitions = _load_json(competitions_path)
    downloaded_matches = 0
    downloaded_events = 0
    selected = []

    for comp in competitions:
        league = STATSBOMB_LEAGUES.get(comp.get("competition_name"))
        if not league or league not in leagues:
            continue
        comp_id = comp.get("competition_id")
        season_id = comp.get("season_id")
        if comp_id is None or season_id is None:
            continue
        matches_path = os.path.join(STATSBOMB_DATA_DIR, "matches", str(comp_id), f"{season_id}.json")
        changed = _download(f"{RAW_BASE}/matches/{comp_id}/{season_id}.json", matches_path, overwrite=overwrite)
        downloaded_matches += int(changed)
        selected.append({
            "league": league,
            "competition_id": comp_id,
            "season_id": season_id,
            "season_name": comp.get("season_name"),
        })
        try:
            matches = _load_json(matches_path)
        except Exception:
            continue
        for match in matches:
            if max_matches is not None and downloaded_events >= max_matches:
                break
            match_id = match.get("match_id")
            if match_id is None:
                continue
            events_path = os.path.join(STATSBOMB_DATA_DIR, "events", f"{match_id}.json")
            changed = _download(f"{RAW_BASE}/events/{match_id}.json", events_path, overwrite=overwrite)
            downloaded_events += int(changed)
        if max_matches is not None and downloaded_events >= max_matches:
            break

    return {
        "competitions_seen": len(competitions),
        "selected_competitions": selected,
        "matches_files_downloaded": downloaded_matches,
        "event_files_downloaded": downloaded_events,
    }


def _goal_result(home_goals: Optional[int], away_goals: Optional[int]) -> Optional[str]:
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if away_goals > home_goals:
        return "A"
    return "D"


def _canonical(league: str, team_name: object) -> str:
    return resolve_team_name(league, str(team_name or "").strip())


def _training_example_id(league: str, match_date: str, home_team: str, away_team: str) -> Optional[str]:
    if not os.path.exists(DB_PATH):
        return None
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        """
        SELECT id
        FROM training_examples
        WHERE league = ?
          AND match_date = ?
          AND home_team = ?
          AND away_team = ?
        ORDER BY source_confidence DESC, updated_at DESC
        LIMIT 1
        """,
        (league, match_date, home_team, away_team),
    ).fetchone()
    con.close()
    return row[0] if row else None


def _fallback_match_id(source: str, league: str, match_date: str, home_team: str, away_team: str) -> str:
    raw = f"event_{source}_{league}_{match_date}_{home_team}_{away_team}"
    return raw.replace("/", "-").replace("\\", "-")


def _merge_features(existing_json: str, event_features: Dict[str, object]) -> str:
    try:
        payload = json.loads(existing_json or "{}")
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    existing_event = payload.get("event_learning") or {}
    if not isinstance(existing_event, dict):
        existing_event = {}
    existing_event.update(event_features)
    payload["event_learning"] = existing_event
    return json.dumps(payload)


def _update_training_example(match_id: str, stats: Dict[str, object], features: Dict[str, object]) -> bool:
    if not match_id or not os.path.exists(DB_PATH):
        return False
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT features_json FROM training_examples WHERE id = ?", (match_id,)).fetchone()
    if not row:
        con.close()
        return False
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    set_stats = ", ".join(f'"{col}" = COALESCE(?, "{col}")' for col in STAT_COLUMNS)
    merged_features = _merge_features(row["features_json"], features)
    values = [stats.get(col) for col in STAT_COLUMNS]
    con.execute(
        f"""
        UPDATE training_examples
        SET {set_stats},
            features_json = ?,
            stats_scraped_at = COALESCE(stats_scraped_at, ?),
            updated_at = ?
        WHERE id = ?
        """,
        (*values, merged_features, now, now, match_id),
    )
    con.commit()
    con.close()
    return True


def _persist_event_match(
    *,
    source: str,
    source_url: str,
    source_match_id: object,
    league: str,
    match_date: str,
    home_team: str,
    away_team: str,
    stats: Dict[str, object],
    features: Dict[str, object],
    raw_summary: Dict[str, object],
    result_row: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    match_id = _training_example_id(league, match_date, home_team, away_team)
    matched_training_example = bool(match_id)
    if not match_id:
        match_id = _fallback_match_id(source, league, match_date, home_team, away_team)

    patterns = derive_post_match_patterns(stats, result_row or {})
    if features:
        for key, value in sorted(features.items()):
            if key.startswith("pattern_") and value:
                patterns.append({
                    "pattern_key": key.replace("pattern_", ""),
                    "confidence": 0.62,
                    "details": {"source_feature": key, "value": value},
                })

    upsert_raw_match_evidence({
        "id": f"{source}:{source_match_id}:event_aggregate",
        "match_id": match_id,
        "league": league,
        "match_date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "source": source,
        "source_url": source_url,
        "evidence_type": "event_aggregate",
        "raw": raw_summary,
        "extracted": {"stats": stats, "features": features, "patterns": patterns},
        "data_quality": event_data_quality(stats, features),
    })
    pattern_count = upsert_match_learning_patterns(match_id, patterns, source=f"{source}_event_import")
    updated_training = _update_training_example(match_id, stats, features) if matched_training_example else False
    return {
        "match_id": match_id,
        "matched_training_example": matched_training_example,
        "updated_training_example": updated_training,
        "patterns": pattern_count,
    }


def event_data_quality(stats: Dict[str, object], features: Dict[str, object]) -> float:
    core = ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY"]
    advanced = ["HXG", "AXG", "HPoss", "APoss", "HF", "AF"]
    core_rate = sum(1 for col in core if stats.get(col) is not None) / len(core)
    advanced_rate = sum(1 for col in advanced if stats.get(col) is not None) / len(advanced)
    event_bonus = 0.08 if features.get("total_events", 0) else 0.0
    return round(min(1.0, 0.18 + 0.58 * core_rate + 0.16 * advanced_rate + event_bonus), 4)


def _statsbomb_team_name(match: Dict[str, object], side: str) -> str:
    key = f"{side}_team"
    payload = match.get(key) or {}
    return payload.get(f"{side}_team_name") or payload.get("name") or ""


def _statsbomb_score(match: Dict[str, object], side: str) -> Optional[int]:
    value = match.get(f"{side}_score")
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _event_team(event: Dict[str, object]) -> str:
    team = event.get("team") or {}
    return team.get("name") or ""


def _event_type(event: Dict[str, object]) -> str:
    typ = event.get("type") or {}
    return typ.get("name") or ""


def _sub_name(event: Dict[str, object], key: str) -> str:
    payload = event.get(key) or {}
    value = payload.get("name")
    if value is not None:
        return str(value)
    typ = payload.get("type") or {}
    return str(typ.get("name") or "")


def _is_statsbomb_sot(event: Dict[str, object]) -> bool:
    shot = event.get("shot") or {}
    outcome = _sub_name(shot, "outcome")
    return outcome in {"Goal", "Saved", "Saved to Post"}


def _statsbomb_xg(event: Dict[str, object]) -> Optional[float]:
    shot = event.get("shot") or {}
    return _safe_float(shot.get("statsbomb_xg"))


def _aggregate_statsbomb_events(
    events: Sequence[Dict[str, object]],
    league: str,
    home_team: str,
    away_team: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    teams = {home_team: "home", away_team: "away"}
    counters = {
        "home": Counter(),
        "away": Counter(),
    }
    xg = {"home": 0.0, "away": 0.0}
    possession = Counter()
    big_chances = {"home": 0, "away": 0}
    early_goals = 0
    late_goals = 0
    pressure_events = Counter()

    for event in events:
        team = _canonical(league, _event_team(event))
        side = teams.get(team)
        if not side:
            continue
        event_type = _event_type(event)
        counters[side]["events"] += 1
        poss_team = event.get("possession_team") or {}
        poss_side = teams.get(_canonical(league, poss_team.get("name")))
        if poss_side:
            possession[poss_side] += 1
        if event_type == "Shot":
            counters[side]["shots"] += 1
            if _is_statsbomb_sot(event):
                counters[side]["sot"] += 1
            shot_xg = _statsbomb_xg(event)
            if shot_xg is not None:
                xg[side] += shot_xg
            if bool((event.get("shot") or {}).get("one_on_one")):
                big_chances[side] += 1
            if _sub_name(event.get("shot") or {}, "outcome") == "Goal":
                minute = int(event.get("minute") or 0)
                early_goals += int(minute <= 15)
                late_goals += int(minute >= 75)
        elif event_type == "Pass":
            pass_type = _sub_name(event, "pass")
            if pass_type == "Corner":
                counters[side]["corners"] += 1
        elif event_type == "Foul Committed":
            counters[side]["fouls"] += 1
            card = _sub_name(event.get("foul_committed") or {}, "card")
            if "Yellow" in card:
                counters[side]["yellow"] += 1
            if "Red" in card:
                counters[side]["red"] += 1
        elif event_type == "Bad Behaviour":
            card = _sub_name(event.get("bad_behaviour") or {}, "card")
            if "Yellow" in card:
                counters[side]["yellow"] += 1
            if "Red" in card:
                counters[side]["red"] += 1
        elif event_type == "Offside":
            counters[side]["offsides"] += 1
        elif event_type in {"Pressure", "Duel", "Interception"}:
            pressure_events[side] += 1

    poss_total = possession["home"] + possession["away"]
    total_pressure = pressure_events["home"] + pressure_events["away"]
    stats = {
        "HS": counters["home"]["shots"],
        "AS": counters["away"]["shots"],
        "HST": counters["home"]["sot"],
        "AST": counters["away"]["sot"],
        "HF": counters["home"]["fouls"],
        "AF": counters["away"]["fouls"],
        "HO": counters["home"]["offsides"],
        "AO": counters["away"]["offsides"],
        "HPoss": _round(100.0 * possession["home"] / poss_total, 2) if poss_total else None,
        "APoss": _round(100.0 * possession["away"] / poss_total, 2) if poss_total else None,
        "HXG": _round(xg["home"], 3),
        "AXG": _round(xg["away"], 3),
        "HBC": big_chances["home"] or None,
        "ABC": big_chances["away"] or None,
        "HC": counters["home"]["corners"],
        "AC": counters["away"]["corners"],
        "HY": counters["home"]["yellow"],
        "AY": counters["away"]["yellow"],
        "HR": counters["home"]["red"],
        "AR": counters["away"]["red"],
    }
    features = {
        "total_events": len(events),
        "home_pressure_events": pressure_events["home"],
        "away_pressure_events": pressure_events["away"],
        "home_pressure_share": _round(pressure_events["home"] / total_pressure, 4) if total_pressure else None,
        "pattern_early_goal_state_change": early_goals > 0,
        "pattern_late_goal_state_change": late_goals > 0,
        "pattern_transition_intensity": total_pressure >= 90,
    }
    return stats, features


def import_statsbomb(
    leagues: Optional[Sequence[str]] = None,
    max_matches: Optional[int] = None,
) -> Dict[str, object]:
    competitions_path = os.path.join(STATSBOMB_DATA_DIR, "competitions.json")
    if not os.path.exists(competitions_path):
        return {"source": "statsbomb", "status": "missing", "missing_path": competitions_path}

    leagues = set(leagues or STATSBOMB_LEAGUES.values())
    competitions = _load_json(competitions_path)
    report = {
        "source": "statsbomb",
        "status": "imported",
        "matches_seen": 0,
        "matches_imported": 0,
        "matched_training_examples": 0,
        "updated_training_examples": 0,
        "patterns_written": 0,
        "per_league": defaultdict(int),
    }

    for comp in competitions:
        league = STATSBOMB_LEAGUES.get(comp.get("competition_name"))
        if not league or league not in leagues:
            continue
        matches_path = os.path.join(
            STATSBOMB_DATA_DIR,
            "matches",
            str(comp.get("competition_id")),
            f"{comp.get('season_id')}.json",
        )
        if not os.path.exists(matches_path):
            continue
        matches = _load_json(matches_path)
        for match in matches:
            if max_matches is not None and report["matches_imported"] >= max_matches:
                break
            report["matches_seen"] += 1
            source_match_id = match.get("match_id")
            events_path = os.path.join(STATSBOMB_DATA_DIR, "events", f"{source_match_id}.json")
            if not os.path.exists(events_path):
                continue
            try:
                events = _load_json(events_path)
            except Exception:
                continue
            home_team = _canonical(league, _statsbomb_team_name(match, "home"))
            away_team = _canonical(league, _statsbomb_team_name(match, "away"))
            match_date = str(match.get("match_date") or "")[:10]
            if not home_team or not away_team or not match_date:
                continue
            stats, features = _aggregate_statsbomb_events(events, league, home_team, away_team)
            home_goals = _statsbomb_score(match, "home")
            away_goals = _statsbomb_score(match, "away")
            result_row = {
                "home_goals": home_goals,
                "away_goals": away_goals,
                "actual_ftr": _goal_result(home_goals, away_goals),
            }
            persisted = _persist_event_match(
                source="statsbomb_open_data",
                source_url=f"https://github.com/statsbomb/open-data/blob/master/data/events/{source_match_id}.json",
                source_match_id=source_match_id,
                league=league,
                match_date=match_date,
                home_team=home_team,
                away_team=away_team,
                stats=stats,
                features=features,
                raw_summary={
                    "competition_name": comp.get("competition_name"),
                    "season_name": comp.get("season_name"),
                    "source_match_id": source_match_id,
                    "event_count": len(events),
                },
                result_row=result_row,
            )
            report["matches_imported"] += 1
            report["matched_training_examples"] += int(persisted["matched_training_example"])
            report["updated_training_examples"] += int(persisted["updated_training_example"])
            report["patterns_written"] += int(persisted["patterns"])
            report["per_league"][league] += 1
        if max_matches is not None and report["matches_imported"] >= max_matches:
            break

    report["per_league"] = dict(sorted(report["per_league"].items()))
    return report


def _wyscout_zip_json(zip_name: str, member_name: str) -> Optional[object]:
    path = os.path.join(WYSCOUT_DIR, zip_name)
    if not os.path.exists(path):
        return None
    with zipfile.ZipFile(path) as archive:
        with archive.open(member_name) as handle:
            return json.loads(handle.read().decode("utf-8"))


def _load_wyscout_teams() -> Dict[int, str]:
    path = os.path.join(WYSCOUT_DIR, "teams.json")
    if not os.path.exists(path):
        return {}
    teams = _load_json(path)
    out = {}
    for team in teams:
        try:
            out[int(team["wyId"])] = team.get("name") or team.get("officialName") or ""
        except Exception:
            continue
    return out


def _wyscout_match_teams(match: Dict[str, object], teams: Dict[int, str]) -> Tuple[Optional[int], Optional[int]]:
    teams_data = match.get("teamsData") or {}
    home_id = away_id = None
    for raw_id, data in teams_data.items():
        side = str((data or {}).get("side") or "").lower()
        try:
            team_id = int(raw_id)
        except Exception:
            continue
        if side == "home":
            home_id = team_id
        elif side == "away":
            away_id = team_id
    return home_id, away_id


def _wyscout_scores(match: Dict[str, object], home_id: Optional[int], away_id: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    teams_data = match.get("teamsData") or {}
    def score(team_id: Optional[int]) -> Optional[int]:
        if team_id is None:
            return None
        data = teams_data.get(str(team_id)) or teams_data.get(team_id) or {}
        try:
            return int(data.get("score")) if data.get("score") is not None else None
        except Exception:
            return None
    return score(home_id), score(away_id)


def _tag_ids(event: Dict[str, object]) -> set:
    tags = event.get("tags") or []
    out = set()
    for tag in tags:
        try:
            out.add(int(tag.get("id")))
        except Exception:
            continue
    return out


def _aggregate_wyscout_events(
    events: Sequence[Dict[str, object]],
    home_id: int,
    away_id: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    side_for_id = {home_id: "home", away_id: "away"}
    counters = {"home": Counter(), "away": Counter()}
    early_goals = 0
    late_goals = 0
    progressive_events = Counter()

    for event in events:
        try:
            team_id = int(event.get("teamId"))
        except Exception:
            continue
        side = side_for_id.get(team_id)
        if not side:
            continue
        event_name = str(event.get("eventName") or "")
        sub_name = str(event.get("subEventName") or "")
        tags = _tag_ids(event)
        counters[side]["events"] += 1
        if event_name == "Shot":
            counters[side]["shots"] += 1
            if 1801 in tags or 101 in tags:
                counters[side]["sot"] += 1
            if 101 in tags:
                minute = int(float(event.get("eventSec") or 0) // 60)
                early_goals += int(minute <= 15)
                late_goals += int(minute >= 75)
        elif event_name == "Free Kick" and sub_name == "Corner":
            counters[side]["corners"] += 1
        elif event_name == "Foul":
            counters[side]["fouls"] += 1
            if 1702 in tags:
                counters[side]["yellow"] += 1
            if 1701 in tags or 1703 in tags:
                counters[side]["red"] += 1
        elif event_name == "Offside":
            counters[side]["offsides"] += 1

        if event_name in {"Duel", "Pass"} and 1801 in tags:
            progressive_events[side] += 1

    possession_total = counters["home"]["events"] + counters["away"]["events"]
    progressive_total = progressive_events["home"] + progressive_events["away"]
    stats = {
        "HS": counters["home"]["shots"],
        "AS": counters["away"]["shots"],
        "HST": counters["home"]["sot"],
        "AST": counters["away"]["sot"],
        "HF": counters["home"]["fouls"],
        "AF": counters["away"]["fouls"],
        "HO": counters["home"]["offsides"],
        "AO": counters["away"]["offsides"],
        "HPoss": _round(100.0 * counters["home"]["events"] / possession_total, 2) if possession_total else None,
        "APoss": _round(100.0 * counters["away"]["events"] / possession_total, 2) if possession_total else None,
        "HXG": None,
        "AXG": None,
        "HBC": None,
        "ABC": None,
        "HC": counters["home"]["corners"],
        "AC": counters["away"]["corners"],
        "HY": counters["home"]["yellow"],
        "AY": counters["away"]["yellow"],
        "HR": counters["home"]["red"],
        "AR": counters["away"]["red"],
    }
    features = {
        "total_events": len(events),
        "home_accurate_progressive_events": progressive_events["home"],
        "away_accurate_progressive_events": progressive_events["away"],
        "home_progressive_share": _round(progressive_events["home"] / progressive_total, 4) if progressive_total else None,
        "pattern_early_goal_state_change": early_goals > 0,
        "pattern_late_goal_state_change": late_goals > 0,
        "pattern_transition_intensity": progressive_total >= 600,
    }
    return stats, features


def import_wyscout(
    leagues: Optional[Sequence[str]] = None,
    max_matches: Optional[int] = None,
) -> Dict[str, object]:
    leagues = set(leagues or WYSCOUT_COUNTRIES.values())
    teams = _load_wyscout_teams()
    if not teams:
        return {"source": "wyscout", "status": "missing", "missing_path": os.path.join(WYSCOUT_DIR, "teams.json")}

    report = {
        "source": "wyscout",
        "status": "imported",
        "matches_seen": 0,
        "matches_imported": 0,
        "matched_training_examples": 0,
        "updated_training_examples": 0,
        "patterns_written": 0,
        "per_league": defaultdict(int),
    }

    for country, league in WYSCOUT_COUNTRIES.items():
        if league not in leagues:
            continue
        matches = _wyscout_zip_json("matches.zip", f"matches_{country}.json")
        events = _wyscout_zip_json("events.zip", f"events_{country}.json")
        if not matches or not events:
            continue
        events_by_match = defaultdict(list)
        for event in events:
            events_by_match[event.get("matchId")].append(event)
        for match in matches:
            if max_matches is not None and report["matches_imported"] >= max_matches:
                break
            report["matches_seen"] += 1
            source_match_id = match.get("wyId")
            home_id, away_id = _wyscout_match_teams(match, teams)
            if home_id is None or away_id is None:
                continue
            match_events = events_by_match.get(source_match_id, [])
            if not match_events:
                continue
            home_team = _canonical(league, teams.get(home_id))
            away_team = _canonical(league, teams.get(away_id))
            match_date = str(match.get("dateutc") or match.get("date") or "")[:10]
            if not match_date or not home_team or not away_team:
                continue
            stats, features = _aggregate_wyscout_events(match_events, home_id, away_id)
            home_goals, away_goals = _wyscout_scores(match, home_id, away_id)
            persisted = _persist_event_match(
                source="wyscout_public_events",
                source_url=f"https://figshare.com/collections/Soccer_match_event_dataset/4415000",
                source_match_id=source_match_id,
                league=league,
                match_date=match_date,
                home_team=home_team,
                away_team=away_team,
                stats=stats,
                features=features,
                raw_summary={
                    "country": country,
                    "source_match_id": source_match_id,
                    "event_count": len(match_events),
                    "competitionId": match.get("competitionId"),
                },
                result_row={
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "actual_ftr": _goal_result(home_goals, away_goals),
                },
            )
            report["matches_imported"] += 1
            report["matched_training_examples"] += int(persisted["matched_training_example"])
            report["updated_training_examples"] += int(persisted["updated_training_example"])
            report["patterns_written"] += int(persisted["patterns"])
            report["per_league"][league] += 1
        if max_matches is not None and report["matches_imported"] >= max_matches:
            break

    report["per_league"] = dict(sorted(report["per_league"].items()))
    return report


def run_event_data_import(
    sources: Sequence[str] = ("statsbomb", "wyscout"),
    download: bool = False,
    leagues: Optional[Sequence[str]] = None,
    max_matches_per_source: Optional[int] = None,
    overwrite_downloads: bool = False,
) -> Dict[str, object]:
    init_db()
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report: Dict[str, object] = {
        "created_at": datetime.utcnow().isoformat(),
        "sources_requested": list(sources),
        "download": {},
        "imports": {},
    }
    if download and "wyscout" in sources:
        report["download"]["wyscout"] = download_wyscout(overwrite=overwrite_downloads)
    if download and "statsbomb" in sources:
        report["download"]["statsbomb"] = download_statsbomb(
            leagues=leagues,
            max_matches=max_matches_per_source,
            overwrite=overwrite_downloads,
        )

    if "statsbomb" in sources:
        report["imports"]["statsbomb"] = import_statsbomb(
            leagues=leagues,
            max_matches=max_matches_per_source,
        )
    if "wyscout" in sources:
        report["imports"]["wyscout"] = import_wyscout(
            leagues=leagues,
            max_matches=max_matches_per_source,
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["statsbomb", "wyscout"], choices=["statsbomb", "wyscout"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite-downloads", action="store_true")
    parser.add_argument("--leagues", default=None, help="Comma-separated canonical leagues, e.g. PremierLeague,LaLiga")
    parser.add_argument("--max-matches-per-source", type=int, default=None)
    args = parser.parse_args()
    leagues = [item.strip() for item in args.leagues.split(",") if item.strip()] if args.leagues else None
    print(json.dumps(
        run_event_data_import(
            sources=args.sources,
            download=args.download,
            leagues=leagues,
            max_matches_per_source=args.max_matches_per_source,
            overwrite_downloads=args.overwrite_downloads,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
