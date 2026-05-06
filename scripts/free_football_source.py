"""Free football data source adapters.

This module provides optional ingestion from public, no-API datasets:
 - openfootball / football.json style JSON schedules
 - football-data.co.uk CSV mirrors such as datasets/football-datasets
 - local cached archives already present in this repo

The module is intentionally defensive: if a source is unavailable, it
falls back to the next one without failing the whole pipeline.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

from team_name_normalizer import canonical_fixture_key, canonical_team_name


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXTERNAL_REPOS_DIR = os.path.join(PROJECT_ROOT, "external_repos")
SOCCERDATA_DIR = os.path.join(EXTERNAL_REPOS_DIR, "soccerdata_cache")
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")
os.environ.setdefault("SOCCERDATA_DIR", SOCCERDATA_DIR)


@dataclass
class FreeMatch:
    league: str
    match_date: str
    home_team: str
    away_team: str
    kickoff_utc: Optional[str] = None
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    actual_ftr: Optional[str] = None
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None
    source_url: Optional[str] = None
    source_id: Optional[str] = None
    source: str = "free_dataset"


def _safe_date(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text[:10], fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _normalize_date_key(value: object) -> Optional[str]:
    date_value = _safe_date(value)
    if date_value:
        return date_value
    text = str(value or "").strip()
    return text[:10] if text else None


def _ftr_from_goals(home_goals: Optional[int], away_goals: Optional[int]) -> Optional[str]:
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _match_row_from_csv_row(league: str, row: dict) -> Optional[FreeMatch]:
    match_date = _safe_date(row.get("Date") or row.get("MatchDate"))
    home_team = str(row.get("HomeTeam") or row.get("Home") or "").strip()
    away_team = str(row.get("AwayTeam") or row.get("Away") or "").strip()
    if not match_date or not home_team or not away_team:
        return None

    home_goals = _safe_int(row.get("FTHG") or row.get("FTHome"))
    away_goals = _safe_int(row.get("FTAG") or row.get("FTAway"))
    actual_ftr = str(row.get("FTR") or row.get("FTResult") or _ftr_from_goals(home_goals, away_goals) or "").strip()[:1]

    return FreeMatch(
        league=league,
        match_date=match_date,
        home_team=home_team,
        away_team=away_team,
        home_goals=home_goals,
        away_goals=away_goals,
        actual_ftr=actual_ftr if actual_ftr in {"H", "D", "A"} else None,
        home_odds=_safe_float(row.get("B365H") or row.get("OddHome") or row.get("home_odds")),
        draw_odds=_safe_float(row.get("B365D") or row.get("OddDraw") or row.get("draw_odds")),
        away_odds=_safe_float(row.get("B365A") or row.get("OddAway") or row.get("away_odds")),
        source=str(row.get("source") or "free_dataset"),
    )


def load_from_local_csv(path: str, league: str) -> List[FreeMatch]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, low_memory=False)
    matches: List[FreeMatch] = []
    for row in df.to_dict("records"):
        item = _match_row_from_csv_row(league, row)
        if item:
            matches.append(item)
    return matches


def load_from_zip(zip_path: str) -> List[FreeMatch]:
    if not os.path.exists(zip_path):
        return []
    matches: List[FreeMatch] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            league = os.path.basename(name).split("_", 1)[0]
            with zf.open(name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
                reader = csv.DictReader(text)
                for row in reader:
                    item = _match_row_from_csv_row(league, row)
                    if item:
                        matches.append(item)
    return matches


def load_from_openfootball_json(url: str, league: str) -> List[FreeMatch]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    matches: List[FreeMatch] = []
    for item in payload if isinstance(payload, list) else payload.get("matches", []):
        home = item.get("home_team") or item.get("homeTeam") or item.get("home")
        away = item.get("away_team") or item.get("awayTeam") or item.get("away")
        date_value = item.get("date") or item.get("match_date") or item.get("Date")
        match_date = _safe_date(date_value)
        if not match_date or not home or not away:
            continue
        home_goals = _safe_int(item.get("home_goals") or item.get("FTHG"))
        away_goals = _safe_int(item.get("away_goals") or item.get("FTAG"))
        matches.append(
            FreeMatch(
                league=league,
                match_date=match_date,
                home_team=str(home).strip(),
                away_team=str(away).strip(),
                home_goals=home_goals,
                away_goals=away_goals,
                actual_ftr=_ftr_from_goals(home_goals, away_goals),
                source="openfootball",
            )
        )
    return matches


def load_openfootball_repo(
    repo_root: Optional[str] = None,
    leagues: Optional[List[str]] = None,
    match_date: Optional[str] = None,
) -> List[FreeMatch]:
    """Load all matches from a local openfootball/football.json clone."""
    repo_root = repo_root or os.path.join(EXTERNAL_REPOS_DIR, "football.json")
    if not os.path.isdir(repo_root):
        return []
    allowed = set(leagues) if leagues else None
    target_date = _normalize_date_key(match_date)
    matches: List[FreeMatch] = []
    league_map = {
        "en.1.json": "PremierLeague",
        "en.2.json": "Championship",
        "es.1.json": "LaLiga",
        "es.2.json": "LaLiga2",
        "de.1.json": "Bundesliga",
        "de.2.json": "2Bundesliga",
        "it.1.json": "SerieA",
        "it.2.json": "SerieB",
        "fr.1.json": "Ligue1",
        "fr.2.json": "Ligue2",
        "uefa.cl.json": "ChampionsLeague",
    }
    for root, _, files in os.walk(repo_root):
        for name in files:
            if not name.endswith(".json"):
                continue
            league = league_map.get(name)
            if not league or (allowed and league not in allowed):
                continue
            path = os.path.join(root, name)
            try:
                payload = json.loads(open(path, "r", encoding="utf-8").read())
            except Exception:
                continue
            items = payload.get("matches", []) if isinstance(payload, dict) else payload
            for item in items:
                home = item.get("team1") or item.get("home_team") or item.get("homeTeam")
                away = item.get("team2") or item.get("away_team") or item.get("awayTeam")
                date_value = item.get("date") or item.get("match_date") or item.get("Date")
                match_date = _safe_date(date_value)
                if not match_date or not home or not away:
                    continue
                if target_date and match_date != target_date:
                    continue
                score = item.get("score", {})
                ft = score.get("ft") if isinstance(score, dict) else None
                home_goals = _safe_int(ft[0]) if isinstance(ft, (list, tuple)) and len(ft) == 2 else None
                away_goals = _safe_int(ft[1]) if isinstance(ft, (list, tuple)) and len(ft) == 2 else None
                matches.append(
                    FreeMatch(
                        league=league,
                        match_date=match_date,
                        home_team=str(home).strip(),
                        away_team=str(away).strip(),
                        home_goals=home_goals,
                        away_goals=away_goals,
                        actual_ftr=_ftr_from_goals(home_goals, away_goals),
                        source="openfootball_repo",
                    )
                )
    return matches


def load_football_datasets_repo(
    repo_root: Optional[str] = None,
    leagues: Optional[List[str]] = None,
    match_date: Optional[str] = None,
) -> List[FreeMatch]:
    """Load football-datasets repo CSV packages from a local clone."""
    repo_root = repo_root or os.path.join(EXTERNAL_REPOS_DIR, "football-datasets")
    if not os.path.isdir(repo_root):
        return []
    allowed = set(leagues) if leagues else None
    target_date = _normalize_date_key(match_date)
    folder_map = {
        "premier-league": "PremierLeague",
        "la-liga": "LaLiga",
        "serie-a": "SerieA",
        "bundesliga": "Bundesliga",
        "ligue-1": "Ligue1",
    }
    matches: List[FreeMatch] = []
    datasets_root = os.path.join(repo_root, "datasets")
    for folder, league in folder_map.items():
        if allowed and league not in allowed:
            continue
        league_dir = os.path.join(datasets_root, folder)
        if not os.path.isdir(league_dir):
            continue
        for path in sorted([p for p in os.listdir(league_dir) if p.lower().endswith(".csv")]):
            full_path = os.path.join(league_dir, path)
            try:
                df = pd.read_csv(full_path, low_memory=False)
            except Exception:
                continue
            for _, row in df.iterrows():
                item = _match_row_from_csv_row(league, row.to_dict())
                if item and (not target_date or item.match_date == target_date):
                    item.source = "football_datasets_repo"
                    matches.append(item)
    return matches


def load_training_examples_for_date(
    match_date: str,
    leagues: Optional[List[str]] = None,
) -> List[FreeMatch]:
    """Load historical fixtures from SQLite for a selected date."""
    target_date = _normalize_date_key(match_date)
    if not target_date or not os.path.exists(DB_PATH):
        return []

    query = """
        SELECT league, match_date, home_team, away_team,
               home_goals, away_goals, actual_ftr,
               home_odds, draw_odds, away_odds
        FROM training_examples
        WHERE actual_ftr IN ('H', 'D', 'A')
          AND match_date = ?
    """
    params: List[object] = [target_date]
    if leagues:
        placeholders = ",".join("?" * len(leagues))
        query += f" AND league IN ({placeholders})"
        params.extend(leagues)
    query += " ORDER BY league ASC, home_team ASC, away_team ASC"

    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute(query, tuple(params)).fetchall()
    finally:
        con.close()

    matches: List[FreeMatch] = []
    for league, row_date, home_team, away_team, home_goals, away_goals, actual_ftr, home_odds, draw_odds, away_odds in rows:
        matches.append(
            FreeMatch(
                league=str(league),
                match_date=_normalize_date_key(row_date) or target_date,
                home_team=str(home_team).strip(),
                away_team=str(away_team).strip(),
                home_goals=_safe_int(home_goals),
                away_goals=_safe_int(away_goals),
                actual_ftr=str(actual_ftr).strip()[:1] if actual_ftr else None,
                home_odds=_safe_float(home_odds),
                draw_odds=_safe_float(draw_odds),
                away_odds=_safe_float(away_odds),
                source="historical",
            )
        )
    return matches


def resolve_fixtures_for_date(
    match_date: str,
    leagues: Optional[List[str]] = None,
    include_sqlite_history: bool = True,
    include_openfootball: bool = True,
    include_football_datasets: bool = True,
    include_zip: bool = True,
    include_browser: bool = False,
    browser_config_path: Optional[str] = None,
) -> List[FreeMatch]:
    """Resolve fixtures for a selected date from the available free sources."""
    target_date = _normalize_date_key(match_date)
    if not target_date:
        return []

    resolved: List[FreeMatch] = []
    if include_sqlite_history:
        resolved.extend(load_training_examples_for_date(target_date, leagues=leagues))
    if include_openfootball:
        resolved.extend(load_openfootball_repo(leagues=leagues, match_date=target_date))
    if include_football_datasets:
        resolved.extend(load_football_datasets_repo(leagues=leagues, match_date=target_date))
    if include_zip:
        for item in load_from_zip(os.path.join(DATA_DIR, "football_data.zip")):
            if item.match_date == target_date and (not leagues or item.league in set(leagues)):
                resolved.append(item)
    if include_browser:
        try:
            from browser_scraper import discover_browser_fixtures_for_date

            resolved.extend(
                discover_browser_fixtures_for_date(
                    match_date=target_date,
                    leagues=leagues,
                    config_path=browser_config_path,
                )
            )
        except Exception:
            pass

    seen = {}
    deduped: List[FreeMatch] = []
    for item in resolved:
        item.league = str(item.league or "").strip()
        item.home_team = canonical_team_name(item.league, item.home_team)
        item.away_team = canonical_team_name(item.league, item.away_team)
        key = canonical_fixture_key(item.league, item.match_date, item.home_team, item.away_team)
        if key in seen:
            existing = seen[key]
            if existing.home_goals is None and item.home_goals is not None:
                existing.home_goals = item.home_goals
            if existing.away_goals is None and item.away_goals is not None:
                existing.away_goals = item.away_goals
            if not existing.actual_ftr and item.actual_ftr:
                existing.actual_ftr = item.actual_ftr
            if existing.home_odds is None and item.home_odds is not None:
                existing.home_odds = item.home_odds
            if existing.draw_odds is None and item.draw_odds is not None:
                existing.draw_odds = item.draw_odds
            if existing.away_odds is None and item.away_odds is not None:
                existing.away_odds = item.away_odds
            if not existing.source_url and item.source_url:
                existing.source_url = item.source_url
            if not existing.source_id and item.source_id:
                existing.source_id = item.source_id
            if not existing.kickoff_utc and item.kickoff_utc:
                existing.kickoff_utc = item.kickoff_utc
            if str(item.source or "").lower().startswith("browser") and not str(existing.source or "").lower().startswith("browser"):
                existing.source = item.source
                if item.source_url:
                    existing.source_url = item.source_url
                if item.source_id:
                    existing.source_id = item.source_id
            continue
        seen[key] = item
        deduped.append(item)
    return deduped


def discover_public_sources() -> Dict[str, str]:
    """Return a mapping of source labels to URLs for optional manual fetches.

    These are intentionally not hard-coded into the pipeline; the caller can
    choose to download them when network access is available.
    """
    return {
        "openfootball_football_json": "https://github.com/openfootball/football.json",
        "openfootball_leagues": "https://github.com/openfootball/leagues",
        "football_datasets": "https://github.com/datasets/football-datasets",
        "soccerdata": "https://github.com/probberechts/soccerdata",
        "fbref_scraper": "https://github.com/hale46/fbref-scraper",
        "sofascore_scraper": "https://github.com/tunjayoff/sofascore_scraper",
        "odds_harvester": "https://github.com/jordantete/OddsHarvester",
        "playwright": "https://playwright.dev/python/",
        "seleniumbase": "https://seleniumbase.io/",
    }


def discover_local_repos() -> Dict[str, str]:
    return {
        "football_json_repo": os.path.join(EXTERNAL_REPOS_DIR, "football.json"),
        "football_datasets_repo": os.path.join(EXTERNAL_REPOS_DIR, "football-datasets"),
        "leagues_repo": os.path.join(EXTERNAL_REPOS_DIR, "leagues"),
        "soccerdata_repo": os.path.join(EXTERNAL_REPOS_DIR, "soccerdata"),
        "fbref_scraper_repo": os.path.join(EXTERNAL_REPOS_DIR, "fbref-scraper"),
        "sofascore_scraper_repo": os.path.join(EXTERNAL_REPOS_DIR, "sofascore_scraper"),
        "odds_harvester_repo": os.path.join(EXTERNAL_REPOS_DIR, "OddsHarvester"),
    }


def probe_optional_sources() -> Dict[str, bool]:
    try:
        import importlib.util
    except Exception:
        return {}

    return {
        "soccerdata": importlib.util.find_spec("soccerdata") is not None,
        "fbref_scraper": importlib.util.find_spec("fbref") is not None,
        "sofascore_scraper": importlib.util.find_spec("sofascore") is not None,
        "odds_harvester": importlib.util.find_spec("oddsharvester") is not None,
        "playwright": importlib.util.find_spec("playwright") is not None,
        "seleniumbase": importlib.util.find_spec("seleniumbase") is not None,
    }


def build_source_registry() -> Dict[str, dict]:
    registry = {}
    availability = probe_optional_sources()
    local_repos = discover_local_repos()
    for key, url in discover_public_sources().items():
        local_key = {
            "openfootball_football_json": "football_json_repo",
            "openfootball_leagues": "leagues_repo",
            "football_datasets": "football_datasets_repo",
            "soccerdata": "soccerdata_repo",
            "fbref_scraper": "fbref_scraper_repo",
            "sofascore_scraper": "sofascore_scraper_repo",
            "odds_harvester": "odds_harvester_repo",
        }.get(key)
        registry[key] = {
            "url": url,
            "available": bool(availability.get(key, False) or (local_key and os.path.isdir(local_repos.get(local_key, "")))),
        }
    browser_available = bool(availability.get("playwright") or availability.get("seleniumbase"))
    registry["playwright"] = {
        "url": discover_public_sources()["playwright"],
        "available": browser_available and availability.get("playwright", False),
    }
    registry["seleniumbase"] = {
        "url": discover_public_sources()["seleniumbase"],
        "available": browser_available and availability.get("seleniumbase", False),
    }
    return registry
