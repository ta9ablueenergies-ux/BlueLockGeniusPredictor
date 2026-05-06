"""Team-name canonicalization shared by fixture discovery and persistence.

External fixture sources disagree on suffixes, abbreviations, accents, and
historic short names. The prediction stack is trained mostly on football-data
team names, so fixture identity should collapse to those canonical names before
dedupe, prediction, and SQLite writes.
"""

from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")


DEFAULT_TEAM_ALIASES: Tuple[Tuple[str, str, str], ...] = (
    ("PremierLeague", "Everton FC", "Everton"),
    ("PremierLeague", "Liverpool FC", "Liverpool"),
    ("PremierLeague", "Manchester City", "Man City"),
    ("PremierLeague", "Manchester City FC", "Man City"),
    ("PremierLeague", "Manchester United FC", "Man United"),
    ("PremierLeague", "Man Utd", "Man United"),
    ("PremierLeague", "Newcastle United", "Newcastle"),
    ("PremierLeague", "Nottingham", "Nottm Forest"),
    ("PremierLeague", "Nottingham Forest", "Nottm Forest"),
    ("PremierLeague", "Nottingham Forest FC", "Nottm Forest"),
    ("PremierLeague", "Nott'm Forest", "Nottm Forest"),
    ("PremierLeague", "Tottenham Hotspur", "Tottenham"),
    ("PremierLeague", "Tottenham Hotspur FC", "Tottenham"),
    ("LaLiga", "Athletic Club", "Ath Bilbao"),
    ("LaLiga", "Athletic Bilbao", "Ath Bilbao"),
    ("LaLiga", "Atlético Madrid", "Ath Madrid"),
    ("LaLiga", "Atletico Madrid", "Ath Madrid"),
    ("LaLiga", "Club Atlético de Madrid", "Ath Madrid"),
    ("LaLiga", "Real Sociedad", "Sociedad"),
    ("LaLiga", "Real Sociedad de Fútbol", "Sociedad"),
    ("LaLiga", "Real Sociedad de Futbol", "Sociedad"),
    ("LaLiga", "Real Sociedad de F?tbol", "Sociedad"),
    ("LaLiga", "Real Sociedad de F·tbol", "Sociedad"),
    ("LaLiga", "Sevilla FC", "Sevilla"),
    ("LaLiga", "Real Betis", "Betis"),
    ("LaLiga2", "UD Almería", "Almeria"),
    ("LaLiga2", "UD Almeria", "Almeria"),
    ("LaLiga2", "CD Mirandés", "Mirandes"),
    ("LaLiga2", "CD Mirandes", "Mirandes"),
    ("SerieA", "AS Roma", "Roma"),
    ("SerieA", "A.S. Roma", "Roma"),
    ("SerieA", "ACF Fiorentina", "Fiorentina"),
    ("SerieA", "SS Lazio", "Lazio"),
    ("SerieA", "S.S. Lazio", "Lazio"),
    ("SerieA", "US Cremonese", "Cremonese"),
    ("SerieA", "U.S. Cremonese", "Cremonese"),
    ("LigaNOS", "Sporting", "Sp Lisbon"),
    ("LigaNOS", "Sporting CP", "Sp Lisbon"),
    ("LigaNOS", "Sporting Clube de Portugal", "Sp Lisbon"),
    ("LigaNOS", "Sporting Lisbon", "Sp Lisbon"),
    ("LigaNOS", "Vitória SC", "Guimaraes"),
    ("LigaNOS", "Vitoria SC", "Guimaraes"),
    ("LigaNOS", "Vit?ria SC", "Guimaraes"),
    ("LigaNOS", "Vit·ria SC", "Guimaraes"),
    ("LigaNOS", "Vitória Guimarães", "Guimaraes"),
    ("LigaNOS", "Vitoria Guimaraes", "Guimaraes"),
    ("ScottishPremiership", "Heart of Midlothian", "Hearts"),
    ("ScottishPremiership", "Heart of Midlothian FC", "Hearts"),
    ("ScottishPremiership", "Rangers FC", "Rangers"),
)

PREFIX_TOKENS = {"a", "as", "ac", "acf", "ss", "us", "ud", "fc", "cf", "rc", "cd", "sd", "sv"}
SUFFIX_TOKENS = {
    "afc",
    "calcio",
    "cf",
    "club",
    "cp",
    "fc",
    "football",
    "futbol",
    "sc",
    "ud",
}


def clear_team_name_cache() -> None:
    _build_lookup.cache_clear()


def fold_accents(value: object) -> str:
    text = str(value or "")
    folded = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def normalize_team_key(value: object) -> str:
    text = fold_accents(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"['`´’]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def stripped_team_key(value: object) -> str:
    tokens = normalize_team_key(value).split()
    while tokens and tokens[0] in PREFIX_TOKENS:
        tokens.pop(0)
    while tokens and tokens[-1] in SUFFIX_TOKENS:
        tokens.pop()
    tokens = [token for token in tokens if token not in {"de", "of", "the"}]
    return " ".join(tokens)


def _safe_rows(query: str, params: tuple = (), db_path: str = DB_PATH) -> List[tuple]:
    if not os.path.exists(db_path):
        return []
    con = sqlite3.connect(db_path)
    try:
        return list(con.execute(query, params).fetchall())
    except sqlite3.Error:
        return []
    finally:
        con.close()


@lru_cache(maxsize=8)
def _build_lookup(league: str, db_path: str = DB_PATH) -> dict:
    lookup = {}

    def add(alias: object, canonical: object) -> None:
        alias_text = str(alias or "").strip()
        canonical_text = str(canonical or "").strip()
        if not alias_text or not canonical_text:
            return
        for key in {normalize_team_key(alias_text), stripped_team_key(alias_text)}:
            if key:
                lookup.setdefault(key, canonical_text)

    rows: List[Tuple[str, str, str]] = list(DEFAULT_TEAM_ALIASES)
    rows.extend(
        (str(l), str(a), str(c))
        for l, a, c in _safe_rows(
            "SELECT league, alias_name, canonical_name FROM team_aliases",
            db_path=db_path,
        )
    )
    for row_league, alias, canonical in rows:
        if row_league == league:
            add(alias, canonical)
            add(canonical, canonical)

    known_rows = _safe_rows(
        """
        SELECT home_team FROM training_examples WHERE league = ?
        UNION
        SELECT away_team FROM training_examples WHERE league = ?
        """,
        (league, league),
        db_path=db_path,
    )
    for (team_name,) in known_rows:
        add(team_name, team_name)

    return lookup


def canonical_team_name(league: object, team_name: object, db_path: str = DB_PATH) -> str:
    text = " ".join(str(team_name or "").split()).strip()
    if not text:
        return ""
    league_key = str(league or "").strip()
    if not league_key:
        return text
    lookup = _build_lookup(league_key, db_path)
    for key in (normalize_team_key(text), stripped_team_key(text)):
        if key in lookup:
            return lookup[key]
    return text


def canonical_fixture_key(league: object, match_date: object, home_team: object, away_team: object) -> Tuple[str, str, str, str]:
    league_text = str(league or "").strip()
    home = canonical_team_name(league_text, home_team)
    away = canonical_team_name(league_text, away_team)
    return (
        league_text,
        str(match_date or "").strip()[:10],
        normalize_team_key(home),
        normalize_team_key(away),
    )


def default_alias_rows() -> Iterable[Tuple[str, str, str]]:
    return DEFAULT_TEAM_ALIASES
