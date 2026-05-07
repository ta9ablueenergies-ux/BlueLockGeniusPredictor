# ANTIGRAVITY FOOTBALL DATA SCRAPER V5.9
"""
Fetches real match data using football-data.org v4 API.
Primary: football-data.org (token)
Fallback: mock data (enhanced with realistic matches)
"""
import pandas as pd
import os
import json
import time
import requests
import urllib3
import logging
from datetime import datetime, timedelta

from components.ssl_utils import get_unsafe_session

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger(__name__)

# football-data.org API token
FDO_TOKEN = os.environ.get('FDO_TOKEN', '')

# football-data.org v4 competition codes and IDs
FDO_COMPETITIONS = {
    'PremierLeague':      {'code': 'PL',  'id': 2021},
    'SerieA':             {'code': 'SA',  'id': 2019},
    'LaLiga':             {'code': 'PD',  'id': 2014},
    'Bundesliga':         {'code': 'BL1', 'id': 2002},
    'Ligue1':             {'code': 'FL1', 'id': 2015},
    'Championship':       {'code': 'ELC', 'id': 2016},
    'ChampionsLeague':    {'code': 'CL',  'id': 2001},
    'ScottishPremiership':{'code': 'SPL', 'id': 2084},
    'Eredivisie':         {'code': 'DED', 'id': 2003},
    'LigaNOS':            {'code': 'PPL', 'id': 2017},
    'BelgianProLeague':   {'code': 'BJL', 'id': 2009},
    'SuperLig':           {'code': 'TSL', 'id': 2070},
    '2Bundesliga':        {'code': 'BL2', 'id': 2004},
    'Ligue2':             {'code': 'FL2', 'id': 2142},
    'LaLiga2':            {'code': 'SD',  'id': 2077},
    'SerieB':             {'code': 'SB',  'id': 2121},
    'WorldCup':           {'code': 'WC',  'id': 2000},
}

BASE_URL = 'https://api.football-data.org/v4'

def fetch_fdo_matches(league, date_from, date_to=None, timeout=15):
    """Fetch fixtures from football-data.org with Smart Cache."""
    # 1. Setup Cache
    CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    
    if not date_to: date_to = date_from
    cache_key = f"{league}_{date_from}_{date_to}.json".replace('-', '').replace(':', '')
    cache_path = os.path.join(CACHE_DIR, cache_key)
    
    # 2. Check Cache (24 hour freshness for raw fixtures)
    if os.path.exists(cache_path):
        try:
            mtime = os.path.getmtime(cache_path)
            if (time.time() - mtime) < 86400:
                with open(cache_path, 'r') as f:
                    logger.info(f"FDO: Using cached data for {league}")
                    return pd.DataFrame(json.load(f))
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

    comp = FDO_COMPETITIONS.get(league)
    if not comp:
        logger.warning(f"Unknown league: {league}")
        return pd.DataFrame()

    if not date_to:
        date_to = date_from

    # Extract just the date part (YYYY-MM-DD) for the API
    api_start = date_from.split('T')[0]
    api_end = date_to.split('T')[0]

    url = f"{BASE_URL}/competitions/{comp['code']}/matches"
    params = {
        'dateFrom': api_start,
        'dateTo': api_end,
    }
    headers = {
        'X-Auth-Token': FDO_TOKEN,
        'Accept': 'application/json',
    }

    try:
        logger.info(f"FDO: Fetching {league} ({comp['code']}) from {api_start} to {api_end}")
        session = get_unsafe_session()
        response = session.get(url, params=params, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            matches = data.get('matches', [])
            logger.info(f"FDO: Found {len(matches)} matches for {league}")

            if not matches:
                return pd.DataFrame()

            rows = []
            for m in matches:
                home = m.get('homeTeam', {})
                away = m.get('awayTeam', {})
                score = m.get('score', {})
                utc_date = m.get('utcDate', '')
                
                # Parse time from UTC date
                match_time = ''
                if utc_date:
                    try:
                        dt = datetime.fromisoformat(utc_date.replace('Z', '+00:00'))
                        match_time = dt.strftime('%H:%M')
                    except:
                        match_time = '15:00'

                rows.append({
                    'hometeam': home.get('shortName', home.get('name', '')),
                    'awayteam': away.get('shortName', away.get('name', '')),
                    'date': utc_date[:10] if utc_date else api_start,
                    'utcDate': utc_date, # Keep original for precise filtering later
                    'Time': match_time,
                    'fthg': score.get('fullTime', {}).get('home'),
                    'ftag': score.get('fullTime', {}).get('away'),
                    'status': m.get('status', ''),
                    'league': league,
                })

            if rows:
                with open(cache_path, 'w') as f:
                    json.dump(rows, f)
            return pd.DataFrame(rows)

        elif response.status_code == 429:
            logger.warning(f"FDO: Rate limited (429)")
        elif response.status_code == 403:
            logger.warning(f"FDO Error: 403 Forbidden. This league ({league}) might be restricted on your current API plan (e.g. Free Tier).")
        else:
            logger.warning(f"FDO Error: {response.status_code}")

    except requests.RequestException as e:
        logger.error(f"FDO request failed: {e}")

    return pd.DataFrame()


def generate_enhanced_mock(league, target_date):
    """Realistic mock data when API fails — all 15 leagues."""
    date_only = target_date.split('T')[0]
    date_obj = datetime.strptime(date_only, '%Y-%m-%d')
    date_str = date_obj.strftime('%d/%m/%Y')

    # All 15 leagues with realistic matchups (sourced from main_script.py pool)
    MOCK_FIXTURES = {
        'PremierLeague': [
            ('Arsenal', 'Manchester City', '17:30'),
            ('Liverpool', 'Chelsea', '15:00'),
            ('Manchester United', 'Tottenham', '15:00'),
            ('Newcastle', 'Brighton', '15:00'),
            ('Aston Villa', 'West Ham', '15:00'),
            ('Crystal Palace', 'Wolves', '15:00'),
            ('Fulham', 'Everton', '15:00'),
            ('Brentford', 'Leicester', '15:00'),
            ('Southampton', 'Nottm Forest', '15:00'),
            ('Bournemouth', 'Leeds', '15:00'),
        ],
        'LaLiga': [
            ('Barcelona', 'Real Madrid', '21:00'),
            ('Atletico Madrid', 'Real Betis', '18:30'),
            ('Sevilla', 'Valencia', '16:00'),
            ('Villarreal', 'Celta Vigo', '16:00'),
            ('Girona', 'Athletic Bilbao', '16:00'),
            ('Mallorca', 'Osasuna', '16:00'),
            ('Real Sociedad', 'Getafe', '16:00'),
            ('Betis', 'Sevilla', '16:00'),
            ('Alaves', 'Las Palmas', '16:00'),
            ('Granada', 'Almeria', '16:00'),
        ],
        'ChampionsLeague': [
            ('Real Madrid', 'Manchester City', '20:00'),
            ('Bayern Munich', 'Arsenal', '20:00'),
            ('Paris Saint-Germain', 'Barcelona', '20:00'),
            ('Atletico Madrid', 'Borussia Dortmund', '20:00'),
            ('Inter Milan', 'Porto', '20:00'),
            ('Napoli', 'Eintracht Frankfurt', '20:00'),
            ('Liverpool', 'AC Milan', '20:00'),
            ('Juventus', 'Sevilla', '20:00')
        ],
        'WorldCup': [
            ('Brazil', 'Germany', '21:00'),
            ('France', 'Argentina', '18:00'),
            ('Spain', 'England', '21:00'),
            ('Portugal', 'Netherlands', '18:00'),
            ('USA', 'Mexico', '21:00'),
            ('Morocco', 'Senegal', '15:00'),
            ('Japan', 'South Korea', '15:00'),
            ('Australia', 'Croatia', '18:00'),
        ],
        'SerieA': [
            ('Inter Milan', 'AC Milan', '20:45'),
            ('Juventus', 'Napoli', '18:00'),
            ('Roma', 'Lazio', '18:00'),
            ('Atalanta', 'Fiorentina', '15:00'),
            ('Lazio', 'Roma', '15:00'),
            ('Udinese', 'Bologna', '15:00'),
            ('Monza', 'Lecce', '15:00'),
            ('Torino', 'Empoli', '15:00'),
            ('Verona', 'Spezia', '15:00'),
            ('Frosinone', 'Sampdoria', '15:00'),
        ],
        'Bundesliga': [
            ('Bayern Munich', 'Borussia Dortmund', '18:30'),
            ('Bayer Leverkusen', 'RB Leipzig', '15:30'),
            ('Stuttgart', 'Wolfsburg', '15:30'),
            ('Eintracht Frankfurt', 'Union Berlin', '15:30'),
            ('Mainz', 'Koln', '15:30'),
            ('Hoffenheim', 'Monchengladbach', '15:30'),
            ('Freiburg', 'Augsburg', '15:30'),
            ('Bochum', 'Darmstadt', '15:30'),
            ('Heidenheim', 'Werder Bremen', '15:30'),
            ('Leverkusen', 'Ein Frankfurt', '15:30'),
        ],
        'Ligue1': [
            ('Paris Saint-Germain', 'Monaco', '21:00'),
            ('Marseille', 'Lyon', '20:00'),
            ('Lille', 'Lens', '20:00'),
            ('Rennes', 'Nantes', '20:00'),
            ('Nice', 'Monaco', '20:00'),
            ('Strasbourg', 'Lorient', '20:00'),
            ('Montpellier', 'Brest', '20:00'),
            ('Lyon', 'Toulouse', '20:00'),
            ('Metz', 'Clermont', '20:00'),
            ('Auxerre', 'Le Havre', '20:00'),
        ],
        'Championship': [
            ('Leicester', 'Leeds', '17:00'),
            ('Southampton', 'West Brom', '16:00'),
            ('Sunderland', 'Middlesbrough', '16:00'),
            ('Coventry', 'Swansea', '16:00'),
            ('Hull', 'Cardiff', '16:00'),
            ('Bristol City', 'Luton', '16:00'),
            ('Birmingham', 'Preston', '16:00'),
            ('QPR', 'Sheffield Utd', '16:00'),
            ('Derby', 'Blackburn', '16:00'),
            ('Watford', 'Norwich', '16:00'),
        ],
        'ScottishPremiership': [
            ('Celtic', 'Rangers', '12:30'),
            ('Aberdeen', 'Hearts', '15:00'),
            ('Hibernian', 'Motherwell', '15:00'),
            ('Ross County', 'St Johnstone', '15:00'),
            ('Dundee Utd', 'Livingston', '15:00'),
            ('Kilmarnock', 'St Mirren', '15:00'),
            ('Partick Thistle', 'Dundee', '15:00'),
            ('St Johnstone', 'Ross County', '15:00'),
        ],
        'Eredivisie': [
            ('Ajax', 'PSV', '19:00'),
            ('Feyenoord', 'AZ', '19:00'),
            ('PSV Eindhoven', 'Twente', '19:00'),
            ('Utrecht', 'NEC', '19:00'),
            ('Sparta Rotterdam', 'Go Ahead Eagles', '19:00'),
            ('Vitesse', 'FC Groningen', '19:00'),
            ('Heerenveen', 'Excelsior', '19:00'),
            ('PEC Zwolle', 'RKC Waalwijk', '19:00'),
        ],
        'LigaNOS': [
            ('Benfica', 'Porto', '20:30'),
            ('Sporting CP', 'Braga', '20:30'),
            ('Portimonense', 'Vitoria', '20:30'),
            ('Gil Vicente', 'Moreirense', '20:30'),
            ('Santa Clara', 'Arouca', '20:30'),
            ('Casa Pia', 'Rio Ave', '20:30'),
            ('Estrela', 'Famalicao', '20:30'),
            ('Vizela', 'Boavista', '20:30'),
        ],
        'BelgianProLeague': [
            ('Club Brugge', 'Antwerp', '20:00'),
            ('Gent', 'Standard Liege', '20:00'),
            ('Genk', 'Charleroi', '20:00'),
            ('Anderlecht', 'Mechelen', '20:00'),
            ('Leuven', 'Kortrijk', '20:00'),
            ('Oostende', 'Sint-Truiden', '20:00'),
            ('Cercle Brugge', 'Westerlo', '20:00'),
            ('Eupen', 'Oud-Heverlee Leuven', '20:00'),
        ],
        'SuperLig': [
            ('Galatasaray', 'Fenerbahce', '19:00'),
            ('Besiktas', 'Trabzonspor', '19:00'),
            ('Basaksehir', 'Adana Demirspor', '19:00'),
            ('Konyaspor', 'Alanyaspor', '19:00'),
            ('Sivasspor', 'Gaziantep', '19:00'),
            ('Antalyaspor', 'Kasimpasa', '19:00'),
            ('Fenerbahce', 'Besiktas', '19:00'),
            ('Rizespor', 'Hatayspor', '19:00'),
        ],
        '2Bundesliga': [
            ('Hamburger SV', 'Schalke', '12:00'),
            ('Hertha Berlin', 'Hannover', '12:00'),
            ('Dusseldorf', 'Nurnberg', '12:00'),
            ('Kaiserslautern', 'Paderborn', '12:00'),
            ('Karlsruher', 'Holstein Kiel', '12:00'),
            ('Eintracht Braunschweig', 'Wehen Wiesbaden', '12:00'),
            ('Magdeburg', 'Elversberg', '12:00'),
            ('Munich 1860', 'Lubeck', '12:00'),
        ],
        'Ligue2': [
            ('Bordeaux', 'Saint-Etienne', '18:00'),
            ('Rodez', 'Paris FC', '18:00'),
            ('Laval', 'Annecy', '18:00'),
            ('Concarneau', 'Guingamp', '18:00'),
            ('SC Bastia', 'Dunkerque', '18:00'),
            ('Troyes', 'Grenoble', '18:00'),
            ('Pau', 'Caen', '18:00'),
            ('Amiens', 'Niort', '18:00'),
        ],
        'LaLiga2': [
            ('Eibar', 'Valladolid', '16:00'),
            ('Levante', 'Albacete', '16:00'),
            ('Burgos', 'Leganes', '16:00'),
            ('Racing Santander', 'Real Zaragoza', '16:00'),
            ('Espanyol', 'Tenerife', '16:00'),
            ('Granada', 'Almeria', '16:00'),
            ('Alcorcon', 'Mirandes', '16:00'),
            ('Huesca', 'Racing Ferrol', '16:00'),
        ],
        'SerieB': [
            ('Parma', 'Cremonese', '14:00'),
            ('Cittadella', 'Spezia', '14:00'),
            ('Brescia', 'Cagliari', '14:00'),
            ('Pisa', 'Reggina', '14:00'),
            ('Frosinone', 'Sampdoria', '14:00'),
            ('Modena', 'Ascoli', '14:00'),
            ('Bari', 'Palermo', '14:00'),
            ('Venezia', 'Ternana', '14:00'),
        ],
    }

    teams = MOCK_FIXTURES.get(league, [('Team A', 'Team B', '15:00')])
    rows = []
    for home, away, time_str in teams:
        rows.append({
            'date': date_str,
            'utcDate': f"{date_only}T{time_str}:00Z",
            'Time': time_str,
            'hometeam': home,
            'awayteam': away,
            'fthg': None, 'ftag': None,
            'b365h': 2.0, 'b365d': 3.2, 'b365a': 3.5,
        })
    return pd.DataFrame(rows)


def normalize_to_predictions_format(df, target_date=None):
    """Normalize API response to match predict_results() format"""
    if df.empty:
        return pd.DataFrame()

    result = pd.DataFrame()
    result['HomeTeam'] = df['hometeam']
    result['AwayTeam'] = df['awayteam']
    result['Time'] = df.get('Time', pd.Series(['15:00'] * len(df)))
    result['utcDate'] = df.get('utcDate', pd.Series([''] * len(df)))

    # Full-time results (None for upcoming matches)
    result['FTHG'] = df.get('fthg', pd.Series([None] * len(df)))
    result['FTAG'] = df.get('ftag', pd.Series([None] * len(df)))
    result['FTR'] = pd.Series([None] * len(df))

    # Default odds
    result['h_course'] = pd.Series([2.0] * len(df))
    result['d_course'] = pd.Series([3.2] * len(df))
    result['a_course'] = pd.Series([3.5] * len(df))
    result['h_open'] = pd.Series([2.0] * len(df))
    result['d_open'] = pd.Series([3.2] * len(df))
    result['a_open'] = pd.Series([3.5] * len(df))

    return result


def scrape_matches_for_date(league, start_date, end_date=None):
    """
    Main entry point: scrape real matches for a range.
    start_date/end_date: 'YYYY-MM-DDTHH:MM' or 'YYYY-MM-DD'
    """
    logger.info(f"Scraping {league} from {start_date} to {end_date or start_date}")

    # Try football-data.org API
    df = fetch_fdo_matches(league, start_date, end_date)

    if df.empty:
        logger.info(f"FDO API returned no data for {league}. Skipping mock generation.")
        return pd.DataFrame()

    return normalize_to_predictions_format(df, start_date)



def get_historical_for_analysis(league, target_date, lookback_days=30):
    """Get historical match data for Bayesian ratings"""
    # For historical context, we use the same API with date range
    date_obj = datetime.strptime(target_date.split('T')[0], '%Y-%m-%d')
    start_date = (date_obj - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    df = fetch_fdo_matches(league, start_date)
    if df.empty:
        return []

    # Filter to within lookback_days
    df['date_obj'] = pd.to_datetime(df['date']).dt.date
    date_obj_date = date_obj.date()
    cutoff = date_obj_date - timedelta(days=lookback_days)
    df = df[df['date_obj'] < date_obj_date]

    history_list = []
    for _, row in df.iterrows():
        try:
            history_list.append({
                'Date': datetime.strftime(pd.to_datetime(row['date']), '%d/%m/%Y') if pd.notna(row['date']) else '',
                'Home': row['hometeam'],
                'Away': row['awayteam'],
                'FTHG': row.get('fthg', 1.5) if pd.notna(row.get('fthg', None)) else 1.5,
                'FTAG': row.get('ftag', 1.2) if pd.notna(row.get('ftag', None)) else 1.2
            })
        except Exception:
            continue

    return history_list


def fetch_league_data(league, timeout=15):
    """Fetch full season data from FDO"""
    comp = FDO_COMPETITIONS.get(league)
    if not comp:
        return pd.DataFrame()

    url = f"{BASE_URL}/competitions/{comp['code']}/matches"
    headers = {'X-Auth-Token': FDO_TOKEN, 'Accept': 'application/json'}

    try:
        session = get_unsafe_session()
        response = session.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            matches = data.get('matches', [])
            rows = []
            for m in matches:
                home = m.get('homeTeam', {})
                away = m.get('awayTeam', {})
                score = m.get('score', {})
                rows.append({
                    'date': m.get('utcDate', '')[:10],
                    'hometeam': home.get('shortName', home.get('name', '')),
                    'awayteam': away.get('shortName', away.get('name', '')),
                    'fthg': score.get('fullTime', {}).get('home'),
                    'ftag': score.get('fullTime', {}).get('away'),
                })
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            logger.info(f"FDO: Fetched {len(df)} total matches for {league}")
            return df
    except Exception as e:
        logger.error(f"FDO fetch failed: {e}")

    return pd.DataFrame()