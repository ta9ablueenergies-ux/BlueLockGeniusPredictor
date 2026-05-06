# ANTIGRAVITY FIRECRAWL AGENT — Data-First Pipeline
"""
Firecrawl Agent runs FIRST — discovers today's matches and extracts
all intelligence (xG, injuries, form, referee, H2H) before any
prediction is made. ML + Bayesian then use this as the foundation.

ARCHITECTURE:
  Firecrawl Agent (discover + enrich)
       ↓ enriched match list
  FDO API (odds + historical results) — FILL IN only
       ↓ enriched match list + odds
  ML + Bayesian Engine (predict) — SEES real xG + squad data
       ↓ prediction with full context

Pydantic schemas capture xG-adjusted team strength and are
designed to feed directly into the Bayesian engine's feature set.
"""
import os
import time
import json
from typing import Optional
from datetime import datetime, timedelta
import hashlib

from pydantic import BaseModel, Field
from components.ssl_utils import get_unsafe_session
try:
    from firecrawl import Firecrawl
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False

# =============================================================================
# SSL Fix for Zscaler: patch Firecrawl's HttpClient to use unverified session
# =============================================================================
if FIRECRAWL_AVAILABLE:
    try:
        from firecrawl.v2.utils import http_client as _hc
        _orig_post = _hc.HttpClient.post
        def _unverified_post(self, endpoint, data, **kwargs):
            session = get_unsafe_session()
            headers = self._prepare_headers()
            url = self._build_url(endpoint)
            payload = dict(data)
            try:
                from firecrawl.v2.utils.get_version import get_version
                version = get_version()
            except Exception:
                version = "1.0.0"
            payload['origin'] = f'python-sdk@{version}'
            return session.post(url, headers=headers, json=payload, timeout=kwargs.get('timeout', 30))
        _hc.HttpClient.post = _unverified_post
    except Exception:
        pass


# =============================================================================
# Pydantic Schemas — feed directly into Bayesian engine
# =============================================================================

class Injury(BaseModel):
    player: str = Field(description="Player name")
    position: str = Field(description="GK, DF, MF, FW")
    reason: str = Field(description="injury, suspension, red_card, doubtful")
    return_date: Optional[str] = Field(None, description="YYYY-MM-DD if known")
    severity: Optional[str] = Field(None, description="low, medium, high, season_ending")


class FormGuideEntry(BaseModel):
    result: str = Field(description="W, D, L")
    score: Optional[str] = Field(None, description="e.g. '2-1', '0-0'")
    venue: Optional[str] = Field(description="home or away")
    xg: Optional[float] = Field(None, description="Team xG in this specific match")
    xga: Optional[float] = Field(None, description="Team xG Against in this specific match")
    date: Optional[str] = Field(None, description="YYYY-MM-DD")


class ExpectedGoals(BaseModel):
    """
    xG and xT (Expected Threat) from sofascore/whoscored/StatsBomb.
    xG = goal probability; xT = possession zone value (how dangerous each possession is).
    xT is spatial/possession-based, xG is shot-based — both inform expected goals.
    """
    home_xg: Optional[float] = Field(None, description="Home team expected goals (season avg)")
    away_xg: Optional[float] = Field(None, description="Away team expected goals (season avg)")
    home_xg_last5: Optional[list[float]] = Field(default_factory=list)
    away_xg_last5: Optional[list[float]] = Field(default_factory=list)
    home_xga_last5: Optional[list[float]] = Field(default_factory=list)
    away_xga_last5: Optional[list[float]] = Field(default_factory=list)
    source: Optional[str] = Field(None, description="sofascore, whoscored, footystats")

    # xT (Expected Threat) — zone-based possession value from StatsBomb/Metrica
    home_xt: Optional[float] = Field(None, description="Home team avg xT per possession (0-0.2 typical range)")
    away_xt: Optional[float] = Field(None, description="Away team avg xT per possession")
    home_xt_last5: Optional[list[float]] = Field(default_factory=list, description="Home team xT last 5 matches")
    away_xt_last5: Optional[list[float]] = Field(default_factory=list, description="Away team xT last 5 matches")


class NewsSentiment(BaseModel):
    headline: str = Field(description="Recent news headline")
    sentiment_score: float = Field(description="-1.0 (very negative) to 1.0 (very positive)")
    impact_area: str = Field(description="defense, attack, morale, tactical")
    relevance: float = Field(description="0.0 to 1.0 relevance to upcoming match")


class MatchIntel(BaseModel):
    """
    Complete match intelligence — the FOUNDATION for prediction.
    Every field here directly shapes the Bayesian + ML prediction.

    Named MatchIntel (not MatchIntelligence) to distinguish from the
    enrichment-only schema used in the old Tavily code path.
    """
    # ── Identity ──────────────────────────────────────────────────────────
    league: str
    match_date: str           # YYYY-MM-DD
    kickoff_utc: Optional[str] = None  # HH:MM

    home_team: str
    away_team: str

    # ── Expected Goals (replaces raw FTHG/FTAG in Bayesian prior) ───────
    xg: Optional[ExpectedGoals] = None

    # ── Squad availability (adjusts team strength rating) ─────────────────
    home_injuries: list[Injury] = Field(default_factory=list)
    away_injuries: list[Injury] = Field(default_factory=list)
    home_suspensions: list[str] = Field(default_factory=list)
    away_suspensions: list[str] = Field(default_factory=list)
    missing_key_player_home: bool = False
    missing_key_player_away: bool = False

    # ── News Sentiment ───────────────────────────────────────────────────
    news: list[NewsSentiment] = Field(default_factory=list, description="Tactical sentiment from news sources")

    # ── Referee (shapes card/over-under markets) ──────────────────────────
    referee: Optional[str] = None
    referee_avg_cards: Optional[float] = Field(None, description="avg cards per game")

    # ── Form guide (last 5 league matches) ───────────────────────────────
    home_form_last5: list[FormGuideEntry] = Field(default_factory=list)
    away_form_last5: list[FormGuideEntry] = Field(default_factory=list)

    # Points tally from form_last5 (computed, not directly asked to Agent)
    @property
    def home_pts_last5(self) -> int:
        return sum(3 if e.result == 'W' else 1 if e.result == 'D' else 0
                   for e in self.home_form_last5)

    @property
    def away_pts_last5(self) -> int:
        return sum(3 if e.result == 'W' else 1 if e.result == 'D' else 0
                   for e in self.away_form_last5)

    @property
    def home_goals_scored_last5(self) -> list[float]:
        return [float(e.score.split('-')[0]) if e.score and '-' in e.score else 0.0
                for e in self.home_form_last5]

    @property
    def away_goals_scored_last5(self) -> list[float]:
        return [float(e.score.split('-')[1]) if e.score and '-' in e.score else 0.0
                for e in self.away_form_last5]

    # ── Head-to-head ─────────────────────────────────────────────────────
    h2h_matches: int = 0
    h2h_home_wins: int = 0
    h2h_draws: int = 0
    h2h_away_wins: int = 0
    h2h_home_goals_total: int = 0
    h2h_away_goals_total: int = 0
    h2h_last_result: Optional[str] = None  # "2-1 Home Win"

    # ── Formation (shapes attack/defense weighting) ───────────────────────
    formation_home: Optional[str] = None  # "4-3-3"
    formation_away: Optional[str] = None

    # ── Motivation (shapes win-draw probability) ───────────────────────────
    home_motivation: Optional[str] = Field(
        None, description="must_win, derby, europe_spot, relegation_fight, routine")
    away_motivation: Optional[str] = Field(
        None, description="must_win, derby, europe_spot, relegation_fight, routine")

    # ── League table context ───────────────────────────────────────────────
    home_position: Optional[int] = None  # league table position
    away_position: Optional[int] = None
    home_points: Optional[int] = None
    away_points: Optional[int] = None

    # ── Betting odds (pulled from FDO, stored here for reference) ─────────
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None

    # ── NEW: Advanced Intelligence (V5.9) ────────────────────────────────
    weather_condition: Optional[str] = Field(None, description="rain, snow, extreme_heat, windy, clear")
    pitch_condition: Optional[str] = Field(None, description="excellent, poor, waterlogged, artificial")
    attendance_expected: Optional[int] = None
    
    home_fatigue_index: float = Field(1.0, description="1.0=Fresh, >1.0=Fatigued (e.g. played 3 days ago in CL)")
    away_fatigue_index: float = Field(1.0, description="1.0=Fresh, >1.0=Fatigued")
    
    tactical_style_home: Optional[str] = Field(None, description="tiki_taka, counter_attack, high_press, park_the_bus")
    tactical_style_away: Optional[str] = Field(None, description="tiki_taka, counter_attack, high_press, park_the_bus")
    
    defensive_line_h: Optional[str] = Field(None, description="high, medium, deep")
    defensive_line_a: Optional[str] = Field(None, description="high, medium, deep")
    
    bench_strength_h: float = Field(1.0, description="Multiplier for late-game goal potential")
    bench_strength_a: float = Field(1.0, description="Multiplier for late-game goal potential")

    # ── V6.0 QUANTUM VARIABLES (Beyond simple fatigue) ──────────────────
    ppda_h: Optional[float] = Field(None, description="Passes Per Defensive Action (Lower = Higher Press)")
    ppda_a: Optional[float] = Field(None, description="Passes Per Defensive Action")
    field_tilt_h: Optional[float] = Field(None, description="Final Third Touch share %")
    field_tilt_a: Optional[float] = Field(None, description="Final Third Touch share %")
    travel_km_h: float = 0.0
    travel_km_a: float = 0.0
    new_manager_h: bool = False
    new_manager_a: bool = False

    # ── V6.0+ HYPER-NICHE VARIABLES (Corners, Cards, Player Props) ─────
    wing_play_intensity_h: float = Field(0.0, description="0-10 scale of reliance on crosses/wing play")
    wing_play_intensity_a: float = Field(0.0, description="0-10 scale")
    tactical_fouls_h: float = Field(0.0, description="Avg tactical fouls per game")
    tactical_fouls_a: float = Field(0.0, description="Avg tactical fouls per game")
    
    # Player Props (Key specific players to watch)
    penalty_taker_h: Optional[str] = None
    penalty_taker_a: Optional[str] = None
    assist_specialist_h: Optional[str] = None
    assist_specialist_a: Optional[str] = None
    expected_top_scorer_h: Optional[str] = None
    expected_top_scorer_a: Optional[str] = None
    
    # Corner/Card Tendencies
    avg_corners_h: float = 5.0
    avg_corners_a: float = 4.5
    referee_fouls_per_card: float = 6.0 # How many fouls before this ref cards?

    # ── Derived adjustments (computed from fields above) ───────────────────

    def injury_factor(self, team: str) -> float:
        """
        Returns a strength multiplier based on squad availability.
        1.0 = full strength, 0.7 = severely depleted.
        """
        is_home = team.lower() == 'home'
        injuries = self.home_injuries if is_home else self.away_injuries
        suspensions = self.home_suspensions if is_home else self.away_suspensions
        missing_key = self.missing_key_player_home if is_home else self.missing_key_player_away

        # Weight by position importance (attackers matter more for xG)
        severity_map = {'low': 0.02, 'medium': 0.05, 'high': 0.10, 'season_ending': 0.15}
        impact = sum(severity_map.get(i.severity or 'medium', 0.05) for i in injuries)
        impact += len(suspensions) * 0.05
        if missing_key:
            impact += 0.15

        return max(0.5, 1.0 - impact)

    def form_factor(self, team: str) -> float:
        """
        Recent form as a strength multiplier (0.8 = terrible, 1.2 = red-hot).
        Based on points per game over last 5 matches.
        """
        is_home = team.lower() == 'home'
        pts = self.home_pts_last5 if is_home else self.away_pts_last5
        games = len(self.home_form_last5 if is_home else self.away_form_last5)
        if games == 0:
            return 1.0
        ppg = pts / games
        # PPG 0.0 → 0.8, PPG 1.5 → 1.0, PPG 3.0 → 1.2
        return 0.8 + (ppg / 3.0) * 0.4

    def referee_card_factor(self) -> float:
        """
        Referee tendency: high card avg → more over bets, fewer clean sheets.
        Returns a multiplier for over 2.5 and BTTS probability.
        """
        if self.referee_avg_cards is None:
            return 1.0
        # baseline 3.0 cards = 1.0, each additional card adds 0.05 to over
        return 1.0 + (self.referee_avg_cards - 3.0) * 0.05

    def h2h_home_win_rate(self) -> float:
        if self.h2h_matches == 0:
            return 0.33
        return self.h2h_home_wins / self.h2h_matches

    def to_bayesian_features(self) -> dict:
        """
        Convert intelligence into the features Bayesian engine + ML need.
        This is what gets passed INTO the prediction pipeline.
        """
        # V6.0 Adjustments
        travel_penalty_h = 1.0 - (self.travel_km_h / 5000.0) if self.travel_km_h > 500 else 1.0
        travel_penalty_a = 1.0 - (self.travel_km_a / 5000.0) if self.travel_km_a > 500 else 1.0
        
        manager_bounce_h = 1.05 if self.new_manager_h else 1.0
        manager_bounce_a = 1.05 if self.new_manager_a else 1.0
        
        press_h = 1.1 if (self.ppda_h and self.ppda_h < 10.0) or self.tactical_style_home == 'high_press' else 1.0
        press_a = 1.1 if (self.ppda_a and self.ppda_a < 10.0) or self.tactical_style_away == 'high_press' else 1.0

        return {
            # Team strength adjusted by injuries + form + V6.0 Quantum
            'home_att': 1.2 * self.injury_factor('home') * self.form_factor('home') * press_h * manager_bounce_h * travel_penalty_h,
            'away_att': 1.2 * self.injury_factor('away') * self.form_factor('away') * press_a * manager_bounce_a * travel_penalty_a,
            'home_def': 1.0 * self.injury_factor('away') * self.away_fatigue_index * (1.1 if self.field_tilt_a and self.field_tilt_a > 60 else 1.0), 
            'away_def': 1.0 * self.injury_factor('home') * self.home_fatigue_index * (1.1 if self.field_tilt_h and self.field_tilt_h > 60 else 1.0),

            # Fatigue & Pitch adjustments
            'fatigue_h': self.home_fatigue_index,
            'fatigue_a': self.away_fatigue_index,
            'is_poor_pitch': 1 if self.pitch_condition in ('poor', 'waterlogged') else 0,
            'is_rainy': 1 if self.weather_condition in ('rain', 'snow') else 0,
            
            # V6.0 Raw
            'ppda_h': self.ppda_h,
            'ppda_a': self.ppda_a,
            'field_tilt_h': self.field_tilt_h,
            'field_tilt_a': self.field_tilt_a,
            'travel_h': self.travel_km_h,
            'travel_a': self.travel_km_a,

            # xG-based priors (primary when available)
            'home_xg': self.xg.home_xg if self.xg else None,
            'away_xg': self.xg.away_xg if self.xg else None,
            'home_xg_avg5': sum(self.xg.home_xg_last5) / max(1, len(self.xg.home_xg_last5)) if self.xg and self.xg.home_xg_last5 else None,
            'away_xg_avg5': sum(self.xg.away_xg_last5) / max(1, len(self.xg.away_xg_last5)) if self.xg and self.xg.away_xg_last5 else None,

            # xT (Expected Threat) — possession zone danger, complements xG
            'home_xt': self.xg.home_xt if self.xg and self.xg.home_xt else None,
            'away_xt': self.xg.away_xt if self.xg and self.xg.away_xt else None,
            'home_xt_avg5': sum(self.xg.home_xt_last5) / max(1, len(self.xg.home_xt_last5)) if self.xg and self.xg.home_xt_last5 else None,
            'away_xt_avg5': sum(self.xg.away_xt_last5) / max(1, len(self.xg.away_xt_last5)) if self.xg and self.xg.away_xt_last5 else None,

            # Form
            'home_pts_last5': self.home_pts_last5,
            'away_pts_last5': self.away_pts_last5,

            # H2H
            'h2h_home_win_rate': self.h2h_home_win_rate(),
            'h2h_home_goals_avg': self.h2h_home_goals_total / max(1, self.h2h_matches),
            'h2h_away_goals_avg': self.h2h_away_goals_total / max(1, self.h2h_matches),

            # Referee
            'referee_card_factor': self.referee_card_factor(),

            # Formation attack/defense weighting
            'home_formation_att': 1.1 if self.formation_home in ('4-3-3', '3-4-3', '4-2-3-1') else 1.0,
            'away_formation_att': 1.1 if self.formation_away in ('4-3-3', '3-4-3', '4-2-3-1') else 1.0,

            # Availability counts
            'home_injury_count': len(self.home_injuries) + len(self.home_suspensions),
            'away_injury_count': len(self.away_injuries) + len(self.away_suspensions),

            # Motivation
            'home_must_win': 1 if self.home_motivation in ('must_win', 'relegation_fight') else 0,
            'away_must_win': 1 if self.away_motivation in ('must_win', 'relegation_fight') else 0,
            'is_derby': 1 if self.home_motivation == 'derby' or self.away_motivation == 'derby' else 0,
            
            # Bench impact
            'bench_impact_h': self.bench_strength_h,
            'bench_impact_a': self.bench_strength_a,
        }


# =============================================================================
# Firecrawl Agent — Key Rotation
# =============================================================================

FIRECRAWL_KEYS = [
    os.environ.get('FIRECRAWL_API_KEY', '').strip(),
    'fc-13385e02de0744aa8367813ea2b418c4',
    'fc-4589b8c19d6b49b0bf9c9da0e338c1d9',
    'fc-68142f4be23a4adeb3ad919e708cdc1b',
    'fc-ef9817579e4e4b2980fb5560e90236bc',
    'fc-d1cc96ae5ada46f0b1f16467675f11b6',
    'fc-9f835d7e0747491b996558965781f6cf',
]


class FirecrawlAgentPool:
    """
    Manages a pool of Firecrawl API keys with automatic rotation on failure.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, keys: list[str] = None):
        if self._initialized:
            return
        self._initialized = True
        self._keys = keys or FIRECRAWL_KEYS
        self._app = None
        self._active_key_index = 0
        self._connect()

    def _connect(self):
        if not FIRECRAWL_AVAILABLE:
            print("[Firecrawl] SDK not installed — run: pip install firecrawl-py")
            return
        for i, key in enumerate(self._keys):
            if not key:
                continue
            try:
                app = Firecrawl(api_key=key)
                # Test with a quick map call instead of starting an agent
                from firecrawl.v2.utils import http_client as _hc
                test_resp = _hc.HttpClient(api_key=key, api_url="https://api.firecrawl.dev").get("/v1/ping")
                if test_resp.ok or test_resp.status_code == 404:  # 404 means endpoint exists
                    self._app = app
                    self._active_key_index = i
                    print(f"[Firecrawl] Connected — key index {i} ({key[:12]}...)")
                    return
            except Exception as e:
                print(f"[Firecrawl] Key {i} ({key[:12]}...) failed: {e}")
                continue
        print("[Firecrawl] No working keys — agent disabled")

    def get_app(self):
        return self._app

    def is_available(self) -> bool:
        return self._app is not None

    def rotate_on_failure(self):
        """Marks current key as failed/exhausted and tries to connect to next one."""
        print(f"[Firecrawl] Rotating key due to failure (Index {self._active_key_index} exhausted)")
        if not self._keys:
            self._app = None
            return
        if self._active_key_index < 0 or self._active_key_index >= len(self._keys):
            self._active_key_index = 0
        try:
            self._keys.pop(self._active_key_index)
        except Exception:
            self._keys = []
        self._app = None
        if self._keys:
            self._connect()
        else:
            print("[Firecrawl] No keys remain â€” agent disabled")


_agent_pool: Optional[FirecrawlAgentPool] = None


def get_firecrawl_agent() -> FirecrawlAgentPool:
    global _agent_pool
    if _agent_pool is None:
        _agent_pool = FirecrawlAgentPool()
    return _agent_pool


# =============================================================================
# Stage 1: Discover today's matches for a league (Agent finds them, no FDO needed)
# =============================================================================

def discover_league_matches(league: str, target_date: str) -> list[MatchIntel]:
    return discover_multiple_leagues([league], target_date)

def discover_multiple_leagues(leagues: list[str], target_date: str) -> list[MatchIntel]:
    """
    CREDIT OPTIMIZED: Discovers matches for a LIST of leagues in a single call.
    This saves significant Firecrawl credits by bundling extraction into one agent session.
    """
    pool = get_firecrawl_agent()

    if not pool.is_available():
        print(f"[Firecrawl] Agent unavailable — cannot discover matches")
        return []

    leagues_str = ", ".join(leagues)
    prompt = f"""
You are a football data researcher. Find ALL football matches scheduled for
Leagues: {leagues_str}
Date: {target_date}

For EACH match, return:
1. Home team full name
2. Away team full name
3. League name
4. Kickoff time in UTC (e.g. '17:30')
5. Expected Goals (xG) AND Expected Threat (xT) for BOTH teams — from sofascore, whoscored, StatsBomb, or footystats:
   - Overall xG this season (home and away separately)
   - Last 5 match xG for each team (list of floats)
   - Last 5 match xGA (expected goals against) for each team (list of floats)
   - xT (Expected Threat) per possession for BOTH teams — season average and last 5 matches
     (xT is a zone-based model: e.g., 0.08 means each possession is worth 0.08 expected goals;
      typical range: 0.05-0.15 for top teams; provide as float)
   - Source of xG and xT data (e.g. 'sofascore', 'statsbomb')
6. Injured players for EACH team: name, position (GK/DF/MF/FW), severity (low/medium/high/season_ending)
7. Suspended players for EACH team: name only
8. Any star/key player missing (injury or suspension): yes/no for EACH team
9. Referee name and their average cards per game this season
10. Form guide for last 5 league matches for EACH team: 
    - result (W/D/L), score (e.g. '2-1'), venue (home/away), xG, xGA, and Date (YYYY-MM-DD)
11. Head-to-head record (last 5 meetings): result for each, overall W/D/A for home team, total goals home/away
12. Expected formations for EACH team (e.g. '4-3-3')
13. Match motivation for EACH team: must_win, derby, europe_spot, relegation_fight, routine (or null)
14. Current league table position and total points for EACH team
15. REAL-TIME BETTING ODDS (Home, Draw, Away) from a major bookmaker (e.g. Bet365, Pinnacle).
16. V6.0+ QUANTUM & NICHE VARIABLES:
    - Weather, Pitch, Attendance, Fatigue Index (1.0-1.3)
    - Tactical Style, Defensive Line, Bench Strength (0.8-1.2)
    - PPDA (Numerical), Field Tilt %, Travel Distance (km), New Manager (true/false)
    - Wing Play Intensity (0-10), Tactical Fouling (Avg), Avg Corners, Referee's Fouls per Card.

Be precise. Return ALL matches found for these leagues on this date.
"""


    attempts = 0
    while attempts < 2:
        try:
            app = pool.get_app()
            job = app.start_agent(
                prompt=prompt,
                model="spark-1-pro",
                max_credits=20,
            )
            job_id = getattr(job, 'id', None)
            if not job_id:
                print("[Firecrawl] Discovery started but no job ID returned")
                return []

            # Poll for completion
            import time
            status = None
            for _ in range(30):  # 60s timeout
                status = app.get_agent_status(job_id)
                if status.status in ('completed', 'failed', 'cancelled'):
                    print(f"[Firecrawl] Discovery {status.status}: {status.data}")
                    break
                time.sleep(2)

            if status and status.status == 'completed' and status.data:
                import json as _json
                import re
                try:
                    data_str = str(status.data)
                    if "```json" in data_str:
                        data_str = re.search(r"```json\s*(.*?)\s*```", data_str, re.DOTALL).group(1)
                    elif "```" in data_str:
                        data_str = re.search(r"```\s*(.*?)\s*```", data_str, re.DOTALL).group(1)

                    if '[' in data_str:
                        json_start = data_str.index('[')
                        json_end = data_str.rindex(']') + 1
                        json_str = data_str[json_start:json_end]
                        raw_matches = _json.loads(json_str)
                        matches = [MatchIntel(**m) for m in raw_matches]
                        print(f"[Firecrawl] Discovered {len(matches)} matches for {leagues_str} on {target_date}")
                        return matches
                except Exception as parse_err:
                    print(f"[Firecrawl] JSON parse error: {parse_err}, data preview: {data_str[:200]}...")
            return []
        except Exception as e:
            if "Insufficient credits" in str(e) or "402" in str(e):
                try:
                    pool.rotate_on_failure()
                except Exception as rotate_err:
                    print(f"[Firecrawl] Rotation skipped after credit failure: {rotate_err}")
                    pool._app = None
                attempts += 1
                continue
            print(f"[Firecrawl] Discovery error for {leagues_str}: {e}")
            return []
    return []


# =============================================================================
# Stage 2: Enrich an existing match list with Firecrawl intelligence
#          (used when FDO already found the match list)
# =============================================================================

ENRICHMENT_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache', 'firecrawl_enrichment')

def _cache_key(home_team: str, away_team: str, league: str, match_date: str) -> str:
    raw = f"{league}|{match_date}|{home_team}|{away_team}".lower().strip()
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

def _cache_path(home_team: str, away_team: str, league: str, match_date: str) -> str:
    return os.path.join(ENRICHMENT_CACHE_DIR, _cache_key(home_team, away_team, league, match_date) + '.json')

def _model_to_dict(model):
    if hasattr(model, 'model_dump'):
        return model.model_dump()
    if hasattr(model, 'dict'):
        return model.dict()
    return dict(model)

def _read_enrichment_cache(home_team: str, away_team: str, league: str, match_date: str) -> Optional[MatchIntel]:
    ttl_seconds = int(os.environ.get('FIRECRAWL_ENRICHMENT_CACHE_TTL_SECONDS', str(6 * 60 * 60)))
    path = _cache_path(home_team, away_team, league, match_date)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        cached_at = payload.get('cached_at_epoch', 0)
        if ttl_seconds > 0 and time.time() - cached_at > ttl_seconds:
            return None
        data = payload.get('intel') or {}
        data['_cache_hit'] = True
        return MatchIntel(**data)
    except Exception as exc:
        print(f"[Firecrawl] Cache read failed: {exc}")
        return None

def _write_enrichment_cache(home_team: str, away_team: str, league: str, match_date: str, intel: MatchIntel):
    try:
        os.makedirs(ENRICHMENT_CACHE_DIR, exist_ok=True)
        data = _model_to_dict(intel)
        data.pop('_cache_hit', None)
        with open(_cache_path(home_team, away_team, league, match_date), 'w', encoding='utf-8') as f:
            json.dump({
                "cached_at": datetime.utcnow().isoformat(),
                "cached_at_epoch": time.time(),
                "league": league,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "intel": data,
            }, f, indent=2)
    except Exception as exc:
        print(f"[Firecrawl] Cache write failed: {exc}")

def enrich_match(home_team: str, away_team: str, league: str, match_date: str, retry_count: int = 0) -> Optional[MatchIntel]:
    """
    Enrich a single match with full intelligence.
    Used when the match list is already known (from FDO) but
    intelligence is missing.
    """
    cached = _read_enrichment_cache(home_team, away_team, league, match_date)
    if cached:
        print(f"[Firecrawl] Cache hit: {home_team} vs {away_team} ({league}, {match_date})")
        return cached

    pool = get_firecrawl_agent()

    if not pool.is_available():
        return None

    prompt = f"""
Find detailed match intelligence for:
Home: {home_team}
Away: {away_team}
League: {league}
Date: {match_date}

Return:
1. Expected Goals (xG) home and away - season total and last 5 matches each (xG and xGA lists)
2. Injured players for EACH team: name, position, severity
3. Suspended players for EACH team: name
4. Key player missing: yes/no for EACH team
5. News Sentiment: Find at least 2 recent news headlines for EACH team and assign a sentiment_score (-1 to 1), impact_area (defense/morale/tactical), and relevance.
6. Referee + avg cards per game
7. Form guide last 5 league matches: result, score, venue
8. H2H last 5: results, overall W/D/A for home team
9. Expected formations
10. Motivation for EACH team
11. League position and points for EACH team
"""

    attempts = max(1, retry_count + 1)
    while attempts <= 4:
        try:
            app = pool.get_app()
            job = app.start_agent(
                prompt=prompt,
                model="spark-1-pro",
                max_credits=20,
            )
            job_id = getattr(job, 'id', None)
            if not job_id:
                return None

            import time
            status = None
            for _ in range(20):  # 40s timeout
                status = app.get_agent_status(job_id)
                if status.status in ('completed', 'failed', 'cancelled'):
                    break
                time.sleep(2)

            if status and status.status == 'completed' and status.data:
                import json as _json
                try:
                    data_str = str(status.data)
                    if '[' in data_str:
                        json_start = data_str.index('[')
                        json_end = data_str.rindex(']') + 1
                        json_str = data_str[json_start:json_end]
                        raw = _json.loads(json_str)
                        if isinstance(raw, list) and len(raw) > 0:
                            intel = MatchIntel(**raw[0])
                            _write_enrichment_cache(home_team, away_team, league, match_date, intel)
                            return intel
                except Exception as parse_err:
                    print(f"[Firecrawl] Enrich parse error: {parse_err}")
            return None
        except Exception as e:
            error_str = str(e)
            if "Insufficient credits" in error_str or "402" in error_str:
                pool.rotate_on_failure()
                attempts += 1
                continue
            if "429" in error_str or "Rate limit" in error_str or "Rate Limit" in error_str:
                if attempts < 4:
                    wait_time = (2 ** (attempts - 1)) * 5
                    print(f"[Firecrawl] Rate limited. Backing off for {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                    attempts += 1
                    continue
            print(f"[Firecrawl] Enrich error: {home_team} vs {away_team}: {e}")
            return None
    return None


def enrich_matchday(matches: list[dict], league: str) -> list[dict]:
    """
    Take a list of match dicts (from FDO scraper) and enrich each one.
    Adds 'intel' key with full MatchIntel data to each match dict.
    """
    enriched = []
    for i, m in enumerate(matches):
        print(f"[Firecrawl] Enriching {i+1}/{len(matches)}: {m.get('Home','?')} vs {m.get('Away','?')}")
        intel = enrich_match(
            home_team=m.get('Home', ''),
            away_team=m.get('Away', ''),
            league=league,
            match_date=m.get('Date', ''),
        )
        m['intel'] = intel
        enriched.append(m)
        time.sleep(1)  # rate limit respect
    return enriched
