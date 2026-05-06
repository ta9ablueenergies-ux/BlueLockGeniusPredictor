# ANTIGRAVITY V6.0 SQLITE PERSISTENCE HUB
import sqlite3
import json
import os
from datetime import datetime
from team_name_normalizer import (
    canonical_team_name,
    clear_team_name_cache,
    default_alias_rows,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")

WHY_STAT_COLUMNS = [
    'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HO', 'AO',
    'HPoss', 'APoss', 'HXG', 'AXG', 'HBC', 'ABC',
]

def init_db():
    """Initializes the SQLite database with the V6.0 Quantum schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create team_aliases table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_aliases (
            league TEXT,
            alias_name TEXT,
            canonical_name TEXT,
            PRIMARY KEY (league, alias_name)
        )
    ''')
    for league, alias_name, canonical_name in default_alias_rows():
        cursor.execute(
            '''
            INSERT OR IGNORE INTO team_aliases (league, alias_name, canonical_name)
            VALUES (?, ?, ?)
            ''',
            (league, alias_name, canonical_name),
        )
    
    # Matches Table: Stores everything from identity to niche metrics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            id TEXT PRIMARY KEY,
            league TEXT,
            match_date TEXT,
            match_time TEXT,
            home_team TEXT,
            away_team TEXT,
            prediction TEXT,
            eqi_score REAL,
            stake_pct REAL,
            primary_market TEXT,
            secondary_market TEXT,
            p_btts REAL,
            p_o25 REAL,
            over15_prob REAL,
            under15_prob REAL,
            over35_prob REAL,
            under35_prob REAL,
            p_1x REAL,
            p_x2 REAL,
            p_dnb REAL,
            value_edge REAL,
            cld_delta REAL,
            corners_exp REAL,
            cards_exp REAL,
            player_props TEXT,
            momentum_h TEXT,
            momentum_a TEXT,
            trust_breakdown TEXT,
            
            -- V6.0 Quantum Intel (JSON Blob for flexibility)
            intel_raw TEXT,
            
            -- Meta
            last_updated TEXT,
            is_mock INTEGER DEFAULT 0,
            source_url TEXT,
            flashscore_id TEXT,
            referee TEXT,
            venue TEXT,
            HY REAL,
            AY REAL,
            HR REAL,
            AR REAL,
            HC REAL,
            AC REAL,
            HS REAL,
            "AS" REAL,
            HST REAL,
            AST REAL,
            HF REAL,
            AF REAL,
            HO REAL,
            AO REAL,
            HPoss REAL,
            APoss REAL,
            HXG REAL,
            AXG REAL,
            HBC REAL,
            ABC REAL,
            stats_scraped_at TEXT,
            market_model_json TEXT,
            rationale_json TEXT
        )
    ''')
    
    # History Table: For Plan 2 Sequence Modeling
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sequence_history (
            team_name TEXT,
            match_date TEXT,
            result TEXT,
            score TEXT,
            xg REAL,
            xga REAL,
            PRIMARY KEY (team_name, match_date)
        )
    ''')
    
    # Migration: Add columns if they don't exist (safe to re-run)
    migrations = [
        'ALTER TABLE matches ADD COLUMN momentum_h TEXT',
        'ALTER TABLE matches ADD COLUMN momentum_a TEXT',
        'ALTER TABLE matches ADD COLUMN trust_breakdown TEXT',
        'ALTER TABLE matches ADD COLUMN centrality_score REAL',
        'ALTER TABLE matches ADD COLUMN v8_motif INTEGER DEFAULT 0',
        'ALTER TABLE matches ADD COLUMN data_source TEXT DEFAULT \'mock\'',
        'ALTER TABLE matches ADD COLUMN source_confidence REAL DEFAULT 0.2',
        'ALTER TABLE matches ADD COLUMN run_id TEXT',
        'ALTER TABLE matches ADD COLUMN graph_snapshot_id TEXT',
        'ALTER TABLE matches ADD COLUMN over15_prob REAL',
        'ALTER TABLE matches ADD COLUMN under15_prob REAL',
        'ALTER TABLE matches ADD COLUMN over35_prob REAL',
        'ALTER TABLE matches ADD COLUMN under35_prob REAL',
        'ALTER TABLE matches ADD COLUMN source_url TEXT',
        'ALTER TABLE matches ADD COLUMN flashscore_id TEXT',
        'ALTER TABLE matches ADD COLUMN referee TEXT',
        'ALTER TABLE matches ADD COLUMN venue TEXT',
        'ALTER TABLE matches ADD COLUMN HY REAL',
        'ALTER TABLE matches ADD COLUMN AY REAL',
        'ALTER TABLE matches ADD COLUMN HR REAL',
        'ALTER TABLE matches ADD COLUMN AR REAL',
        'ALTER TABLE matches ADD COLUMN HC REAL',
        'ALTER TABLE matches ADD COLUMN AC REAL',
        'ALTER TABLE matches ADD COLUMN stats_scraped_at TEXT',
        'ALTER TABLE matches ADD COLUMN market_model_json TEXT',
        'ALTER TABLE matches ADD COLUMN rationale_json TEXT',
        'ALTER TABLE training_examples ADD COLUMN api_football_id TEXT',
        'ALTER TABLE training_examples ADD COLUMN referee TEXT',
        'ALTER TABLE training_examples ADD COLUMN venue TEXT',
        'ALTER TABLE training_examples ADD COLUMN HY REAL',
        'ALTER TABLE training_examples ADD COLUMN AY REAL',
        'ALTER TABLE training_examples ADD COLUMN HR REAL',
        'ALTER TABLE training_examples ADD COLUMN AR REAL',
        'ALTER TABLE training_examples ADD COLUMN HC REAL',
        'ALTER TABLE training_examples ADD COLUMN AC REAL',
        'ALTER TABLE training_examples ADD COLUMN flashscore_id TEXT',
        'ALTER TABLE training_examples ADD COLUMN source_url TEXT',
        'ALTER TABLE training_examples ADD COLUMN stats_scraped_at TEXT',
    ]
    for table in ('matches', 'training_examples'):
        for col in WHY_STAT_COLUMNS:
            migrations.append(f'ALTER TABLE {table} ADD COLUMN "{col}" REAL')
    for migration in migrations:
        try:
            cursor.execute(migration)
        except sqlite3.OperationalError:
            pass  # Column already exists

    # ── Match Results Table (U1: for calibration tracking) ─────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_match_evidence (
            id TEXT PRIMARY KEY,
            match_id TEXT,
            league TEXT,
            match_date TEXT,
            home_team TEXT,
            away_team TEXT,
            source TEXT,
            source_url TEXT,
            evidence_type TEXT,
            raw_json TEXT,
            extracted_json TEXT,
            data_quality REAL,
            scraped_at TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_learning_patterns (
            match_id TEXT,
            pattern_key TEXT,
            confidence REAL,
            source TEXT,
            details_json TEXT,
            created_at TEXT,
            PRIMARY KEY (match_id, pattern_key)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_results (
            id TEXT PRIMARY KEY,
            actual_fthg INTEGER,
            actual_ftag INTEGER,
            actual_ftr TEXT,
            closing_home_odds REAL,
            closing_draw_odds REAL,
            closing_away_odds REAL,
            closing_prob_home REAL,
            closing_prob_draw REAL,
            closing_prob_away REAL,
            result_written_at TEXT
        )
    ''')

    # ── Closing Line Tracking Table (U5: CLV monitoring) ──────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS closing_line_tracking (
            id TEXT PRIMARY KEY,
            match_date TEXT,
            league TEXT,
            home_team TEXT,
            away_team TEXT,
            predicted_prob REAL,
            closing_prob REAL,
            open_prob REAL,
            cld REAL,
            actual_ftr TEXT,
            hit INTEGER,
            edge REAL,
            stake_pct REAL,
            kelly_fraction REAL,
            pnl REAL,
            recorded_at TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_examples (
            id TEXT PRIMARY KEY,
            league TEXT NOT NULL,
            match_date TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_goals INTEGER,
            away_goals INTEGER,
            actual_ftr TEXT NOT NULL,
            home_odds REAL,
            draw_odds REAL,
            away_odds REAL,
            market_prob_home REAL,
            market_prob_draw REAL,
            market_prob_away REAL,
            closing_home_odds REAL,
            closing_draw_odds REAL,
            closing_away_odds REAL,
            source TEXT DEFAULT 'historical_csv',
            source_confidence REAL DEFAULT 0.90,
            features_json TEXT,
            api_football_id TEXT,
            referee TEXT,
            venue TEXT,
            HY REAL,
            AY REAL,
            HR REAL,
            AR REAL,
            HC REAL,
            AC REAL,
            HS REAL,
            "AS" REAL,
            HST REAL,
            AST REAL,
            HF REAL,
            AF REAL,
            HO REAL,
            AO REAL,
            HPoss REAL,
            APoss REAL,
            HXG REAL,
            AXG REAL,
            HBC REAL,
            ABC REAL,
            flashscore_id TEXT,
            source_url TEXT,
            stats_scraped_at TEXT,
            yc_home_avg5 REAL,
            yc_away_avg5 REAL,
            yc_home_avg10 REAL,
            yc_away_avg10 REAL,
            corners_home_avg5 REAL,
            corners_away_avg5 REAL,
            corners_home_avg10 REAL,
            corners_away_avg10 REAL,
            ref_yc_avg REAL,
            created_at TEXT,
            updated_at TEXT
        )
    ''')

    # Performance indexes for hot paths
    index_statements = [
        'CREATE INDEX IF NOT EXISTS idx_matches_league_mock_p1x_date_time ON matches(league, is_mock, p_1x, match_date, match_time)',
        'CREATE INDEX IF NOT EXISTS idx_matches_home_date ON matches(home_team, match_date)',
        'CREATE INDEX IF NOT EXISTS idx_matches_away_date ON matches(away_team, match_date)',
        'CREATE INDEX IF NOT EXISTS idx_clv_league_date ON closing_line_tracking(league, match_date)',
        'CREATE INDEX IF NOT EXISTS idx_clv_recorded_at ON closing_line_tracking(recorded_at)',
        'CREATE INDEX IF NOT EXISTS idx_match_results_id ON match_results(id)',
        'CREATE INDEX IF NOT EXISTS idx_training_examples_league_date ON training_examples(league, match_date)',
        'CREATE INDEX IF NOT EXISTS idx_training_examples_home_date ON training_examples(home_team, match_date)',
        'CREATE INDEX IF NOT EXISTS idx_training_examples_away_date ON training_examples(away_team, match_date)',
        'CREATE INDEX IF NOT EXISTS idx_training_examples_api_football_id ON training_examples(api_football_id)',
        'CREATE INDEX IF NOT EXISTS idx_matches_flashscore_id ON matches(flashscore_id)',
        'CREATE INDEX IF NOT EXISTS idx_training_examples_flashscore_id ON training_examples(flashscore_id)',
        'CREATE INDEX IF NOT EXISTS idx_raw_match_evidence_match_id ON raw_match_evidence(match_id)',
        'CREATE INDEX IF NOT EXISTS idx_raw_match_evidence_type ON raw_match_evidence(evidence_type, scraped_at)',
        'CREATE INDEX IF NOT EXISTS idx_match_learning_patterns_key ON match_learning_patterns(pattern_key)',
    ]
    for stmt in index_statements:
        try:
            cursor.execute(stmt)
        except sqlite3.OperationalError:
            pass

    cursor.execute("""
        UPDATE matches
        SET source_confidence = CASE
            WHEN data_source = 'fdo' THEN 0.90
            WHEN data_source = 'firecrawl' THEN 0.75
            WHEN data_source = 'mock' THEN 0.20
            ELSE COALESCE(source_confidence, 0.40)
        END
        WHERE source_confidence IS NULL
           OR (source_confidence = 0.20 AND data_source != 'mock')
    """)

    conn.commit()
    clear_team_name_cache()
    conn.close()

def calculate_source_confidence(data_source, intel_feats=None):
    """Normalize data provenance into a confidence score used by the UI and staking layer."""
    base = {
        'fdo': 0.90,
        'firecrawl': 0.75,
        'historical': 0.85,
        'sqlite_history': 0.85,
        'free_repo': 0.80,
        'browser': 0.78,
        'mock': 0.20,
    }.get(data_source or 'mock', 0.40)
    intel_feats = intel_feats or {}
    if intel_feats:
        base += 0.05
    if isinstance(intel_feats, dict) and intel_feats.get('_cache_hit'):
        base -= 0.03
    return round(max(0.0, min(1.0, base)), 3)

def normalized_market_probs(home_odds, draw_odds, away_odds):
    inv = []
    for odd in (home_odds, draw_odds, away_odds):
        try:
            odd = float(odd)
            inv.append(1.0 / odd if odd > 1.01 else 0.0)
        except Exception:
            inv.append(0.0)
    total = sum(inv)
    if total <= 0:
        return None, None, None
    return tuple(x / total for x in inv)

def trust_score_from_eqi(value):
    """Translate legacy EQI storage into a clear 1-100 Trust Score."""
    try:
        score = float(value)
    except Exception:
        score = 0.0
    return round(max(1.0, min(100.0, score)), 1)

_ALIAS_CACHE = None

def resolve_team_name(league, team_name):
    global _ALIAS_CACHE
    if _ALIAS_CACHE is None:
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute('SELECT league, alias_name, canonical_name FROM team_aliases')
            _ALIAS_CACHE = {}
            for l, a, c in cur.fetchall():
                _ALIAS_CACHE[(l, a)] = c
            conn.close()
        except sqlite3.OperationalError:
            _ALIAS_CACHE = {}

    exact = _ALIAS_CACHE.get((league, team_name), team_name)
    return canonical_team_name(league, exact, db_path=DB_PATH)

def upsert_training_example(example):
    """Insert a normalized supervised training example for neural/GNN learning."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    home_odds = example.get('home_odds')
    draw_odds = example.get('draw_odds')
    away_odds = example.get('away_odds')
    p_h, p_d, p_a = normalized_market_probs(home_odds, draw_odds, away_odds)
    features_json = json.dumps(example.get('features', {}))
    
    league = example.get('league')
    home_team = resolve_team_name(league, example.get('home_team'))
    away_team = resolve_team_name(league, example.get('away_team'))
    
    # If id is provided but uses old aliases, we don't change the id generation here
    # since it's already generated by the caller, but we probably should.
    # We will just write the resolved home/away teams.
    
    cursor.execute('''
        INSERT INTO training_examples (
            id, league, match_date, home_team, away_team,
            home_goals, away_goals, actual_ftr,
            home_odds, draw_odds, away_odds,
            market_prob_home, market_prob_draw, market_prob_away,
            closing_home_odds, closing_draw_odds, closing_away_odds,
            source, source_confidence, features_json,
            api_football_id, HY, AY, HR, AR, HC, AC,
            flashscore_id, source_url, stats_scraped_at,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            league=excluded.league,
            match_date=excluded.match_date,
            home_team=excluded.home_team,
            away_team=excluded.away_team,
            home_goals=excluded.home_goals,
            away_goals=excluded.away_goals,
            actual_ftr=excluded.actual_ftr,
            home_odds=excluded.home_odds,
            draw_odds=excluded.draw_odds,
            away_odds=excluded.away_odds,
            market_prob_home=excluded.market_prob_home,
            market_prob_draw=excluded.market_prob_draw,
            market_prob_away=excluded.market_prob_away,
            closing_home_odds=excluded.closing_home_odds,
            closing_draw_odds=excluded.closing_draw_odds,
            closing_away_odds=excluded.closing_away_odds,
            source=excluded.source,
            source_confidence=excluded.source_confidence,
            features_json=excluded.features_json,
            api_football_id=COALESCE(excluded.api_football_id, training_examples.api_football_id),
            HY=COALESCE(excluded.HY, training_examples.HY),
            AY=COALESCE(excluded.AY, training_examples.AY),
            HR=COALESCE(excluded.HR, training_examples.HR),
            AR=COALESCE(excluded.AR, training_examples.AR),
            HC=COALESCE(excluded.HC, training_examples.HC),
            AC=COALESCE(excluded.AC, training_examples.AC),
            flashscore_id=COALESCE(excluded.flashscore_id, training_examples.flashscore_id),
            source_url=COALESCE(excluded.source_url, training_examples.source_url),
            stats_scraped_at=COALESCE(excluded.stats_scraped_at, training_examples.stats_scraped_at),
            updated_at=excluded.updated_at
    ''', (
        example.get('id'), example.get('league'), example.get('match_date'),
        home_team, away_team,
        example.get('home_goals'), example.get('away_goals'), example.get('actual_ftr'),
        home_odds, draw_odds, away_odds, p_h, p_d, p_a,
        example.get('closing_home_odds'), example.get('closing_draw_odds'), example.get('closing_away_odds'),
        example.get('source', 'historical_csv'), example.get('source_confidence', 0.90),
        features_json,
        example.get('api_football_id'),
        example.get('HY'), example.get('AY'), example.get('HR'), example.get('AR'),
        example.get('HC'), example.get('AC'),
        example.get('flashscore_id'), example.get('source_url'), example.get('stats_scraped_at'),
        example.get('created_at', now), now
    ))
    cursor.execute('''
        UPDATE training_examples
        SET referee = COALESCE(?, referee),
            venue = COALESCE(?, venue)
        WHERE id = ?
    ''', (
        example.get('referee') or None,
        example.get('venue') or None,
        example.get('id'),
    ))
    why_values = [example.get(col) for col in WHY_STAT_COLUMNS]
    if any(value is not None for value in why_values):
        set_clause = ', '.join([f'"{col}" = COALESCE(?, "{col}")' for col in WHY_STAT_COLUMNS])
        cursor.execute(
            f'UPDATE training_examples SET {set_clause} WHERE id = ?',
            (*why_values, example.get('id')),
        )
    conn.commit()
    conn.close()

def upsert_raw_match_evidence(evidence):
    """Store raw and extracted scraper evidence for explainability/autopsy."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    scraped_at = evidence.get('scraped_at') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    evidence_id = evidence.get('id') or '|'.join([
        str(evidence.get('match_id') or ''),
        str(evidence.get('evidence_type') or 'unknown'),
        str(evidence.get('source') or 'unknown'),
    ])
    cursor.execute('''
        INSERT INTO raw_match_evidence (
            id, match_id, league, match_date, home_team, away_team,
            source, source_url, evidence_type, raw_json, extracted_json,
            data_quality, scraped_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            league=excluded.league,
            match_date=excluded.match_date,
            home_team=excluded.home_team,
            away_team=excluded.away_team,
            source=excluded.source,
            source_url=excluded.source_url,
            evidence_type=excluded.evidence_type,
            raw_json=excluded.raw_json,
            extracted_json=excluded.extracted_json,
            data_quality=excluded.data_quality,
            scraped_at=excluded.scraped_at
    ''', (
        evidence_id,
        evidence.get('match_id'),
        evidence.get('league'),
        evidence.get('match_date'),
        evidence.get('home_team'),
        evidence.get('away_team'),
        evidence.get('source'),
        evidence.get('source_url'),
        evidence.get('evidence_type'),
        json.dumps(evidence.get('raw', {})),
        json.dumps(evidence.get('extracted', {})),
        evidence.get('data_quality'),
        scraped_at,
    ))
    conn.commit()
    conn.close()

def upsert_match_learning_patterns(match_id, patterns, source='post_match_evidence'):
    """Persist derived post-match pattern tags for model learning."""
    if not match_id or not patterns:
        return 0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    written = 0
    for pattern in patterns:
        key = pattern.get('pattern_key') or pattern.get('key')
        if not key:
            continue
        cursor.execute('''
            INSERT INTO match_learning_patterns (
                match_id, pattern_key, confidence, source, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id, pattern_key) DO UPDATE SET
                confidence=excluded.confidence,
                source=excluded.source,
                details_json=excluded.details_json,
                created_at=excluded.created_at
        ''', (
            match_id,
            key,
            pattern.get('confidence', 0.5),
            source,
            json.dumps(pattern.get('details', {})),
            now,
        ))
        written += 1
    conn.commit()
    conn.close()
    return written

def get_training_examples(league=None):
    """Return normalized completed examples, ordered chronologically."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if league:
        cursor.execute('''
            SELECT * FROM training_examples
            WHERE league = ? AND actual_ftr IN ('H', 'D', 'A')
            ORDER BY match_date ASC, id ASC
        ''', (league,))
    else:
        cursor.execute('''
            SELECT * FROM training_examples
            WHERE actual_ftr IN ('H', 'D', 'A')
            ORDER BY league ASC, match_date ASC, id ASC
        ''')
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def upsert_match_sqlite(m, data_source='mock', run_id=None, graph_snapshot_id=None):
    """
    Inserts or updates a match in the SQLite hub.
    
    data_source: 'fdo' | 'firecrawl' | 'mock'
      - 'fdo'       = confirmed real fixture from Football-Data.org API
      - 'firecrawl' = discovered via Firecrawl web scrape (real but unverified)
      - 'mock'      = synthetic/fallback data (NEVER shown on frontend)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    league = m.get('League')
    home_team = resolve_team_name(league, m.get('Home'))
    away_team = resolve_team_name(league, m.get('Away'))
    m['Home'] = home_team
    m['Away'] = away_team
    
    # Create a unique ID: DATE_HOME_AWAY
    m_id = f"{m.get('Date')}_{home_team}_{away_team}".replace("/", "-")
    
    intel_feats = m.get('intel_feats', {})
    source_confidence = m.get('source_confidence')
    if source_confidence is None:
        source_confidence = calculate_source_confidence(data_source, intel_feats)
    intel_json = json.dumps(intel_feats)
    trust_json = json.dumps(m.get('trust_breakdown', {}))
    
    cursor.execute('''
        INSERT INTO matches (
            id, league, match_date, match_time, home_team, away_team,
            prediction, eqi_score, stake_pct, primary_market, secondary_market,
            p_btts, p_o25, over15_prob, under15_prob, over35_prob, under35_prob,
            p_1x, p_x2, p_dnb, value_edge, cld_delta,
            corners_exp, cards_exp, player_props, momentum_h, momentum_a,
            trust_breakdown, centrality_score, v8_motif, intel_raw, last_updated,
            data_source, source_confidence, is_mock, run_id, graph_snapshot_id, rationale_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            prediction=excluded.prediction,
            eqi_score=excluded.eqi_score,
            stake_pct=excluded.stake_pct,
            primary_market=excluded.primary_market,
            secondary_market=excluded.secondary_market,
            p_btts=excluded.p_btts,
            p_o25=excluded.p_o25,
            over15_prob=excluded.over15_prob,
            under15_prob=excluded.under15_prob,
            over35_prob=excluded.over35_prob,
            under35_prob=excluded.under35_prob,
            p_1x=excluded.p_1x,
            p_x2=excluded.p_x2,
            p_dnb=excluded.p_dnb,
            value_edge=excluded.value_edge,
            cld_delta=excluded.cld_delta,
            corners_exp=excluded.corners_exp,
            cards_exp=excluded.cards_exp,
            player_props=excluded.player_props,
            momentum_h=excluded.momentum_h,
            momentum_a=excluded.momentum_a,
            trust_breakdown=excluded.trust_breakdown,
            centrality_score=excluded.centrality_score,
            v8_motif=excluded.v8_motif,
            intel_raw=excluded.intel_raw,
            data_source=excluded.data_source,
            source_confidence=excluded.source_confidence,
            is_mock=excluded.is_mock,
            run_id=excluded.run_id,
            graph_snapshot_id=excluded.graph_snapshot_id,
            rationale_json=excluded.rationale_json,
            last_updated=excluded.last_updated
    ''', (
        m_id, m.get('League'), m.get('Date'), m.get('Time'), m.get('Home'), m.get('Away'),
        m.get('prediction'), m.get('execution_trust'), m.get('stake_pct'),
        m.get('primary_market'), m.get('secondary_market'),
        m.get('P(BTTS)'), m.get('P(O2.5)'),
        m.get('over15_prob'), m.get('under15_prob'), m.get('over35_prob'), m.get('under35_prob'),
        m.get('P(1X)'), m.get('P(X2)'), m.get('DNB'),
        m.get('Value Edge'), m.get('CLD'), m.get('E(Corners)'), m.get('E(Cards)'),
        m.get('Player Prop'), m.get('Momentum H'), m.get('Momentum A'),
        trust_json, m.get('centrality_score', 0.5), m.get('v8_motif', False),
        intel_json, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        data_source, source_confidence, 1 if data_source == 'mock' else 0, run_id, graph_snapshot_id,
        json.dumps(m.get('rationale', {}))
    ))

    cursor.execute('''
        UPDATE matches
        SET source_url = COALESCE(?, source_url),
            flashscore_id = COALESCE(?, flashscore_id),
            referee = COALESCE(?, referee),
            venue = COALESCE(?, venue),
            market_model_json = COALESCE(?, market_model_json)
        WHERE id = ?
    ''', (
        m.get('source_url') or None,
        m.get('flashscore_id') or None,
        m.get('referee') or None,
        m.get('venue') or None,
        json.dumps(m.get('market_model')) if m.get('market_model') else None,
        m_id,
    ))
    
    conn.commit()
    conn.close()

def get_league_json_from_sqlite(league_name):
    """Exports SQLite data to JSON for the UI. NEVER returns mock/synthetic fixtures."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    include_mock = os.environ.get('ALLOW_MOCK_FALLBACK', '').strip().lower() in {'1', 'true', 'yes', 'on'}
    
    # By default the UI only shows real fixtures. When mock fallback is explicitly
    # enabled for development/demo runs, allow synthetic fixtures through.
    cursor.execute(f'''
        SELECT id, league, match_date, match_time, home_team, away_team, prediction, eqi_score, stake_pct,
               primary_market, secondary_market, p_btts, p_o25, over15_prob, under15_prob, over35_prob, under35_prob,
               p_1x, p_x2, p_dnb, value_edge, cld_delta,
               corners_exp, cards_exp, player_props, momentum_h, momentum_a, trust_breakdown, centrality_score,
               v8_motif, intel_raw, data_source, source_confidence, run_id, graph_snapshot_id,
               source_url, flashscore_id, referee, venue, HY, AY, HR, AR, HC, AC,
               HS, "AS" AS "AS", HST, AST, HF, AF, HO, AO, HPoss, APoss, HXG, AXG, HBC, ABC,
               stats_scraped_at,
               market_model_json,
               (SELECT actual_fthg FROM match_results mr WHERE mr.id = matches.id) AS actual_fthg,
               (SELECT actual_ftag FROM match_results mr WHERE mr.id = matches.id) AS actual_ftag,
               (SELECT actual_ftr FROM match_results mr WHERE mr.id = matches.id) AS actual_ftr,
               (SELECT result_written_at FROM match_results mr WHERE mr.id = matches.id) AS result_written_at
        FROM matches
        WHERE league = ? AND p_1x IS NOT NULL {"AND is_mock = 0" if not include_mock else ""}
        ORDER BY match_date DESC, match_time DESC
    ''', (league_name,))
    rows = cursor.fetchall()
    conn.close()
    
    # Map SQL rows back to the JSON keys the UI uses
    result = []
    for r in rows:
        match_dict = dict(r)
        # Rename keys for UI compatibility
        match_dict['Date'] = r['match_date']
        match_dict['Time'] = r['match_time']
        match_dict['Home'] = r['home_team']
        match_dict['Away'] = r['away_team']
        match_dict['League'] = r['league']
        trust_score = trust_score_from_eqi(r['eqi_score'])
        match_dict['eqi_raw'] = r['eqi_score']
        match_dict['trust_score'] = trust_score
        match_dict['Trust Score'] = trust_score
        match_dict['execution_trust'] = trust_score
        match_dict['Value Edge'] = r['value_edge']
        match_dict['CLD'] = r['cld_delta']
        match_dict['P(BTTS)'] = r['p_btts']
        match_dict['P(O2.5)'] = r['p_o25']
        match_dict['over15_prob'] = r['over15_prob']
        match_dict['under15_prob'] = r['under15_prob']
        match_dict['over35_prob'] = r['over35_prob']
        match_dict['under35_prob'] = r['under35_prob']
        match_dict['P(1X)'] = r['p_1x']
        match_dict['P(X2)'] = r['p_x2']
        match_dict['DNB'] = r['p_dnb']
        match_dict['E(Corners)'] = r['corners_exp']
        match_dict['E(Cards)'] = r['cards_exp']
        match_dict['Player Prop'] = r['player_props']
        match_dict['Momentum H'] = r['momentum_h']
        match_dict['Momentum A'] = r['momentum_a']
        match_dict['centrality'] = r['centrality_score']
        match_dict['v8_motif'] = bool(match_dict.get('v8_motif', 0))
        match_dict['source_confidence'] = r['source_confidence']
        match_dict['Source Confidence'] = r['source_confidence']
        match_dict['source_url'] = r['source_url']
        match_dict['flashscore_id'] = r['flashscore_id']
        match_dict['referee'] = r['referee']
        match_dict['venue'] = r['venue']
        match_dict['HY'] = r['HY']
        match_dict['AY'] = r['AY']
        match_dict['HR'] = r['HR']
        match_dict['AR'] = r['AR']
        match_dict['HC'] = r['HC']
        match_dict['AC'] = r['AC']
        for col in WHY_STAT_COLUMNS:
            match_dict[col] = r[col]
        match_dict['stats_scraped_at'] = r['stats_scraped_at']
        try:
            match_dict['market_model'] = json.loads(r['market_model_json']) if r['market_model_json'] else {}
        except Exception:
            match_dict['market_model'] = {}
        try:
            intel_raw = json.loads(r['intel_raw']) if r['intel_raw'] else {}
        except Exception:
            intel_raw = {}
        situation = intel_raw.get('league_situation') if isinstance(intel_raw, dict) else {}
        if not isinstance(situation, dict):
            situation = {}
        match_dict['league_situation'] = situation
        match_dict['motivation_score'] = situation.get('match_pressure_score', 0)
        match_dict['motivation_label'] = situation.get('label', 'Normal table context')
        match_dict['league_situation_adjustment'] = (
            intel_raw.get('league_situation_adjustment', {}) if isinstance(intel_raw, dict) else {}
        )
        market_router = intel_raw.get('market_router', {}) if isinstance(intel_raw, dict) else {}
        if not isinstance(market_router, dict):
            market_router = {}
        match_dict['market_router'] = market_router
        match_dict['selected_market_probability'] = market_router.get('selected_probability')
        match_dict['market_router_action'] = market_router.get('action')
        match_dict['actual_fthg'] = r['actual_fthg']
        match_dict['actual_ftag'] = r['actual_ftag']
        match_dict['actual_ftr'] = r['actual_ftr']
        match_dict['result_written_at'] = r['result_written_at']
        match_dict['trust_breakdown'] = json.loads(r['trust_breakdown']) if r['trust_breakdown'] else {"model":0, "market":0, "stability":0, "context":0}
        match_dict['has_intel'] = True if intel_raw else False
        result.append(match_dict)
    
    return result

def write_match_result(m_id, actual_fthg, actual_ftag, actual_ftr,
                        closing_home=None, closing_draw=None, closing_away=None):
    """
    Write actual match result to match_results table.
    Also computes closing probabilities from odds.
    Call this after a match completes to close the feedback loop.
    """
    closing_prob_home = 1/closing_home if closing_home else None
    closing_prob_draw = 1/closing_draw if closing_draw else None
    closing_prob_away = 1/closing_away if closing_away else None

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO match_results (
            id, actual_fthg, actual_ftag, actual_ftr,
            closing_home_odds, closing_draw_odds, closing_away_odds,
            closing_prob_home, closing_prob_draw, closing_prob_away,
            result_written_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            actual_fthg=excluded.actual_fthg,
            actual_ftag=excluded.actual_ftag,
            actual_ftr=excluded.actual_ftr,
            closing_home_odds=excluded.closing_home_odds,
            closing_draw_odds=excluded.closing_draw_odds,
            closing_away_odds=excluded.closing_away_odds,
            closing_prob_home=excluded.closing_prob_home,
            closing_prob_draw=excluded.closing_prob_draw,
            closing_prob_away=excluded.closing_prob_away,
            result_written_at=excluded.result_written_at
    ''', (
        m_id, actual_fthg, actual_ftag, actual_ftr,
        closing_home, closing_draw, closing_away,
        closing_prob_home, closing_prob_draw, closing_prob_away,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    conn.close()


def write_clv_record(prediction_id, match_date, league, home_team, away_team,
                      predicted_prob, closing_prob, open_prob, cld,
                      actual_ftr, hit, edge, stake_pct, kelly_fraction, pnl):
    """
    Write a closing line value tracking record.
    Called after result writeback to record P&L from CLV perspective.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO closing_line_tracking (
            id, match_date, league, home_team, away_team,
            predicted_prob, closing_prob, open_prob, cld,
            actual_ftr, hit, edge, stake_pct, kelly_fraction, pnl,
            recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            predicted_prob=excluded.predicted_prob,
            closing_prob=excluded.closing_prob,
            open_prob=excluded.open_prob,
            cld=excluded.cld,
            actual_ftr=excluded.actual_ftr,
            hit=excluded.hit,
            edge=excluded.edge,
            stake_pct=excluded.stake_pct,
            kelly_fraction=excluded.kelly_fraction,
            pnl=excluded.pnl,
            recorded_at=excluded.recorded_at
    ''', (
        prediction_id, match_date, league, home_team, away_team,
        predicted_prob, closing_prob, open_prob, cld,
        actual_ftr, hit, edge, stake_pct, kelly_fraction, pnl,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    conn.close()

def _move_auxiliary_match_ids(cur, old_id, new_id, home_team=None, away_team=None):
    try:
        cur.execute('UPDATE match_results SET id = ? WHERE id = ?', (new_id, old_id))
    except sqlite3.IntegrityError:
        cur.execute('DELETE FROM match_results WHERE id = ?', (old_id,))
    try:
        cur.execute(
            'UPDATE closing_line_tracking SET id = ?, home_team = COALESCE(?, home_team), away_team = COALESCE(?, away_team) WHERE id = ?',
            (new_id, home_team, away_team, old_id),
        )
    except sqlite3.IntegrityError:
        cur.execute('DELETE FROM closing_line_tracking WHERE id = ?', (old_id,))


def _merge_duplicate_match_row(cur, keep_id, drop_id, home_team, away_team):
    row = cur.execute(
        '''
        SELECT source_url, flashscore_id, referee, venue,
               HY, AY, HR, AR, HC, AC, stats_scraped_at, market_model_json,
               data_source, source_confidence, run_id, graph_snapshot_id
        FROM matches
        WHERE id = ?
        ''',
        (drop_id,),
    ).fetchone()
    if not row:
        return

    (
        source_url,
        flashscore_id,
        referee,
        venue,
        hy,
        ay,
        hr,
        ar,
        hc,
        ac,
        stats_scraped_at,
        market_model_json,
        data_source,
        source_confidence,
        run_id,
        graph_snapshot_id,
    ) = row
    target_source = cur.execute('SELECT data_source FROM matches WHERE id = ?', (keep_id,)).fetchone()
    target_source = target_source[0] if target_source else None
    merged_source = 'browser' if data_source == 'browser' or target_source == 'browser' else (target_source or data_source)

    cur.execute(
        '''
        UPDATE matches
        SET home_team = ?,
            away_team = ?,
            source_url = COALESCE(source_url, ?),
            flashscore_id = COALESCE(flashscore_id, ?),
            referee = COALESCE(referee, ?),
            venue = COALESCE(venue, ?),
            HY = COALESCE(HY, ?),
            AY = COALESCE(AY, ?),
            HR = COALESCE(HR, ?),
            AR = COALESCE(AR, ?),
            HC = COALESCE(HC, ?),
            AC = COALESCE(AC, ?),
            stats_scraped_at = COALESCE(stats_scraped_at, ?),
            market_model_json = COALESCE(market_model_json, ?),
            data_source = COALESCE(?, data_source),
            source_confidence = MAX(COALESCE(source_confidence, 0), COALESCE(?, 0)),
            run_id = COALESCE(run_id, ?),
            graph_snapshot_id = COALESCE(graph_snapshot_id, ?)
        WHERE id = ?
        ''',
        (
            home_team,
            away_team,
            source_url,
            flashscore_id,
            referee,
            venue,
            hy,
            ay,
            hr,
            ar,
            hc,
            ac,
            stats_scraped_at,
            market_model_json,
            merged_source,
            source_confidence,
            run_id,
            graph_snapshot_id,
            keep_id,
        ),
    )
    _move_auxiliary_match_ids(cur, drop_id, keep_id, home_team, away_team)
    cur.execute('DELETE FROM matches WHERE id = ?', (drop_id,))


def auto_dedupe_aliases():
    """
    Automatically scrubs the DB using the team_aliases table.
    Called by the orchestrator at the start of a sweep to ensure any new aliases
    are retroactively resolved across all historical data and shadow logs.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Reload cache in case it changed
    global _ALIAS_CACHE
    _ALIAS_CACHE = None
    resolve_team_name('dummy', 'dummy')  # forces cache load
    
    cur.execute('SELECT league, alias_name, canonical_name FROM team_aliases')
    aliases = cur.fetchall()
    
    for league, alias, canon in aliases:
        # Dedupe matches
        cur.execute('SELECT id FROM matches WHERE league = ? AND id LIKE ?', (league, '%' + alias + '%'))
        for (m_id,) in cur.fetchall():
            parts = m_id.split('_')
            if len(parts) >= 3:
                date, h, a = parts[0], parts[1], parts[2]
                if h == alias: h = canon
                if a == alias: a = canon
                new_id = f'{date}_{h}_{a}'
                if new_id != m_id:
                    if cur.execute('SELECT 1 FROM matches WHERE id = ?', (new_id,)).fetchone():
                        _merge_duplicate_match_row(cur, new_id, m_id, h, a)
                    else:
                        _move_auxiliary_match_ids(cur, m_id, new_id, h, a)
                        cur.execute('UPDATE matches SET id = ?, home_team = ?, away_team = ? WHERE id = ?', (new_id, h, a, m_id))

        # Dedupe training_examples
        cur.execute('SELECT id FROM training_examples WHERE league = ? AND id LIKE ?', (league, '%' + alias + '%'))
        for (m_id,) in cur.fetchall():
            new_id = m_id.replace('_' + alias + '_', '_' + canon + '_')
            if new_id.endswith('_' + alias):
                new_id = new_id[:-len('_' + alias)] + '_' + canon
            if new_id != m_id:
                try: cur.execute('UPDATE training_examples SET id = ? WHERE id = ?', (new_id, m_id))
                except sqlite3.IntegrityError: cur.execute('DELETE FROM training_examples WHERE id = ?', (m_id,))

        cur.execute('UPDATE training_examples SET home_team = ? WHERE league = ? AND home_team = ?', (canon, league, alias))
        cur.execute('UPDATE training_examples SET away_team = ? WHERE league = ? AND away_team = ?', (canon, league, alias))
        cur.execute('UPDATE matches SET home_team = ? WHERE league = ? AND home_team = ?', (canon, league, alias))
        cur.execute('UPDATE matches SET away_team = ? WHERE league = ? AND away_team = ?', (canon, league, alias))

    conn.commit()
    conn.close()
    
    # --- Backend Data Hygiene ---
    # 1. Purge stale mock fixtures from the SQLite DB (older than 3 days) to prevent bloat
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM matches WHERE is_mock = 1 AND match_date < date('now', '-3 days')")
        conn.commit()
    except Exception:
        pass
    conn.close()
    
    # 2. Scrub shadow_log.json aliases and time out stale PENDING entries
    shadow_path = os.path.join(PROJECT_ROOT, 'web', 'data', 'shadow_log.json')
    if os.path.exists(shadow_path):
        import hashlib
        from datetime import datetime, timedelta
        try:
            with open(shadow_path, 'r') as f:
                entries = json.load(f)
            updated = False
            
            fourteen_days_ago = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
            
            for e in entries:
                # Resolve aliases
                if e.get('league') and e.get('home') and e.get('away'):
                    l = e['league']
                    h = e['home']
                    a = e['away']
                    canon_h = resolve_team_name(l, h)
                    canon_a = resolve_team_name(l, a)
                    if canon_h != h or canon_a != a:
                        e['home'] = canon_h
                        e['away'] = canon_a
                        e['match'] = f"{canon_h} v {canon_a}"
                        d = e.get('date', '')
                        p = e.get('prediction', '')
                        e['id'] = hashlib.md5(f'{d}_{canon_h}_{canon_a}_{p}'.encode()).hexdigest()[:8]
                        updated = True
                
                # Auto-timeout stale PENDING matches
                if e.get('result') == 'PENDING':
                    date_val = e.get('date', '')
                    if date_val:
                        # Extract YYYY-MM-DD
                        date_str = date_val.split('T')[0]
                        if date_str < fourteen_days_ago:
                            e['result'] = 'VOID'
                            e['resolved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            updated = True
                            
            if updated:
                with open(shadow_path, 'w') as f:
                    json.dump(entries, f, indent=4)
        except Exception:
            pass

def get_clv_summary(league=None, lookback_days=30):
    """
    Compute CLV summary statistics from closing_line_tracking.
    Returns win rate, P&L, CLV hit rate per market.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if league:
        cursor.execute('''
            SELECT * FROM closing_line_tracking
            WHERE league = ? AND recorded_at >= date('now', '-' || ? || ' days')
            ORDER BY recorded_at DESC
        ''', (league, lookback_days))
    else:
        cursor.execute('''
            SELECT * FROM closing_line_tracking
            WHERE recorded_at >= date('now', '-' || ? || ' days')
            ORDER BY recorded_at DESC
        ''', (lookback_days,))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {'n_bets': 0, 'total_pnl': 0, 'hit_rate': 0, 'clv_hit_rate': 0}

    n = len(rows)
    total_pnl = sum(r['pnl'] or 0 for r in rows)
    hits = sum(r['hit'] or 0 for r in rows)
    clv_hits = sum(1 for r in rows if r['cld'] and r['cld'] > 0 and r['hit'])

    # Per-market breakdown
    market_breakdown = {}
    for r in rows:
        mkt = r.get('market', 'unknown')
        if mkt not in market_breakdown:
            market_breakdown[mkt] = {'n': 0, 'hits': 0, 'pnl': 0}
        market_breakdown[mkt]['n'] += 1
        market_breakdown[mkt]['hits'] += (r['hit'] or 0)
        market_breakdown[mkt]['pnl'] += (r['pnl'] or 0)

    return {
        'n_bets': n,
        'total_pnl': round(total_pnl, 2),
        'hit_rate': round(hits / n, 3),
        'clv_hit_rate': round(clv_hits / max(1, sum(1 for r in rows if r['cld'] and r['cld'] > 0)), 3),
        'avg_stake': round(sum(r['stake_pct'] or 0 for r in rows) / n, 2),
        'market_breakdown': {
            k: {
                'n': v['n'],
                'hit_rate': round(v['hits'] / v['n'], 3) if v['n'] > 0 else 0,
                'pnl': round(v['pnl'], 2)
            }
            for k, v in market_breakdown.items()
        }
    }


def get_all_pending_predictions():
    """
    Returns all predictions from shadow_log.json that are still PENDING.
    Used by result_collector to close the loop.
    """
    shadow_path = os.path.join(os.path.dirname(__file__), '..', 'web', 'data', 'shadow_log.json')
    if not os.path.exists(shadow_path):
        return []

    try:
        with open(shadow_path, 'r') as f:
            entries = json.load(f)
    except Exception:
        return []

    return [e for e in entries if e.get('result') in ('PENDING', None, '')]

_SEQ_ALIAS_CACHE: dict = {}

def _get_sequence_aliases(cursor, team_name: str) -> set:
    """Return all name variants for a team from team_aliases (both directions)."""
    if team_name in _SEQ_ALIAS_CACHE:
        return _SEQ_ALIAS_CACHE[team_name]
    aliases = {team_name}
    try:
        # alias_name → canonical_name direction
        rows = cursor.execute(
            "SELECT canonical_name FROM team_aliases WHERE alias_name = ?", (team_name,)
        ).fetchall()
        for r in rows:
            aliases.add(r[0])
        # canonical_name → all alias_names direction
        rows = cursor.execute(
            "SELECT alias_name FROM team_aliases WHERE canonical_name = ?", (team_name,)
        ).fetchall()
        for r in rows:
            aliases.add(r[0])
        # Also expand through any canonical we found (one extra hop covers FDO↔Flashscore gaps)
        for canonical in list(aliases - {team_name}):
            rows = cursor.execute(
                "SELECT alias_name FROM team_aliases WHERE canonical_name = ?", (canonical,)
            ).fetchall()
            for r in rows:
                aliases.add(r[0])
    except Exception:
        pass
    _SEQ_ALIAS_CACHE[team_name] = aliases
    return aliases


def get_team_match_history(team_name, limit=15):
    """
    Retrieves the last N completed matches for a team from SQLite.
    Used for temporal sequence modeling in V11.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Prefer completed predictions joined to actual results. Pending fixtures are
    # deliberately excluded because they inject default goals/results into V11.
    cursor.execute('''
        SELECT
            m.id, m.match_date, m.match_time, m.home_team, m.away_team,
            r.actual_fthg, r.actual_ftag, r.actual_ftr,
            m.p_1x, m.p_x2, m.p_dnb, m.intel_raw
        FROM matches m
        JOIN match_results r ON r.id = m.id
        WHERE (m.home_team = ? OR m.away_team = ?)
        ORDER BY m.match_date DESC, m.match_time DESC
        LIMIT ?
    ''', (team_name, team_name, limit))

    rows = cursor.fetchall()

    # Fallback to sequence_history for historical data.
    # Query using all known alias variants so cross-source name differences are handled.
    if len(rows) < limit:
        aliases = _get_sequence_aliases(cursor, team_name)
        placeholders = ','.join('?' * len(aliases))
        cursor.execute(f'''
            SELECT team_name, match_date, result, score, xg, xga
            FROM sequence_history
            WHERE team_name IN ({placeholders})
            ORDER BY match_date DESC
            LIMIT ?
        ''', (*sorted(aliases), limit - len(rows)))
        seq_rows = cursor.fetchall()
    else:
        seq_rows = []

    conn.close()

    history = []
    for r in rows:
        history.append(dict(r))

    for r in seq_rows:
        item = dict(r)
        score = item.get('score') or '0-0'
        try:
            gf, ga = [int(x.strip()) for x in score.split('-', 1)]
        except Exception:
            gf, ga = 0, 0
        result = item.get('result')
        actual_ftr = {'W': 'H', 'D': 'D', 'L': 'A'}.get(result, result)
        history.append({
            'id': f"sequence_{team_name}_{item.get('match_date')}",
            'match_date': item.get('match_date'),
            'match_time': '',
            'home_team': team_name,
            'away_team': '',
            'actual_fthg': gf,
            'actual_ftag': ga,
            'actual_ftr': actual_ftr,
            'xg': item.get('xg'),
            'xga': item.get('xga'),
            'intel_raw': '{}',
        })

    return history

# Initialize on import
init_db()
