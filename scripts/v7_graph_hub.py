# ANTIGRAVITY V7.0 + V7.1 SPECTRAL-NEURAL GRAPH HUB
# V7.1: Time-decay on edge weights (half_life_days=90)
import os
import numpy as np
import pandas as pd
import sqlite3
import json
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")


def _time_decay_weight(match_date_str, reference_date, half_life_days=90):
    """
    Compute time-decay multiplier for an edge.
    0.5^(days / half_life_days) — half weight every `half_life_days` days.
    """
    if not match_date_str:
        return 1.0

    # Try multiple date formats
    for fmt in ('%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y'):
        try:
            mdate = datetime.strptime(str(match_date_str), fmt)
            break
        except ValueError:
            continue

    delta = (reference_date - mdate).days
    if delta < 0:
        delta = 0  # future dates treated as "now"
    return 0.5 ** (delta / half_life_days)


class V7GraphHub:
    def __init__(self, db_path=DB_PATH,
                 half_life_days=90, temporal_days=None):
        """
        half_life_days: days after which an edge has half its original weight.
        temporal_days: if set, only consider matches within this many days of reference_date.
                       Default: 2 years (730 days)
        """
        self.db_path = db_path
        self.teams = []
        self.team_to_idx = {}
        self.adj_matrix = None
        self.centrality = None
        self.half_life_days = half_life_days
        self.temporal_days = temporal_days or 730  # ~2 years default lookback

    def build_relational_graph(self, league=None, reference_date=None):
        """
        Constructs the spectral graph from the SQLite Intelligence Hub.
        Applies time-decay to all edge weights.

        Args:
            league: filter by league (optional)
            reference_date: datetime used as "now" for decay calculation.
                           Defaults to datetime.now().
        """
        if reference_date is None:
            reference_date = datetime.now()

        conn = sqlite3.connect(self.db_path)
        query = "SELECT home_team, away_team, eqi_score, league, match_date FROM matches"
        params = []

        if league:
            query += " WHERE league = ?"
            params.append(league)

        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()

        if df.empty:
            return False

        # Filter to temporal window
        if 'match_date' in df.columns:
            def within_window(d):
                if not d: return True
                for fmt in ('%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y'):
                    try:
                        mdate = datetime.strptime(str(d), fmt)
                        delta = (reference_date - mdate).days
                        return 0 <= delta <= self.temporal_days
                    except ValueError:
                        continue
                return True  # unparseable dates included

            df = df[df['match_date'].apply(within_window)].copy()

        if df.empty:
            return False

        # 1. Identify Unique Nodes (Teams)
        self.teams = sorted(list(set(df['home_team'].tolist() + df['away_team'].tolist())))
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
        n = len(self.teams)

        # 2. Build Time-Decayed Weighted Adjacency Matrix
        # Weight = EQI_Score / 100.0 * time_decay
        row = []
        col = []
        data = []

        for _, r in df.iterrows():
            u = self.team_to_idx[r['home_team']]
            v = self.team_to_idx[r['away_team']]
            base_w = r['eqi_score'] / 100.0 if pd.notna(r['eqi_score']) else 1.0
            decay = _time_decay_weight(r.get('match_date'), reference_date, self.half_life_days)
            w = base_w * decay

            row.extend([u, v])
            col.extend([v, u])
            data.extend([w, w])

        self.adj_matrix = csr_matrix((data, (row, col)), shape=(n, n))

        # 3. Calculate Spectral Centrality (Leading Eigenvector)
        try:
            vals, vecs = eigsh(self.adj_matrix, k=1, which='LM')
            self.centrality = np.abs(vecs.flatten())
            max_c = np.max(self.centrality)
            self.centrality = self.centrality / max_c if max_c > 0 else self.centrality
        except Exception:
            self.centrality = np.ones(n)

        return True

    def get_relational_tilt(self, team_name):
        """
        Returns the V7.0 Relational Influence score for a team (0.0–1.0).
        """
        idx = self.team_to_idx.get(team_name)
        if idx is not None and self.centrality is not None:
            return float(self.centrality[idx])
        return 0.5

    def team_centralities(self):
        """Return dict of {team_name: centrality} for all teams."""
        return {t: self.get_relational_tilt(t) for t in self.teams}


class V7TemporalGraph(V7GraphHub):
    """
    V7 Temporal Graph — builds separate graphs for different time windows.
    Useful for comparing recent vs. historical team relationships.
    """
    def __init__(self, db_path=DB_PATH, half_life_days=90):
        super().__init__(db_path, half_life_days)
        self.recent_graph = None
        self.historical_graph = None

    def build_split_graphs(self, league=None, split_days=180, reference_date=None):
        """
        Build two separate graphs: recent (within split_days) and historical (older).
        Useful for detecting which teams have changed relationship patterns recently.
        """
        if reference_date is None:
            reference_date = datetime.now()

        recent_df = None
        historical_df = None

        conn = sqlite3.connect(self.db_path)
        query = "SELECT home_team, away_team, eqi_score, league, match_date FROM matches"
        params = []
        if league:
            query += " WHERE league = ?"
            params.append(league)

        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()

        if df.empty:
            return False

        def in_recent(d):
            if not d: return False
            for fmt in ('%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y'):
                try:
                    mdate = datetime.strptime(str(d), fmt)
                    delta = (reference_date - mdate).days
                    return 0 <= delta <= split_days
                except ValueError:
                    continue
            return False

        recent_df = df[df['match_date'].apply(in_recent)]
        historical_df = df[~df.index.isin(recent_df.index)]

        # Build recent graph
        self.recent_graph = V7GraphHub(self.db_path, self.half_life_days)
        self.recent_graph.build_relational_graph(league=league, reference_date=reference_date)

        # Build historical graph
        self.historical_graph = V7GraphHub(self.db_path, self.half_life_days)
        self.historical_graph.build_relational_graph(league=league, reference_date=reference_date)

        return True

    def get_relationship_change(self, team_a, team_b):
        """
        Returns how much a team's relationship with another has changed recently.
        Positive = stronger recent, Negative = historical was stronger.
        """
        if not self.recent_graph or not self.historical_graph:
            return 0.0

        recent_tilt_a = self.recent_graph.get_relational_tilt(team_a)
        recent_tilt_b = self.recent_graph.get_relational_tilt(team_b)
        hist_tilt_a = self.historical_graph.get_relational_tilt(team_a)
        hist_tilt_b = self.historical_graph.get_relational_tilt(team_b)

        recent_avg = (recent_tilt_a + recent_tilt_b) / 2
        hist_avg = (hist_tilt_a + hist_tilt_b) / 2

        return recent_avg - hist_avg


if __name__ == "__main__":
    hub = V7GraphHub()
    now = datetime.now()

    for league in ['PremierLeague', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1']:
        if hub.build_relational_graph(league=league, reference_date=now):
            top5 = sorted(hub.teams, key=lambda t: hub.get_relational_tilt(t), reverse=True)[:5]
            print(f"\nV7.1 {league} (half_life=90d): {len(hub.teams)} teams")
            for team in top5:
                print(f"  {team}: {hub.get_relational_tilt(team):.3f}")
        else:
            print(f"\nV7.1 {league}: No data")
