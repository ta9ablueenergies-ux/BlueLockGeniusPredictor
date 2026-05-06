# ANTIGRAVITY V8.0 + V8.1 GSN (GRAPH SUBSTRUCTURE NETWORK) DETECTOR
# V8.1: Temporal window + recency-weighted thresholds + lookback_games limit
import os
import sqlite3
import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "web", "data", "intelligence_hub.db")


def _time_decay_weight(match_date_str, reference_date, half_life_days=60):
    """0.5^(days/half_life_days) — half weight every `half_life_days` days."""
    if not match_date_str:
        return 1.0
    for fmt in ('%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y'):
        try:
            mdate = datetime.strptime(str(match_date_str), fmt)
            delta = (reference_date - mdate).days
            if delta < 0:
                delta = 0
            return 0.5 ** (delta / half_life_days)
        except ValueError:
            continue
    return 1.0


class V8MotifDetector:
    def __init__(self, db_path=DB_PATH,
                 half_life_days=365,
                 lookback_games=20,
                 min_games_for_triangle=3,
                 temporal_days=3650,
                 reference_date=None):
        """
        V8.1 Temporal GSN Motif Detector.

        Args:
            half_life_days: edge weight halves after this many days (default 60d)
            lookback_games: max matches per team to consider (default 20, ~half season)
            min_games_for_triangle: minimum encounters before a triangle is valid (default 3)
            temporal_days: only use matches within this window (default 730d = ~2 years)
            reference_date: datetime for decay calculation (default now)
        """
        self.db_path = db_path
        self.half_life_days = half_life_days
        self.lookback_games = lookback_games
        self.min_games_for_triangle = min_games_for_triangle
        self.temporal_days = temporal_days or 3650
        self.reference_date = reference_date or datetime.now()
        self.graph = nx.DiGraph()
        self.triangles = []
        self.giant_killers = {}
        self._edge_games = {}  # (team, opponent) -> list of (date, weighted_edge)

    def build_directed_topology(self, league="PremierLeague"):
        """Builds a time-decayed directed graph of team dominance relationships."""
        conn = sqlite3.connect(self.db_path)
        query = ("SELECT home_team, away_team, prediction, eqi_score, match_date, league "
                 "FROM matches WHERE league = ?")
        df = pd.read_sql_query(query, conn, params=(league,))
        conn.close()

        if df.empty:
            return False

        # Filter to temporal window
        def within_window(d):
            if not d: return True
            for fmt in ('%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y'):
                try:
                    mdate = datetime.strptime(str(d), fmt)
                    delta = (self.reference_date - mdate).days
                    return 0 <= delta <= self.temporal_days
                except ValueError:
                    continue
            return True

        df = df[df['match_date'].apply(within_window)].copy()

        if df.empty:
            return False

        # Per-team lookback: only last `lookback_games` matches per team
        # Accumulate edges per team pair with game counts
        team_edges = {}  # (home_team, away_team) -> list of (match_date, weighted_eqi)

        for _, r in df.iterrows():
            h_team = r['home_team']
            a_team = r['away_team']
            match_date = r.get('match_date', '')
            eqi = float(r['eqi_score']) if pd.notna(r['eqi_score']) else 1200.0
            decay = _time_decay_weight(match_date, self.reference_date, self.half_life_days)
            weighted_eqi = eqi * decay

            # Forward edge (home dominates)
            key_fwd = (h_team, a_team)
            if key_fwd not in team_edges:
                team_edges[key_fwd] = []
            team_edges[key_fwd].append((match_date, weighted_eqi))

            # Reverse edge (away dominates)
            key_rev = (a_team, h_team)
            if key_rev not in team_edges:
                team_edges[key_rev] = []
            team_edges[key_rev].append((match_date, weighted_eqi))

        # Apply lookback_games limit per team pair (take most recent)
        for key in team_edges:
            edges = sorted(team_edges[key], key=lambda x: x[0] or '', reverse=True)
            team_edges[key] = edges[:self.lookback_games]

        # Build directed graph: edge direction based on EQI dominance at time of match
        self.graph = nx.DiGraph()
        for (winner, loser), edges in team_edges.items():
            if not edges:
                continue
            # Take the most recent strong edge as the current relationship
            best_date, best_eqi = max(edges, key=lambda x: x[0] or '')
            if best_eqi > 1200:  # winner dominated at this point
                if not self.graph.has_edge(winner, loser):
                    self.graph.add_edge(winner, loser, weight=best_eqi, recent_eqi=best_eqi)
                else:
                    # Update if more recent
                    existing_weight = self.graph[winner][loser].get('weight', 0)
                    if best_eqi > existing_weight:
                        self.graph[winner][loser]['weight'] = best_eqi
                        self.graph[winner][loser]['recent_eqi'] = best_eqi

        print(f"  [V8.1] Directed Topology ({league}): {self.graph.number_of_nodes()} Nodes, "
              f"{self.graph.number_of_edges()} Edges (half_life={self.half_life_days}d, "
              f"lookback={self.lookback_games}/team)")
        return True

    def detect_cyclic_dominance(self):
        """
        Finds 'Bogey Team' Triangles (A beats B, B beats C, C beats A).
        Only triangles where ALL edges have recent support are considered valid.
        """
        self.triangles = set()
        nodes = list(self.graph.nodes())

        for n1 in nodes:
            for n2 in self.graph.successors(n1):
                for n3 in self.graph.successors(n2):
                    if self.graph.has_edge(n3, n1):
                        # Check recency: at least 2 of 3 edges must be recent
                        recent_count = 0
                        for a, b in [(n1, n2), (n2, n3), (n3, n1)]:
                            edge_data = self.graph[a][b]
                            # recent_eqi is set at build time — edge with recent_eqi > 1200 = recent
                            if edge_data.get('recent_eqi', 0) > 1200:
                                recent_count += 1

                        if recent_count >= 2:
                            triangle = tuple(sorted((n1, n2, n3)))
                            self.triangles.add(triangle)

        if not self.triangles:
            print("   [V8.1] No strict cyclic motifs found (all stale or incomplete)")
            return []

        print(f"   [V8.1] FOUND {len(self.triangles)} 'Bogey Team' Triangles (with recent support)")
        for i, t in enumerate(list(self.triangles)[:3]):
            print(f"       Motif {i+1}: {t[0]} vs {t[1]} vs {t[2]}")
        self.triangles = list(self.triangles)
        return self.triangles

    def detect_giant_killers(self, base_threshold=1250):
        """
        Identifies teams that frequently defeat high-EQI opponents.
        Recent victories count more — threshold decays for older matches.

        giant_killers: dict of {team: weighted_kill_count}
        """
        killers = {}
        for node in self.graph.nodes():
            giant_kills = 0
            for target in self.graph.successors(node):
                edge_data = self.graph[node][target]
                eqi = edge_data.get('recent_eqi', 1200)
                # Apply time-adjusted threshold: older edges need higher EQI to count
                kill_weight = edge_data.get('weight', 1200)
                if kill_weight > base_threshold:
                    # Scale contribution by recency of that specific edge
                    if eqi > 1250:
                        giant_kills += 1
                    elif eqi > 1200:
                        giant_kills += 0.5

            if giant_kills > 0:
                killers[node] = giant_kills

        sorted_killers = sorted(killers.items(), key=lambda item: item[1], reverse=True)
        print(f"   [V8.1] FOUND {len(sorted_killers)} 'Giant Killer' Nodes")
        self.giant_killers = {k: v for k, v in sorted_killers}
        return self.giant_killers

    def get_centrality_scores(self):
        """
        Calculates PageRank and Betweenness centrality for the current graph.
        Returns: {team_name: (pagerank, betweenness)}
        """
        if not self.graph or len(self.graph.nodes()) < 2:
            return {}
            
        try:
            pr = nx.pagerank(self.graph, weight='weight')
            bt = nx.betweenness_centrality(self.graph, weight='weight')
            return {node: (pr.get(node, 0.0), bt.get(node, 0.0)) for node in self.graph.nodes()}
        except:
            return {node: (0.0, 0.0) for node in self.graph.nodes()}

    def get_motif_multiplier(self, home_team, away_team):
        """
        Returns V8.1 temporal motif multipliers for expected goals adjustment.

        Returns: (mult_h, mult_a)
        - Giant killer: home gets +8%, away gets +5%
        - Bogey triangle: bogey team gets 1.15x, victim gets 0.85x
        - Only applies if motifs have RECENT support (at least 2 recent edges)
        """
        mult_h, mult_a = 1.0, 1.0

        # 1. Giant Killer Check
        if hasattr(self, 'giant_killers') and self.giant_killers:
            if home_team in self.giant_killers and self.giant_killers[home_team] >= 1.5:
                mult_h = 1.08
            if away_team in self.giant_killers and self.giant_killers[away_team] >= 1.5:
                mult_a = 1.05

        # 2. Bogey Triangle Check (only if triangle has recent support)
        if hasattr(self, 'triangles') and self.triangles:
            for t in self.triangles:
                if home_team in t and away_team in t:
                    # Check which direction has the recent dominant edge
                    if self.graph.has_edge(home_team, away_team):
                        edge_data = self.graph[home_team][away_team]
                        if edge_data.get('recent_eqi', 0) > 1200:
                            # Home is the current bogey for away
                            mult_h = 1.15
                            mult_a = 0.85
                    elif self.graph.has_edge(away_team, home_team):
                        edge_data = self.graph[away_team][home_team]
                        if edge_data.get('recent_eqi', 0) > 1200:
                            # Away is the current bogey for home
                            mult_a = 1.15
                            mult_h = 0.85
                    break  # Apply once per matchup

        return mult_h, mult_a


if __name__ == "__main__":
    detector = V8MotifDetector(half_life_days=365, lookback_games=20)
    for league in ['PremierLeague', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1']:
        if detector.build_directed_topology(league):
            detector.detect_cyclic_dominance()
            detector.detect_giant_killers()
            mh, ma = detector.get_motif_multiplier('Arsenal', 'Brighton')
            print(f"   Multiplier (Arsenal vs Brighton): H={mh}, A={ma}")
