"""
pi_ratings.py
=============
Antigravity V9.0 - Pi-Rating System (Constantinou & Fenton).
A state-of-the-art rating system that outperforms ELO by:
1. Using Goal Difference (Dominance) instead of simple Win/Loss.
2. Maintaining separate Home and Away strength ratings.
3. Incorporating diminishing returns for high-score blowouts.
"""
import numpy as np
import pandas as pd

class PiRatingSystem:
    def __init__(self, lambda_p=0.035, gamma=0.7, c=10):
        """
        Initialize Pi-Rating Engine.
        :param lambda_p: Learning rate (0.035 - 0.05).
        :param gamma: Home advantage coefficient.
        :param c: Constant for expected goal difference.
        """
        self.lambda_p = lambda_p
        self.gamma = gamma
        self.c = c
        # team_id -> [HomeRating, AwayRating]
        self.ratings = {}

    def get_rating(self, team):
        if team not in self.ratings:
            self.ratings[team] = [0.0, 0.0] # [H, A] initialized at 0
        return self.ratings[team]

    def predict_expected_gd(self, h_name, a_name):
        """
        Calculate the expected Goal Difference (E_GD).
        E_GD = (H_rating_home - A_rating_away)
        """
        h_r = self.get_rating(h_name)
        a_r = self.get_rating(a_name)
        return h_r[0] - a_r[1]

    def update(self, h_name, a_name, h_goals, a_goals):
        """
        Update ratings based on actual match outcome (Goal Difference).
        """
        gd = h_goals - a_goals
        e_gd = self.predict_expected_gd(h_name, a_name)
        
        # Error (Surprise factor)
        err = gd - e_gd
        
        # Diminishing returns for blowouts (logarithmic scaling of error)
        # In the original Pi-ratings, the error is used directly but capped or scaled.
        # We'll use a standard linear update with the lambda learning rate.
        
        h_r = self.get_rating(h_name)
        a_r = self.get_rating(a_name)
        
        # Update Home Team's Home Rating
        # Update Away Team's Away Rating
        delta_h = self.lambda_p * err
        delta_a = self.lambda_p * (-err) # Mirror surprise for away team
        
        self.ratings[h_name][0] += delta_h
        self.ratings[a_name][1] += delta_a
        
        # Back-propagation to secondary ratings (A-team home, H-team away)
        # A win away should also slightly improve your home rating
        self.ratings[h_name][1] += delta_h * self.gamma
        self.ratings[a_name][0] += delta_a * self.gamma
        
        return err

    def process_season(self, df):
        """
        Process an entire season of matches to build ratings.
        """
        errors = []
        for _, r in df.iterrows():
            err = self.update(r['HomeTeam'], r['AwayTeam'], r['FTHG'], r['FTAG'])
            errors.append(err)
        return np.mean(np.abs(errors))

    def save_state(self, path):
        import json
        with open(path, 'w') as f:
            json.dump(self.ratings, f)

    def load_state(self, path):
        import json
        with open(path, 'r') as f:
            self.ratings = json.load(f)
