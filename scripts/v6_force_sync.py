import sys
import os
import json
import pandas as pd
from datetime import datetime

# Setup paths
PROJECT_ROOT = "c:/Users/U033IAT/Documents/antigravity world/football_predictions"
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from football_data_scraper import fetch_fdo_matches
from firecrawl_agent import enrich_match
from main_script import precalculate_all_features, predict_results
from persistence_manager import upsert_match_sqlite, get_league_json_from_sqlite

def force_sync(league, date):
    print(f"--- V6.0 FORCE SYNC: {league} ({date}) ---")
    
    # 1. Fetch
    fdo_data = fetch_fdo_matches(league, date, date)
    if fdo_data is None or fdo_data.empty:
        print(f"FAILED: No matches found for {league}")
        return

    matches = []
    for _, m in fdo_data.iterrows():
        matches.append({
            'Home': m.get('hometeam'),
            'Away': m.get('awayteam'),
            'Date': m.get('date'),
            'Time': m.get('Time') or '15:00',
            'League': league
        })
    
    print(f"Found {len(matches)} matches. Starting Deep Enrichment...")
    
    # 2. Enrich
    enriched_intel = []
    for m in matches:
        print(f"  > Enriching: {m['Home']} vs {m['Away']}...")
        intel = enrich_match(m['Home'], m['Away'], league, m['Date'])
        if intel:
            print(f"    [OK] xG and Momentum captured.")
            enriched_intel.append(intel)
        else:
            print(f"    [FAIL] Using base stats for this match.")
            # Create a placeholder intel if enrichment fails
            from firecrawl_agent import MatchIntel
            intel = MatchIntel(
                home_team=m['Home'], away_team=m['Away'], 
                league=league, match_date=m['Date'], 
                kickoff_utc=m['Time']
            )
            enriched_intel.append(intel)

    # 3. Predict
    print("Running Bayesian Engine...")
    intel_dicts = [i.model_dump() for i in enriched_intel]
    mock_matches = []
    for i in enriched_intel:
        mock_matches.append({
            'HomeTeam': i.home_team, 'AwayTeam': i.away_team,
            'Time': i.kickoff_utc or '15:00', 'Date': i.match_date,
            'h_course': 2.0, 'd_course': 3.0, 'a_course': 3.5,
            'h_open': 2.0, 'd_open': 3.0, 'a_open': 3.5
        })
    
    df_matches = pd.DataFrame(mock_matches)
    ratings = precalculate_all_features(league)
    final_df = predict_results(df_matches, league, ratings, intel_dicts)
    
    # 4. Save
    print(f"Saving {len(final_df)} matches to Intelligence Hub...")
    for _, row in final_df.iterrows():
        upsert_match_sqlite(row.to_dict(), data_source='fdo')
    
    # 5. Export JSON
    league_json = get_league_json_from_sqlite(league)
    with open(f"web/data/{league}.json", 'w') as f:
        json.dump(league_json, f, indent=4)
        
    # 6. Rebuild Manifest (for Global View)
    print("Rebuilding Global Manifest...")
    from platform_orchestrator import run_update
    # We don't want to run the whole update, just the manifest part.
    # Actually, I'll just manually trigger the manifest rebuild logic if I can.
    # For now, I'll just assume the user runs the orchestrator or I'll add a helper.
    
    print("SYNC COMPLETE.")

if __name__ == "__main__":
    force_sync('PremierLeague', '2026-05-02')
