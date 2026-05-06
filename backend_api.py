from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os
import json
from datetime import datetime
from typing import List, Optional

app = FastAPI(title="Antigravity Neural Gateway", version="5.0.0")

# Enable CORS for the frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.getcwd(), "web", "data", "intelligence_hub.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/")
async def health_check():
    return {
        "status": "online",
        "engine": "V11 Phase 5 Hybrid",
        "timestamp": datetime.now().isoformat(),
        "database_size_mb": round(os.path.getsize(DB_PATH) / (1024 * 1024), 2)
    }

@app.get("/api/v5/predictions/{league}")
async def get_predictions(league: str, limit: int = 50):
    """Fetch live predictions from the Intelligence Hub."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if league == "global":
            cursor.execute('''
                SELECT id, league, match_date, match_time, home_team, away_team, prediction, eqi_score, 
                       primary_market, secondary_market, p_btts, p_o25, corners_exp, cards_exp, 
                       source_confidence, last_updated, rationale_json, p_1x, p_x2, p_dnb, value_edge
                FROM matches
                WHERE is_mock = 0
                ORDER BY match_date DESC, match_time DESC
                LIMIT ?
            ''', (limit,))
        else:
            cursor.execute('''
                SELECT id, league, match_date, match_time, home_team, away_team, prediction, eqi_score, 
                       primary_market, secondary_market, p_btts, p_o25, corners_exp, cards_exp, 
                       source_confidence, last_updated, rationale_json, p_1x, p_x2, p_dnb, value_edge
                FROM matches
                WHERE league = ? AND is_mock = 0
                ORDER BY match_date DESC, match_time DESC
                LIMIT ?
            ''', (league, limit))
        
        rows = cursor.fetchall()
        predictions = []
        for r in rows:
            p = dict(r)
            # Parse JSON fields
            if p.get('rationale_json'):
                try:
                    p['rationale'] = json.loads(p['rationale_json'])
                except:
                    p['rationale'] = {}
            
            # Re-map to UI expected keys if necessary
            p['Home'] = p.get('home_team')
            p['Away'] = p.get('away_team')
            p['Date'] = p.get('match_date')
            p['Time'] = p.get('match_time')
            p['P(BTTS)'] = p.get('p_btts')
            p['P(O2.5)'] = p.get('p_o25')
            p['P(1X)'] = p.get('p_1x')
            p['P(X2)'] = p.get('p_x2')
            p['DNB'] = p.get('p_dnb')
            p['Value Edge'] = p.get('value_edge')
            p['E(Corners)'] = p.get('corners_exp')
            p['E(Cards)'] = p.get('cards_exp')
            
            p['trust_score'] = round(max(1.0, min(100.0, float(r['eqi_score']))), 1)
            predictions.append(p)
            
        return {
            "league": league,
            "count": len(predictions),
            "data": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/v5/leagues")
async def list_leagues():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT league FROM matches WHERE is_mock = 0")
    leagues = [r[0] for r in cursor.fetchall()]
    conn.close()
    return {"leagues": leagues}

if __name__ == "__main__":
    import uvicorn
    print(f"Launching Antigravity Neural Gateway on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
