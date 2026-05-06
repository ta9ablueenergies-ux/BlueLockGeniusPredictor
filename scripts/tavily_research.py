# ANTIGRAVITY TAVILY DEEP RESEARCH INTEGRATION
"""
Uses Tavily AI for comprehensive match analysis:
- Team news, injuries, lineup predictions
- Head-to-head historical analysis
- Form and momentum indicators
- Manager quotes and tactical setup
"""
import os
import json
import time
import requests
import urllib3
from datetime import datetime, timedelta

from components.ssl_utils import get_unsafe_session

urllib3.disable_warnings()

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# Tavily API key - set via environment variable
# Prioritized API keys
TAVILY_BACKUP_KEYS = [
    os.environ.get('TAVILY_API_KEY', ''),
    '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',
    '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',
    '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'
]

class TavilyResearcher:
    """
    Deep research layer using Tavily AI search and Ollama Sentiment.
    Provides contextual intelligence for match predictions.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = None
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Key Rotation Logic
        keys_to_try = [api_key] if api_key else TAVILY_BACKUP_KEYS
        for key in keys_to_try:
            if not key: continue
            try:
                if TAVILY_AVAILABLE:
                    test_client = TavilyClient(api_key=key)
                    # Simple test search
                    test_client.search("test", max_results=1)
                    self.client = test_client
                    self.api_key = key
                    print(f"[Tavily] Connected with key: {key[:15]}...")
                    break
            except Exception as e:
                print(f"[Tavily] Key {key[:15]}... failed: {e}")
                continue
        
        if not self.client:
            print("[Tavily] No working keys found - running in demo mode")

    def analyze_sentiment_with_ollama(self, snippets, team_name):
        """Uses local Ollama to analyze sentiment of news snippets (-1.0 to 1.0)"""
        if not snippets: return 0.0
        
        context = "\n".join([s.get('content', '')[:300] for s in snippets])
        prompt = f"""
        Analyze the following sports news for {team_name}. 
        Is it positive or negative for their upcoming match performance?
        Focus on injuries, team morale, and tactics.
        Return ONLY a single numeric value between -1.0 (extremely negative) and 1.0 (extremely positive).
        0.0 is neutral.
        
        NEWS:
        {context}
        
        SCORE:
        """
        
        try:
            payload = {
                "model": "mistral", # Default to mistral, common for Ollama
                "prompt": prompt,
                "stream": False
            }
            session = get_unsafe_session()
            response = session.post(self.ollama_url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json().get('response', '0.0').strip()
                # Clean result to extract float
                try:
                    score = float(''.join(c for c in result if c in '0123456789.-'))
                    print(f"[Ollama] Sentiment for {team_name}: {score}")
                    return max(-1.0, min(1.0, score))
                except: return 0.0
        except Exception as e:
            # print(f"[Ollama] Connection error: {e}")
            return 0.0
        return 0.0

    def search(self, query, **kwargs):
        """Wrapper for Tavily search with fallback"""
        if not self.client:
            return self._demo_response(query)
        try:
            return self.client.search(query=query, max_results=5, **kwargs)
        except Exception as e:
            print(f"[Tavily] Search error: {e}")
            return self._demo_response(query)

    def _demo_response(self, query):
        """Demo response when no API key"""
        return {
            'results': [{
                'title': f'Demo: {query}',
                'url': 'https://example.com',
                'content': 'Tavily API key required for real research. Set TAVILY_API_KEY environment variable.'
            }],
            'answer': 'Research module requires Tavily API key. Visit https://app.tavily.com to get one (1000 free credits/month).'
        }

    def research_match(self, home_team, away_team, league, target_date):
        """
        Perform deep research on a specific match.
        Returns contextual intelligence for the prediction engine.
        """
        if not self.client:
            return self._demo_match_research(home_team, away_team, league)

        research = {
            'home_news': [],
            'away_news': [],
            'h2h': [],
            'form': [],
            'sentiment': {'home': 0.0, 'away': 0.0},
            'timestamp': datetime.now().isoformat()
        }

        # 1. Search Home Team News
        try:
            h_res = self.search(f"{home_team} injuries team news lineup {target_date}", search_depth='advanced', topic='news')
            research['home_news'] = h_res.get('results', [])
            research['sentiment']['home'] = self.analyze_sentiment_with_ollama(research['home_news'], home_team)
        except: pass

        # 2. Search Away Team News
        try:
            a_res = self.search(f"{away_team} injuries team news lineup {target_date}", search_depth='advanced', topic='news')
            research['away_news'] = a_res.get('results', [])
            research['sentiment']['away'] = self.analyze_sentiment_with_ollama(research['away_news'], away_team)
        except: pass

        # 3. Search H2H
        try:
            h2h_res = self.search(f"{home_team} vs {away_team} head to head history", search_depth='advanced', topic='news')
            research['h2h'] = h2h_res.get('results', [])[:3]
        except: pass

        return research

    def _demo_match_research(self, home_team, away_team, league):
        """Demo research when no API key"""
        return {
            'home_news': [{'title': f'{home_team} Squad Update', 'content': 'Demo mode - Tavily API key required'}],
            'away_news': [{'title': f'{away_team} Squad Update', 'content': 'Demo mode - Tavily API key required'}],
            'h2h': [{'title': f'{home_team} vs {away_team} H2H', 'content': 'Historical data requires Tavily API key'}],
            'form': [{'title': f'{league} Form Analysis', 'content': 'Research module - get API key at app.tavily.com'}],
            'sentiment': {'home': 0.0, 'away': 0.0},
            'timestamp': datetime.now().isoformat()
        }

    def enrich_prediction(self, match_data):
        """
        Take a prediction and enrich it with Tavily research and Ollama sentiment.
        """
        home = match_data.get('Home', '')
        away = match_data.get('Away', '')
        league = match_data.get('League', '')
        date = match_data.get('Date', datetime.now().strftime('%d/%m/%Y'))

        research = self.research_match(home, away, league, date)

        # Add research to match data
        enriched = match_data.copy()
        enriched['research'] = {
            'has_research': self.client is not None,
            'sentiment_home': research['sentiment']['home'],
            'sentiment_away': research['sentiment']['away'],
            'sources': len(research['home_news']) + len(research['away_news']) + len(research['h2h']),
            'timestamp': research['timestamp']
        }
        
        # Calculate net sentiment bonus (-2.0 to +2.0)
        net_sentiment = (research['sentiment']['home'] - research['sentiment']['away']) * 2.0
        enriched['research']['sentiment_bonus'] = round(net_sentiment, 2)

        return enriched


def get_tavily_researcher():
    """Factory function to get Tavily researcher instance"""
    return TavilyResearcher()