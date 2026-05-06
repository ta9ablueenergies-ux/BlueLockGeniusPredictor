import os
import subprocess
import time
import datetime
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PATH = os.path.join(PROJECT_ROOT, "venv39")
PYTHON_EXE = os.path.join(PROJECT_ROOT, "micromamba.exe")
MAIN_SCRIPT = "scripts/main_script.py"
LOG_FILE = os.path.join(PROJECT_ROOT, "web", "data", "update_log.json")
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from persistence_manager import upsert_match_sqlite, get_league_json_from_sqlite
from graph_registry import get_active_graph_snapshot_id

LEAGUES = [
    'PremierLeague', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1',
    'ChampionsLeague', 'Championship', 'ScottishPremiership',
    'Eredivisie', 'LigaNOS', 'BelgianProLeague', 'SuperLig',
    '2Bundesliga', 'Ligue2', 'LaLiga2', 'SerieB'
]

CLUSTERS = [
    ['PremierLeague', 'LaLiga', 'SerieA', 'Bundesliga'], # Elite Core
    ['ChampionsLeague'], # Mid-Week Europe
    ['Championship', 'ScottishPremiership'], # British Tier 2
    ['Eredivisie', 'LigaNOS', 'BelgianProLeague', 'SuperLig'], # European T2
    ['Ligue1', '2Bundesliga', 'Ligue2', 'LaLiga2', 'SerieB'] # French/Lower
]

def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def get_sweep_mode():
    """
    SWEEP_MODE controls the discovery boundary:
      - internal: no external discovery; rebuild public files from SQLite only
      - free: run free-source historical backfill, then rebuild public files
      - fdo: Football-Data.org only
      - hybrid: FDO first, then Firecrawl if FDO has no fixture
      - external: Firecrawl first, then FDO fallback
    """
    mode = os.environ.get("SWEEP_MODE", "hybrid").strip().lower()
    return mode if mode in {"internal", "free", "fdo", "hybrid", "external"} else "hybrid"

def maybe_run_free_source_backfill():
    if not env_bool("ENABLE_FREE_SOURCE_BACKFILL", False) and get_sweep_mode() != "free":
        return None
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
    from free_source_pipeline import run_free_source_pipeline
    print("[Orchestrator] Running free-source backfill suite...")
    report = run_free_source_pipeline(
        include_extra_leagues=env_bool("INCLUDE_EXTRA_LEAGUES", False),
        import_temp_extract=env_bool("IMPORT_TEMP_EXTRACT", True),
    )
    print(f"[Orchestrator] Free-source backfill complete: {report}")
    return report


def maybe_refresh_market_models():
    """Refresh probabilistic goals/corners/cards models from closed examples."""
    if not env_bool("ENABLE_MARKET_COUNT_MODELS", True):
        return None
    try:
        from market_count_models import run_market_count_pipeline
        report = run_market_count_pipeline(
            enrich_limit=int(os.environ.get("MARKET_MODEL_ENRICH_LIMIT", "0")) or None,
        )
        trained = sum(1 for row in report.get("leagues", {}).values() if row.get("status") == "trained")
        print(f"[Orchestrator] Market-count models refreshed: {trained} leagues")
        return {
            "trained_leagues": trained,
            "updated_matches": (report.get("match_enrichment") or {}).get("updated", 0),
            "error": report.get("error"),
        }
    except Exception as exc:
        print(f"[Orchestrator] Market-count refresh skipped: {exc}")
        return {"error": str(exc)}


def maybe_refresh_training_data_guard(stage=""):
    """Build canonical training views and model staleness report before training."""
    if not env_bool("ENABLE_TRAINING_DATA_GUARD", True):
        return None
    try:
        from training_data_guard import ensure_training_data_guard

        report = ensure_training_data_guard(write_report=True)
        summary = report.get("summary") or {}
        print(
            "[Orchestrator] Training data guard"
            f" {stage}: raw={summary.get('raw_rows')} canonical={summary.get('canonical_rows')}"
            f" duplicate_extra={summary.get('duplicate_extra_rows')}"
            f" stale_v11={summary.get('v11_stale_leagues')}"
            f" stale_market={summary.get('market_count_stale_leagues')}"
        )
        return {
            "stage": stage,
            "status": report.get("status"),
            "report_path": report.get("report_path"),
            "summary": summary,
        }
    except Exception as exc:
        print(f"[Orchestrator] Training data guard skipped: {exc}")
        return {"stage": stage, "error": str(exc)}


def _free_match_to_fixture(item, league):
    source = str(item.source or "").lower()
    if source == "historical":
        source_confidence = 0.88
    elif source in {"openfootball_repo", "football_datasets_repo", "free_dataset"}:
        source_confidence = 0.82
    elif source.startswith("browser"):
        source_confidence = 0.78
    else:
        source_confidence = 0.80
    return {
        'Home': item.home_team,
        'Away': item.away_team,
        'Time': item.kickoff_utc or '15:00',
        'Date': item.match_date,
        'League': league or item.league,
        'h_course': item.home_odds or 2.0,
        'd_course': item.draw_odds or 3.0,
        'a_course': item.away_odds or 3.5,
        'h_open': item.home_odds or 2.0,
        'd_open': item.draw_odds or 3.0,
        'a_open': item.away_odds or 3.5,
        'source_url': item.source_url,
        'flashscore_id': item.source_id if str(item.source or "").lower().startswith("browser") else None,
        'source_confidence': source_confidence,
    }

def log_event(status, message, run_id=None, stage=None, meta=None):
    log_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "message": message,
        "run_id": run_id,
        "stage": stage,
        "meta": meta or {}
    }
    history = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try: history = json.load(f)
            except: history = []
    history.insert(0, log_data)
    with open(LOG_FILE, "w") as f:
        json.dump(history[:10], f, indent=4)

def export_all_league_json():
    """Refresh every league JSON from SQLite so stale files cannot survive a sweep."""
    data_dir = os.path.join(PROJECT_ROOT, "web", "data")
    os.makedirs(data_dir, exist_ok=True)
    exported = {}
    for league in LEAGUES:
        league_json = get_league_json_from_sqlite(league)
        out_path = os.path.join(data_dir, f"{league}.json")
        with open(out_path, "w") as f:
            json.dump(league_json, f, indent=4)
        exported[league] = len(league_json)
    return exported

def build_global_manifest(window_past_days=7, window_future_days=3):
    """Build global_manifest.json from refreshed league JSON files."""
    print("[Orchestrator] Generating Global Manifest...")
    all_manifest_matches = []
    now = datetime.datetime.now()
    for league in LEAGUES:
        json_path = os.path.join(PROJECT_ROOT, 'web', 'data', f'{league}.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                    def in_manifest_window(m):
                        m_date = m.get('Date', '')
                        if '/' in m_date:
                            parts = m_date.split('/')
                            m_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                        try:
                            m_dt = datetime.datetime.strptime(m_date.split('T')[0], '%Y-%m-%d')
                            diff = (m_dt - now).days
                            return -window_past_days <= diff <= window_future_days
                        except Exception:
                            return True

                    all_manifest_matches.extend([m for m in data if in_manifest_window(m)])
            except Exception as exc:
                print(f"[Orchestrator] Manifest read error for {league}: {exc}")

    manifest_path = os.path.join(PROJECT_ROOT, 'web', 'data', 'global_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "matches": all_manifest_matches
        }, f, indent=4)
    return len(all_manifest_matches)

def write_sweep_report(run_id, target_start, mode, source_counts, exported_counts, manifest_count, extra=None):
    report = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "target_date": target_start,
        "mode": mode,
        "source_counts": source_counts,
        "exported_counts": exported_counts,
        "manifest_count": manifest_count,
        "mock_fallback_enabled": env_bool("ALLOW_MOCK_FALLBACK", False),
    }
    if extra:
        report.update(extra)
    report_path = os.path.join(PROJECT_ROOT, "web", "data", "last_sweep_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    return report

def rebuild_public_outputs(run_id, target_start, mode, source_counts):
    exported_counts = export_all_league_json()
    print(f"[Orchestrator] League JSON refreshed from SQLite: {exported_counts}")
    build_global_tickets()
    manifest_count = build_global_manifest()
    report = write_sweep_report(run_id, target_start, mode, source_counts, exported_counts, manifest_count)
    print(f"[Orchestrator] Sweep report: {report}")
    return report

def collect_recent_results():
    """Close completed match feedback loops before fresh predictions are generated."""
    try:
        from result_collector import collect_persisted_results, collect_all_pending, collect_flashscore_results
        flashscore = 0
        if env_bool("ENABLE_BROWSER_SCRAPING", False) or env_bool("ENABLE_FLASHSCORE_RESULT_COLLECTION", False):
            flashscore = collect_flashscore_results(days_back=int(os.environ.get("RESULT_LOOKBACK_DAYS", "14")))
        persisted = collect_persisted_results(days_back=int(os.environ.get("RESULT_LOOKBACK_DAYS", "14")))
        pending = collect_all_pending()
        try:
            from prediction_failure_analyzer import analyze_failures
            failure_report = analyze_failures(limit=int(os.environ.get("FAILURE_ANALYSIS_LIMIT", "500")))
        except Exception as analysis_exc:
            failure_report = {"error": str(analysis_exc)}
        try:
            from probability_quality_report import build_probability_quality_report
            probability_report = build_probability_quality_report(limit=int(os.environ.get("PROBABILITY_QUALITY_LIMIT", "2000")))
        except Exception as quality_exc:
            probability_report = {"error": str(quality_exc)}
        return {
            "flashscore": flashscore,
            "persisted": persisted,
            "shadow_pending": pending,
            "failure_analysis": {
                "resolved_predictions": failure_report.get("resolved_predictions", 0),
                "overall_hit_rate": failure_report.get("overall_hit_rate", 0),
                "error": failure_report.get("error"),
            },
            "probability_quality": {
                "rows_scanned": probability_report.get("rows_scanned", 0),
                "model_log_loss": ((probability_report.get("overall") or {}).get("model") or {}).get("log_loss"),
                "model_brier": ((probability_report.get("overall") or {}).get("model") or {}).get("brier"),
                "error": probability_report.get("error"),
            },
        }
    except Exception as exc:
        print(f"[Orchestrator] Result collection skipped: {exc}")
        return {"persisted": 0, "shadow_pending": 0, "error": str(exc)}


def maybe_backfill_flashscore_stats(target_date):
    """Populate cards/corners for Flashscore-backed rows after browser discovery."""
    if not (env_bool("ENABLE_BROWSER_SCRAPING", False) or env_bool("ENABLE_FLASHSCORE_STATS_BACKFILL", False)):
        return None
    try:
        from flashscore_stats_backfill import run_backfill
        report = run_backfill(
            date=target_date,
            limit=int(os.environ.get("FLASHSCORE_STATS_LIMIT", "200")),
            only_missing=not env_bool("FLASHSCORE_STATS_REFRESH_ALL", False),
            engine=os.environ.get("BROWSER_ENGINE"),
            sleep_seconds=float(os.environ.get("FLASHSCORE_STATS_SLEEP", "0.3")),
        )
        print(f"[Orchestrator] Flashscore stats backfill: requested={report.get('requested')} updated={report.get('updated')} failed={report.get('failed')}")
        return {k: report.get(k) for k in ("requested", "updated", "failed")}
    except Exception as exc:
        print(f"[Orchestrator] Flashscore stats backfill skipped: {exc}")
        return {"error": str(exc)}


def maybe_collect_flashscore_results_for_target(target_date):
    """Close selected-date browser rows after a sweep creates or refreshes them."""
    if not (env_bool("ENABLE_BROWSER_SCRAPING", False) or env_bool("ENABLE_FLASHSCORE_RESULT_COLLECTION", False)):
        return None
    try:
        from result_collector import collect_flashscore_results
        resolved = collect_flashscore_results(target_date=target_date)
        try:
            from prediction_failure_analyzer import analyze_failures
            analyze_failures(limit=int(os.environ.get("FAILURE_ANALYSIS_LIMIT", "500")))
        except Exception:
            pass
        return {"resolved": resolved}
    except Exception as exc:
        print(f"[Orchestrator] Flashscore result close skipped: {exc}")
        return {"error": str(exc)}

def build_global_tickets():
    """
    Collects all predictions across leagues and builds quantum accumulator tickets.
    Writes to web/data/tickets.json
    """
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
    from execution_engine import construct_quantum_ticket_bundle, trust_score_from_eqi
    
    start_date_str = os.environ.get('PIPELINE_START_DATE')
    end_date_str = os.environ.get('PIPELINE_END_DATE')
    
    # Target date for filtering (just the YYYY-MM-DD part)
    target_date = start_date_str.split('T')[0] if start_date_str else None

    all_matches = []
    for league in LEAGUES:
        json_path = os.path.join(PROJECT_ROOT, 'web', 'data', f'{league.replace(" ", "")}.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                    # Temporal Filtering: Only include matches within the current window
                    if target_date:
                        # Match 'Date' format 'DD/MM/YYYY' or 'YYYY-MM-DD'
                        def matches_date(m):
                            m_date = m.get('Date', '')
                            if '/' in m_date: # Convert DD/MM/YYYY to YYYY-MM-DD
                                parts = m_date.split('/')
                                m_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                            return m_date == target_date

                        data = [m for m in data if matches_date(m)]
                    
                    all_matches.extend(data)
            except Exception as e:
                print(f"[Orchestrator] Error loading {league}: {e}")

    def serialize_leg(t, fallback_league='GLOBAL'):
        trust_score = trust_score_from_eqi(
            t.get('trust_score')
            if t.get('trust_score') is not None
            else t.get('execution_trust')
            if t.get('execution_trust') is not None
            else t.get('eqi')
        )
        return {
            "Home": t.get('Home') or t.get('home_team'),
            "Away": t.get('Away') or t.get('away_team'),
            "prediction": t.get('prediction', 'N/A'),
            "best_market": t.get('best_market') or t.get('primary_market') or t.get('prediction', 'N/A'),
            "trust_score": trust_score,
            "Trust Score": trust_score,
            "eqi": trust_score,
            "ticket_score": t.get('ticket_score'),
            "market_probability": t.get('market_probability'),
            "stake_pct": t.get('stake_pct', 0),
            "League": t.get('League', fallback_league),
            "Date": t.get('Date', ''),
        }

    def serialize_bundle(bundle, scope):
        return {
            "TYPE_A_ULTRA_SAFE": [serialize_leg(t, scope) for t in bundle.get("TYPE_A_ULTRA_SAFE", [])],
            "TYPE_B_BALANCED": [serialize_leg(t, scope) for t in bundle.get("TYPE_B_BALANCED", [])],
            "TYPE_C_VALUE": [serialize_leg(t, scope) for t in bundle.get("TYPE_C_VALUE", [])],
            "scope": scope,
            "target_date": target_date,
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    if not all_matches:
        print(f"[Orchestrator] No matches found for tickets (Target: {target_date})")
        ticket_path = os.path.join(PROJECT_ROOT, 'web', 'data', 'tickets.json')
        with open(ticket_path, 'w') as f:
            json.dump(serialize_bundle({}, "GLOBAL"), f, indent=4)
        return

    # Build league-specific tickets
    for league in LEAGUES:
        league_matches = []
        json_path = os.path.join(PROJECT_ROOT, 'web', 'data', f'{league.replace(" ", "")}.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if target_date:
                        def matches_date(m):
                            m_date = m.get('Date', '')
                            if '/' in m_date:
                                parts = m_date.split('/')
                                m_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                            return m_date == target_date
                        data = [m for m in data if matches_date(m)]
                    league_matches = data
            except: pass

        l_bundle = construct_quantum_ticket_bundle(league_matches)
        l_ticket_path = os.path.join(PROJECT_ROOT, 'web', 'data', f'tickets_{league.replace(" ", "")}.json')
        with open(l_ticket_path, 'w') as f:
            json.dump(serialize_bundle(l_bundle, league), f, indent=4)

    # Build Global Ticket
    print(f"[Orchestrator] Building global tickets from {len(all_matches)} temporal matches")
    ticket_bundle = construct_quantum_ticket_bundle(all_matches)
    
    ticket_path = os.path.join(PROJECT_ROOT, 'web', 'data', 'tickets.json')
    with open(ticket_path, 'w') as f:
        json.dump(serialize_bundle(ticket_bundle, "GLOBAL"), f, indent=4)

    print(f"[Orchestrator] Generated {len(ticket_bundle.get('TYPE_A_ULTRA_SAFE', []))}-leg global trust ticket")

def run_update():
    try:
        from persistence_manager import auto_dedupe_aliases
        auto_dedupe_aliases()
    except Exception as e:
        print(f"[Orchestrator] Auto-dedupe failed: {e}")
    market_model_refresh = None
    training_data_guard = maybe_refresh_training_data_guard("startup")
        
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Target date YYYY-MM-DD")
    parser.add_argument("--mode", help="Sweep mode")
    args, unknown = parser.parse_known_args()

    run_id = os.environ.get("PIPELINE_RUN_ID") or datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
    graph_snapshot_id = os.environ.get("GRAPH_SNAPSHOT_ID") or get_active_graph_snapshot_id()
    mode = args.mode or get_sweep_mode()
    print(f"[{datetime.datetime.now()}] Starting V5.9 Global Sharp Sweep... run_id={run_id} mode={mode}")

    SWEEP_LOG = os.path.join(PROJECT_ROOT, "web", "data", "sweep.log")
    source_counts = {"fdo": 0, "firecrawl": 0, "historical": 0, "free_repo": 0, "browser": 0, "mock": 0, "skipped": 0}
    result_collection = collect_recent_results()
    training_data_guard = maybe_refresh_training_data_guard("post_results") or training_data_guard
    market_model_refresh = maybe_refresh_market_models()
    
    with open(SWEEP_LOG, "a") as log_f:
        target_start = args.date or os.environ.get('PIPELINE_START_DATE')
        if not target_start:
            target_start = datetime.datetime.now().strftime('%Y-%m-%d')
        elif 'T' in target_start:
            target_start = target_start.split('T')[0]

        if get_sweep_mode() == "free" or env_bool("ENABLE_FREE_SOURCE_BACKFILL", False):
            free_report = maybe_run_free_source_backfill()
            training_data_guard = maybe_refresh_training_data_guard("free_source") or training_data_guard
            market_model_refresh = maybe_refresh_market_models()
            log_event("Success", "Free-source backfill complete.", run_id=run_id, stage="free_source", meta={"report": free_report})
            if get_sweep_mode() == "free":
                report = rebuild_public_outputs(run_id, target_start, mode, source_counts)
                write_sweep_report(
                    run_id, target_start, mode, source_counts,
                    report["exported_counts"], report["manifest_count"],
                    extra={"result_collection": result_collection, "free_report": free_report, "market_model_refresh": market_model_refresh, "training_data_guard": training_data_guard},
                )
                return

        if mode == "internal":
            print("[Orchestrator] Internal mode: rebuilding public outputs from SQLite only.")
            log_f.write(f"\n\n=== INTERNAL REFRESH: {datetime.datetime.now()} (Target: {target_start}, run_id={run_id}) ===\n")
            flashscore_stats = maybe_backfill_flashscore_stats(target_start)
            flashscore_results = maybe_collect_flashscore_results_for_target(target_start)
            if flashscore_results and flashscore_results.get("resolved"):
                training_data_guard = maybe_refresh_training_data_guard("flashscore_results") or training_data_guard
                market_model_refresh = maybe_refresh_market_models()
            report = rebuild_public_outputs(run_id, target_start, mode, source_counts)
            write_sweep_report(
                run_id, target_start, mode, source_counts,
                report["exported_counts"], report["manifest_count"],
                extra={"result_collection": result_collection, "flashscore_stats": flashscore_stats, "flashscore_results": flashscore_results, "market_model_refresh": market_model_refresh, "training_data_guard": training_data_guard},
            )
            log_event("Success", "Internal refresh complete. Public files rebuilt from SQLite.", run_id=run_id, stage="internal")
            return
        
        log_f.write(f"\n\n=== NEW V6.0 CLUSTERED SWEEP: {datetime.datetime.now()} (Target: {target_start}, run_id={run_id}, mode={mode}) ===\n")
        print(f"--- ANTIGRAVITY V6.0 CLUSTERED SYNC (Target: {target_start}, mode={mode}) ---")
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
        from firecrawl_agent import discover_multiple_leagues
        from main_script import precalculate_all_features, predict_results
        
        for cluster in CLUSTERS:
            print(f"\n[Orchestrator] Processing Cluster: {', '.join(cluster)}...")
            
            for league in cluster:
                print(f"[Orchestrator] Discovering matches for {league}...")
                
                # 1. DISCOVERY (Date-scoped first, then hybrid live sources)
                from football_data_scraper import fetch_fdo_matches
                from free_football_source import resolve_fixtures_for_date
                matches = []
                counted_source_breakdown = False
                date_snapshot = resolve_fixtures_for_date(
                    target_start,
                    leagues=[league],
                    include_browser=env_bool("ENABLE_BROWSER_SCRAPING", False),
                )
                if date_snapshot:
                    matches = [_free_match_to_fixture(item, league) for item in date_snapshot]
                    if any(str(item.source or "").lower().startswith("browser") for item in date_snapshot):
                        match_source = "browser"
                    elif any(str(item.source or "").lower() == "historical" for item in date_snapshot):
                        match_source = "historical"
                    else:
                        match_source = "free_repo"
                    for item in date_snapshot:
                        item_source = str(item.source or "").lower()
                        if item_source == "historical":
                            source_counts["historical"] += 1
                        elif item_source.startswith("browser"):
                            source_counts["browser"] += 1
                        else:
                            source_counts["free_repo"] += 1
                    counted_source_breakdown = True

                if not matches and mode == "external":
                    discovered_intel = discover_multiple_leagues([league], target_start)
                    if discovered_intel:
                        matches = [{
                            'Home': i.home_team,
                            'Away': i.away_team,
                            'Time': i.kickoff_utc or '00:00',
                            'Date': i.match_date,
                            'League': league
                        } for i in discovered_intel]
                        match_source = 'firecrawl'
                    else:
                        fdo_data = fetch_fdo_matches(league, target_start, target_start)
                elif not matches:
                    fdo_data = fetch_fdo_matches(league, target_start, target_start)

                if not matches and mode in {"hybrid", "fdo", "external"} and fdo_data is not None and not fdo_data.empty:
                    for _, m in fdo_data.iterrows():
                        matches.append({
                            'Home': m.get('hometeam'),
                            'Away': m.get('awayteam'),
                            'Time': m.get('Time') or '15:00',
                            'Date': m.get('date'),
                            'League': league
                        })
                    match_source = 'fdo'  # Real confirmed fixtures

                if not matches and env_bool("ENABLE_BROWSER_SCRAPING", False):
                    print(f"[{league}] Trying browser scraping via Playwright/SeleniumBase...")
                    try:
                        from browser_scraper import discover_browser_fixtures_for_date

                        browser_fixtures = discover_browser_fixtures_for_date(
                            target_start,
                            leagues=[league],
                        )
                        if browser_fixtures:
                            matches = [_free_match_to_fixture(item, league) for item in browser_fixtures]
                            match_source = "browser"
                            source_counts["browser"] += len(browser_fixtures)
                            counted_source_breakdown = True
                    except Exception as exc:
                        print(f"[{league}] Browser scraping skipped: {exc}")

                if not matches and mode == "hybrid":
                    print(f"[{league}] No matches found in FDO. Trying Firecrawl Discovery...")
                    discovered_intel = discover_multiple_leagues([league], target_start)
                    if not discovered_intel:
                        print(f"[{league}] Discovery failed.")
                    else:
                        # Map discovered intel to match skeleton
                        matches = [{
                            'Home': i.home_team,
                            'Away': i.away_team,
                            'Time': i.kickoff_utc or '00:00',
                            'Date': i.match_date,
                            'League': league
                        } for i in discovered_intel]
                        match_source = 'firecrawl'  # Web-scraped real fixtures

                if not matches and env_bool("ALLOW_MOCK_FALLBACK", False):
                    print(f"[{league}] Using localized mock schedules because ALLOW_MOCK_FALLBACK is enabled.")
                    from football_data_scraper import generate_enhanced_mock
                    mock_df = generate_enhanced_mock(league, target_start)
                    if not mock_df.empty:
                        matches = [{
                            'Home': r['hometeam'],
                            'Away': r['awayteam'],
                            'Time': r['Time'],
                            'Date': target_start,
                            'League': league
                        } for _, r in mock_df.iterrows()]
                    match_source = 'mock'
                
                if not matches:
                    print(f"[{league}] No real fixtures found. Skipping.")
                    source_counts["skipped"] += 1
                    continue

                if not counted_source_breakdown:
                    if match_source in source_counts:
                        source_counts[match_source] += len(matches)
                    elif match_source.startswith("browser"):
                        source_counts["browser"] += len(matches)
                    else:
                        source_counts["free_repo"] += len(matches)
                
                print(f"[{league}] Found {len(matches)} matches. Enriching with V6.0 Quantum Intel...")
                
                # 2. ENRICHMENT (Firecrawl Neural Edge)
                from firecrawl_agent import enrich_match
                from pydantic import BaseModel
                class MockIntel(BaseModel):
                    home_team: str
                    away_team: str
                    match_date: str
                    kickoff_utc: str = "15:00"
                    odds_home: float = 2.0
                    odds_draw: float = 3.0
                    odds_away: float = 3.5

                final_intel_list = []
                use_external_enrichment = mode in {"hybrid", "external"} and match_source not in {"mock", "historical"}
                if use_external_enrichment:
                    max_workers = min(6, max(1, len(matches)))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(enrich_match, m['Home'], m['Away'], league, m['Date']): m
                            for m in matches
                        }
                        for future in as_completed(futures):
                            m = futures[future]
                            try:
                                intel = future.result()
                                if intel:
                                    final_intel_list.append(intel)
                                else:
                                    final_intel_list.append(MockIntel(home_team=m['Home'], away_team=m['Away'], match_date=m['Date']))
                            except Exception as e:
                                print(f"[{league}] Enrichment error for {m['Home']} vs {m['Away']}: {e}")
                                final_intel_list.append(MockIntel(home_team=m['Home'], away_team=m['Away'], match_date=m['Date']))
                else:
                    final_intel_list = [
                        MockIntel(
                            home_team=m['Home'],
                            away_team=m['Away'],
                            match_date=m['Date'],
                            kickoff_utc=m.get('Time') or '15:00',
                            odds_home=m.get('h_course') or 2.0,
                            odds_draw=m.get('d_course') or 3.0,
                            odds_away=m.get('a_course') or 3.5,
                        )
                        for m in matches
                    ]
                
                if not final_intel_list:
                    print(f"[{league}] Discovery failed. Skipping.")
                    continue
                
                # 3. PREDICT AND PERSIST
                # Map intel objects back to dicts for the engine
                intel_dicts = [i.model_dump() for i in final_intel_list]
                
                # Convert FDO matches to DataFrame
                mock_matches = []
                for i in final_intel_list:
                    fixture = next(
                        (m for m in matches if m['Home'] == i.home_team and m['Away'] == i.away_team and m['Date'] == i.match_date),
                        {},
                    )
                    # Match identity from enriched intel
                    mock_matches.append({
                        'HomeTeam': i.home_team,
                        'AwayTeam': i.away_team,
                        'Time': i.kickoff_utc or '00:00',
                        'Date': i.match_date,
                        'h_course': i.odds_home or fixture.get('h_course') or 2.0,
                        'd_course': i.odds_draw or fixture.get('d_course') or 3.0,
                        'a_course': i.odds_away or fixture.get('a_course') or 3.5,
                        'h_open': i.odds_home or fixture.get('h_open') or 2.0,
                        'd_open': i.odds_draw or fixture.get('d_open') or 3.0,
                        'a_open': i.odds_away or fixture.get('a_open') or 3.5,
                        'source_url': fixture.get('source_url'),
                        'flashscore_id': fixture.get('flashscore_id'),
                    })
                
                import pandas as pd
                df_matches = pd.DataFrame(mock_matches)
                
                ratings = precalculate_all_features(league)
                final_df = predict_results(df_matches, league, ratings, intel_dicts)
                
                # PERSIST with correct data_source tag
                for _, row in final_df.iterrows():
                    upsert_match_sqlite(
                        row.to_dict(),
                        data_source=match_source,
                        run_id=run_id,
                        graph_snapshot_id=graph_snapshot_id,
                    )
                
                print(f"[{league}] V6.0 Hybrid Sync Complete.")
                log_f.write(f"[{league}] V6.0 Sync Complete: {len(final_intel_list)} matches. source={match_source} run_id={run_id}\n")
                log_f.flush()
    
    flashscore_stats = maybe_backfill_flashscore_stats(target_start)
    flashscore_results = maybe_collect_flashscore_results_for_target(target_start)
    if flashscore_results and flashscore_results.get("resolved"):
        training_data_guard = maybe_refresh_training_data_guard("flashscore_results") or training_data_guard
        market_model_refresh = maybe_refresh_market_models()
    report = rebuild_public_outputs(run_id, target_start, mode, source_counts)
    write_sweep_report(
        run_id, target_start, mode, source_counts,
        report["exported_counts"], report["manifest_count"],
        extra={"result_collection": result_collection, "flashscore_stats": flashscore_stats, "flashscore_results": flashscore_results, "market_model_refresh": market_model_refresh, "training_data_guard": training_data_guard},
    )

    log_event("Success", "V5.9 Global Sharp Sweep Complete. Manifest generated.", run_id=run_id, stage="finalize")

if __name__ == "__main__":
    run_update()
