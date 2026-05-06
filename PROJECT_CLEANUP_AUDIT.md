# Project Cleanup Audit

Generated: 2026-05-04

## Goal

Reduce project noise without losing useful prediction logic. This is an audit only:
do not delete the listed files until they are either archived or mined for useful
logic.

## Active Core

These files are on the current API, sweep, prediction, scraping, persistence,
training, ticket, and reporting paths. Keep them in place.

- `scripts/api_server.py`
- `scripts/platform_orchestrator.py`
- `scripts/main_script.py`
- `scripts/persistence_manager.py`
- `scripts/browser_scraper.py`
- `scripts/free_football_source.py`
- `scripts/result_collector.py`
- `scripts/build_training_examples.py`
- `scripts/market_count_models.py`
- `scripts/training_runner.py`
- `scripts/run_v11_hybrid.py`
- `scripts/refresh_situation_predictions.py`
- `scripts/probability_quality_report.py`
- `scripts/prediction_failure_analyzer.py`
- `scripts/gnn_graph_layer.py`
- `scripts/gnn_walk_forward_validation.py`
- `scripts/graph_snapshot_builder.py`
- `scripts/model_registry.py`
- `scripts/team_name_normalizer.py`
- `scripts/league_situation.py`
- `scripts/execution_engine.py`
- `scripts/v11_hybrid_model.py`
- `scripts/neural_v9_backbone.py`
- `scripts/v8_gsn_architect.py`
- `scripts/football_data_scraper.py`
- `scripts/firecrawl_agent.py`
- `scripts/flashscore_stats_backfill.py`
- `scripts/league_market_profiles.py`
- `scripts/rolling_card_corner_features.py` is not on the main import path but is still useful for count-market feature refresh.

## Mine Before Archive

These files are not reached from the current core graph, but contain ideas that
can improve confidence, calibration, or market selection.

- `scripts/calibration_tracker.py`  
  Useful: Brier, ECE, resolved-prediction calibration, recent calibration windows.
  Mine this for the new true 1-100 Trust Score.
- `scripts/v102_deep_calibration_study.py`  
  Useful: calibration curves, market bias, ECE analysis, ROI recommendations.
  Mine this before replacing EQI with Trust Score.
- `scripts/v102_calibrate_temperature.py`  
  Useful: temperature scaling. Keep the idea, not necessarily the file.
- `scripts/clv_dashboard.py`  
  Useful: closing-line value summaries and drift analysis. This should feed trust.
- `scripts/bayesian_engine.py`  
  Useful: Bayesian team ratings and bivariate Poisson draw logic.
- `backtest_engine.py`, `backtest_engine_v2.py`  
  Useful: older walk-forward backtesting and enhanced features. Compare against current failure reports.
- `scripts/roi_simulator_v93.py`  
  Useful: ROI simulation framing for staking thresholds.
- `scripts/train_ou_btts.py`  
  Useful: older over/under and BTTS feature logic, including table-style aggregates.
- `scripts/components/aggvarfun.py`  
  Useful: aggregate/trend helper functions.
- `scripts/components/fifa_loader.py`  
  Useful: FIFA/player/team attribute enrichment if we add squad-quality features.
- `scripts/download_missing_leagues.py`  
  Useful: missing-league import and rolling YC/corners feature logic.
- `scripts/api_football_stats_backfill.py`  
  Useful: referee, venue, cards, corners, and fixture-id backfill patterns.
- `scripts/import_xgabora.py`  
  Useful: xG/rolling feature import logic.
- `scripts/v9_preprocessor.py`  
  Useful: Dixon-Coles/time-decay ideas.
- `scripts/trainer_v82.py`  
  Useful: classical model calibration and probability quality checks.

## Optional Data Source Archive

These are not active in the current browser/free-source pipeline. Archive after
confirming the current API-Football path is no longer needed.

- `scripts/api_football_scraper.py`
- `scripts/api_football_id_matcher.py`
- `scripts/api_football_stats_backfill.py`
- `scripts/daily_backfill_runner.py`
- `scripts/download_missing_leagues.py`
- `scripts/real_discovery_runner.py`
- `scripts/v7_data_ingest.py`

## Legacy Model/Training Archive

These are older training experiments. Keep only if we need to reproduce old
results; otherwise archive after mining the useful calibration ideas above.

- `scripts/run_v11_neural_training.py`
- `scripts/run_v11_optimized.py`
- `scripts/run_v11_validation.py`
- `scripts/train_temporal_model.py`
- `scripts/train_v11_comprehensive.py`
- `scripts/train_v11_enhanced.py`
- `scripts/train_v11_temporal.py`
- `scripts/train_v9_prototype.py`
- `scripts/trainer_v9.py`
- `scripts/v11_integration.py`
- `scripts/v11_neural_backbone.py`
- `scripts/v11_temporal_integration.py`
- `scripts/optimize_v9_neural.py`
- `scripts/backtest_v9.py`
- `scripts/v82_audit_tool.py`
- `scripts/v94_fine_tune.py`
- `scripts/v95_quantum_shield.py`
- `scripts/v95_simulation_audit.py`
- `scripts/v100_singularity_core.py`
- `scripts/v101_recursive_teacher.py`
- `scripts/v102_teacher_sim.py`

## Safe Archive Candidates

These look like one-off diagnostics, old demos, scratch queries, or manual checks.
They should be moved to an archive folder rather than deleted immediately.

- `build_team_aliases.py` - superseded by `team_name_normalizer.py` and DB aliases.
- `check_db.py`
- `check_pl_db.py`
- `inspect_db.py`
- `diag_phase4.py`
- `phase4_report.py`
- `audit_features.py`
- `explore_github.py`
- `fast_sync.py`
- `run_predictions_demo.py`
- `run_v11_phase4.py`
- `run_v11_phase5.py`
- `run_v9_focal.py`
- `verify_T.py`
- `test_env.py`
- `test_fetch.py`
- `test_imports.py`
- `test_pickle.py`
- `test_requests.py`
- `test_scraping.py`
- `test_scraping2.py`
- `test_skysports.py`
- `test_tavily.py`
- `test_tavily2.py`
- `test_v8_performance.py`
- `scratch/check_db.py`
- `scratch/list_ids.py`
- `scratch/list_tables.py`
- `scratch/patch_db.py`
- `scratch/query_brentford.py`
- `scratch/query_db.py`
- `scratch/query_db2.py`
- `scratch/query_new.py`
- `scratch/query_odds.py`
- `scratch/test_upsert.py`
- `output_tables/*.html`
- root `index.html`
- `web/init_index.html`
- `web/assets/**`
- `web/vendor/**`

## Other Orphaned Scripts

These were not reached from the current active graph. Some may still be useful
as manual commands, but they are not part of the current end-to-end path.

- `scripts/__init__.py`
- `scripts/enrich/__init__.py`
- `scripts/ingest/__init__.py`
- `scripts/persist/__init__.py`
- `scripts/predict/__init__.py`
- `scripts/serve/__init__.py`
- `scripts/create_hist_data.py`
- `scripts/data_sanitizer.py`
- `scripts/deep_study_diagnostic.py`
- `scripts/edit_index_html.py`
- `scripts/evaluate_candidate.py`
- `scripts/execute_v11_training.py`
- `scripts/extensive_teaching_v93.py`
- `scripts/full_system_audit_v93.py`
- `scripts/generate_synthetic_data.py`
- `scripts/import_xgabora.py`
- `scripts/main_script_executor.py`
- `scripts/main_script_v4.py`
- `scripts/performance.py`
- `scripts/prepare_index_html.py`
- `scripts/rebuild_all_json.py`
- `scripts/rebuild_manifest.py`
- `scripts/run_all_leagues.py`
- `scripts/sanitize_data.py`
- `scripts/shadow_tracker.py`
- `scripts/smoke_check.py`
- `scripts/train_elite_squad.py`
- `scripts/v11_config.py`

Keep note:

- `scripts/shadow_tracker.py` overlaps with current failure-analysis concepts. If we rely only on SQLite and `prediction_failure_analyzer.py`, archive it.
- `scripts/smoke_check.py` is useful as a quick CI-style contract check. Consider rewriting it into a maintained `scripts/smoke_check_current.py` instead of keeping it stale.
- `scripts/evaluate_candidate.py` may become useful if we formalize champion/challenger promotion gates.

## Dangerous Archive Only

These contain destructive database cleanup logic. Do not keep them in the main
project root where they can be run by mistake.

- `purge_all_mocks.py`
- `purge_fake_pl.py`

## Generated/Disposable Files

These can be regenerated and should be excluded from normal project review.

- `cache/**` - browser and fixture cache, about 40 MB.
- `graphify-out/**` - static analysis output, about 141 MB.
- `__pycache__/**`
- `*.log`, `*.out.log`, `*.err.log`
- `err.txt`, `out.log`, `output.txt`, `exec.log`
- `diag_output.txt`

## Environment Folders

- `venv39/` is huge but appears to be the active project environment. Keep.
- `venv/` looks like an older environment. Archive/delete only after confirming no process uses it.
- `micromamba.exe` is used by some scripts. Keep.

## Model Artifact Cleanup

Keep active registry paths:

- `model/v11/*_hybrid_v11_draww_v5.pt`
- `model/v9/v9_*_draww_v3.pt`
- current `model/model_registry.json`
- `model/market_counts/*_market_count_model.json`
- `model/gnn/*_team_gnn.pt` and meta files if GNN remains enabled.

Archive candidates:

- old V11 checkpoints: `*_hybrid_v11.pt`, `*_draww_v2.pt`, `*_draww_v3.pt`, `*_draww_v4.pt`, `*_pre_*_backup.pt`, failed/quarantined checkpoints.
- old V9 checkpoints: `*_backbone*.pt`, `*_v94.pt`, `*_v102.pt`, `*_draww_v2.pt`, `*_draww_v4.pt`, backup checkpoints.
- old XGBoost JSON/pickle models in `model/` if V8.2 is no longer served by the UI/API.

Before removing old models, preserve a small metrics index with filename,
league, validation accuracy, log loss, draw recall, corners MAE, cards MAE, and
timestamp.

## How To Raise Trust Properly

Cleaning files will not directly make predictions better, but it reduces the
risk of using stale scripts, mock artifacts, and wrong models. Real confidence
should come from a new calibrated Trust Score.

Proposed Trust Score inputs:

- model confidence: max 1X2 probability, double-chance probability, BTTS/O/U probability.
- calibration quality: league/market ECE, Brier, log loss from resolved predictions.
- source quality: browser/API/free-source confidence, fixture/result/stat completeness.
- market quality: positive edge, CLV, odds availability, drift direction.
- data context: current standings, relegation pressure, team form, rest days, referee, venue, cards/corners history.
- penalties: mock/synthetic source, missing odds, negative edge, poor league calibration, low sample size.

The target behavior should be:

- Trust 80-95: high probability, calibrated league/market, real data, positive edge.
- Trust 60-79: playable but not elite.
- Trust 35-59: watchlist only.
- Trust 1-34: low trust, incomplete data, weak/negative edge, or poor calibration.

## Priority Upgrade Order

1. Replace EQI-as-Trust with a calibrated Trust Score service.
2. Mine `calibration_tracker.py`, `v102_deep_calibration_study.py`, and `clv_dashboard.py`.
3. Add current-season standings ingestion so the motivation layer uses real live table data, not previous-season proxy.
4. Keep only active model checkpoints and archive old checkpoints after metrics extraction.
5. Move one-off tests, scratch files, old UI template assets, and dangerous purge scripts out of the main root.
6. Add `.gitignore` entries for caches, logs, graphify output, and local env folders.
