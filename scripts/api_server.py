from flask import Flask, send_from_directory, jsonify, request, abort
import subprocess, os, sys, logging, threading, datetime, json, uuid

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR      = os.path.join(PROJECT_ROOT, "web")

app = Flask(__name__, static_folder=WEB_DIR)

from flask_cors import CORS
CORS(app)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MICROMAMBA   = os.path.join(PROJECT_ROOT, "micromamba.exe")
ORCHESTRATOR = os.path.join(PROJECT_ROOT, "scripts", "platform_orchestrator.py")

# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------
pipeline_lock = threading.Lock()
is_running    = False
PIPELINE_TIMEOUT_SECONDS = 60 * 45
TRAINING_DATA_HEALTH_REPORT = os.path.join(WEB_DIR, "data", "training_data_health_report.json")


def latest_training_data_guard_summary():
    try:
        with open(TRAINING_DATA_HEALTH_REPORT, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        return {
            "status": report.get("status"),
            "created_at": report.get("created_at"),
            "summary": report.get("summary") or {},
        }
    except Exception:
        return None

# ===========================================================================
# API ROUTES  (must come BEFORE the catch-all static route)
# ===========================================================================

@app.route('/api/status')
def pipeline_status():
    """Quick health-check — frontend polls this every 2 s while a run is live."""
    return jsonify({
        "is_running": is_running,
        "training_data_guard": latest_training_data_guard_summary(),
    })


@app.route('/api/launch-pipeline', methods=['POST'])
def launch_pipeline():
    """
    Accepts { startDate, endDate } from the frontend Pipeline Control UI.
    Starts the orchestrator in a background thread (instant response).
    Client polls /api/status to track progress.
    """
    global is_running
    if is_running:
        return jsonify({"status": "busy",
                        "message": "Pipeline already running — check progress bar."}), 409

    body       = request.get_json(silent=True) or {}
    start_date = body.get('startDate', '')
    end_date   = body.get('endDate',   '')
    sweep_mode = body.get('sweepMode') or body.get('mode') or os.environ.get('SWEEP_MODE', 'hybrid')
    allow_mock = body.get('allowMockFallback')
    enable_browser = body.get('enableBrowserScraping')

    run_id = body.get('runId') or datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]

    def _run():
        global is_running
        is_running = True
        try:
            env = os.environ.copy()
            if start_date: env['PIPELINE_START_DATE'] = start_date
            if end_date:   env['PIPELINE_END_DATE']   = end_date
            env['PIPELINE_RUN_ID'] = run_id
            env['SWEEP_MODE'] = str(sweep_mode).strip().lower()
            if allow_mock is not None:
                env['ALLOW_MOCK_FALLBACK'] = '1' if bool(allow_mock) else '0'
            browser_enabled = enable_browser
            if browser_enabled is None:
                browser_enabled = str(sweep_mode).strip().lower() in {'hybrid', 'external'}
            env['ENABLE_BROWSER_SCRAPING'] = '1' if bool(browser_enabled) else '0'
            if bool(browser_enabled):
                env.setdefault('BROWSER_ENGINE', 'auto')
                env.setdefault('BROWSER_AUTO_PREFER', 'seleniumbase')
                env.setdefault('BROWSER_HEADLESS', '1')
                env.setdefault('PLAYWRIGHT_USE_BUNDLED', '0')
                env.setdefault('SELENIUM_USE_BINARY_LOCATION', '0')
            env.setdefault('OPENBLAS_NUM_THREADS', '1')
            env.setdefault('OMP_NUM_THREADS', '1')
            env.setdefault('MKL_NUM_THREADS', '1')
            env.setdefault('NUMEXPR_NUM_THREADS', '1')

            logging.info(f">>> Launch-Pipeline | run_id={run_id} start={start_date} end={end_date} mode={sweep_mode} browser={browser_enabled}")
            result = subprocess.run(
                [MICROMAMBA, "run", "-p", "./venv39", "python", ORCHESTRATOR],
                cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=PIPELINE_TIMEOUT_SECONDS
            )
            if result.returncode != 0:
                logging.error("Pipeline failed (run_id=%s): %s", run_id, result.stderr[-1000:])
                run_log_path = os.path.join(PROJECT_ROOT, "web", "data", f"pipeline_error_{run_id}.log")
                with open(run_log_path, "w", encoding="utf-8") as f:
                    f.write(result.stderr or "")
                    if result.stdout:
                        f.write("\n\n--- STDOUT ---\n")
                        f.write(result.stdout)
        except subprocess.TimeoutExpired:
            logging.error("Pipeline timed out after %ss (run_id=%s)", PIPELINE_TIMEOUT_SECONDS, run_id)
        except Exception as exc:
            logging.error(f"Pipeline error: {exc}")
        finally:
            is_running = False

    threading.Thread(target=_run, daemon=True).start()
    label = f"{start_date or 'today'} → {end_date or 'today'}"
    return jsonify({"status": "launched",
                    "run_id": run_id,
                    "mode": sweep_mode,
                    "message": f"Pipeline started ({label}, mode={sweep_mode})"}), 200


@app.route('/api/refresh')
def refresh_data():
    """On-demand synchronous refresh (used by auto-run on page load)."""
    global is_running
    if is_running:
        return jsonify({"status": "busy", "message": "Pipeline already running..."})

    with pipeline_lock:
        is_running = True
        logging.info(">>> On-Demand Refresh triggered")
        try:
            result = subprocess.run(
                [MICROMAMBA, "run", "-p", "./venv39", "python", ORCHESTRATOR],
                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=PIPELINE_TIMEOUT_SECONDS
            )
            is_running = False
            if result.returncode == 0:
                return jsonify({"status": "success",
                                "message": "Pipeline updated successfully"})
            else:
                return jsonify({"status": "error",
                                "message": result.stderr}), 500
        except Exception as exc:
            is_running = False
            return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/api/purge-history', methods=['POST'])
def purge_history():
    """Purge all historical data and start fresh"""
    try:
        import glob
        import shutil

        # Get data directory path
        data_dir = os.path.join(WEB_DIR, 'data')

        # Remove all JSON files except tickets.json and update_log.json
        if os.path.exists(data_dir):
            for file_path in glob.glob(os.path.join(data_dir, '*.json')):
                filename = os.path.basename(file_path)
                if filename not in ['tickets.json', 'update_log.json']:
                    os.remove(file_path)

        # Create a new empty tickets.json
        tickets_path = os.path.join(data_dir, 'tickets.json')
        if os.path.exists(tickets_path):
            os.remove(tickets_path)

        # Create empty tickets file
        with open(tickets_path, 'w') as f:
            json.dump({"TYPE_A_ULTRA_SAFE": [], "TYPE_B_BALANCED": [], "TYPE_C_VALUE": [], "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f)

        # Clear update log but keep it
        log_path = os.path.join(data_dir, 'update_log.json')
        if os.path.exists(log_path):
            with open(log_path, 'w') as f:
                json.dump([{
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "Success",
                    "message": "History purged successfully"
                }], f, indent=4)

        # Remove sweep.log
        sweep_log_path = os.path.join(data_dir, 'sweep.log')
        if os.path.exists(sweep_log_path):
            os.remove(sweep_log_path)

        logging.info(">>> History purged successfully")
        return jsonify({"status": "success", "message": "History purged successfully"}), 200
    except Exception as e:
        logging.error(f"Error purging history: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===========================================================================
# STATIC FILE ROUTES  (catch-all — MUST be last)
# ===========================================================================

@app.route('/')
def index():
    return send_from_directory(WEB_DIR, 'index.html')


@app.route('/data/<path:filename>')
def data_files(filename):
    data_dir = os.path.join(WEB_DIR, 'data')
    try:
        response = send_from_directory(data_dir, filename)
        # DISABLE CACHING FOR INTELLIGENCE FILES
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as exc:
        logging.error(f"Data file not found: {filename} — {exc}")
        return jsonify({"error": "File not found"}), 404


@app.route('/<path:path>')
def static_files(path):
    # Guard: never let static catch-all swallow API requests
    if path.startswith('api/'):
        abort(404)
    try:
        return send_from_directory(WEB_DIR, path)
    except Exception:
        return send_from_directory(WEB_DIR, 'index.html')


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == '__main__':
    print("=" * 52)
    print("  ANTIGRAVITY V5.4 INTELLIGENCE SERVER LIVE")
    print("  Dashboard -> http://localhost:5000")
    print("=" * 52)
    app.run(host='0.0.0.0', port=5000, debug=False)
