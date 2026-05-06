import argparse
import json
import os
import sys
import subprocess
from datetime import datetime

# Path resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from model_registry import set_champion_candidate, load_registry

MODEL_V11_DIR = os.path.join(PROJECT_ROOT, "model", "v11")

CORE_LEAGUES = ["PremierLeague", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]


def probability_quality_score(metrics):
    """Lower is better. Accuracy is diagnostic; calibration quality gates promotion."""
    if not metrics:
        return None
    log_loss = metrics.get("val_log_loss")
    if log_loss is None:
        log_loss = metrics.get("val_loss")
    if log_loss is None:
        return None
    brier = metrics.get("val_brier_1x2", 0.70)
    draw_recall = metrics.get("draw_recall", 0.0)
    return float(log_loss) + (0.35 * float(brier or 0.0)) + (0.03 * max(0.0, 0.10 - float(draw_recall or 0.0)))


def run_training_for_league(league, epochs, dropout=0.2, lr=3e-4, wd=1e-2):
    cmd = [
        sys.executable, os.path.join(SCRIPTS_DIR, "run_v11_hybrid.py"),
        "--league", league,
        "--epochs", str(epochs),
        "--dropout", str(dropout),
        "--lr", str(lr),
        "--wd", str(wd)
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


def register_candidate_from_artifact(league, promote=False):
    artifact = os.path.join(MODEL_V11_DIR, f"{league}_training_artifact.json")
    if not os.path.exists(artifact):
        return False
    with open(artifact, "r", encoding="utf-8") as f:
        data = json.load(f)
    candidate_path = data.get("model_path")
    if not candidate_path:
        return False
    registry = load_registry()
    champion = registry.get("leagues", {}).get(league, {}).get("v11_hybrid", {}).get("champion_model_path")
    
    # Simple gate: don't promote if metrics are worse than champion (basic evaluate_candidate.py logic)
    metrics = data.get("metrics", {})
    active_path = candidate_path if promote else (champion or candidate_path)
    set_champion_candidate(
        league=league,
        family="v11_hybrid",
        champion_path=candidate_path if promote else champion,
        candidate_path=candidate_path,
        active_path=active_path,
        metrics=metrics,
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    args = parser.parse_args()

    leagues = [args.league] if args.league else CORE_LEAGUES
    run_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "epochs": args.epochs,
        "mode": "sweep" if args.sweep else "standard",
        "leagues": {},
    }

    # Hyperparameter Grid for Deep Sweep (Phase 4)
    GRID = {
        "dropout": [0.1, 0.2, 0.3],
        "lr": [1e-4, 3e-4, 5e-4],
        "weight_decay": [1e-3, 1e-2]
    } if args.sweep else {"dropout": [0.2], "lr": [3e-4], "weight_decay": [1e-2]}

    for league in leagues:
        best_acc = 0
        best_quality = None
        best_cfg = None
        
        # Get Current Champion Accuracy for Gating
        registry = load_registry()
        champion_meta = registry.get("leagues", {}).get(league, {}).get("v11_hybrid", {})
        champion_metrics = champion_meta.get("last_metrics") or champion_meta.get("metrics") or {}
        champion_acc = champion_metrics.get("val_accuracy") or champion_metrics.get("val_1x2_acc") or 0
        champion_quality = probability_quality_score(champion_metrics)
        
        for d in GRID["dropout"]:
            for lr in GRID["lr"]:
                for wd in GRID["weight_decay"]:
                    print(f"  [Deep Sweep] Testing {league}: d={d}, lr={lr}, wd={wd}")
                    ok, out, err = run_training_for_league(league, args.epochs, d, lr, wd)
                    if ok:
                        artifact_path = os.path.join(MODEL_V11_DIR, f"{league}_training_artifact.json")
                        with open(artifact_path, "r") as f:
                            meta = json.load(f)
                        metrics = meta.get("metrics", {})
                        val_acc = metrics.get("val_accuracy", 0)
                        quality = probability_quality_score(metrics)
                        if quality is None:
                            quality = 99.0 - float(val_acc or 0)
                        if best_quality is None or quality < best_quality:
                            best_quality = quality
                            best_acc = val_acc
                            best_cfg = (d, lr, wd, out, err)
        
        if best_cfg:
            d, lr, wd, out, err = best_cfg
            # Promotion gate: probability quality first, accuracy second.
            acc_lift = (best_acc - champion_acc) / max(0.01, champion_acc)
            quality_lift = None
            if champion_quality is not None and best_quality is not None:
                quality_lift = champion_quality - best_quality
            print(f"  [Sweep Result] {league}: Best Acc={best_acc:.4f} (Acc Lift: {acc_lift:+.2%}, Quality Lift: {quality_lift})")
            
            if champion_quality is None or (quality_lift is not None and quality_lift > 0.002) or (champion_acc == 0 and best_acc > 0):
                print(f"  [Promotion] Lift confirmed. Promoting {league} to production.")
                registered = register_candidate_from_artifact(league, promote=True)
            else:
                print(f"  [Gate] Quality lift insufficient. Keeping current champion.")
                registered = False
                
            run_report["leagues"][league] = {
                "ok": True,
                "best_acc": best_acc,
                "acc_lift": acc_lift,
                "best_quality": best_quality,
                "quality_lift": quality_lift,
                "best_params": {"dropout": d, "lr": lr, "weight_decay": wd},
                "registered": registered,
            }
        else:
            run_report["leagues"][league] = {"ok": False, "error": "Sweep failed to produce valid artifacts"}

    report_path = os.path.join(MODEL_V11_DIR, "training_runner_report.json")
    os.makedirs(MODEL_V11_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)
    print(f"Training runner report: {report_path}")


if __name__ == "__main__":
    main()
