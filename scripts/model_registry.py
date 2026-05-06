import json
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
REGISTRY_PATH = os.path.join(MODEL_DIR, "model_registry.json")


def _default_registry():
    return {
        "updated_at": datetime.utcnow().isoformat(),
        "leagues": {}
    }


def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return _default_registry()
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return _default_registry()


def save_registry(registry):
    os.makedirs(MODEL_DIR, exist_ok=True)
    registry["updated_at"] = datetime.utcnow().isoformat()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def get_active_model_path(league, family, fallback_path):
    """
    family examples: 'v11_hybrid', 'v9_backbone'
    """
    registry = load_registry()
    league_entry = registry.get("leagues", {}).get(league, {})
    family_entry = league_entry.get(family, {})
    active_path = family_entry.get("active_model_path")
    if active_path and os.path.exists(active_path):
        return active_path
    return fallback_path


def set_champion_candidate(league, family, champion_path=None, candidate_path=None, active_path=None, metrics=None):
    registry = load_registry()
    if "leagues" not in registry:
        registry["leagues"] = {}
    if league not in registry["leagues"]:
        registry["leagues"][league] = {}
    if family not in registry["leagues"][league]:
        registry["leagues"][league][family] = {}

    entry = registry["leagues"][league][family]
    if champion_path is not None:
        entry["champion_model_path"] = champion_path
    if candidate_path is not None:
        entry["candidate_model_path"] = candidate_path
    if active_path is not None:
        entry["active_model_path"] = active_path
    if metrics is not None:
        entry["last_metrics"] = metrics
    entry["updated_at"] = datetime.utcnow().isoformat()
    save_registry(registry)

