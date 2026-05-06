import json
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
REGISTRY_PATH = os.path.join(MODEL_DIR, "graph_registry.json")


def load_graph_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {"updated_at": "", "active_snapshot_id": "latest", "snapshots": {}}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"updated_at": "", "active_snapshot_id": "latest", "snapshots": {}}


def save_graph_registry(registry):
    os.makedirs(MODEL_DIR, exist_ok=True)
    registry["updated_at"] = datetime.utcnow().isoformat()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def get_active_graph_snapshot_id():
    registry = load_graph_registry()
    return registry.get("active_snapshot_id", "latest")


def register_graph_snapshot(snapshot_id, path):
    registry = load_graph_registry()
    registry.setdefault("snapshots", {})
    registry["snapshots"][snapshot_id] = {"path": path, "created_at": datetime.utcnow().isoformat()}
    registry["active_snapshot_id"] = snapshot_id
    save_graph_registry(registry)

