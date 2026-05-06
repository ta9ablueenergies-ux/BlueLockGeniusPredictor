"""Materialize graph snapshots for V7/V8 features."""

import argparse
import json
import os
from datetime import datetime

from graph_registry import register_graph_snapshot
from v7_graph_hub import V7GraphHub
from v8_gsn_architect import V8MotifDetector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "model", "graph_snapshots")

LEAGUES = ["PremierLeague", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]


def build_snapshot(leagues):
    now = datetime.utcnow()
    snapshot_id = now.strftime("%Y%m%d%H%M%S")
    payload = {
        "snapshot_id": snapshot_id,
        "created_at": now.isoformat(),
        "leagues": {},
    }
    for league in leagues:
        hub = V7GraphHub()
        detector = V8MotifDetector()
        v7_ok = hub.build_relational_graph(league=league)
        v8_ok = detector.build_directed_topology(league=league)
        if v8_ok:
            detector.detect_cyclic_dominance()
            detector.detect_giant_killers()
        payload["leagues"][league] = {
            "v7_ok": v7_ok,
            "v8_ok": v8_ok,
            "v7_top_centralities": sorted(hub.team_centralities().items(), key=lambda x: x[1], reverse=True)[:10] if v7_ok else [],
            "v8_triangle_count": len(detector.triangles) if v8_ok else 0,
            "v8_giant_killers": detector.giant_killers if v8_ok else {},
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{snapshot_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    register_graph_snapshot(snapshot_id, out_path)
    return snapshot_id, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default=None)
    args = parser.parse_args()
    leagues = [args.league] if args.league else LEAGUES
    snapshot_id, out_path = build_snapshot(leagues)
    print(json.dumps({"snapshot_id": snapshot_id, "path": out_path}))


if __name__ == "__main__":
    main()

