"""Walk-forward validation for the GNN residual layer."""

import argparse
import json
import os
import sys
from datetime import datetime
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from gnn_graph_layer import (
    LEAGUES,
    TeamGraphOutcomeModel,
    _best_market_graph_blend,
    _fit_temperature,
    _graph_tensors_from_history,
    _log_loss_from_probs,
    _brier_from_probs,
    build_graph_dataset,
)


REPORT_PATH = os.path.join(PROJECT_ROOT, "web", "data", "gnn_walk_forward_report.json")


def year_from_date(value):
    try:
        return int(str(value)[:4])
    except Exception:
        return None


def train_fold(dataset, train_idx, test_idx, epochs=80, learning_rate=0.01):
    train_history = dataset["history"].iloc[train_idx].copy()
    train_node_features, train_adjacency = _graph_tensors_from_history(train_history, dataset["teams"])

    node_features = torch.tensor(train_node_features, dtype=torch.float32)
    adjacency = torch.tensor(train_adjacency, dtype=torch.float32)
    match_index = torch.tensor(dataset["match_index"], dtype=torch.long)
    match_features = torch.tensor(dataset["match_features"], dtype=torch.float32)
    labels = torch.tensor(dataset["labels"], dtype=torch.long)

    torch.manual_seed(42)
    model = TeamGraphOutcomeModel(
        node_feature_dim=node_features.shape[1],
        match_feature_dim=match_features.shape[1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(node_features, adjacency, match_index[train_idx], match_features[train_idx])
        loss = F.cross_entropy(logits, labels[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(node_features, adjacency, match_index[test_idx], match_features[test_idx])
        temperature, _ = _fit_temperature(test_logits, labels[test_idx])
        test_probs = torch.softmax(test_logits / temperature, dim=1).detach().cpu().numpy()

    test_labels = dataset["labels"][test_idx]
    market_probs = dataset["match_features"][test_idx, :3]
    has_market = market_probs.sum(axis=1) > 0
    market_probs = market_probs[has_market]
    graph_probs = test_probs[has_market]
    market_labels = test_labels[has_market]

    graph_acc = float((test_probs.argmax(axis=1) == test_labels).mean())
    graph_log_loss = _log_loss_from_probs(test_probs, test_labels)
    graph_brier = _brier_from_probs(test_probs, test_labels)

    if len(market_labels):
        market_acc = float((market_probs.argmax(axis=1) == market_labels).mean())
        market_log_loss = _log_loss_from_probs(market_probs, market_labels)
        market_brier = _brier_from_probs(market_probs, market_labels)
        blend = _best_market_graph_blend(market_probs, graph_probs, market_labels)
    else:
        market_acc = 0.0
        market_log_loss = 0.0
        market_brier = 0.0
        blend = {"weight": 0.0, "accuracy": graph_acc, "log_loss": graph_log_loss, "brier": graph_brier}

    return {
        "train_examples": int(len(train_idx)),
        "test_examples": int(len(test_idx)),
        "market_coverage": round(float(has_market.mean()) if len(has_market) else 0.0, 4),
        "graph_accuracy": round(graph_acc, 4),
        "graph_log_loss": round(graph_log_loss, 4),
        "graph_brier": round(graph_brier, 4),
        "market_accuracy": round(market_acc, 4),
        "market_log_loss": round(market_log_loss, 4),
        "market_brier": round(market_brier, 4),
        "blend_weight": round(float(blend["weight"]), 4),
        "blend_accuracy": round(float(blend["accuracy"]), 4),
        "blend_log_loss": round(float(blend["log_loss"]), 4),
        "blend_brier": round(float(blend["brier"]), 4),
        "blend_accuracy_lift": round(float(blend["accuracy"] - market_acc), 4),
        "blend_logloss_lift": round(float(market_log_loss - blend["log_loss"]), 4),
        "temperature": round(float(temperature), 4),
    }


def validate_league(league, epochs=80, min_train=500, min_test=80):
    dataset = build_graph_dataset(league)
    if dataset is None:
        return {"league": league, "folds": [], "enabled": False, "reason": "no_dataset"}

    years = dataset["history"]["date"].map(year_from_date).to_numpy()
    unique_years = sorted(y for y in set(years.tolist()) if y)
    folds = []
    for year in unique_years:
        train_idx = np.where(years < year)[0]
        test_idx = np.where(years == year)[0]
        if len(train_idx) < min_train or len(test_idx) < min_test:
            continue
        fold = train_fold(dataset, train_idx, test_idx, epochs=epochs)
        fold["test_year"] = int(year)
        folds.append(fold)

    positive_logloss = sum(1 for f in folds if f["blend_logloss_lift"] > 0)
    positive_accuracy = sum(1 for f in folds if f["blend_accuracy_lift"] > 0)
    avg_logloss_lift = float(np.mean([f["blend_logloss_lift"] for f in folds])) if folds else 0.0
    avg_accuracy_lift = float(np.mean([f["blend_accuracy_lift"] for f in folds])) if folds else 0.0
    required_positive = ceil(len(folds) * 0.60) if folds else 0
    enabled = bool(
        len(folds) >= 3
        and positive_logloss >= required_positive
        and avg_logloss_lift >= 0.0015
    )

    return {
        "league": league,
        "enabled": enabled,
        "fold_count": len(folds),
        "positive_logloss_folds": positive_logloss,
        "positive_accuracy_folds": positive_accuracy,
        "required_positive_logloss_folds": required_positive,
        "avg_blend_logloss_lift": round(avg_logloss_lift, 4),
        "avg_blend_accuracy_lift": round(avg_accuracy_lift, 4),
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="ALL")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    leagues = LEAGUES if args.league == "ALL" else [args.league]
    results = [validate_league(league, epochs=args.epochs) for league in leagues]
    report = {"created_at": datetime.utcnow().isoformat(), "results": results}
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
