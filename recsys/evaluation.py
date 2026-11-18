"""Macro top-K metrics over all positive-label users, including cold starts."""

from __future__ import annotations

import math
import time

import numpy as np

from .data import histories
from .models import Popularity


def ranking_metrics(recommendations: list[int], targets: set[int], k: int) -> tuple[float, float]:
    if k < 1:
        raise ValueError("k must be positive")
    if len(set(recommendations)) != len(recommendations):
        raise ValueError("recommendations must be unique")
    if not targets:
        raise ValueError("ranking metrics require at least one positive target")
    hits = [int(item in targets) for item in recommendations[:k]]
    recall = sum(hits) / len(targets)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    idcg = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(targets))))
    return recall, dcg / idcg


def evaluate(model: Popularity, labels: np.ndarray, observed: np.ndarray,
             threshold: float = 4, k: int = 10) -> dict:
    targets = histories(labels, threshold)
    seen_by_user = histories(observed)
    catalog = set(int(item) for item in model.data.item_ids)
    groups: dict[str, list[tuple[float, float]]] = {"warm": [], "cold": []}
    recommended_items: set[int] = set()
    elapsed_ms = []
    total_targets = reachable_targets = unknown_item_targets = previously_seen_targets = 0
    for user, positives in sorted(targets.items()):
        seen = seen_by_user.get(user, set())
        total_targets += len(positives)
        reachable_targets += len((positives & catalog) - seen)
        unknown_item_targets += len(positives - catalog)
        previously_seen_targets += len(positives & seen)
        start = time.perf_counter()
        recommendations = model.recommend(user, k, seen=seen)
        elapsed_ms.append((time.perf_counter() - start) * 1000)
        recommended_items.update(recommendations)
        group = "warm" if user in model.data.seen else "cold"
        groups[group].append(ranking_metrics(recommendations, positives, k))

    def average(values):
        if not values:
            return {"users": 0, f"recall@{k}": None, f"ndcg@{k}": None}
        return {"users": len(values), f"recall@{k}": float(np.mean([v[0] for v in values])),
                f"ndcg@{k}": float(np.mean([v[1] for v in values]))}

    all_values = groups["warm"] + groups["cold"]
    return {
        **average(all_values),
        "warm_users": average(groups["warm"]), "cold_users": average(groups["cold"]),
        "users_without_positive_labels": len(np.unique(labels["user"])) - len(targets),
        "positive_targets": total_targets, "reachable_targets": reachable_targets,
        "unknown_item_targets": unknown_item_targets,
        "previously_seen_targets": previously_seen_targets,
        "reachable_target_fraction": reachable_targets / total_targets if total_targets else None,
        "catalog_coverage": len(recommended_items) / len(catalog) if catalog else 0.0,
        "latency_ms_mean": float(np.mean(elapsed_ms)) if elapsed_ms else None,
        "latency_ms_p95": float(np.percentile(elapsed_ms, 95)) if elapsed_ms else None,
    }
