"""Popularity and binary cosine ItemCF, using training data only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import histories


@dataclass
class TrainingData:
    item_ids: np.ndarray
    popularity: np.ndarray
    seen: dict[int, set[int]]
    positives: dict[int, set[int]]

    @classmethod
    def from_rows(cls, rows: np.ndarray, positive_threshold: float = 4.0) -> "TrainingData":
        item_ids = np.unique(rows["item"])
        positive_rows = rows[rows["rating"] >= positive_threshold]
        popularity = np.bincount(
            np.searchsorted(item_ids, positive_rows["item"]), minlength=len(item_ids),
        ).astype(np.float32)
        return cls(item_ids, popularity, histories(rows), histories(positive_rows))


class Popularity:
    name = "popularity"

    def __init__(self, data: TrainingData):
        self.data = data

    def scores(self, user: int) -> np.ndarray:
        return self.data.popularity.copy()

    def recommend(self, user: int, k: int = 10, seen: set[int] | None = None) -> list[int]:
        if k < 1:
            raise ValueError("k must be positive")
        scores = self.scores(user)
        excluded = self.data.seen.get(user, set()) if seen is None else seen
        if excluded:
            # Unknown IDs must not exclude the next known item after insertion.
            excluded_ids = np.array(sorted(excluded), dtype=np.int64)
            indices = np.searchsorted(self.data.item_ids, excluded_ids)
            in_bounds = indices < len(self.data.item_ids)
            indices, excluded_ids = indices[in_bounds], excluded_ids[in_bounds]
            known_indices = indices[self.data.item_ids[indices] == excluded_ids]
            scores[known_indices] = -np.inf
        # Similarity first, popularity fallback, then stable ascending movie ID.
        order = np.lexsort((self.data.item_ids, -self.data.popularity, -scores))
        order = order[np.isfinite(scores[order])][:k]
        return [int(item) for item in self.data.item_ids[order]]


class ItemCF(Popularity):
    name = "itemcf"

    def __init__(self, data: TrainingData, neighbors: int = 50, *, min_common_users: int = 1):
        super().__init__(data)
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        if min_common_users < 1:
            raise ValueError("min_common_users must be positive")
        self.neighbors = neighbors
        self.min_common_users = min_common_users
        self.neighbor_indices: np.ndarray | None = None
        self.neighbor_weights: np.ndarray | None = None

    def fit(self) -> "ItemCF":
        count = len(self.data.item_ids)
        if count > 6000:
            raise ValueError("this learning implementation supports up to 6000 items; use sparse ItemCF above that")
        similarity = np.zeros((count, count), dtype=np.float32)
        for positive_items in self.data.positives.values():
            indices = np.searchsorted(self.data.item_ids, sorted(positive_items))
            similarity[np.ix_(indices, indices)] += 1
        similarity[similarity < self.min_common_users] = 0
        norm = np.sqrt(np.maximum(self.data.popularity, 1))
        similarity /= norm[:, None]
        similarity /= norm[None, :]
        np.fill_diagonal(similarity, 0)
        width = min(self.neighbors, count)
        self.neighbor_indices = np.empty((count, width), dtype=np.int32)
        self.neighbor_weights = np.empty((count, width), dtype=np.float32)
        for index, row in enumerate(similarity):
            # Deterministic tie handling matters for reproducibility.
            order = np.lexsort((self.data.item_ids, -row))[:width]
            self.neighbor_indices[index] = order
            self.neighbor_weights[index] = row[order]
        return self

    def scores(self, user: int) -> np.ndarray:
        if self.neighbor_indices is None or self.neighbor_weights is None:
            raise RuntimeError("fit or load the ItemCF model before recommending")
        positive_items = self.data.positives.get(user, set())
        if not positive_items:
            return self.data.popularity.copy()
        indices = np.searchsorted(self.data.item_ids, sorted(positive_items))
        scores = np.zeros(len(self.data.item_ids), dtype=np.float32)
        np.add.at(scores, self.neighbor_indices[indices].ravel(), self.neighbor_weights[indices].ravel())
        return scores
