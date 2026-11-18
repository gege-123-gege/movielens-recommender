"""Read MovieLens 1M and split whole timestamp groups chronologically."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np

RATING_DTYPE = np.dtype([
    ("user", "i8"), ("item", "i8"), ("rating", "f4"), ("timestamp", "i8"),
])
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(output: Path, archive_path: Path | None = None) -> Path:
    """Fetch the official archive; extract only fixed, expected file names."""
    import json

    target = output / "ml-1m"
    if target.exists():
        raise FileExistsError(f"{target} already exists; use it or choose another output directory")
    if archive_path is None:
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "movie-recsys-learning/0.1"})
        with urllib.request.urlopen(request, timeout=90) as response:
            archive_bytes = response.read()
    else:
        archive_bytes = archive_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = ["ratings.dat", "movies.dat", "README"]
        for name in names:
            archive.getinfo(f"ml-1m/{name}")
        target.mkdir(parents=True, exist_ok=False)
        for name in names:
            with archive.open(f"ml-1m/{name}") as source, (target / name).open("wb") as destination:
                shutil.copyfileobj(source, destination)
    manifest = {
        "url": DATA_URL,
        "retrieval": "urllib_https" if archive_path is None else "local_archive_expected_from_official_url",
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "files_sha256": {name: file_sha256(target / name) for name in names},
    }
    (target / "source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def load_ratings(path: Path) -> np.ndarray:
    def records():
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                fields = line.strip().split("::")
                if len(fields) != 4:
                    raise ValueError(f"{path}:{line_no}: expected user::item::rating::timestamp")
                try:
                    yield int(fields[0]), int(fields[1]), float(fields[2]), int(fields[3])
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid numeric field") from exc

    rows = np.fromiter(records(), dtype=RATING_DTYPE)
    if not len(rows):
        raise ValueError("ratings file is empty")
    if (np.any(rows["user"] <= 0) or np.any(rows["item"] <= 0)
            or np.any(rows["timestamp"] < 0)
            or np.any(~np.isfinite(rows["rating"]))
            or np.any((rows["rating"] < 0.5) | (rows["rating"] > 5))):
        raise ValueError("invalid user/item ID, timestamp, or rating (expected 0.5 through 5)")
    pairs = np.rec.fromarrays([rows["user"], rows["item"]], names="user,item")
    if len(np.unique(pairs)) != len(rows):
        raise ValueError("duplicate user-item ratings found; define a causal deduplication policy first")
    return rows[np.argsort(rows["timestamp"], kind="stable")]


def load_movies(path: Path) -> dict[int, str]:
    titles = {}
    # MovieLens 1M movie titles are Latin-1 encoded, not UTF-8.
    with path.open(encoding="latin-1") as source:
        for line in source:
            item, title, _genres = line.rstrip("\r\n").split("::", 2)
            titles[int(item)] = title
    return titles


def temporal_split(rows: np.ndarray, train_fraction: float = 0.8,
                   validation_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("train and validation fractions must be positive and sum to less than 1")
    rows = rows[np.argsort(rows["timestamp"], kind="stable")]
    times = np.unique(rows["timestamp"])
    if len(times) < 3:
        raise ValueError("at least three distinct timestamps are required")
    # Group boundaries can shift row ratios slightly; no timestamp is split.
    train_target = rows["timestamp"][int(len(rows) * train_fraction)]
    test_target = rows["timestamp"][int(len(rows) * (train_fraction + validation_fraction))]
    first = int(np.clip(np.searchsorted(times, train_target), 1, len(times) - 2))
    second = int(np.clip(np.searchsorted(times, test_target), first + 1, len(times) - 1))
    train_end = int(np.searchsorted(rows["timestamp"], times[first]))
    validation_end = int(np.searchsorted(rows["timestamp"], times[second]))
    return rows[:train_end], rows[train_end:validation_end], rows[validation_end:]


def histories(rows: np.ndarray, positive_threshold: float | None = None) -> dict[int, set[int]]:
    if positive_threshold is not None:
        rows = rows[rows["rating"] >= positive_threshold]
    result: dict[int, set[int]] = {}
    for user, item in zip(rows["user"], rows["item"]):
        result.setdefault(int(user), set()).add(int(item))
    return result


def describe(rows: np.ndarray) -> dict:
    users, user_counts = np.unique(rows["user"], return_counts=True)
    items, item_counts = np.unique(rows["item"], return_counts=True)
    scores, counts = np.unique(rows["rating"], return_counts=True)
    return {
        "ratings": len(rows), "users": len(users), "items": len(items),
        "interaction_density": len(rows) / (len(users) * len(items)),
        "timestamp_min": int(rows["timestamp"].min()),
        "timestamp_max": int(rows["timestamp"].max()),
        "rating_distribution": {str(float(s)): int(n) for s, n in zip(scores, counts)},
        "user_interactions_p50_p90_max": [float(v) for v in np.quantile(user_counts, [0.5, 0.9, 1])],
        "item_interactions_p50_p90_max": [float(v) for v in np.quantile(item_counts, [0.5, 0.9, 1])],
    }
