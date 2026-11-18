"""Train a BPR matrix factorization model and score from its frozen factors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluation import evaluate
from .models import Popularity, TrainingData


@dataclass(frozen=True)
class BPRConfig:
    epochs: int
    factors: int = 32
    batch_size: int = 2048
    learning_rate: float = 0.01
    regularization: float = 0.0001
    seed: int = 7

    def __post_init__(self):
        if min(self.epochs, self.factors, self.batch_size) < 1:
            raise ValueError("BPR epochs, factors and batch size must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("BPR learning rate must be finite and positive")
        if not np.isfinite(self.regularization) or self.regularization < 0 or self.seed < 0:
            raise ValueError("BPR regularization and seed must be nonnegative")


class BPR(Popularity):
    name = "bpr"

    def __init__(self, data: TrainingData, user_ids: np.ndarray,
                 user_factors: np.ndarray, item_factors: np.ndarray):
        super().__init__(data)
        self.user_ids = user_ids
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.user_index = {int(user): index for index, user in enumerate(user_ids)}

    def scores(self, user: int) -> np.ndarray:
        index = self.user_index.get(user)
        if index is None:
            return super().scores(user)
        return np.einsum("ij,j->i", self.item_factors, self.user_factors[index])


def fit_bpr(data: TrainingData, train: np.ndarray, validation: np.ndarray,
            threshold: float, k: int, config: BPRConfig) -> tuple[BPR, dict]:
    """Train on one sampled negative per positive per epoch. Select on validation NDCG."""
    try:
        import torch
        from torch.nn.functional import softplus
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise ValueError("BPR training requires PyTorch; see README.md for CPU setup") from exc

    catalog = data.item_ids
    users = sorted(user for user in data.positives if len(data.seen[user]) < len(catalog))
    if not users:
        raise ValueError("BPR requires a user with both a training positive and an unrated catalog item")

    # Each user occupies a contiguous slice before shuffling. Movie IDs and
    # user IDs are mapped to positions in their respective embedding tables.
    positive_groups = [np.searchsorted(catalog, sorted(data.positives[user])) for user in users]
    rated_groups = [np.searchsorted(catalog, sorted(data.seen[user])) for user in users]
    offsets = np.cumsum([0] + [len(group) for group in positive_groups])
    user_rows = torch.from_numpy(np.repeat(np.arange(len(users), dtype=np.int64), np.diff(offsets)))
    positive_rows = torch.from_numpy(np.concatenate(positive_groups).astype(np.int64))
    user_ids = np.asarray(users, dtype=np.int64)

    rng = np.random.default_rng(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    initial_users = 0.1 * torch.randn((len(users), config.factors), generator=generator)
    initial_items = 0.1 * torch.randn((len(catalog), config.factors), generator=generator)
    user_table = torch.nn.Embedding.from_pretrained(initial_users, freeze=False)
    item_table = torch.nn.Embedding.from_pretrained(initial_items, freeze=False)
    optimizer = torch.optim.Adam([user_table.weight, item_table.weight], lr=config.learning_rate)
    history = []
    best_model = None
    best_ndcg = None
    best_epoch = 1
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for epoch in range(1, config.epochs + 1):
            negative_ids = np.empty(len(positive_rows), dtype=np.int64)
            # This is the practice sampler's same unrated pool, reused for
            # all of one user's positives. Resample it every epoch.
            unrated = np.ones(len(catalog), dtype=bool)
            for index, rated in enumerate(rated_groups):
                unrated[rated] = False
                candidates = np.flatnonzero(unrated)
                start, end = offsets[index:index + 2]
                negative_ids[start:end] = rng.choice(candidates, size=end - start)
                unrated[rated] = True
            negative_rows = torch.from_numpy(negative_ids)

            order = torch.from_numpy(rng.permutation(len(positive_rows)))
            loss_total = 0.0
            for batch in order.split(config.batch_size):
                user_vector = user_table(user_rows[batch])
                positive_vector = item_table(positive_rows[batch])
                negative_vector = item_table(negative_rows[batch])
                gap = (user_vector * (positive_vector - negative_vector)).sum(dim=1)
                ranking_loss = softplus(-gap).mean()
                squared_norms = (user_vector.square() + positive_vector.square()
                                 + negative_vector.square()).sum(dim=1)
                loss = ranking_loss + config.regularization * squared_norms.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_total += float(ranking_loss.detach()) * len(batch)

            mean_loss = loss_total / len(positive_rows)
            if not np.isfinite(mean_loss):
                raise ValueError(f"BPR training loss became nonfinite at epoch {epoch}")
            snapshot = BPR(data, user_ids, user_table.weight.detach().numpy().copy(),
                           item_table.weight.detach().numpy().copy())
            metrics = evaluate(snapshot, validation, train, threshold, k)
            ndcg = metrics[f"ndcg@{k}"]
            history.append({"epoch": epoch, "train_ranking_loss": mean_loss,
                            f"recall@{k}": metrics[f"recall@{k}"], f"ndcg@{k}": ndcg})
            score = "N/A" if ndcg is None else f"{ndcg:.6f}"
            print(f"BPR epoch {epoch}/{config.epochs}: loss={mean_loss:.6f}, validation NDCG@{k}={score}",
                  flush=True)
            if best_model is None or (ndcg is not None and (best_ndcg is None or ndcg > best_ndcg)):
                best_model, best_ndcg, best_epoch = snapshot, ndcg, epoch
    finally:
        torch.set_num_threads(previous_threads)

    return best_model, {
        "selected_epoch": best_epoch,
        "selection_rule": f"highest validation NDCG@{k}; ties prefer the earliest epoch",
        "trained_users": len(users),
        "triplets_per_epoch": len(positive_rows),
        "sampling": "one uniformly sampled unrated training-catalog item per training positive per epoch",
        "optimizer": "Adam",
        "cpu_threads": 1,
        "torch": str(torch.__version__),
        "epochs": history,
    }
