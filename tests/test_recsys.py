from __future__ import annotations

import json
from importlib.util import find_spec
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from recsys.bpr import BPRConfig
from recsys.data import RATING_DTYPE, load_movies, load_ratings, temporal_split
from recsys.evaluation import evaluate, ranking_metrics
from recsys.experiment import load_model, run
from recsys.models import ItemCF, Popularity, TrainingData

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = ROOT / "examples" / "synthetic"


def training_rows():
    return np.array([
        (1, 10, 5, 1), (1, 20, 4, 2), (1, 40, 1, 3),
        (2, 10, 5, 4), (2, 30, 5, 5),
        (3, 20, 5, 6), (3, 30, 4, 7),
    ], dtype=RATING_DTYPE)


class DataTests(unittest.TestCase):
    def test_global_split_never_splits_timestamp_group(self):
        rows = np.array([(i + 1, i + 1, 5, t) for i, t in enumerate([3, 1, 1, 2, 3, 4, 4])], dtype=RATING_DTYPE)
        train, validation, test = temporal_split(rows)
        self.assertEqual(len(train) + len(validation) + len(test), len(rows))
        self.assertLess(train["timestamp"].max(), validation["timestamp"].min())
        self.assertLess(validation["timestamp"].max(), test["timestamp"].min())
        self.assertEqual(set(train["timestamp"]) & set(validation["timestamp"]), set())

    def test_split_rejects_impossible_windows(self):
        with self.assertRaises(ValueError):
            temporal_split(np.array([(1, 1, 5, 1)], dtype=RATING_DTYPE))
        with self.assertRaises(ValueError):
            temporal_split(training_rows(), 0.9, 0.2)

    def test_duplicate_pairs_rejected_instead_of_using_future_rating(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ratings.dat"
            path.write_text("1::2::1::10\n1::2::5::20\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_ratings(path)

    def test_malformed_and_nonfinite_ratings_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ratings.dat"
            for text in ["1::2::nan::10", "1::2::6::10", "1::2::5", "0::2::5::10"]:
                path.write_text(text + "\n", encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_ratings(path)

    def test_movielens_movie_titles_use_latin1(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "movies.dat"
            path.write_bytes("1::Amélie::Comedy\n".encode("latin-1"))
            self.assertEqual(load_movies(path)[1], "Amélie")


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.data = TrainingData.from_rows(training_rows())

    def test_itemcf_binary_cosine_matches_hand_computation(self):
        model = ItemCF(self.data, neighbors=4).fit()
        # Each positive item has two users; pairs have one co-occurrence: 1/sqrt(2*2).
        first = np.zeros(4)
        first[model.neighbor_indices[0]] = model.neighbor_weights[0]
        np.testing.assert_allclose(first, [0, 0.5, 0.5, 0])
        self.assertEqual(model.recommend(1, 10), [30])

    def test_low_ratings_are_seen_but_not_positive(self):
        self.assertIn(40, self.data.seen[1])
        self.assertNotIn(40, self.data.positives[1])
        self.assertEqual(Popularity(self.data).recommend(1), [30])

    def test_cold_start_uses_stable_popularity_order(self):
        model = ItemCF(self.data).fit()
        self.assertEqual(model.recommend(999, 3), [10, 20, 30])
        self.assertEqual(model.recommend(999, 3), Popularity(self.data).recommend(999, 3))

    def test_unknown_seen_id_does_not_exclude_next_catalog_item(self):
        self.assertEqual(Popularity(self.data).recommend(999, 3, seen={25, 999}), [10, 20, 30])

    def test_all_seen_produces_empty_recommendations(self):
        self.assertEqual(Popularity(self.data).recommend(1, 10, seen={10, 20, 30, 40}), [])


class EvaluationTests(unittest.TestCase):
    def test_ndcg_and_recall_match_hand_computation(self):
        recall, ndcg = ranking_metrics([2, 9, 1], {1, 2}, 3)
        self.assertEqual(recall, 1)
        self.assertAlmostEqual(ndcg, 1.5 / (1 + 1 / math.log2(3)))
        self.assertEqual(ranking_metrics([], {1, 2}, 10), (0, 0))
        with self.assertRaises(ValueError):
            ranking_metrics([1, 1], {1}, 10)

    def test_cold_and_unreachable_positives_remain_in_macro_denominator(self):
        rows = training_rows()
        model = Popularity(TrainingData.from_rows(rows))
        labels = np.array([(1, 30, 5, 10), (99, 999, 5, 11)], dtype=RATING_DTYPE)
        result = evaluate(model, labels, rows, k=10)
        self.assertEqual(result["users"], 2)
        self.assertEqual(result["positive_targets"], 2)
        self.assertEqual(result["unknown_item_targets"], 1)
        self.assertEqual(result["reachable_target_fraction"], 0.5)
        self.assertEqual(result["recall@10"], 0.5)
        self.assertEqual(result["cold_users"]["recall@10"], 0)

    def test_test_time_filter_includes_validation_low_ratings(self):
        rows = training_rows()
        observed = np.concatenate([rows, np.array([(1, 30, 1, 8)], dtype=RATING_DTYPE)])
        labels = np.array([(1, 30, 5, 9)], dtype=RATING_DTYPE)
        result = evaluate(Popularity(TrainingData.from_rows(rows)), labels, observed)
        self.assertEqual(result["previously_seen_targets"], 1)
        self.assertEqual(result["recall@10"], 0)

    def test_empty_positive_window_has_null_metrics(self):
        rows = training_rows()
        labels = np.array([(1, 999, 1, 10)], dtype=RATING_DTYPE)
        result = evaluate(Popularity(TrainingData.from_rows(rows)), labels, rows)
        self.assertEqual(result["users"], 0)
        self.assertEqual(result["users_without_positive_labels"], 1)
        self.assertIsNone(result["recall@10"])
        json.dumps(result, allow_nan=False)


class PipelineTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("torch"), "BPR training requires optional PyTorch")
    def test_bpr_selects_validation_checkpoint_and_restores_recommendations(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "bpr"
            result = run(SYNTHETIC, output, k=3, synthetic=True, include_test=True,
                         bpr_config=BPRConfig(epochs=3, factors=4, batch_size=8, seed=7))
            training = result["bpr_training"]
            best = max(training["epochs"], key=lambda epoch: epoch["ndcg@3"])
            self.assertEqual(training["selected_epoch"], best["epoch"])

            train, validation, _ = temporal_split(load_ratings(SYNTHETIC / "ratings.dat"))
            bpr, _ = load_model(output, "bpr")
            popularity, _ = load_model(output, "popularity")
            metrics = evaluate(bpr, validation, train, k=3)
            self.assertEqual(metrics["ndcg@3"], best["ndcg@3"])
            self.assertTrue(set(bpr.recommend(1, 3)).isdisjoint(bpr.data.seen[1]))
            self.assertEqual(bpr.recommend(999999, 3), popularity.recommend(999999, 3))
            self.assertIn("bpr", result["test"])

    def test_pipeline_saves_loadable_models_without_evaluating_test_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            result = run(SYNTHETIC, output, neighbors=3, k=3, synthetic=True)
            self.assertNotIn("test", result)
            self.assertEqual(result["dataset"]["kind"], "synthetic_pipeline_check")
            rows = load_ratings(SYNTHETIC / "ratings.dat")
            train, validation, _ = temporal_split(rows)
            context = TrainingData.from_rows(train)
            for name, expected in [("popularity", Popularity(context)), ("itemcf", ItemCF(context, 3).fit())]:
                restored, titles = load_model(output, name)
                self.assertEqual(restored.recommend(1, 3), expected.recommend(1, 3))
                self.assertEqual(set(titles), set(context.item_ids))
                np.testing.assert_array_equal(restored.data.item_ids, np.unique(train["item"]))
            self.assertNotIn(11, context.item_ids)
            with self.assertRaises(FileExistsError):
                run(SYNTHETIC, output)

    def test_explicit_final_evaluation_and_cli_recommendation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            result = run(SYNTHETIC, output, k=3, include_test=True, synthetic=True)
            self.assertIn("test", result)
            command = [sys.executable, "-m", "recsys", "recommend", "--run-dir", str(output),
                       "--model", "itemcf", "--user-id", "1", "--k", "3"]
            response = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
            payload = json.loads(response.stdout)
            self.assertEqual(payload["model"], "itemcf")
            self.assertFalse(payload["new_user"])
            self.assertEqual(payload["recommendations"][0]["movie_id"], 3)


if __name__ == "__main__":
    unittest.main()
