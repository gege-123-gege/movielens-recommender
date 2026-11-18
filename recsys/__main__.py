"""Usage: python -m recsys --help"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .bpr import BPRConfig
from .data import download
from .experiment import load_model, run


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens recommendation with Popularity, ItemCF and optional BPR")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="download MovieLens 1M from GroupLens")
    fetch.add_argument("--output", type=Path, default=Path("data"))
    fetch.add_argument("--archive", type=Path, help="extract a previously downloaded official ml-1m.zip")
    for command in ("run", "demo"):
        experiment = commands.add_parser(command, help="run real-data baselines" if command == "run" else "check pipeline on synthetic data")
        if command == "run":
            experiment.add_argument("--data-dir", type=Path, default=Path("data/ml-1m"))
        experiment.add_argument("--output", type=Path, default=Path("outputs") / ("demo" if command == "demo" else "baseline"))
        experiment.add_argument("--neighbors", type=int, default=50)
        experiment.add_argument("--min-common-users", type=int, default=1,
                                help="minimum number of common positive users per ItemCF movie pair")
        experiment.add_argument("--k", type=int, default=10)
        experiment.add_argument("--threshold", type=float, default=4)
        experiment.add_argument("--bpr-epochs", type=int, help="train BPR and select the epoch on validation")
        experiment.add_argument("--bpr-seed", type=int, default=7, help="seed for BPR initialization and sampling")
        experiment.add_argument("--include-test", action="store_true", help="final evaluation only, after validation-based configuration selection")
    predict = commands.add_parser("recommend", help="recommend using a saved training snapshot")
    predict.add_argument("--run-dir", type=Path, default=Path("outputs/baseline"))
    predict.add_argument("--model", choices=["popularity", "itemcf", "bpr"], default="itemcf")
    predict.add_argument("--user-id", type=int, required=True)
    predict.add_argument("--k", type=int, default=10)
    lesson = commands.add_parser("learn-filter", help="walk through filtering four example movies")
    lesson.add_argument("--seen", nargs="*", type=int, default=[10, 30], help="already-rated movie IDs; default: 10 30")
    commands.add_parser("practice-recall", help="check your own Recall implementation in exercises/recall_practice.py")
    commands.add_parser("practice-itemcf", help="check your own small ItemCF scoring function")
    ndcg_lesson = commands.add_parser("practice-ndcg", help="check your NDCG implementation or the combined recommendation exercise")
    ndcg_lesson.add_argument("--pipeline", action="store_true", help="check evaluate_neighbors instead of ndcg_at_k")
    data_lesson = commands.add_parser("practice-data", help="practice parsing ratings and building user histories")
    data_lesson.add_argument("--part", choices=["parse", "profiles", "real"], default="parse")
    commands.add_parser("practice-similarity", help="check your binary cosine similarity function")
    itemcf_lesson = commands.add_parser("learn-itemcf", help="verify a real similarity or inspect a validation miss")
    itemcf_lesson.add_argument("--part", choices=["pair", "miss"], default="pair")
    bpr_lesson = commands.add_parser("practice-bpr", help="practice BPR triplets and pairwise loss")
    bpr_lesson.add_argument("--part", choices=["sample", "loss", "real", "update", "batch"], default="sample")
    args = parser.parse_args()
    try:
        if args.command == "practice-bpr":
            from exercises.bpr_practice import run_practice
            if not run_practice(args.part):
                parser.exit(1)
        elif args.command == "learn-itemcf":
            from exercises.itemcf_analysis import run_analysis
            if not run_analysis(args.part):
                parser.exit(1)
        elif args.command == "practice-similarity":
            from exercises.similarity_practice import check_practice
            if not check_practice():
                parser.exit(1)
        elif args.command == "practice-data":
            from exercises.data_practice import check_parse, check_profiles, inspect_real_training
            if args.part == "parse":
                success = check_parse()
            elif args.part == "profiles":
                success = check_profiles()
            else:
                success = check_parse() and check_profiles() and inspect_real_training()
            if not success:
                parser.exit(1)
        elif args.command == "practice-ndcg":
            from exercises.ndcg_practice import check_ndcg, check_pipeline
            check = check_pipeline if args.pipeline else check_ndcg
            if not check():
                parser.exit(1)
        elif args.command == "practice-itemcf":
            from exercises.itemcf_practice import check_practice
            if not check_practice():
                parser.exit(1)
        elif args.command == "practice-recall":
            from exercises.recall_practice import check_practice
            if not check_practice():
                parser.exit(1)
        elif args.command == "learn-filter":
            from .learning import explain_filter
            explain_filter(args.seen)
        elif args.command == "download":
            print(f"Data ready at {download(args.output, args.archive).resolve()}")
        elif args.command in ("run", "demo"):
            synthetic_dir = Path(__file__).resolve().parent.parent / "examples" / "synthetic"
            data_dir = synthetic_dir if args.command == "demo" else args.data_dir
            bpr_config = None if args.bpr_epochs is None else BPRConfig(args.bpr_epochs, seed=args.bpr_seed)
            results = run(data_dir, args.output, neighbors=args.neighbors, k=args.k,
                          threshold=args.threshold, min_common_users=args.min_common_users,
                          include_test=args.include_test,
                          synthetic=data_dir.resolve() == synthetic_dir.resolve(), bpr_config=bpr_config)
            k = args.k
            for model, metrics in results["validation"].items():
                print(f"{model}: validation Recall@{k}={metrics[f'recall@{k}']}, NDCG@{k}={metrics[f'ndcg@{k}']}")
        else:
            model, titles = load_model(args.run_dir, args.model)
            start = time.perf_counter()
            items = model.recommend(args.user_id, args.k)
            latency_ms = (time.perf_counter() - start) * 1000
            history = sorted(model.data.positives.get(args.user_id, set()))
            payload = {
                "user_id": args.user_id, "model": model.name,
                "new_user": args.user_id not in model.data.seen,
                "profile": "training snapshot; filters all training-rated items",
                "positive_history_count": len(history),
                "positive_history_sample": [{"movie_id": item, "title": titles[item]} for item in history[:10]],
                "recommendations": [{"movie_id": item, "title": titles[item]} for item in items],
                "recommendation_latency_ms": latency_ms,
            }
            # ASCII escaping also works with Windows consoles using non-UTF-8 encodings.
            print(json.dumps(payload, indent=2, ensure_ascii=True))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
