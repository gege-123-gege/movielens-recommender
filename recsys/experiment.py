"""Reproducible baseline experiment, reports, and non-pickle model snapshots."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .bpr import BPR, BPRConfig, fit_bpr
from .data import describe, file_sha256, load_movies, load_ratings, temporal_split
from .evaluation import evaluate
from .models import ItemCF, Popularity, TrainingData


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run(data_dir: Path, output: Path, *, neighbors: int = 50, k: int = 10,
        threshold: float = 4, min_common_users: int = 1, include_test: bool = False,
        synthetic: bool = False, bpr_config: BPRConfig | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new --output to preserve previous experiments")
    if neighbors < 1 or k < 1 or not 0.5 <= threshold <= 5:
        raise ValueError("neighbors and k must be positive; threshold must be between 0.5 and 5")
    if min_common_users < 1:
        raise ValueError("min_common_users must be positive")
    print("Reading and validating ratings...", flush=True)
    rows = load_ratings(data_dir / "ratings.dat")
    titles = load_movies(data_dir / "movies.dat")
    train, validation, test = temporal_split(rows)
    print(f"Ratings: {len(rows):,}; train/validation/test: {len(train):,}/{len(validation):,}/{len(test):,}", flush=True)
    context = TrainingData.from_rows(train, threshold)
    print(f"Training ItemCF (catalog={len(context.item_ids):,}, neighbors={neighbors})...", flush=True)
    start = time.perf_counter()
    itemcf = ItemCF(context, neighbors, min_common_users=min_common_users).fit()
    fit_seconds = time.perf_counter() - start
    models = [Popularity(context), itemcf]
    results = {
        "schema_version": 1,
        "dataset": {
            "kind": "synthetic_pipeline_check" if synthetic else "external_movielens_1m_format",
            "ratings_sha256": file_sha256(data_dir / "ratings.dat"),
            "movies_sha256": file_sha256(data_dir / "movies.dat"),
            "summary": describe(rows),
        },
        "configuration": {"neighbors": neighbors, "k": k, "positive_threshold": threshold,
                          "min_common_users": min_common_users,
                          "train_fraction_target": 0.8, "validation_fraction_target": 0.1},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform(), "processor": platform.processor()},
        "protocol": {
            "split": "global chronological; timestamp groups stay together",
            "training_timestamp_max": int(train["timestamp"].max()),
            "validation_timestamp_min": int(validation["timestamp"].min()),
            "validation_timestamp_max": int(validation["timestamp"].max()),
            "test_timestamp_min": int(test["timestamp"].min()),
            "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
            "candidate_catalog": "all items with any training rating; no sampled candidates",
            "seen_filter": "all ratings before the evaluated window, including low ratings",
            "profile": "frozen training-only user positive history for both windows",
            "metric_average": "macro over all users with >=1 positive label, including cold users",
            "unreachable_targets": "retained in recall denominator and ideal DCG",
            "cold_user": "no training interaction; popularity fallback",
            "latency": "recommend() including scoring, seen filtering, and sorting; excludes model/data loading",
        },
        "itemcf_fit_seconds": fit_seconds,
        "validation": {},
    }
    source = data_dir / "source.json"
    if source.exists():
        results["dataset"]["source_manifest"] = json.loads(source.read_text(encoding="utf-8"))

    bpr = None
    if bpr_config is not None:
        start = time.perf_counter()
        bpr, training = fit_bpr(context, train, validation, threshold, k, bpr_config)
        training["fit_and_selection_seconds"] = time.perf_counter() - start
        results["configuration"]["bpr"] = asdict(bpr_config)
        results["bpr_training"] = training
        models.append(bpr)

    for model in models:
        print(f"Evaluating {model.name} on validation...", flush=True)
        results["validation"][model.name] = evaluate(model, validation, train, threshold, k)
    if include_test:
        # Frozen training model; validation interactions are used ONLY for seen filtering.
        before_test = np.concatenate([train, validation])
        results["test"] = {}
        for model in models:
            print(f"Evaluating {model.name} on test (final evaluation)...", flush=True)
            results["test"][model.name] = evaluate(model, test, before_test, threshold, k)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "train.npz", rows=train)
    np.savez_compressed(output / "itemcf.npz", indices=itemcf.neighbor_indices, weights=itemcf.neighbor_weights)
    if bpr is not None:
        np.savez_compressed(output / "bpr.npz", user_ids=bpr.user_ids,
                            user_factors=bpr.user_factors, item_factors=bpr.item_factors)
    catalog_titles = {str(item): titles.get(int(item), f"Movie {item}") for item in context.item_ids}
    write_json(output / "movies.json", catalog_titles)
    write_json(output / "results.json", results)
    (output / "report.md").write_text(render_report(results), encoding="utf-8")
    print(f"Saved report and model snapshots to {output.resolve()}", flush=True)
    return results


def render_report(results: dict) -> str:
    config = results["configuration"]
    k = config["k"]
    lines = ["# 推荐实验报告", "", f"数据标记：`{results['dataset']['kind']}`", ""]
    if results["dataset"]["kind"] == "synthetic_pipeline_check":
        lines += ["这是人造小样本，只验证流程；不能作为简历中的 MovieLens 实验结果。", ""]
    lines += [f"评分阈值为 {config['positive_threshold']}。ItemCF 邻居数为 {config['neighbors']}，"
              f"共同喜欢人数门槛为 {config.get('min_common_users', 1)}。", ""]
    if "bpr_training" in results:
        training = results["bpr_training"]
        lines += [f"BPR 使用 {config['bpr']['factors']} 维向量。按验证集 NDCG@{k} "
                  f"选定第 {training['selected_epoch']} 轮，训练曲线见 results.json。", ""]
    lines += ["| 数据区间 | 模型 | 用户数 | Recall@K | NDCG@K | 目录覆盖率 | 推荐平均耗时 ms |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]

    def fmt(value):
        return "N/A" if value is None else f"{value:.6f}"

    for split in ("validation", "test"):
        for model, metrics in results.get(split, {}).items():
            lines.append(f"| {split} | {model} | {metrics['users']} | {fmt(metrics[f'recall@{k}'])} | "
                         f"{fmt(metrics[f'ndcg@{k}'])} | {fmt(metrics['catalog_coverage'])} | "
                         f"{fmt(metrics['latency_ms_mean'])} |")
    lines += ["", f"表中 K={k}。所有正反馈用户参与宏平均；新电影等不可推荐正例也保留在指标分母中。", "",
              "冷启动分组、可达正例比例、样本数量、数据哈希、机器环境和详细协议见 results.json。", "",
              "模型只使用训练区间。测试时额外过滤验证区间已评价的电影，用户偏好向量保持训练快照。", "",
              "若未指定 --include-test，本次未计算测试集指标。请用验证集选定配置后再进行最终测试。", "",
              "## 分析记录", "", "- 与热门推荐相比，ItemCF 的效果如何？原因是什么？",
              "- warm_users 与 cold_users 的效果差异多大？",
              "- 不可达正例占比多大？会怎样影响结论？",
              "- 准备修改哪个因素？依据是什么？一次只修改一个因素。", ""]
    return "\n".join(lines)


def load_model(run_dir: Path, name: str) -> tuple[Popularity, dict[int, str]]:
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    config = results["configuration"]
    with np.load(run_dir / "train.npz", allow_pickle=False) as archive:
        context = TrainingData.from_rows(archive["rows"], config["positive_threshold"])
    if name == "popularity":
        model = Popularity(context)
    elif name == "itemcf":
        model = ItemCF(context, config["neighbors"], min_common_users=config.get("min_common_users", 1))
        with np.load(run_dir / "itemcf.npz", allow_pickle=False) as archive:
            model.neighbor_indices = archive["indices"]
            model.neighbor_weights = archive["weights"]
    elif name == "bpr":
        if "bpr" not in config:
            raise ValueError("this run has no BPR snapshot; train with --bpr-epochs first")
        with np.load(run_dir / "bpr.npz", allow_pickle=False) as archive:
            model = BPR(context, archive["user_ids"], archive["user_factors"], archive["item_factors"])
    else:
        raise ValueError(f"unknown model: {name}")
    titles = json.loads((run_dir / "movies.json").read_text(encoding="utf-8"))
    return model, {int(item): title for item, title in titles.items()}
