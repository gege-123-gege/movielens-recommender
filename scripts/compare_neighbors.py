"""Compare existing validation runs without training or accessing test metrics.

Run from the project root with a Python that can read the JSON result files.
Only the standard library is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ItemCF neighbors on a fixed validation protocol")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/itemcf_neighbors"))
    args = parser.parse_args()
    runs = [(path, json.loads((path / "results.json").read_text(encoding="utf-8"))) for path in args.runs]
    reference = runs[0][1]
    k = reference["configuration"]["k"]
    primary = f"ndcg@{k}"
    secondary = f"recall@{k}"
    expected_config = {key: value for key, value in reference["configuration"].items() if key != "neighbors"}
    expected_config.setdefault("min_common_users", 1)
    compared = []
    seen_neighbors = set()
    for path, result in runs:
        config = {key: value for key, value in result["configuration"].items() if key != "neighbors"}
        config.setdefault("min_common_users", 1)
        if config != expected_config or result["protocol"] != reference["protocol"]:
            parser.error(f"{path}: configurations or evaluation protocols differ")
        for field in ("ratings_sha256", "movies_sha256", "kind"):
            if result["dataset"][field] != reference["dataset"][field]:
                parser.error(f"{path}: dataset field {field} differs")
        neighbors = result["configuration"]["neighbors"]
        if neighbors in seen_neighbors:
            parser.error(f"duplicate neighbors setting: {neighbors}")
        seen_neighbors.add(neighbors)
        metrics = result["validation"]["itemcf"]
        if metrics[primary] is None or metrics[secondary] is None:
            parser.error(f"{path}: validation has no positive-label users")
        for metric in (primary, secondary, "users", "positive_targets", "reachable_targets"):
            if result["validation"]["popularity"][metric] != reference["validation"]["popularity"][metric]:
                parser.error(f"{path}: shared popularity baseline differs for {metric}")
        compared.append({"run_directory": path.as_posix(), "neighbors": neighbors, "validation": metrics})
    compared.sort(key=lambda row: row["neighbors"])
    # Select on the declared primary metric; exact ties prefer a smaller model.
    selected = max(compared, key=lambda row: (row["validation"][primary], -row["neighbors"]))
    payload = {
        "selection_rule": f"maximize validation {primary}; exact ties choose fewer neighbors",
        "dataset_sha256": reference["dataset"]["ratings_sha256"],
        "configuration_except_neighbors": expected_config,
        "selected_neighbors": selected["neighbors"],
        "selected_run_directory": selected["run_directory"],
        "test_metrics_used": False,
        "candidates": compared,
    }
    lines = ["# ItemCF 邻居数对比", "", f"主要选择指标：验证集 {primary}；完全同分时选择更少的邻居。", "",
             "已检查各轮数据哈希、除邻居数以外的配置、评估协议和共享热门推荐基线一致。", "",
             f"| 邻居数 | {secondary} | {primary} | 暖用户 {primary} | 冷用户 {primary} | 目录覆盖率 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for candidate in compared:
        metrics = candidate["validation"]
        def fmt(value):
            return "N/A" if value is None else f"{value:.6f}"
        lines.append(f"| {candidate['neighbors']} | {fmt(metrics[secondary])} | {fmt(metrics[primary])} | "
                     f"{fmt(metrics['warm_users'][primary])} | {fmt(metrics['cold_users'][primary])} | "
                     f"{fmt(metrics['catalog_coverage'])} |")
    lines += ["", f"本轮选定邻居数：**{selected['neighbors']}**。对应快照：`{selected['run_directory']}`。", "",
              "该结论仅表示在已比较设置中验证指标最高，不代表全局最优或统计显著差异。本轮没有使用测试指标。", "",
              "## 解释与后续", "",
              "- 增大邻居数会让更多相似电影参与打分，可能补充信号，也可能引入噪声；效果需要实验验证。",
              "- 冷用户没有训练历史，使用同一个热门兜底，因此这组结果不随邻居数变化。",
              "- 覆盖率是所有用户推荐列表的电影并集占训练目录的比例，和相关性指标不一定同时提高。",
              "- 本次固定推荐列表长度 K，改变的是每部历史电影参与计算的相似邻居数。",
              "- 下一阶段实现 PyTorch BPR，在相同协议下用验证集选配置；完成模型选择后再评估测试集。", "",
              "## 自己动手", "",
              "1. 解释 neighbors=100 和 K=10 的区别。",
              "2. 手算两个电影的二值余弦相似度，再对应 models.py 中的代码。",
              "3. 用选定快照和初始快照分别给用户 6040 推荐，记录列表变化。",
              "4. 用自己的话写三句话：改变了什么、怎么比较、结果有什么局限。", ""]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected neighbors={selected['neighbors']} by validation {primary}; report: {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
