"""今日练习：先实现 NDCG，再把小版 ItemCF 与评估串起来。"""

import math

from exercises.itemcf_practice import score_candidates
from exercises.recall_practice import recall_at_k


def ndcg_at_k(recommendations, targets, k):
    """返回二值相关性的 NDCG@K。

    recommendations 是有序推荐列表；targets 是验证集中喜欢的电影集合。
    保证 k 为正整数、targets 非空、推荐编号不重复。
    推荐列表允许不足 k 个，理想列表长度仍取 min(k, len(targets))。
    """
    # 1. dcg 从 0 开始，遍历 recommendations[:k]。
    # 2. 用 enumerate(..., start=1) 同时取得名次 position 和 movie_id。
    # 3. 命中 targets 时，给 dcg 加上 1 / math.log2(position + 1)。
    # 4. idcg 为理想排列得分：前 min(k, len(targets)) 个位置全部命中。
    # 5. return dcg / idcg。
    dcg = 0
    for position, movie_id in enumerate(recommendations[:k], start=1):
        if movie_id in targets:
            dcg += 1 / math.log2(position + 1)

    ideal_count = min(k, len(targets))
    idcg = 0
    for position in range(1, ideal_count + 1):
        idcg += 1 / math.log2(position + 1)

    return dcg / idcg

def evaluate_neighbors(history, neighbors, targets, n, k):
    """组合昨天和今天的函数，返回推荐列表、Recall 和 NDCG。

    返回格式：{"recommendations": [...], "recall": 数值, "ndcg": 数值}。
    history 与 neighbors 沿用昨天的输入约定；k、targets 同上。
    """
    # 1. 调用 score_candidates(history, neighbors, n) 得到分数字典。
    # 2. 按分数从高到低排序，同分时按电影编号从小到大，取前 k 个编号。
    # 3. 分别调用 recall_at_k 和 ndcg_at_k，使用验证集 targets 作为答案。
    # 4. 按上面的返回格式组织字典。
    scores = score_candidates(history, neighbors, n)
    ranking = sorted(scores, key=lambda movie_id: (-scores[movie_id], movie_id))
    recommendations = ranking[:k]
    recall = recall_at_k(recommendations, targets, k)
    ndcg = ndcg_at_k(recommendations, targets, k)
    return {
        "recommendations": recommendations,
        "recall": recall,
        "ndcg": ndcg,
    }

def check_ndcg(solution=ndcg_at_k):
    """只检查指标函数；期望值来自手算示例。"""
    cases = [
        ("ideal_order", [1, 2, 9], {1, 2}, 3, 1.0),
        ("hit_at_second", [9, 1, 8], {1}, 3, 0.6309297535714575),
        ("hit_at_third", [9, 8, 1], {1}, 3, 0.5),
        ("no_hits", [8, 9], {1, 2}, 2, 0.0),
        ("outside_top_k", [8, 9, 1], {1}, 2, 0.0),
        ("more_targets_than_k", [1, 9, 2], {1, 2, 3}, 2, 0.6131471927654584),
        ("short_recommendations", [1], {1, 2}, 3, 0.6131471927654584),
    ]
    passed = 0
    for name, recommendations, targets, k, expected in cases:
        try:
            actual = solution(list(recommendations), set(targets), k)
            if actual is None:
                print(f"PENDING {name}: replace pass with your implementation")
            elif _same_number(actual, expected):
                passed += 1
                print(f"PASS {name}: {actual:.6f}")
            else:
                print(f"FAIL {name}: actual={ascii(actual)}, expected={expected:.6f}")
        except Exception as exc:
            print(f"FAIL {name}: {ascii(exc)}")
    print(f"Passed: {passed}/{len(cases)}")
    return passed == len(cases)


def check_pipeline(solution=evaluate_neighbors):
    """同一份历史和验证答案，仅改变邻居数 n 或推荐长度 k。"""
    neighbors = {
        "A": [("C", 0.9), ("D", 0.8)],
        "B": [("D", 0.7), ("C", 0.5)],
    }
    cases = [
        (1, 1, ["C"], 0.0, 0.0),
        (2, 1, ["D"], 1.0, 1.0),
        (1, 2, ["C", "D"], 1.0, 0.6309297535714575),
        (2, 2, ["D", "C"], 1.0, 1.0),
    ]
    passed = 0
    for n, k, ranking, recall, ndcg in cases:
        name = f"neighbors={n}, k={k}"
        try:
            actual = solution(["A", "B"], {key: list(value) for key, value in neighbors.items()}, {"D"}, n, k)
            if actual is None:
                print(f"PENDING {name}: replace pass with your implementation")
                continue
            correct = isinstance(actual, dict) and actual.get("recommendations") == ranking
            correct = correct and _same_number(actual.get("recall"), recall)
            correct = correct and _same_number(actual.get("ndcg"), ndcg)
            if correct:
                passed += 1
                print(f"PASS {name}: ranking={ranking}, Recall={recall:.6f}, NDCG={ndcg:.6f}")
            else:
                print(f"FAIL {name}: actual={ascii(actual)}")
                print(f"  expected: ranking={ranking}, Recall={recall:.6f}, NDCG={ndcg:.6f}")
        except Exception as exc:
            print(f"FAIL {name}: {ascii(exc)}")
    print(f"Passed: {passed}/{len(cases)}")
    return passed == len(cases)


def _same_number(actual, expected):
    from numbers import Real

    return isinstance(actual, Real) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
