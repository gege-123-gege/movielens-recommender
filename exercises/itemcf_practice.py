"""小版 ItemCF：只修改 score_candidates()，再运行 run.ps1 practice-itemcf。"""


def score_candidates(history, neighbors, n):
    """返回字典：候选电影编号 -> 累加的推荐分数。

    history：用户喜欢过的电影，例如 ["A", "B"]，没有重复编号。
    neighbors：每部历史电影的相似邻居，已经按相似度从高到低排列。
    例如 {"A": [("C", 0.9), ("D", 0.8)]}。
    n：每部历史电影最多使用前几个相似邻居，为正整数。

    本练习假设已评价电影全部是喜欢的电影，所以 history 也作为排除名单。
    完整项目中，已评价名单还包含低评分电影，与正反馈历史分开保存。
    所有历史电影都能在 neighbors 中找到。
    """
    # 1. 创建空字典 scores。
    # 2. 外层循环：逐个取出 history 中的历史电影。
    # 3. 内层循环：从 neighbors[历史电影][:n] 取出候选编号和相似度。
    # 4. 候选编号如果不在 history 中，就把相似度累加到它的 scores 上。
    # 5. 全部循环结束后 return scores。
    scores = {}

    for history_movie in history:
        for movie_id, similarity in neighbors[history_movie][:n]:
            if movie_id not in history:
                scores[movie_id] = scores.get(movie_id, 0) + similarity

    return scores
# 以下是检查工具。相似度是教学用给定值，不读取真实数据。
def check_practice(solution=score_candidates):
    import math
    from numbers import Real

    neighbors = {
        "A": [("C", 0.9), ("D", 0.8)],
        "B": [("D", 0.7), ("C", 0.5)],
        "C": [("A", 0.9), ("D", 0.6)],
    }
    cases = [
        ("one_neighbor", ["A", "B"], 1, {"C": 0.9, "D": 0.7}),
        ("two_neighbors", ["A", "B"], 2, {"C": 1.4, "D": 1.5}),
        ("one_history_movie", ["B"], 2, {"D": 0.7, "C": 0.5}),
        ("exclude_history", ["A", "C"], 2, {"D": 1.4}),
    ]
    passed = 0
    for name, history, n, expected in cases:
        try:
            actual = solution(list(history), {key: list(value) for key, value in neighbors.items()}, n)
            if actual is None:
                print(f"PENDING {name}: function returned None; expected {expected}")
                continue
            correct = isinstance(actual, dict) and actual.keys() == expected.keys()
            correct = correct and all(
                isinstance(actual[item], Real)
                and math.isclose(actual[item], value, rel_tol=1e-9, abs_tol=1e-9)
                for item, value in expected.items()
            )
            if correct:
                passed += 1
                print(f"PASS {name}: scores={actual}")
                if name in ("one_neighbor", "two_neighbors"):
                    ranking = sorted(actual, key=lambda item: (-actual[item], item))
                    print(f"  neighbors={n}, ranking={ranking}")
            else:
                print(f"FAIL {name}: actual={ascii(actual)}, expected={expected}")
        except Exception as exc:
            print(f"FAIL {name}: {ascii(exc)}")
    print(f"Passed: {passed}/{len(cases)}")
    return passed == len(cases)
