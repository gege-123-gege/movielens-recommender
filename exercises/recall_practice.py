"""练习：自己实现 Recall，再执行 run.ps1 practice-recall 检查。"""


def recall_at_k(recommendations, targets, k):
    """返回前 k 个推荐的 Recall。

    recommendations：按推荐顺序排列的电影编号列表。
    targets：验证记录中，这个用户喜欢的电影编号集合。
    k：检查前几个推荐。

    本练习保证：k 为正整数、targets 非空、推荐列表没有重复编号。
    """
    # 1. 用 recommendations[:k] 取前 k 个推荐。
    # 2. 遍历这些编号，统计其中有几个在 targets 中。
    # 3. 用命中数量除以 len(targets)，并 return 结果。
    print("前k个推荐：", recommendations[:k])

    hit_count = 0

    for movie_id in recommendations[:k]:
        if movie_id in targets:
            hit_count += 1

    recall = hit_count / len(targets)

    return recall


# 以下是已经准备好的检查工具，练习时只修改上面的函数。
def check_practice(solution=recall_at_k):
    import math
    from numbers import Real

    cases = [
        ("top3", [1, 2, 3], {1, 3, 5, 7}, 3, 0.5),
        ("top2", [1, 2, 3], {1, 3, 5, 7}, 2, 0.25),
        ("no_hits", [2, 4], {1, 3}, 2, 0.0),
        ("all_hits", [1, 3, 5], {1, 3}, 2, 1.0),
    ]
    passed = 0
    for name, recommendations, targets, k, expected in cases:
        try:
            actual = solution(recommendations, targets, k)
        except Exception as exc:
            print(f"FAIL {name}: {ascii(exc)}")
            continue
        if actual is None:
            print(f"PENDING {name}: function returned None; expected {expected}")
        elif isinstance(actual, Real) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            print(f"PASS {name}: {actual}")
            passed += 1
        else:
            print(f"FAIL {name}: actual={ascii(actual)}, expected={expected}")
    print(f"Passed: {passed}/{len(cases)}")
    return passed == len(cases)
