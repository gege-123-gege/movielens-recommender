"""练习：用喜欢两部电影的用户集合，计算二值余弦相似度。"""

import math


def cosine_similarity(users_a, users_b):
    """users_a/users_b 分别为喜欢电影 A/B 的用户编号集合。

    公式：共同喜欢的人数 / sqrt(喜欢 A 的人数 * 喜欢 B 的人数)。
    任一集合为空则返回 0，避免除以零。输入不会被修改。
    这是通用相似度函数；正式模型会另外把电影与自身的相似度设为 0。
    """
    # 1. 如果任一集合为空，return 0.0。
    # 2. users_a & users_b 得到两个集合的交集。
    # 3. len(...) 统计人数，math.sqrt(...) 计算平方根。
    # 4. 按公式返回相似度。
    if len(users_a) == 0 or len(users_b) == 0:
        return 0.0
    users_ab = users_a & users_b
    similarity = len(users_ab) / math.sqrt(len(users_a) * len(users_b))
    return similarity

def check_practice(solution=cosine_similarity):
    from numbers import Real

    cases = [
        ("partial_overlap", {1, 2, 3}, {1, 2, 4}, 2 / 3),
        ("same_audience", {1, 2}, {1, 2}, 1.0),
        ("no_overlap", {1, 2}, {3, 4}, 0.0),
        ("popular_movie", {1, 2, 3}, set(range(1, 13)), 0.5),
        ("empty_left", set(), {1, 2}, 0.0),
        ("empty_right", {1, 2}, set(), 0.0),
    ]
    passed = 0
    for name, users_a, users_b, expected in cases:
        try:
            input_a, input_b = set(users_a), set(users_b)
            actual = solution(input_a, input_b)
            if actual is None:
                print(f"PENDING {name}: implement cosine_similarity")
                continue
            correct = isinstance(actual, Real) and math.isclose(actual, expected, abs_tol=1e-9, rel_tol=1e-9)
            correct = correct and input_a == users_a and input_b == users_b
            if correct:
                passed += 1
                print(f"PASS {name}: {actual:.6f}")
            else:
                print(f"FAIL {name}: actual={ascii(actual)}, expected={expected:.6f}; keep inputs unchanged")
        except Exception as exc:
            print(f"FAIL {name}: {ascii(exc)}")
    print(f"Passed: {passed}/{len(cases)}")
    return passed == len(cases)
