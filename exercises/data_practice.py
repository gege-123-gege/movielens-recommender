"""数据处理练习：只修改前两个函数，检查工具保留。"""


def parse_rating(line):
    """把一行 user::item::rating::timestamp 转为四元组。

    例如 "1::1193::5::978300760\n" -> (1, 1193, 5.0, 978300760)。
    类型依次为 int、int、float、int。保证输入合法，本练习先不处理坏行。
    """
    # 1. line.strip() 去掉首尾空白，再用 .split("::") 拆成四个字符串。
    # 2. 用 int(...) 或 float(...) 转为对应数字。
    # 3. 返回 (user_id, movie_id, rating, timestamp)。
    fields = line.strip().split("::")
    user_id = int(fields[0])
    movie_id = int(fields[1])
    rating = float(fields[2])
    timestamp = int(fields[3])
    return user_id, movie_id, rating, timestamp

def build_profiles(rows, threshold=4):
    """返回 (seen, positives)，均为 用户编号 -> 电影编号集合 的字典。

    rows 中每条记录都是 (user_id, movie_id, rating, timestamp)。
    seen 收集全部已评分电影；positives 只收集 rating >= threshold 的电影。
    没有正反馈的用户不放入 positives 字典。空输入返回两个空字典。
    只处理传入的记录；调用方负责事先切分数据。
    """
    # 1. 创建 seen = {} 和 positives = {}。
    # 2. 用 for user_id, movie_id, rating, timestamp in rows: 逐条读取。
    # 3. 用户首次出现在 seen 时，先给他创建空集合 set()。
    # 4. 用 seen[user_id].add(movie_id) 收集全部评分电影。
    # 5. 如果 rating >= threshold，再在 positives 中创建集合并添加电影。
    # 6. 全部记录处理完后 return seen, positives。
    seen = {}
    positives = {}
    for user_id, movie_id, rating, timestamp in rows:
        if user_id not in seen:
            seen[user_id] = set()
        seen[user_id].add(movie_id)

        if rating >= threshold:
            if user_id not in positives:
                positives[user_id] = set()
            positives[user_id].add(movie_id)

    return seen, positives

def check_parse(solution=parse_rating):
    cases = [
        ("normal", "1::1193::5::978300760\n", (1, 1193, 5.0, 978300760)),
        ("low_rating", "2::10::2::100", (2, 10, 2.0, 100)),
        ("whitespace", " 3::20::4::200\r\n", (3, 20, 4.0, 200)),
    ]
    passed = 0
    for name, line, expected in cases:
        try:
            actual = solution(line)
            if actual is None:
                print(f"PENDING {name}: implement parse_rating")
                continue
            correct = isinstance(actual, tuple) and actual == expected
            correct = correct and tuple(type(value) for value in actual) == (int, int, float, int)
            if correct:
                passed += 1
                print(f"PASS {name}: {actual}")
            else:
                print(f"FAIL {name}: actual={ascii(actual)}, expected={expected} with numeric types")
        except Exception as exc:
            print(f"FAIL {name}: {ascii(exc)}")
    print(f"Passed: {passed}/{len(cases)}")
    return passed == len(cases)


def check_profiles(solution=build_profiles):
    rows = [(1, 10, 5.0, 100), (1, 20, 2.0, 110),
            (1, 30, 4.0, 120), (2, 20, 3.0, 130)]
    cases = [
        ("mixed_ratings", rows, 4, ({1: {10, 20, 30}, 2: {20}}, {1: {10, 30}})),
        ("threshold_five", rows, 5, ({1: {10, 20, 30}, 2: {20}}, {1: {10}})),
        ("empty", [], 4, ({}, {})),
        ("all_low", [(3, 40, 2.0, 140)], 4, ({3: {40}}, {})),
    ]
    passed = 0
    for name, records, threshold, expected in cases:
        try:
            actual = solution(list(records), threshold)
            if actual is None:
                print(f"PENDING {name}: implement build_profiles")
                continue
            correct = isinstance(actual, tuple) and actual == expected
            correct = correct and all(isinstance(group, dict) for group in actual)
            correct = correct and all(type(items) is set for group in actual for items in group.values())
            if correct:
                passed += 1
                print(f"PASS {name}: seen={actual[0]}, positives={actual[1]}")
            else:
                print(f"FAIL {name}: actual={ascii(actual)}, expected={expected}")
        except Exception as exc:
            print(f"FAIL {name}: {ascii(exc)}")
    print(f"Passed: {passed}/{len(cases)}")
    return passed == len(cases)


def inspect_real_training(parser=parse_rating, builder=build_profiles):
    """使用已有训练快照，不把验证或测试记录混入用户历史。"""
    from pathlib import Path
    import numpy as np
    from recsys.data import histories

    root = Path(__file__).resolve().parents[1]
    with (root / "data/ml-1m/ratings.dat").open(encoding="utf-8") as source:
        for _ in range(3):
            print(f"RAW SAMPLE: {parser(next(source))}")
    # 上面的三行只用于展示解析，不会加入下面的训练历史。
    with np.load(root / "outputs/neighbors100/train.npz", allow_pickle=False) as archive:
        train = archive["rows"]
    records = ((int(row["user"]), int(row["item"]), float(row["rating"]), int(row["timestamp"]))
               for row in train)
    seen, positives = builder(records, 4)
    if seen != histories(train) or positives != histories(train, 4):
        print("FAIL real_training: profiles differ from the project implementation")
        return False
    print(f"PASS real_training: rows={len(train)}, users={len(seen)}, positive_users={len(positives)}")
    user_id = 6040
    rated = seen.get(user_id, set())
    liked = positives.get(user_id, set())
    print(f"user={user_id}: seen={len(rated)}, positives={len(liked)}, below_threshold={len(rated - liked)}")
    print(f"seen_sample={sorted(rated)[:10]}")
    print(f"positive_sample={sorted(liked)[:10]}")
    print(f"below_threshold_sample={sorted(rated - liked)[:10]}")
    return True
