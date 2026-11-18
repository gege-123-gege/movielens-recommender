"""在同一份 ItemCF 快照上核对一对电影，再分析一个验证集漏推案例。"""

import json
import math
from pathlib import Path

import numpy as np

from exercises.similarity_practice import cosine_similarity
from recsys.data import load_ratings
from recsys.experiment import load_model


def users_who_liked(positives, movie_id):
    """返回喜欢指定电影的用户编号集合。

    positives 是你上次练过的字典，例如 {1: {10, 20}, 2: {20}}。
    传入电影 20 应返回 {1, 2}；传入电影 10 应返回 {1}。
    """
    # 1. 创建空集合 users = set()。
    # 2. 遍历 positives.items()，每次取出 user_id 和 liked_movies。
    # 3. 如果 movie_id 在 liked_movies 中，把 user_id 加入 users。
    # 4. 循环结束后返回 users。
    users = set()
    for user_id, liked_movies in positives.items():
        if movie_id in liked_movies:
            users.add(user_id)
    return users


def run_analysis(part):
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "outputs/neighbors100"
    model, titles = load_model(run_dir, "itemcf")
    if part == "pair":
        return compare_pair(model, titles)

    metadata = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    rows = load_ratings(root / "data/ml-1m/ratings.dat")
    return explain_miss(model, titles, rows, metadata)


def compare_pair(model, titles):
    movie_a, movie_b = 2571, 260
    users_a = users_who_liked(model.data.positives, movie_a)
    users_b = users_who_liked(model.data.positives, movie_b)
    if users_a is None or users_b is None:
        print("PENDING: implement users_who_liked in exercises/itemcf_analysis.py")
        return False
    if not isinstance(users_a, set) or not isinstance(users_b, set):
        print("FAIL: users_who_liked must return a set of user IDs")
        return False

    index_a, index_b = np.searchsorted(model.data.item_ids, [movie_a, movie_b])
    weights = model.neighbor_weights[index_a][model.neighbor_indices[index_a] == index_b]
    if len(weights) != 1:
        print("FAIL: this pair is missing from the saved top-100 neighbor list")
        return False

    calculated = cosine_similarity(users_a, users_b)
    saved = float(weights[0])
    print(f"A: {movie_a} {titles[movie_a]}, positive_users={len(users_a)}")
    print(f"B: {movie_b} {titles[movie_b]}, positive_users={len(users_b)}")
    print(f"shared_users={len(users_a & users_b)}, calculated={calculated:.9f}, saved={saved:.9f}")
    counts_match = (len(users_a) == model.data.popularity[index_a]
                    and len(users_b) == model.data.popularity[index_b])
    matches = counts_match and math.isclose(calculated, saved, rel_tol=1e-6)
    print("PASS real_pair" if matches else "FAIL real_pair")
    return matches


def explain_miss(model, titles, rows, metadata):
    user_id, movie_id = 973, 318
    protocol = metadata["protocol"]
    config = metadata["configuration"]
    k = config["k"]
    label = rows[(rows["user"] == user_id) & (rows["item"] == movie_id)
                 & (rows["timestamp"] >= protocol["validation_timestamp_min"])
                 & (rows["timestamp"] <= protocol["validation_timestamp_max"])
                 & (rows["rating"] >= config["positive_threshold"])]

    ids = model.data.item_ids
    ranked = model.recommend(user_id, k=len(ids))
    if len(label) != 1 or movie_id not in ranked or movie_id in ranked[:k]:
        print("FAIL: the selected validation miss could not be reproduced")
        return False

    scores = model.scores(user_id)
    target_index = np.searchsorted(ids, movie_id)
    boundary = ranked[k - 1]
    boundary_index = np.searchsorted(ids, boundary)
    print(f"user={user_id}, target={movie_id} {titles[movie_id]}, validation_rating={float(label['rating'][0]):g}")
    print(f"rank={ranked.index(movie_id) + 1}, target_score={scores[target_index]:.6f}")
    print(f"rank_{k}={boundary} {titles[boundary]}, cutoff_score={scores[boundary_index]:.6f}")

    contributions = []
    for history_id in sorted(model.data.positives[user_id]):
        index = np.searchsorted(ids, history_id)
        mask = model.neighbor_indices[index] == target_index
        weight = float(model.neighbor_weights[index][mask].sum())
        contributions.append(weight)
        print(f"  history={history_id} {titles[history_id]}, contribution={weight:.6f}")
    total = sum(contributions)
    print(f"contribution_sum={total:.6f}")
    matches = math.isclose(total, float(scores[target_index]), rel_tol=1e-6, abs_tol=1e-9)
    print("PASS validation_miss" if matches else "FAIL validation_miss")
    return matches
