"""Small, artificial examples for learning the recommendation code."""


def explain_filter(seen_ids: list[int]) -> None:
    # Scores are given for this lesson; no real model is trained or evaluated.
    scores = {10: 0.8, 20: 0.6, 30: 0.9, 40: 0.2}
    seen = set(seen_ids)
    candidates = {}

    print("过滤练习：4 部电影的人造小例子，分数已经给定。")
    print(f"已经评价过的电影编号：{sorted(seen)}")
    for movie_id, score in scores.items():
        if movie_id not in seen:
            candidates[movie_id] = score
            print(f"电影 {movie_id}：不在已评价名单中，保留，分数 {score}")
        else:
            print(f"电影 {movie_id}：在已评价名单中，跳过")

    print(f"过滤后留下的电影及分数：{candidates}")
