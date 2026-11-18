"""BPR 练习，包括采样、打分和训练更新。"""

import math
from pathlib import Path
from random import Random

import numpy as np


def sample_triplet(user_id, liked_movies, rated_movies, catalog, rng):
    """返回 (用户编号, 正例电影编号, 对照电影编号)，无可用样本时返回 None。

    liked_movies 是训练正反馈集合，rated_movies 是所有训练已评分电影的集合。
    catalog 是训练电影目录中的编号列表，rng 提供 choice 方法。
    """
    # 1. 正例从 liked_movies 里选。没有喜欢的电影就返回 None。
    # 2. 遍历 catalog，把不在 rated_movies 里的电影加入对照候选列表。
    # 3. 对照候选为空时返回 None。
    # 4. rng.choice(sorted(liked_movies)) 选正例，rng.choice(候选列表) 选对照。
    # 5. 返回 (user_id, positive_id, negative_id)。
    candidates = []
    for movie_id in catalog:
        if movie_id not in rated_movies:
            candidates.append(movie_id)
    if not liked_movies or not candidates:
        return None
    positive_id = rng.choice(sorted(liked_movies))
    negative_id = rng.choice(candidates)
    return user_id, positive_id, negative_id

def score_and_loss(user_vector, positive_vector, negative_vector):
    """返回 (正例分数, 对照分数, 单条样本的 BPR 排序损失)。"""
    # 1. np.dot(user_vector, positive_vector) 计算正例分数。
    # 2. 用相同方法计算对照分数。
    # 3. np.logaddexp(0.0, negative_score - positive_score) 计算排序损失。
    # 4. 按顺序返回三个值。
    positive_score = np.dot(user_vector, positive_vector)
    negative_score = np.dot(user_vector, negative_vector)
    loss = np.logaddexp(0.0, negative_score - positive_score)
    return positive_score, negative_score, loss

def run_practice(part):
    if part == "update":
        return show_torch_update()
    if part == "batch":
        return show_batch_training()
    if part == "sample":
        cases = [
            ("exclude_low_rating", (7, {10}, {10, 20}, [10, 20, 30], Random(7)), (7, 10, 30)),
            ("no_positive", (7, set(), {20}, [10, 20, 30], Random(7)), None),
            ("no_negative", (7, {10}, {10, 20, 30}, [10, 20, 30], Random(7)), None),
        ]
        return check_cases(sample_triplet, cases)
    if part == "loss":
        cases = [
            ("positive_ahead", ([1, 2], [1, 1], [2, 0]), (3, 2, 0.3132616875)),
            ("negative_ahead", ([1, 2], [2, 0], [1, 1]), (2, 3, 1.3132616875)),
            ("tie", ([1, 2], [1, 0], [1, 0]), (1, 1, math.log(2))),
        ]
        return check_cases(score_and_loss, cases)
    return show_real_triplet()


def check_cases(function, cases):
    passed = 0
    for name, args, expected in cases:
        try:
            actual = function(*args)
            if expected is None:
                matches = actual is None
            else:
                matches = (isinstance(actual, tuple) and len(actual) == len(expected)
                           and np.allclose(actual, expected, rtol=1e-7, atol=1e-9))
            if matches:
                passed += 1
                print(f"PASS {name}: {actual}")
            elif actual is None:
                print(f"PENDING {name}: expected {expected}")
            else:
                print(f"FAIL {name}: got {actual}, expected {expected}")
        except Exception as exc:
            print(f"FAIL {name}: {ascii(exc)}")
    print(f"Passed: {passed}/{len(cases)}")
    return passed == len(cases)


def show_real_triplet():
    from recsys.experiment import load_model

    root = Path(__file__).resolve().parents[1]
    model, titles = load_model(root / "outputs/neighbors100", "popularity")
    user_id = 6040
    liked = model.data.positives[user_id]
    seen = model.data.seen[user_id]
    catalog = model.data.item_ids.tolist()
    triplet = sample_triplet(user_id, liked, seen, catalog, Random(7))
    if triplet is None:
        print("PENDING: complete sample_triplet before checking real data")
        return False

    sampled_user, positive, negative = triplet
    if sampled_user != user_id or positive not in liked or negative in seen or negative not in catalog:
        print(f"FAIL invalid triplet: {triplet}")
        return False
    print(f"PASS real_triplet: {triplet}")
    print(f"user={user_id}, train_rated={len(seen)}, train_positive={len(liked)}")
    print(f"positive={positive} {ascii(titles[positive])}")
    print(f"sampled_unrated={negative} {ascii(titles[negative])}")
    return True


def show_torch_update():
    """用一条人造样本演示 PyTorch 的一次参数更新。"""
    try:
        import torch
        from torch.nn import functional as F
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print("PyTorch is required for --part update. Install the CPU build in the project .venv.")
        return False

    user = torch.tensor([1.0, 2.0], requires_grad=True)
    positive = torch.tensor([2.0, 0.0], requires_grad=True)
    negative = torch.tensor([1.0, 1.0], requires_grad=True)
    learning_rate = 0.05
    optimizer = torch.optim.SGD([user, positive, negative], lr=learning_rate)

    def scores_and_loss():
        positive_score = torch.dot(user, positive)
        negative_score = torch.dot(user, negative)
        loss = F.softplus(negative_score - positive_score)
        return positive_score, negative_score, loss

    optimizer.zero_grad()
    positive_before, negative_before, loss_before = scores_and_loss()
    user_before = user.detach().tolist()
    loss_before.backward()
    optimizer.step()

    with torch.no_grad():
        positive_after, negative_after, loss_after = scores_and_loss()
    gap_before = (positive_before - negative_before).item()
    gap_after = (positive_after - negative_after).item()

    print(f"PyTorch {torch.__version__} | toy triplet | learning_rate={learning_rate}")
    print(f"before: positive={positive_before.item():.6f}, negative={negative_before.item():.6f}, "
          f"gap={gap_before:.6f}, loss={loss_before.item():.6f}")
    print(f"after:  positive={positive_after.item():.6f}, negative={negative_after.item():.6f}, "
          f"gap={gap_after:.6f}, loss={loss_after.item():.6f}")
    print(f"user_vector: {user_before} -> {[round(value, 6) for value in user.detach().tolist()]}")
    improved = gap_after > gap_before and loss_after.item() < loss_before.item()
    print("PASS one_update" if improved else "FAIL one_update")
    return improved


def update_batch(positive_scores, negative_scores, optimizer):
    """用一批样本更新一次向量，返回更新前的平均损失。"""
    from torch.nn import functional as F

    # 1. 对每对分数用 F.softplus(negative_scores - positive_scores) 算损失。
    # 2. 用 .mean() 把所有样本的损失合成一个数。
    # 3. 清掉上一次的梯度，反向传播，再调用优化器更新。
    # 4. 用 loss.item() 返回更新前的平均损失。
    loss = F.softplus(negative_scores - positive_scores).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def show_batch_training():
    """在固定的一小批真实训练样本上观察多次更新。"""
    try:
        import torch
        from torch.nn import functional as F
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print("PyTorch is required for --part batch. Install the CPU build in the project .venv.")
        return False

    from recsys.experiment import load_model

    root = Path(__file__).resolve().parents[1]
    model, _ = load_model(root / "outputs/neighbors100", "popularity")
    catalog = model.data.item_ids.tolist()
    movie_index = {movie: index for index, movie in enumerate(catalog)}
    user_ids = (973, 6040)
    rng = Random(7)
    triples = []
    for index, user_id in enumerate(user_ids):
        liked = model.data.positives[user_id]
        seen = model.data.seen[user_id]
        for _ in range(16):
            sample = sample_triplet(user_id, liked, seen, catalog, rng)
            if sample is None:
                print(f"FAIL: no triplet for user {user_id}")
                return False
            _, positive, negative = sample
            triples.append((index, movie_index[positive], movie_index[negative]))

    users, positives, negatives = torch.tensor(triples, dtype=torch.long).unbind(dim=1)
    torch.manual_seed(7)
    user_table = torch.nn.Embedding(len(user_ids), 8)
    movie_table = torch.nn.Embedding(len(catalog), 8)
    optimizer = torch.optim.SGD([user_table.weight, movie_table.weight], lr=0.2)

    def score_batch():
        user_vectors = user_table(users)
        positive_scores = (user_vectors * movie_table(positives)).sum(dim=1)
        negative_scores = (user_vectors * movie_table(negatives)).sum(dim=1)
        return positive_scores, negative_scores

    def measure_loss():
        with torch.no_grad():
            positive_scores, negative_scores = score_batch()
            return F.softplus(negative_scores - positive_scores).mean().item()

    print(f"fixed training batch: users={len(user_ids)}, triplets={len(triples)}, vector_size=8")
    start_loss = measure_loss()
    print(f"before training: mean_loss={start_loss:.6f}")
    for step in range(1, 11):
        positive_scores, negative_scores = score_batch()
        loss = update_batch(positive_scores, negative_scores, optimizer)
        if loss is None:
            print("PENDING: complete update_batch in exercises/bpr_practice.py")
            return False
        if step == 1 and not math.isclose(loss, start_loss, rel_tol=1e-6):
            print(f"FAIL: first batch mean loss should be {start_loss:.6f}, got {loss}")
            return False
        if step in (1, 5, 10):
            print(f"step={step:02d}: mean_loss_before_step={loss:.6f}")

    final_loss = measure_loss()
    print(f"after training:  mean_loss={final_loss:.6f}")
    improved = final_loss < start_loss
    print("PASS batch_training" if improved else "FAIL batch_training")
    return improved
