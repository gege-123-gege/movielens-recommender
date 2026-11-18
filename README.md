# 个性化电影推荐系统

在 MovieLens 1M 上比较热门推荐、ItemCF 和 PyTorch BPR 矩阵分解。三种模型使用同一时间切分、训练电影目录及 Recall@10 和 NDCG@10 评估，可从保存的模型直接演示推荐。

目标是能解释数据、算法和实验结论。完整路线见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。

完整结果见 [最终实验报告](FINAL_RESULTS.md)，演示流程、简历条目与面试问答见 [项目展示指南](docs/05_portfolio.md)。初始实验见 [BASELINE_RESULTS.md](BASELINE_RESULTS.md)。

已有最终快照时，直接在项目目录运行下面的命令。把 `--model` 换成 `bpr` 或 `popularity` 可换模型。

```powershell
.\run.ps1 recommend --run-dir outputs/final_20261116 --model itemcf --user-id 6040 --k 5
```

## 环境与启动

Python 3.10+ 和 NumPy 2.x 可运行基线与快照推理。训练 BPR 需要 PyTorch。本次使用 Python 3.12.14、NumPy 2.3.5 和 CPU 版 PyTorch 2.9.1。以下命令在项目根目录执行。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m recsys --help
```

Windows 的 `python` 如果指向商店占位程序，请使用已安装的解释器完整路径创建 `.venv`。`run.ps1` 优先使用项目虚拟环境，其次查找 PATH 和 Codex 自带的解释器。

## 先检查流程

```powershell
.\run.ps1 demo --bpr-epochs 3 --output outputs/demo_bpr
.\run.ps1 recommend --run-dir outputs/demo_bpr --model bpr --user-id 1 --k 3
```

`examples/synthetic` 是用来检查流程的人造小样本。省略 `--bpr-epochs` 时运行热门推荐和 ItemCF 两个基线。

## 真实数据实验

使用 GroupLens 发布的固定版本 [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)，官方说明包含 1,000,209 条评分、6,040 名用户。数据字段和使用条件见[官方 README](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)。下载器记录来源及 SHA-256，数据和运行产物不提交到代码仓库。

```powershell
.\.venv\Scripts\python.exe -m recsys download
.\.venv\Scripts\python.exe -m recsys run --data-dir data/ml-1m --neighbors 50 --output outputs/baseline
.\.venv\Scripts\python.exe -m recsys recommend --run-dir outputs/baseline --user-id 6040 --model itemcf
```

也可以从官方页面手动下载并解压，使目录包含 `data/ml-1m/ratings.dat` 和 `movies.dat`。

若本机 Python 缺少可用的证书链，可以使用 Windows 的系统证书校验下载，再导入本地官方压缩包：

```powershell
New-Item -ItemType Directory -Path data -Force
Invoke-WebRequest -Uri 'https://files.grouplens.org/datasets/movielens/ml-1m.zip' -OutFile 'data/ml-1m.zip'
.\run.ps1 download --archive data/ml-1m.zip
```

每次实验指定一个新输出目录，保留历史结果。调参只看验证集，例如：

```powershell
.\.venv\Scripts\python.exe -m recsys run --neighbors 20 --output outputs/neighbors20
.\.venv\Scripts\python.exe -m recsys run --neighbors 100 --output outputs/neighbors100
```

本机已有上述实验。重新运行时可换一个输出名，例如 `outputs/neighbors20_repeat`。下面的命令读取先前的验证结果，检查数据和协议是否一致，再汇总对比。

```powershell
.\.venv\Scripts\python.exe scripts/compare_neighbors.py --runs outputs/neighbors20 outputs/baseline outputs/neighbors100 --output experiments/itemcf_neighbors_repeat
```

既有报告位于 `experiments/itemcf_neighbors`，其中 `comparison.json` 保存选择规则、数据哈希、各候选指标及选中的训练快照路径。

最终实验使用 `neighbors=100`、共同喜欢人数门槛 1 和 `K=10`。BPR 在第 1～12 轮中按验证集 NDCG@10 选取快照（同分取较早一轮），随后对三个冻结模型做最终测试。已存在同名目录时，复现请换一个输出名。

```powershell
.\run.ps1 run --neighbors 100 --min-common-users 1 --k 10 --bpr-epochs 12 --bpr-seed 7 --include-test --output outputs/final_reproduction
.\run.ps1 recommend --run-dir outputs/final_reproduction --model bpr --user-id 6040 --k 5
```

用第二个随机种子复查验证指标波动。

```powershell
.\run.ps1 run --neighbors 100 --min-common-users 1 --k 10 --bpr-epochs 12 --bpr-seed 11 --output outputs/bpr_seed11_reproduction
```

## 输出内容

- `report.md`：模型指标对比及主要配置。
- `results.json`：配置、数据统计、哈希、环境、分组指标、时间边界及详细评估协议。
- `train.npz`、`itemcf.npz`：训练快照与相似电影邻居，可直接恢复推荐；读取禁用 pickle。
- `bpr.npz`：已选轮次的用户与电影向量（启用 BPR 时保存），可用 NumPy 加载推理。
- `movies.json`：训练候选目录对应的电影名称。

推荐命令使用保存的训练时点快照；历史展示是按电影编号排序的最多 10 个正反馈样例，不表示最近观看。

## 算法和评估约定

评分不低于 4 分视为正反馈。热门推荐按训练用户正反馈人数排序。ItemCF 使用二值正反馈的余弦相似度：两部电影共同喜欢用户数除以各自喜欢用户数乘积的平方根。每部电影保存最多 `neighbors` 个相似邻居，用户推荐分数为其历史正反馈电影邻居相似度之和。

已评价电影全部过滤，包括低评分电影。相似度分数为零时按热门程度补位；没有训练正反馈的新用户或用户采用热门兜底；同分时按电影编号稳定排序。

BPR 对每个训练正例从该用户未评分的训练电影中均匀采一个对照项，每轮重新采样。模型给同一用户的正例和对照项做点积打分，最小化 `softplus(对照分数 - 正例分数)` 的批平均加 L2 正则。使用 Adam 更新用户与电影向量。无可训练向量的用户采用热门兜底。算法来源见 [BPR 原论文](https://proceedings.mlr.press/r7/rendle09a.html)。

时间划分目标为 80% / 10% / 10%，同一时间戳不会跨区间，因此实际行数比例可能有偏移。候选目录为训练区间出现过的全部电影，评估全量候选，不采样负例。

验证时过滤训练历史。测试时过滤训练和验证历史，但使用冻结的训练模型和用户偏好，不重新拟合。指标对所有有正反馈标签的用户做宏平均，包括训练中未出现的新用户；新电影等不可推荐正例仍保留在 Recall 和理想 DCG 分母中。没有正反馈标签的用户不参与这两个指标，数量单独记录。冷用户指没有任何训练交互，暖用户中的“仅有低评分历史”也会用热门兜底。

报告 `Recall@10`、`NDCG@10`、目录覆盖率、推荐平均及 P95 耗时，并分开记录冷／暖用户与不可达正例。耗时包含单次推荐的计算、过滤和排序，排除模型加载，没有做严格硬件基准测试。

MovieLens 是评分数据，未评分不等于不喜欢；离线结果不能推导线上点击率提升。全局时间切分可能产生较多新用户，这也是需要分析的实验现象。

当前 ItemCF 为便于理解采用稠密共现矩阵，仅支持最多 6,000 部训练电影，训练内存随电影数平方增长；面向更大目录时应替换为稀疏实现。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖手算指标、相似度、时间戳隔离、冷启动、低评分历史过滤、不可达正例计分、BPR 轮次选择及模型保存恢复。缺少 PyTorch 时跳过 BPR 训练测试。

## 从哪里开始学

已理解 Recall 后，按 [亲手实现与实验任务](docs/02_recall_practice.md) 完成 `exercises/recall_practice.py`，使用 `.\run.ps1 practice-recall` 检查自己的实现。

初学时可先运行逐行展示的小练习，观察 `for`、`if` 和 `not in` 的执行结果：

```powershell
.\run.ps1 learn-filter
.\run.ps1 learn-filter --seen 10 20 30
```

第一条把 10、30 视为已评价，留下 20、40；第二条增加了已评价的电影 20，因此只留下 40。练习使用 4 部人造电影和给定分数，不读取或修改真实实验产物。对应代码为 `recsys/learning.py`。

1. 先运行小样本，打开报告，找到数据量、时间边界、两个模型的验证指标。
2. 阅读 `recsys/data.py`，解释为何不能随机划分后直接使用全量数据统计相似度。
3. 阅读 `recsys/models.py`，用三名用户、三部电影手算一组相似度。
4. 阅读 `recsys/evaluation.py`，手算一个推荐列表的 Recall 与 NDCG。
5. 在真实数据上只改变邻居数，对比验证指标和目录覆盖率，填写分析记录。

跑 `.\run.ps1 practice-bpr --part update` 或 `.\run.ps1 practice-bpr --part batch`，可以复现单步更新与固定批次损失的练习。正式训练入口是 `run --bpr-epochs`，每轮重新采样训练对照项。
