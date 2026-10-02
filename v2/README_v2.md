# Semi-vast v2：数据设计修正 + 可检验的消融实验

v1 消融实验中 A / B / C 三组几乎没有差别（测试集只差 3 个样本）。原因主要有三点：

1. 伪标签数据（只有新闻标题、没有证据）和被评估的任务（claim + evidence 句对）不一致。
2. 有标签数据本身存在证据泄漏。
3. 消融实验的设置有混杂变量。

v2 把数据统一为 **claim + evidence 句对**，修复泄漏，并改用标准的半监督实验协议：

- 只保留一部分训练标签，其余样本隐藏标签后作为无标签池。
- 每组跑多个 seed。
- 做显著性检验。
- 加入随机对照组和 Oracle 上界组。

v2 是完全独立的新目录，**原项目代码一行都没有改**。

---

## 1. 目录结构

```
v2/
├── run_all.py                     一键运行（跨平台，Windows 下直接 python run_all.py）
├── run_all.sh                     bash 包装（会自动激活 ../.venv）
├── configs/config.yaml            唯一配置文件；相对路径都相对于 v2/ 解析
├── common/
│   ├── paths.py                   路径解析 + 每个 run 的目录布局（RunPaths）
│   └── data_utils.py              统一的句对 Dataset 与 JSONL 工具
├── data_prep/
│   ├── build_labeled.py           重建有标签数据（修复三处泄漏 / 重复）
│   └── make_ssl_split.py          按比例隐藏标签，切出无标签池
├── models/                        复制自 v1；extractor / detector / rl_selector 有修改
├── training/
│   ├── train_extractor.py         只用有标签部分训练
│   ├── generate_pseudolabels.py   对句对打伪标签，LogicScore 使用真实证据
│   ├── train_rl_selector.py       重写后的 PPO 选择流程
│   ├── build_baseline_sets.py     生成 B / W / R / O 组的伪标签集合
│   └── train_detector.py          伪标签走推理通道；所有组使用同一个监督损失
└── evaluation/
    ├── pseudo_label_quality.py    用隐藏的金标签评估伪标签与各选择策略
    └── aggregate_results.py       跨 seed 计算 mean±std、配对 bootstrap，并画图
```

运行时生成的目录：

```
v2/processed/labeled/{train,dev,test}.jsonl   修复后的有标签数据（所有 run 共用）
v2/runs/r0.10_s42/                            一个 (标注比例, seed) 组合
    data/        labeled_train.jsonl, unlabeled_pool.jsonl（不含标签）, unlabeled_gold.jsonl（只用于评估）
    pseudo/      pseudo_pool.jsonl, pseudo_filtered.jsonl, set_{B,W,R,C,O}*.jsonl
    outputs/     extractor/, rl_selector/, detector/<组别>/, pseudo_label_quality.json
    checkpoints/ 默认在该 run 跑完后删除
v2/results/                                   summary.md / summary.csv / significance.csv / pseudo_quality.csv / macro_f1_vs_ratio.png
```

---

## 2. 本次修改

### 2.1 数据泄漏与重复（`data_prep/build_labeled.py`）

| 问题 | v1 位置 | v2 做法 |
|---|---|---|
| Climate-FEVER 用**金标签挑证据**：优先选 `evidence_label == claim_label` 的句子，train/dev/test 都这样处理 | `process_labeled.py:63-64` | 不看任何标签，用 TF-IDF 余弦相似度从 5 条候选中选 top-3 |
| PUBHEALTH 用 `explanation`（核查记者为结论写的解释）当证据，结论基本写在输入里 | `process_labeled.py:272` | 从 `main_text`（原文正文）中按相似度选 top-3 句；需要复现旧做法时把 `data.pubhealth_evidence` 设为 `explanation` |
| Climate-FEVER 在 **dev/test 中也被 2× 重复**，测试时每条样本被计算两次 | `merge_labeled_datasets(cf_weight=2.0)` | dev/test 不再重复；过采样只作用于每个 run 的有标签训练部分 |
| PUBHEALTH 缺少 `claim_id` 时用 Python `hash()` 生成 ID，每次运行结果不同 | `process_labeled.py:270` | 改用 md5，结果稳定 |

> 修复泄漏后，**v2 的测试集和 v1 不同**，所以 v1 的数字（0.72 F1）不能直接拿来对比。纯监督基线大概率会下降，这是正常的。

### 2.2 无标签数据的设计（`data_prep/make_ssl_split.py`）

- 从修复后的 train 中，按 (来源, 类别) 分层抽取 `label_ratio` 比例的样本保留标签，其余样本去掉标签，作为无标签池。
- 无标签池保留 claim + evidence，与 dev/test 格式一致。
- 金标签单独存放在 `unlabeled_gold.jsonl`。**训练流程中的 extractor、伪标签生成、RL、detector 都不会读取这个文件**，只有 `evaluation/` 和 Oracle 组会用到。
- Guardian 新闻标题暂时不再使用。方案二（为真实无标签数据检索证据）见第 4 节。

### 2.3 伪标签生成（`training/generate_pseudolabels.py`、`models/extractor.py`）

- Extractor 在句对上打伪标签，与它的训练输入一致；v1 是在只有 claim 的输入上打标签，属于分布外预测。
- **LogicScore 用样本自己的证据作为 NLI 前提**。v1 用的是空字符串，导致约 70% 的 |LogicScore| 饱和在 1.0。
- 伪标签记录中带上 `evidence`，RL 探针和 detector 看到的都是同一个句对。
- 计算类别先验只使用当前 run 的有标签部分。

### 2.4 RL Selector（`models/rl_selector.py`、`training/train_rl_selector.py`）

| v1 | v2 |
|---|---|
| `baseline_f1` 每个 episode 换成上一个 episode 的 F1，ΔF1 比较的是两个随机子集 | `baseline_f1` 固定为"只用有标签数据"时的探针 F1 |
| 对池子只遍历一遍，每 2048 步只有一个奖励、只做一次更新，γ=0.99 使远处的动作几乎拿不到信号 | 每次更新从池中随机采 `episodes_per_iter` 个 episode（每个 `episode_size` 条），共 `n_iterations` 次更新；γ=1，episode 之间的对比提供 advantage |
| 最终结果是训练过程中随机采样动作的累加，包括早期接近 50/50 的策略 | 训练结束后用确定性策略（P(keep) > 0.5）在整个池上重新筛一遍 |
| diversity 用 `[confidence, entropy]` 这个二维向量代替 | 使用样本真实的 [CLS] 句向量 |
| 探针使用 saga 求解器 | 改用 lbfgs（在 1024 维稠密特征上更快） |
| — | 保留比例低于 `min_keep_ratio` 时，按 keep 概率补足并给出警告；训练曲线保存在 `outputs/rl_selector/` |

### 2.5 Detector（`training/train_detector.py`、`models/detector.py`）

- **伪标签样本走推理通道（claim + evidence）**，与验证和测试的输入一致。v1 的伪标签只走内容通道，而评估时从不使用内容通道。
- **所有组使用同一个监督损失**（`compute_supervised_loss`：focal + logit adjust）。v1 的 A 组走的是普通 CE。
- 伪标签损失默认用 CE（`algorithm.pseudo_loss_type`）。focal 会把高置信伪标签的损失压到接近 0。
- 样本权重的归一化默认为 `mean(w·L)`（`algorithm.pseudo_weight_norm`），让权重的绝对值起作用。v1 是 `Σ w·L / Σ w`，只有相对权重起作用。
- λ 默认从 0.3 线性增加到 0.7（v1 是 0.1 到 0.3），可在 `hyperparameters` 中调整。
- warmup 改为比例（`training.warmup_ratio=0.1`）。低标注比例下总步数只有一两百步，固定 100 步的 warmup 会占满整个训练过程。
- checkpoint 只保存模型权重（v1 还保存了优化器状态，文件大约是现在的 3 倍），并在测试结束后删除。测试集逐样本预测保存在 `test_predictions.jsonl`，用于显著性检验。
- 某个组的伪标签集合为空时（比如没有样本超过置信度阈值），写入 `skipped.json` 并继续执行，不会中断整个实验矩阵。

### 2.6 Extractor 训练（`training/train_extractor.py`）

- 只使用 run 的有标签部分训练；增加 `--run_dir` 和 `--seed` 参数；warmup 改为比例。
- 随机删词增强默认关闭（`training.extractor_text_augment`）。它会把句对 decode 成一句话再重新编码，破坏 claim / evidence 的结构。
- 不再保存 `final_model.pt`，下游没有用到它。

### 2.7 消融组别与评估

| 组 | 含义 | 用来回答的问题 |
|---|---|---|
| A | 纯监督 | 基线 |
| B | 置信度 ≥ 0.7，权重为 1 | 最简单的半监督方法有没有用 |
| W | 复合权重 ≥ 阈值，全部保留（不用 RL） | LogicScore / Discourse 加权本身有没有用 |
| R | 从 W 的池子中随机抽取与 C 同样数量的样本 | **C 和 R 只在"选哪些样本"上不同**，用来检验 RL 策略是否比随机好 |
| C | 完整模型（RL 选择） | 主方法 |
| O | 整个无标签池用金标签 | 半监督方法能达到的上界 |

- `pseudo_label_quality.py`：每个 run 的伪标签准确率、每组选中样本的准确率，以及 confidence、熵、|LogicScore|、discourse、复合权重各自区分伪标签对错的 **AUROC**。
- `aggregate_results.py`：每个 (标注比例, 组) 的 mean ± std；每个 seed 内的**配对 bootstrap**（ΔmacroF1 的 95% 置信区间）；以及 F1 随标注比例变化的曲线。

---

## 3. 运行

### 3.1 环境与数据

- 直接使用原项目的环境，`requirements.txt` 不需要改，也没有新增依赖。
- 原始数据默认读取 `../Data/`（即原项目的 `Data/` 目录），路径和 v1 的 `process_labeled.py` 一致：
  - `../Data/Climate Fever Dataset/archive/climate-fever.jsonl`
  - `../Data/PUBHEALTH-DATASET/archive/{train,dev,test}.tsv`

  数据放在其他位置时，修改 `configs/config.yaml` 的 `paths.raw_*`，可以写绝对路径，比如 `D:/LIUBOProj/Sophia/Data/...`。
- HF 模型缓存在 `../model_cache`，与原项目的 extractor 共用。v1 的 detector 用的是 HF 默认缓存，所以第一次运行时可能会重新下载 deberta-v3-large。

### 3.2 命令

```bash
cd v2

# 先跑一个 run 看效果（强烈建议）
python run_all.py --ratios 0.1 --seeds 42

# 完整实验矩阵（config 中的 label_ratios × seeds × methods）
python run_all.py

# 只跑部分组别
python run_all.py --methods A,R,C

# 只重新汇总结果
python run_all.py --aggregate_only
```

- **断点续跑**：每一步的产物已存在就自动跳过，中断后重新执行同一条命令即可继续。`--force` 会重跑所有步骤，`--rebuild_data` 只重建有标签数据。
- `--keep_checkpoints`：保留所有 `.pt` 文件，默认删除。每个 DeBERTa-large 的 checkpoint 约 1.7GB。

### 3.3 计算量

每个 run 要训练 1 个 extractor、1 个 RL selector，外加 6 个 detector。

- detector 现在每个有标签 batch 会额外处理 `pseudo_ratio × pseudo_batch_size = 48` 个句对。按 v1 的 detector 每轮耗时估算，单个 detector 大约是 v1 的 3~4 倍。
- 默认矩阵是 3 个比例 × 3 个 seed × 6 组 = 54 个 detector。

建议先跑一个 run 估计时间，再决定矩阵大小。

---

## 4. 执行清单

### 第一步：重建数据并检查

- [ ] `python data_prep/build_labeled.py`
- [ ] 查看 `processed/labeled/build_stats.json`：
  - PUBHEALTH 的 `dropped.no_evidence` 不应占很大比例。如果大量样本没有 `main_text`，需要检查 TSV 的版本。
  - dev/test 中 `climate_fever` 的数量约为 v1 的一半（v1 重复了一次）。
- [ ] 随机抽 10 条 PUBHEALTH 记录，确认 evidence 是正文中的相关句子，而不是"This claim is false because…"这类解释。

### 第二步：跑一个 run（`--ratios 0.1 --seeds 42`），逐项检查

- [ ] `runs/r0.10_s42/data/split_stats.json`：有标签和无标签部分的规模、类别分布是否合理。
- [ ] `outputs/extractor/extractor_train_metrics.json`：验证集 F1 应明显高于随机（三分类随机约 0.33）。
- [ ] `pseudo/pseudo_stats.json`：
  - `retention_rate`：v1 的权重被饱和的 |LogicScore| 抬高了 0.3，v2 的复合权重会整体偏低。如果低于 10%，脚本会给出警告，此时应调低 `experiment.weight_threshold`。
  - `avg_abs_logic_score` 不应再接近 1.0。
- [ ] `outputs/pseudo_label_quality.json`（**最关键**）：
  - 整个池的伪标签准确率。如果接近 1/3，说明伪标签基本是噪声，后面不用再看。
  - `abs_logic_score` 的 AUROC：明显大于 0.5，才能说明 LogicScore 有助于挑出正确的伪标签。
  - C 组的准确率是否高于 R 组。如果不高，说明 RL 选择没有比随机更好。
- [ ] `outputs/rl_selector/selection_info.json` 和 `rl_selector_training_history.png`：
  - 奖励和 ΔF1 是否随迭代上升。
  - `warning` 不为空，说明策略几乎全部丢弃（或几乎全部保留）样本，需要调整 reward 中的 `beta`，或者增加 `n_iterations`。
- [ ] `outputs/detector/*/test_results.json`：O 组明显优于 A 组，说明这个设置下半监督有提升空间；如果 O 组和 A 组差不多，伪标签方法本身就没有发挥的余地。
- [ ] 显存：如果 OOM，把 `training.pseudo_ratio` 改为 1 或 2，或者减小 `pseudo_batch_size`。

### 第三步：完整矩阵

- [ ] 根据第一个 run 的耗时确定 `label_ratios` 和 `seeds`，至少 3 个 seed。
- [ ] `python run_all.py`
- [ ] 查看 `results/summary.md`、`results/significance.csv` 和 `results/macro_f1_vs_ratio.png`。

### 第四步：解读结果

- [ ] 半监督有效：在低标注比例下，B/W/R/C 中至少一组优于 A，且多数 seed 的 95% 置信区间不包含 0。
- [ ] RL 有效：C 优于 R（`C vs R`），并且 C 组伪标签的准确率高于 R 组。
- [ ] LogicScore 有效：W 优于 B，或者 `abs_logic_score` 的 AUROC 明显大于 0.5。
- [ ] 上界：O 和 A 之间的差距就是半监督方法最多能提升的空间。

---

## 5. 已验证与未验证

- **已验证**：本机没有真实数据，也没有 GPU。我用合成数据（150 条 Climate-FEVER 格式、420 条 PUBHEALTH 格式）和一个随机初始化的微型 BERT（代替 DeBERTa 和 NLI 模型），在 CPU 上把 `run_all.py` 完整跑通了：2 个 seed × 6 组 + 结果汇总。同时确认了以下几点：
  - dev/test 中没有重复 ID；
  - PUBHEALTH 的证据来自正文，不包含 explanation；
  - 无标签池中没有标签，且与有标签部分没有交集；
  - 伪标签损失确实参与训练；
  - A 组与其他组使用同一个监督损失；
  - 断点续跑和空集合跳过都能正常工作。
- **未验证**：
  - 真实数据下的字段细节，特别是 PUBHEALTH TSV 中是否有 `main_text` 列。没有这一列时脚本会报错并给出提示。
  - DeBERTa-large 在 GPU 上的显存占用和耗时。
  - 新超参数（λ、阈值、RL 迭代次数）的效果。这些都是合理的起点，需要根据第二步的检查结果调整。

## 6. 尚未实现

- **方案二（扩展实验）**：为 Guardian 新闻抽取可核查的句子，再从 Wikipedia 或新闻正文中检索证据，构造真实的无标签句对。需要先在本地准备检索语料，建议在方案一得到稳定结论之后再做。
- RL 状态目前没有加入伪标签的类别信息；Discourse 特征的构成没有改动。
