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
│   ├── build_baseline_sets.py     生成 B / L / Q / K / W / R / O 组的伪标签集合
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
    pseudo/      pseudo_pool.jsonl, pseudo_filtered.jsonl, set_{B,L,Q,K,W,R,C,O}*.jsonl
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
- **LogicScore 用样本自己的证据作为 NLI 前提**。v1 用的是空字符串。
- **LogicScore 的 NLI 标签下标从模型配置读取**（修复 v1 起把"中立"当成"蕴含"的 bug，见 7.2.0）。
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
- **训练预算（第二步检查后新增）**：`training.epoch_mode: cover_pseudo`。原做法每个 epoch 只遍历一遍有标签数据（ratio=0.1 时 68 个 batch），8 个 epoch 只有 136 次参数更新，O 组（8798 条金标签）也学不进去，O ≈ A。现在每个 epoch 的长度为 `max(有标签 batch 数, ceil(伪标签 batch 数 / pseudo_ratio))`，有标签数据循环使用；伪标签 batch 在各步之间均匀分配，每个 epoch 抽取的伪标签条数等于集合大小，小集合（R/C）不再被整体重复约 10 次。注意：开启 `use_balanced_pseudo_sampler` 时是有放回抽样，少数类样本仍会被重复（如 B 组 REF 只占 5%，平衡后每条每 epoch 约被抽 6.8 次）。设为 `labeled` 可复现原做法。
- **动态补齐（第二步检查后新增）**：`training.dynamic_padding: true`，每个 batch 只补到 batch 内最长句对（平均 129 token），不再统一补到 384。detector 取 [CLS] 且有 attention mask，结果不变（A 组测试 F1 0.5627 → 0.5586，在单 seed 波动范围内），A 组耗时 6 → 3 分钟。只作用于 detector；extractor / 伪标签 / RL 仍为固定补齐。
- `training.gradient_checkpointing` 改为可配置，但必须保持开启：实测关闭后 64 条 × 256 token 即 OOM（32GB），开启时 384 token 峰值约 10GB。

### 2.6 Extractor 训练（`training/train_extractor.py`）

- 只使用 run 的有标签部分训练；增加 `--run_dir` 和 `--seed` 参数；warmup 改为比例。
- 随机删词增强默认关闭（`training.extractor_text_augment`）。它会把句对 decode 成一句话再重新编码，破坏 claim / evidence 的结构。
- 不再保存 `final_model.pt`，下游没有用到它。

### 2.7 消融组别与评估

| 组 | 含义 | 用来回答的问题 |
|---|---|---|
| A | 纯监督 | 基线 |
| B | 置信度 ≥ 0.7，权重为 1 | 最简单的半监督方法有没有用 |
| L | B+L：置信度 ≥ 0.7 且方向一致性 c > 0.3（`experiment.consistency_threshold`），权重为 1 | **L 和 B 只差 LogicScore 一个条件**，直接检验 LogicScore 有没有用（7.3 第二档） |
| Q | L-q：置信度 ≥ 0.7；SUPPORTS / REFUTES 各自在类内按 c 保留前 50%（`experiment.consistency_quantile`），NEI 不用 c；权重为 1 | 改进后的 L（8.6）：不再用固定阈值，SUPPORTS 不会被大量砍掉 |
| K | 按置信度取前 \|Q\| 条，权重为 1 | **Q 和 K 规模相同、只差选样规则**，把"挑得更准"和"样本更少"分开 |
| W | 复合权重 ≥ 阈值，全部保留（不用 RL） | LogicScore / Discourse 加权本身有没有用 |
| R | 从 W 的池子中随机抽取与 C 同样数量的样本 | **C 和 R 只在"选哪些样本"上不同**，用来检验 RL 策略是否比随机好 |
| C | 完整模型（RL 选择） | 主方法 |
| O | 整个无标签池用金标签 | 半监督方法能达到的上界 |

- `pseudo_label_quality.py`：每个 run 的伪标签准确率、每组选中样本的准确率，以及 confidence、熵、|LogicScore|、方向一致性 c、discourse、复合权重各自区分伪标签对错的 **AUROC**（整体和按来源分别计算）。
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

- [x] `python data_prep/build_labeled.py`
  > **执行情况（2026-10-02）**：原始数据直接放在 `../Data/` 默认路径下，无需预处理。PUBHEALTH 为官方版本（train/dev/test = 9832/1221/1235 行，含 `main_text`），约 40 条脏行（label 为空或错位）由 label 白名单自动过滤。输出 train/dev/test = **9775 / 1306 / 1321**（Climate-FEVER 1074/230/231，PUBHEALTH 8701/1076/1090），无重复 ID、无空证据、每条 3 句证据、split 之间 ID 与 claim 均无重叠。
- [x] 查看 `processed/labeled/build_stats.json`：
  - PUBHEALTH 的 `dropped.no_evidence` 不应占很大比例。如果大量样本没有 `main_text`，需要检查 TSV 的版本。
    > **执行情况**：`no_evidence` = **0**；丢弃的全是 covid 过滤（train 1103 / dev 138 / test 143，与 v1 规则一致）。
  - dev/test 中 `climate_fever` 的数量约为 v1 的一半（v1 重复了一次）。
    > **执行情况**：用 `../process_labeled.py` 重新生成了 v1 数据（`../processed/labeled/`）对比。v1 dev/test 中 Climate-FEVER 为 460/462，v2 为 230/231，正好一半。去重后 v1 与 v2 的样本集合、split 归属、标签**完全一致**，差异只在证据与重复。
- [x] 随机抽 10 条 PUBHEALTH 记录，确认 evidence 是正文中的相关句子，而不是"This claim is false because…"这类解释。
  > **执行情况**：10 条均为正文中与 claim 相关的句子，非 explanation。残留问题：PolitiFact/Snopes 正文本身包含记者结论，约 5% 的 PUBHEALTH 样本（564/10867）的证据句在 explanation 中原文出现，23 条含 "we rate" 等判定用语；程度远低于 v1，暂不处理。

> **v1 vs v2 泄漏对比（补充）**
> - Climate-FEVER 被选证据的 `evidence_label` 与 `claim_label` 一致率：v1 **87–91%**（SUPPORTS 100%），v2 61–65%（SUPPORTS 55%）。
> - v1 新发现的泄漏：**证据条数暴露标签**——NEI 永远是 3 条，1–2 条的一定不是 NEI；仅用条数做特征在 Climate-FEVER 测试集上 macro-F1 = 0.411（v2 固定 3 条，该信号消失）。
> - TF-IDF + LR 探针（测试集 macro-F1，已去重）：
>
>   | 输入 | v1 | v2 |
>   |---|---|---|
>   | 只用 claim | 0.520 | 0.523 |
>   | 只用证据 | **0.576** | 0.531 |
>   | 只用证据（PUBHEALTH） | **0.608** | 0.559 |
>   | claim + 证据 | 0.585 | 0.543 |
>
>   只用 claim 两版相同（对照），差异全部来自证据；修复后纯监督基线下降符合预期。

### 第二步：跑一个 run（`--ratios 0.1 --seeds 42`），逐项检查

> **执行情况（2026-10-02，PUBHEALTH 版）**：`python run_all.py --ratios 0.1 --seeds 42` 完整跑通，无报错。日志 `logs/step2_r0.10_s42.log`。
>
> ⚠️ 这个 run 的 LogicScore 有下标 bug（见 7.2.0），所有依赖 LogicScore 的结果（复合权重、W / R / C 组、RL）都已从本文档删除，需要在修正后的代码上重跑。下面只保留不依赖 LogicScore 的 A / B / O 和数据检查。之后数据集也已更换（第 8 节）。
>
> | 组 | 伪标签条数 | 伪标签准确率 | 测试 macro-F1 | AUC | Δ vs A（95% CI） |
> |---|---|---|---|---|---|
> | A 纯监督 | 0 | — | **0.5627** | 0.7835 | — |
> | B 置信度 ≥ 0.7 | 2416 | 0.621 | 0.5489 | 0.7714 | −0.014（−0.038, +0.009） |
> | O 金标签 | 8798 | 1.000 | 0.5607 | 0.7887 | −0.002（−0.026, +0.020） |

- [x] `runs/r0.10_s42/data/split_stats.json`：有标签和无标签部分的规模、类别分布是否合理。
  > ✅ 有标签 977 条（Climate-FEVER 过采样后 1084），无标签池 8798 条；两边 SUP:REF:NEI 均约 51:30:20，分层生效。
- [x] `outputs/extractor/extractor_train_metrics.json`：验证集 F1 应明显高于随机（三分类随机约 0.33）。
  > ✅ 最佳验证 macro-F1 = **0.487**（第 8 epoch）。前 2 个 epoch 几乎全预测 NEI（0.14），第 3 epoch 起上升；第 4 epoch 后 val_loss 上升（1.02→1.08），有轻微过拟合。
- [x] `pseudo/pseudo_stats.json`：
  - `retention_rate`：复合权重 ≥ `experiment.weight_threshold` 的比例。如果低于 10%，脚本会给出警告，此时应调低阈值。
    > 新数据（LogicScore 已修正，`runs_sci/r0.10_s42`）：0.348（1340/3853）。
  - `avg_abs_logic_score`：用真实证据作前提后，不应再大量饱和在 1.0。
    > 新数据：平均 0.272；53% 小于 0.01（蕴含与矛盾概率相当，基本判为中立），12% 大于 0.99。
- [x] `outputs/pseudo_label_quality.json`（**最关键**）：
  - 整个池的伪标签准确率。如果接近 1/3，说明伪标签基本是噪声，后面不用再看。
    > ✅ **0.514**。分类别精度：SUPPORTS 0.80、REFUTES 0.40、NEI 0.27。金标签 SUPPORTS 有 4453 条，伪标签只给出 3364 条 SUPPORTS——类别先验校正（`apply_prior_adjust_in_pseudolabels`）把大量样本推向 REFUTES/NEI，是 REFUTES/NEI 精度低的主要来源之一。
  - 各信号区分伪标签对错的 AUROC：LogicScore 相关信号明显大于 0.5，才能说明 LogicScore 有助于挑出正确的伪标签。
    > 不依赖 LogicScore 的信号：confidence 0.600、neg_entropy 0.582、discourse_score **0.476**（略反向）。LogicScore 修正后的结果见 7.2.2：在 PUBHEALTH 上没有信号，在新数据上带方向的 c 有信号。
  - C 组的准确率是否高于 R 组。
    > 待修正后的 RL 重跑（7.3 第三档）。
- [x] `outputs/rl_selector/selection_info.json` 和 `rl_selector_training_history.png`：奖励和 ΔF1 是否随迭代上升；`warning` 不为空说明策略几乎全部丢弃（或保留）样本。
  > 待修正后的 RL 重跑。修正 LogicScore 后奖励仍被近似常数项主导，见 7.2.3。
- [x] `outputs/detector/*/test_results.json`：O 组明显优于 A 组，说明这个设置下半监督有提升空间；如果 O 组和 A 组差不多，伪标签方法本身就没有发挥的余地。
  > ❌ O（0.5607）≈ A（0.5627）。O 组标签已核对为 100% 金标签，排除了程序错误。原因更可能是**训练预算**而非数据：每个 epoch 的长度由有标签 loader 决定（68 batch，grad_accum=4 → 17 次更新），8 个 epoch 共只有 **136 次优化器更新**；O 组到最后伪标签 CE 仍有 0.80，明显欠拟合，验证 F1 第 6 epoch 仍在上升。在当前设置下，即使给 9 倍金标签也学不进去，所以这个 run 无法判断任何半监督方法的效果。**进入第三步前需要先修正训练预算，让 O 明显高于 A。**
  >
  > 另外，小的伪标签集合在每个有标签 batch 配 48 条的旧做法下，每个 epoch 每条会被重复约 10 次，容易过拟合。
  >
  > **修正后复测（`runs/r0.10_s42_fix`，同一份数据与伪标签）**：改用 `epoch_mode: cover_pseudo` + 动态补齐。日志 `logs/fix_AO_r0.10_s42.log`、`logs/fix_BWRC_r0.10_s42.log`；显著性 `runs/r0.10_s42_fix/outputs/significance.csv`（与 `aggregate_results.py` 同一函数、1000 次重抽样、种子 0）。
  >
  > | 组 | 伪标签条数 | 伪标签准确率 | 参数更新次数 | 最佳 epoch | 测试 macro-F1 | 旧设置 | AUC | 耗时 |
  > |---|---|---|---|---|---|---|---|---|
  > | A 纯监督 | 0 | — | 136 | 7 | 0.5586 | 0.5627 | 0.777 | 3 min |
  > | B 置信度 ≥ 0.7 | 2416 | 0.621 | 136 | 8 | **0.5654** | 0.5489 | 0.785 | 8 min |
  > | O 金标签 | 8798 | 1.000 | 368 | 8 | **0.6292** | 0.5607 | 0.843 | 23 min |
  >
  > | 比较 | Δ macro-F1 | 95% CI | 显著 | 旧设置 Δ |
  > |---|---|---|---|---|
  > | B vs A | +0.007 | (−0.015, +0.030) | 否 | −0.014 |
  > | O vs A | **+0.071** | (+0.043, +0.103) | **是** | −0.002 |
  >
  > - ✅ **O − A = +0.071（显著）**：这个设置下半监督有约 7 个点的提升空间。O 的验证 F1 到第 8 epoch 仍在上升；有标签部分的监督 loss 降到 0.006，有标签数据已被记住。
  > - 训练预算修正后，瓶颈转移到**伪标签质量**：B 有上升趋势但不显著。
  > - 单 seed 噪声：同一配置下 A 组仅因数值差异就从 0.5627 变为 0.5586。**1–2 个点的差距需要多 seed 才能下结论。**
  >
  > **离线分析：类别先验校正降低了伪标签质量。** `pseudo_pool.jsonl` 保存的是校正后概率 `softmax(logits − τ·log prior)`，可以精确还原校正前的概率（`softmax(log probs + τ·log prior)`，τ=0.5）：
  >
  > | | 整池准确率 | 置信度 AUROC | 置信度 ≥ 0.7 的条数 | 其中准确率 |
  > |---|---|---|---|---|
  > | 校正后（当前） | 0.514 | 0.600 | 2416 | 0.621 |
  > | 校正前 | **0.535** | **0.638** | **2863** | **0.697** |
  >
  > 关闭 `imbalance.apply_prior_adjust_in_pseudolabels` 后，B 组的伪标签多 18%、准确率高 7.6 个点。
- [x] 显存：如果 OOM，把 `training.pseudo_ratio` 改为 1 或 2，或者减小 `pseudo_batch_size`。
  > ✅ 未 OOM。detector 峰值约 13.6GB / 32GB（RTX 5090），extractor 约 8.4GB。
  >
  > 速度问题：`common/data_utils.py` 用 `padding="max_length"` 把所有句对补齐到 384，而真实长度平均 129、p90 205，大部分计算浪费在补齐上。已改为动态补齐（见 2.5 节）：A 组 6 → 3 分钟，带伪标签的组每步约快 1.4 倍。梯度检查点不能关（实测关闭后 64 条 × 256 token 即 OOM）。

### 第三步：完整矩阵

> ⚠️ 进入这一步前，先按第 7.4 节完成伪标签相关的修改和多 seed 筛选；改配置后要用 `--force` 或新的 run 目录（见 7.2.6）。

- [ ] 根据第一个 run 的耗时确定 `label_ratios` 和 `seeds`，至少 3 个 seed。
- [ ] `python run_all.py`
- [ ] 查看 `results/summary.md`、`results/significance.csv` 和 `results/macro_f1_vs_ratio.png`。

### 第四步：解读结果

- [ ] 半监督有效：在低标注比例下，B/W/R/C 中至少一组优于 A，且多数 seed 的 95% 置信区间不包含 0。
- [ ] RL 有效：C 优于 R（`C vs R`），并且 C 组伪标签的准确率高于 R 组。
- [ ] LogicScore 有效：L（B+L）优于 B，且方向一致性 c（`direction_consistency`）的 AUROC 明显大于 0.5。不带方向的 `abs_logic_score` 不能作为判据（7.2.2）。
- [ ] 上界：O 和 A 之间的差距就是半监督方法最多能提升的空间。

---

## 5. 已验证与未验证

> 2026-10-03 更新。原来的内容写于没有 GPU 和真实数据的时候，下面按实际运行情况重写。

- **已验证（真实数据，RTX 5090 32GB）**：
  - `run_all.py` 在 PUBHEALTH 版（6 组）和新数据集版（A / B / L / O）上都完整跑通；断点续跑、只补跑新增组（如 seed 42 只跑 L）正常。
  - 数据：dev/test 无重复 ID，split 之间无相同 claim（新数据按 claim 分组划分）；无标签池不含标签；PUBHEALTH 的 `main_text` 存在，证据来自正文。
  - 显存：detector 开梯度检查点时峰值约 12.5–13.6GB，extractor 约 8.4GB；梯度检查点不能关（见 2.5）。
  - 耗时（新数据，ratio 0.1）：extractor 约 5 分钟，伪标签约 2.5 分钟，A 约 4.5 分钟，B / L 约 7 分钟，O 约 24 分钟。
  - LogicScore 的标签下标已修正并验证（蕴含句对 +0.999，矛盾句对 −1.000，见 7.2.0）。
- **未验证 / 待定**：
  - W、R、C 三组在修正 LogicScore 后还没有重跑过；复合权重和 RL 需要先按 7.3 第一档、第三档修改。
  - ratio 0.25 / 0.5 还没有跑过；ratio 越高，A 组越慢（有标签 batch 越多），但无标签池越小。
  - 超参数（λ、各阈值、RL 迭代次数）没有调过，目前都是默认的起点。

## 6. 尚未实现

- **方案二（扩展实验）**：为 Guardian 新闻抽取可核查的句子，再从 Wikipedia 或新闻正文中检索证据，构造真实的无标签句对。需要先在本地准备检索语料，建议在方案一得到稳定结论之后再做。
- RL 状态目前没有加入伪标签的类别信息；Discourse 特征的构成没有改动。

## 7. 第二步后发现的问题与优化方案

依据的数据：

- PUBHEALTH 版：`runs/r0.10_s42`（旧训练预算）、`runs/r0.10_s42_fix`（新训练预算）。只使用其中不依赖 LogicScore 的结果（A / B / O、置信度、先验校正）。
- 新数据集（第 8 节）：`runs_sci/r0.10_s42`，LogicScore 已修正。LogicScore 相关的数字都来自这里，由伪标签池和隐藏金标签离线重算。

除特别说明外，所有数字都只来自 ratio 0.1、seed 42 这一个 run，单 seed 的训练随机性可达 1–2 个点。

### 7.1 现状

- **已解决**：证据泄漏与 dev/test 重复（第一步）；detector 训练预算不足（PUBHEALTH 上 O − A 从 −0.002 变为 **+0.071**，显著）；固定补齐浪费算力（改为动态补齐）；LogicScore 的下标 bug（7.2.0）。
- PUBHEALTH 上不依赖 LogicScore 的三组：测试 macro-F1 与伪标签准确率的排序一致，O（1.000）0.629 > B（0.621）0.565 > A 0.559。
- **当前的瓶颈是伪标签质量，以及 W / C 两组方法本身的设计（7.2.2、7.2.3），而不是算力。** W / R / C 还没有在修正后的代码上跑过。

### 7.2 存在的问题

#### 7.2.0 LogicScore 的计算从 v1 起就是错的（已修复）

- `models/logic_scorer.py` 把下标 2 当作"蕴含"，但 `cross-encoder/nli-deberta-v3-large` 的标签顺序是 `{0: 矛盾, 1: 蕴含, 2: 中立}`，所以之前的 LogicScore 实际上是 P(中立) − P(矛盾)。
- **修复**：改为从模型配置的 `id2label` 读取下标。验证：蕴含句对 +0.999，矛盾句对 −1.000。
- **影响**：v1 和 v2 中所有用到 LogicScore 的结果（复合权重、W / R / C 组、RL 的状态和奖励）都无效，已从本文档删除，需要在修正后的代码上重跑。A / B / O、置信度、先验校正、Discourse 不受影响。
- 注意：NLI 模型对**无关**的 claim 也会给出 −0.998，所以 LogicScore 为负不一定表示被反驳。

#### 7.2.1 伪标签不够准（根源）

PUBHEALTH 版（不依赖 LogicScore）：

| 现象 | 数据 |
|---|---|
| extractor 弱 | 只用 977 条有标签数据训练，验证 macro-F1 0.487；第 4 epoch 后验证 loss 回升（1.02 → 1.08） |
| 整池伪标签准确率 | 0.514；按伪标签类别的精度：SUPPORTS 0.80 / REFUTES 0.40 / NEI 0.27 |
| 按来源 | Climate-FEVER 子集 **0.383**（接近随机），PUBHEALTH 0.554 |
| 类别先验校正（`apply_prior_adjust_in_pseudolabels`）降低准确率 | 整池 0.535 → 0.514；置信度 ≥ 0.7 子集 2863 条 / 0.697 → 2416 条 / 0.621。校正把 SUPPORTS 的预测数从 3774 压到 3364，而金标签中 SUPPORTS 有 4453 条 |
| 平衡采样器（`use_balanced_pseudo_sampler`）降低 detector 实际看到的准确率 | B 组 REFUTES 只占 5%（119 条），平衡后占 1/3，每条每 epoch 约被抽 6.8 次；期望准确率 0.621 → 0.583 |

> 计算方法：`pseudo_pool.jsonl` 保存的是校正后概率 `softmax(logits − τ·log prior)`，校正前概率由 `softmax(log p + τ·log prior)` 精确还原（τ = 0.5，prior 见 `pseudo/pseudo_stats.json`）。平衡采样后的期望准确率 = 各伪标签类别精度的平均值。

新数据上整池伪标签准确率为 0.497（见 8.4），仍然是主要瓶颈。

#### 7.2.2 LogicScore / Discourse 的用法有设计问题（影响 W、C）

复合权重 = 0.5·置信度 + 0.3·|LogicScore| + 0.2·Discourse（`models/extractor.py`，系数为 `hyperparameters.beta1/2/3`）。LogicScore = P(蕴含) − P(矛盾)，本来带方向；取绝对值后"证据强烈反驳"和"证据强烈支持"得分相同，与伪标签是否一致无关。

方向一致性 c 的定义：伪标签为 SUPPORTS 时 c = LS；为 REFUTES 时 c = −LS；为 NEI 时 c = 1 − |LS|（`common/data_utils.direction_consistency`）。

各信号区分伪标签对错的 AUROC（LogicScore 已修正；关闭先验校正后的伪标签）：

| 数据 | 置信度 | \|LogicScore\|（当前用法） | 方向一致性 c | Discourse | 复合权重 |
|---|---|---|---|---|---|
| PUBHEALTH | 0.649 | 0.478 | 0.507 | 0.466 | — |
| Climate-FEVER（PUBHEALTH 版的池） | 0.522 | 0.472 | 0.573 | 0.500 | — |
| **新数据：全部** | 0.554 | 0.519 | **0.593** | 0.505 | 0.539 |
| 新数据：ClimateCheck | 0.578 | 0.516 | 0.536 | 0.520 | 0.557 |
| 新数据：Climate-FEVER | 0.494 | 0.577 | **0.620** | 0.478 | 0.544 |
| 新数据：SciFact | 0.555 | 0.476 | **0.748** | 0.477 | 0.514 |

- 不带方向的 |LogicScore| 没有区分能力；Discourse 也没有（略反向）。
- 带方向的 c 在 PUBHEALTH 上没有信号（PUBHEALTH 的证据不含判定依据，见 8.1），在新数据上有信号，SciFact 上最强，整体超过置信度。
- 复合权重（0.539）低于单用置信度（0.554），因为它用的是不带方向的 |LS| 和没有信号的 Discourse。这是 W 组设计上的问题。

#### 7.2.3 RL 选择器的奖励和状态有设计问题（影响 C）

- 奖励 = 0.7·ΔF1 + 0.3·被选样本的平均 |LogicScore|（`hyperparameters.alpha/beta`）。新数据过滤池（复合权重 ≥ 0.4）的平均 |LS| 为 0.748；在随机回合（512 条、保留约一半）上，第二项为 0.224 ± 0.006，几乎是常数，而 0.7·ΔF1 只有 ±0.007 量级。**奖励几乎全部来自常数项，PPO 很难拿到有效的梯度。**
- 状态 `[置信度, 熵, |LogicScore|, 多样性]` 不含伪标签类别，也丢掉了 LogicScore 的方向。
- 保留比例低于 `min_keep_ratio` 时按概率补足，C 的规模不由策略决定。
- C 组在修正后的代码上还没有跑过。

#### 7.2.4 组间对比的混杂因素

- **伪标签损失强度不同**：B、L、O 的权重固定为 1（`build_baseline_sets.py`），W/R/C 保留复合权重（新数据过滤池平均 0.572，p10–p90 为 0.42–0.71）。在 `pseudo_weight_norm: mean` 下，W/R/C 的伪标签损失只有 B 的约 0.6 倍，所以 B vs W 同时差在"选哪些样本"和"损失强度"上。
- **集合大小相差很大**（新数据：L 287、B 712、W 1340、O 3853 条）：`cover_pseudo` 下 A / B / L 都是 224 次参数更新，O 是 640 次；集合越大，有标签数据被循环使用越多。O − A 有一部分来自训练更多。
- 小集合在 `cover_pseudo` 下只有部分步带伪标签（L 每个 epoch 28 步中 17 步），对模型的影响较小，检验功效较低。

#### 7.2.5 统计功效

- 单 seed 的训练随机性可达 1–2 个点：同配置的 A 组仅因数值差异就从 0.5627 变为 0.5586。
- 配对 bootstrap 只反映测试样本的抽样误差，不包含训练随机性。
- 新数据的 Climate-FEVER 测试集只有 200 条，SciFact 339 条，分来源的结论噪声很大。

#### 7.2.6 工程注意事项

- **重跑陷阱**：`run_all.py` 遇到已有产物会跳过。改了伪标签相关的配置后直接重跑，旧的伪标签和 detector 结果不会被替换。需要加 `--force`，或者使用新的 run 目录。
- `runs/r0.10_s42_fix` 不符合 `aggregate_results.py` 的目录命名规则（`r{ratio}_s{seed}`），不会被汇总。
- extractor、伪标签生成、RL 仍是固定补齐，每个 run 合计约 15 分钟，优先级低。

### 7.3 优化方案

现有条件：1 张 RTX 5090（32GB）；本地已缓存 deberta-v3-large 和 nli-deberta-v3-large；不新增标注。

伪标签池里已经保存了每条样本的置信度、LogicScore、Discourse 和概率，**权重、过滤规则和分组都可以离线重算**，不需要重跑 extractor 和 NLI 模型。大部分优化的成本只是重跑 detector。以下数字均来自新数据 `runs_sci/r0.10_s42`。

#### 第一档：改配置或离线重算（证据明确，先做）

| 改动 | 位置 | 预期 |
|---|---|---|
| 关闭伪标签的类别先验校正 | `imbalance.apply_prior_adjust_in_pseudolabels: false` | ✅ 已在 `config_sci.yaml` 中关闭（PUBHEALTH 上整池准确率 0.514 → 0.535，B 组 0.621 → 0.697） |
| 复合权重中 LogicScore 改用方向一致性 c，去掉 Discourse | `models/extractor.py` 的权重公式；`hyperparameters.beta3: 0` | LogicScore 项的 AUROC 0.519 → 0.593。c 可以为负，需要截断到 [0, 1] 并重新设定 `experiment.weight_threshold` |
| 对齐各组的伪标签损失强度 | `algorithm.pseudo_weight_norm: sum` | 只保留相对权重，消除 W/R/C 与 B/O 之间约 0.6 倍的差异。代价是撤销 2.5 节"让权重绝对值起作用"的设计。注意 `compute_joint_loss` 会把权重截断到 [0, 1]，不能靠把权重放大到均值 1 来对齐 |
| 平衡采样器降为 `pseudo_sampler_power: 0.5` 或关闭（作为对照） | `imbalance.*` | 减少少数类的重复抽样；关闭后需要实验确认对 REFUTES 召回的影响 |

#### 第二档：新增 B+L 组，直接检验 LogicScore

> ✅ 已实现为方法 **L**（2026-10-03），结果见 8.5 节。

- 定义：置信度 ≥ 0.7 且 c > 0.3（`experiment.consistency_threshold`），权重为 1。
- B 与 B+L 只差 LogicScore 一个条件，比 W vs B 干净得多，是检验"LogicScore 有用"最直接的实验。
- 不同阈值下的离线结果：

  | 置信度阈值 | 条数 / 准确率 | 再加方向一致（c > 0.3） |
  |---|---|---|
  | 不限 | 3853 / 0.497 | 1836 / 0.555 |
  | ≥ 0.5 | 3307 / 0.518 | 1617 / 0.570 |
  | ≥ 0.7 | 712 / 0.522 | **287 / 0.690** |

- 实现时改动的文件：`training/build_baseline_sets.py`、`common/paths.py`、`run_all.py`、`training/train_detector.py`、`evaluation/aggregate_results.py`（`COMPARISONS` 中加入 L vs A、L vs B）、`evaluation/pseudo_label_quality.py`（加入 L 和 c 的 AUROC，并按来源拆分）、两个配置文件（`consistency_threshold`）。

#### 第三档：RL 与伪标签来源（成本较高，等前两档有结论再做）

- **RL**：奖励去掉恒定的 |LogicScore| 项，或换成方向一致性 c；状态加入伪标签类别（one-hot）和 c；加大 `episode_size` 以降低 ΔF1 的噪声；取消 `min_keep_ratio` 补齐，让 C 的规模由策略决定。如果第二档证明信号本身很弱，RL 很难学到东西，届时需要决定是否保留 RL。
- **extractor 自训练**：用 B+L 那批伪标签（287 条 / 0.690）加有标签数据重训 extractor，再重新打伪标签，每轮约 8 分钟。
- **NLI 与 extractor 融合**：NLI 零样本规则（LogicScore > 0.5 判 SUPPORTS，< −0.5 判 REFUTES，其余判 NEI）与 extractor 的伪标签准确率：Climate-FEVER 0.457 vs 0.429，**SciFact 0.630 vs 0.492**，ClimateCheck 0.437 vs 0.524。融合在 SciFact 上最有希望，ClimateCheck 上应以 extractor 为主。

#### 实验协议

- 方法筛选阶段固定 ratio 0.1、跑 3 个 seed，先跑 A / B / L / O，再跑修改后的 W，暂不跑 RL。确定配置后再跑完整矩阵。
- 每组报告跨 seed 的均值 ± 标准差，并按来源拆分报告（注明样本量）。
- 换配置后使用新的 run 目录或 `--force`，避免新旧结果混在一起。

### 7.4 执行顺序

- [x] 1. 新增 B+L 组；用 ratio 0.1 × 3 seeds 跑 A / B / L / O。
  > **执行情况（2026-10-03）**：B > A（3 个 seed 都显著，平均 +0.118）；L > B 不成立（平均 −0.031）。详见 8.5，c 的设计分析见 8.6。
- [ ] 2. 复合权重改用 c、去掉 Discourse，再跑 W。
- [ ] 3. 根据结果判断：
  - B > A：半监督有效；
  - L > B：LogicScore 有效（第一轮不成立；改进为 Q 后，Q − K 平均 +0.007，仍未证明。需要先修正训练预算，见 8.7）；
  - W 不优于 B：复合加权的思路需要调整。
- [ ] 4. 根据第 3 步的结论决定 RL 的改法或是否保留，然后进入第 4 节第三步（完整矩阵）。

## 8. 更换数据集：Climate-FEVER + ClimateCheck + SciFact（不再使用 PUBHEALTH）

### 8.1 为什么更换

- PUBHEALTH 的标签是核查人员对 claim 真假的结论，而证据是从文章正文中按相似度挑出的句子，通常不含判定依据。用（修正后的）NLI 零样本判断，PUBHEALTH 的 macro-F1 只有 0.309，修正后的 LogicScore 方向一致性 AUROC 只有 0.507。**LogicScore 的前提在 PUBHEALTH 上不成立，论文的核心方法无法被检验。**
- 新数据集的证据都是会对 claim 给出支持/反驳/信息不足判断的文本，NLI 零样本 macro-F1：SciFact 0.621、Climate-FEVER 0.421、ClimateCheck 0.408。
- 原始数据位置：`../Data/ClimateCheck/`（HuggingFace `rabuahmad/climatecheck`，MIT 许可，2026-06 公开了测试集金标签）、`../Data/SciFact/data/`（AllenAI 官方 S3 release）。ClimateCheck 的 39.4 万篇文献库（894MB）未下载，只有做检索（方案二）时才需要。

### 8.2 数据构建（`configs/config_sci.yaml` + `data_prep/build_labeled.py`）

| 来源 | 样本单位 | 证据 | train / dev / test | 处理 |
|---|---|---|---|---|
| Climate-FEVER | claim | 全部 5 条候选句 | 935 / 200 / 200 | 丢弃 DISPUTED（154 条，新数据集没有"证据冲突"类）；同一 claim 文本在原始文件中有多个 id，按文本去重（丢弃重复 29 条、标签矛盾 16 条）；70/15/15 分层切分 |
| ClimateCheck | (claim, 摘要) 对 | 整篇摘要（去掉 "Abstract" 标题） | 2564 / 432 / 1904 | 官方 test 保持不变；dev 按 claim 从官方 train 切出 15%。同一 claim 配不同摘要可能有不同标签（train 中 60% 的 claim 如此） |
| SciFact | (claim, 引用摘要) 对 | 整篇摘要（不使用标注的证据句） | 790 / 127 / 339 | 官方 test 没有标签，官方 dev 作为 test；dev 按 claim 从官方 train 切出 15% |

- **跨数据集去重**：ClimateCheck 有部分 claim 改写自 Climate-FEVER（TF-IDF 余弦 ≥ 0.5 的 18 对）。与 ClimateCheck test 匹配的 Climate-FEVER claim 被丢弃（1 条），跨 split 的 ClimateCheck claim 被丢弃（7 个）。
- **跨 split 相同 claim**（如 SciFact 的 870/871）：只保留在 test > dev > train 中优先级最高的一侧。最终三个 split 之间没有相同的 claim 文本。
- 合计：train 4289 / dev 759 / test 2443。test 中 ClimateCheck 占 78%，所以同时报告按来源拆分的结果。
- `make_ssl_split.py` 按 claim 分组（记录的 `group` 字段）：同一 claim 的多个摘要始终在同一侧。没有 `group` 的旧数据，划分结果与原来完全相同（已回归验证）。ratio 0.1 时：有标签 436 对（226 个 claim），无标签池 3853 对。

### 8.3 配置与代码改动

与 `config.yaml` 的差异都在 `config_sci.yaml` 中标注了 `[sci]`：

| 设置 | 值 | 原因 |
|---|---|---|
| `data.sources` | `[climate_fever, climatecheck, scifact]` | 默认仍为 `[climate_fever, pubhealth]` |
| `data.max_evidences` | 5 | Climate-FEVER 选 3 条会丢掉 9%–19% 的关键证据 |
| `data.cf_disputed` / `data.dedup_claims` | `drop` / `true` | 见 8.2 |
| `training.max_length` | 512 | 证据是整篇摘要：384 时约 30% 被截断，512 时 13%。64 条 × 512 token 的显存峰值 12.5GB |
| `training.gradient_accumulation` | 1 | 有标签部分只有 436 对，累积 4 时 extractor 和 A 组都只有 56 次参数更新 |
| `imbalance.use_balanced_extractor_sampler` | `false` | 已有类别权重，再加平衡采样会重复校正 |
| `imbalance.apply_prior_adjust_in_pseudolabels` | `false` | 7.2.1 节 |
| `paths.processed_dir` / `runs_dir` / `results_dir` | `processed/sci` / `runs_sci` / `results_sci` | 与 PUBHEALTH 版的结果分开 |

其他代码改动：LogicScore 下标修复（7.2.0）；证据条数上限默认值 3 → 5（`ClaimEvidenceDataset`、`train_rl_selector.py`、`extractor.py`）；`train_extractor.py` 新增可选的 `training.extractor_max_epochs` / `training.extractor_gradient_accumulation`（不设置时沿用共享值）。

> 第一次尝试（梯度累积 4 + extractor 平衡采样器）中，extractor 把几乎所有样本预测为 REFUTES/NEI，验证 macro-F1 只有 0.221，已中止（日志 `logs/sci_validate_r0.10_s42.accum4_aborted.log`）。修正后为 0.473。

运行：

```bash
cd v2
python run_all.py --config configs/config_sci.yaml --ratios 0.1 --seeds 42 --methods A,B,O
```

### 8.4 低成本验证结果（ratio 0.1，seed 42，A / B / O）

日志 `logs/sci_validate_r0.10_s42.log`；结果 `runs_sci/r0.10_s42/`、`results_sci/`。

**detector 测试集 macro-F1**

| 组 | 伪标签条数 | 参数更新次数 | 全部 | ClimateCheck | Climate-FEVER | SciFact | AUC |
|---|---|---|---|---|---|---|---|
| A 纯监督 | 0 | 224 | 0.431 | 0.419 | 0.427 | 0.446 | 0.620 |
| B 置信度 ≥ 0.7 | 712 | 224 | **0.496** | 0.493 | 0.404 | 0.531 | 0.671 |
| O 金标签 | 3853 | 640 | **0.648** | 0.620 | 0.591 | 0.804 | 0.825 |

| 比较 | Δ macro-F1 | 95% CI（配对 bootstrap） |
|---|---|---|
| O − A | **+0.217** | (+0.189, +0.246) |
| B − A | **+0.065** | (+0.041, +0.091) |
| O − B | +0.151 | (+0.129, +0.178) |

只看 claim 的基线（TF-IDF + LR）：用同样 10% 的标签为 0.421，用全部 train 标签为 0.500。

**伪标签与信号**（LogicScore 已修正；关闭先验校正）

| 来源 | 伪标签准确率 | 置信度 AUROC | 方向一致性 c 的 AUROC | 置信度 ≥ 0.7 | 再加 c > 0.3（B+L） |
|---|---|---|---|---|---|
| 全部 | 0.497 | 0.554 | **0.593** | 712 条 / 0.522 | **287 条 / 0.690** |
| ClimateCheck | 0.524 | 0.578 | 0.536 | 422 条 / 0.604 | 157 条 / 0.688 |
| Climate-FEVER | 0.429 | 0.494 | 0.620 | 173 条 / 0.329 | 71 条 / 0.606 |
| SciFact | 0.492 | 0.555 | **0.748** | 117 条 / 0.513 | 59 条 / 0.797 |

不带方向的 |LogicScore| 和 Discourse 的 AUROC 分别为 0.519 和 0.505，仍然没有信号。

**结论**

- 半监督的提升空间从 PUBHEALTH 版的 +0.071 扩大到 **+0.217**。
- 最简单的半监督方法 B 显著优于 A（+0.065）。两者参数更新次数相同，这是一次干净的对比（PUBHEALTH 版只有 +0.007，不显著）。
- **修正后的 LogicScore（带方向）有信号**：区分伪标签对错的能力超过置信度，在 SciFact 上最明显。B+L 过滤把伪标签准确率从 0.522 提升到 0.690。
- 注意：
  - 只有一个 seed；
  - A 组（0.431）只比只看 claim 的基线（0.421）略高，说明只有 10% 标签时模型几乎没用上证据；
  - B 在 Climate-FEVER 上反而下降（0.427 → 0.404），与该子集伪标签只有 0.329 的准确率一致，而 B+L 能把它提到 0.606；
  - O 的参数更新次数是 A 的近 3 倍，O − A 有一部分来自训练更多。

### 8.5 下一步

- [x] 实现 B+L 组（方法 L，改动文件见 7.3 第二档），ratio 0.1 × 3 seeds 跑 A / B / L / O。
  > **执行情况（2026-10-03）**：`python run_all.py --config configs/config_sci.yaml --ratios 0.1 --seeds 42,43,44 --methods A,B,L,O`，日志 `logs/sci_ABLO_r0.10_s42-44.log`，汇总 `results_sci/`。seed 42 只补跑了 L。主机负载高（load average 25–33），seed 43 用了约 1.9 小时（extractor 20.9 分钟、O 41.7 分钟），总耗时约 3 小时。多 seed 报告：`python analysis/sci_multiseed_report.py`。
  >
  > **测试集 macro-F1（mean ± std，3 个 seed）**
  >
  > | 组 | 伪标签条数 | 参数更新 | 全部 | ClimateCheck | Climate-FEVER | SciFact |
  > |---|---|---|---|---|---|---|
  > | A 纯监督 | 0 | 221 | 0.448 ± 0.039 | 0.440 ± 0.042 | 0.401 ± 0.038 | 0.488 ± 0.050 |
  > | B 置信度 ≥ 0.7 | 1525 | 296 | **0.566 ± 0.066** | 0.557 ± 0.058 | 0.462 ± 0.082 | 0.643 ± 0.114 |
  > | L 置信度 + c > 0.3 | 773 | 232 | 0.535 ± 0.145 | 0.517 ± 0.143 | 0.526 ± 0.074 | 0.605 ± 0.199 |
  > | O 金标签 | 3855 | 645 | **0.661 ± 0.014** | 0.632 ± 0.012 | 0.595 ± 0.018 | 0.825 ± 0.021 |
  >
  > | seed | extractor 验证 F1 | A | B | L | O | B − A | L − B | O − A |
  > |---|---|---|---|---|---|---|---|---|
  > | 42 | 0.473 | 0.431 | 0.496 | 0.368 | 0.648 | +0.065* | **−0.128*** | +0.217* |
  > | 43 | 0.566 | 0.421 | 0.627 | 0.629 | 0.661 | +0.206* | +0.002 | +0.240* |
  > | 44 | 0.512 | 0.492 | 0.575 | 0.608 | 0.675 | +0.083* | +0.032* | +0.183* |
  >
  > （\* 为配对 bootstrap 95% CI 不含 0；每个 seed 单独计算。）
  >
  > **伪标签集合质量**（"平衡后准确率"= 各伪标签类别精度的平均，即平衡采样器下 detector 实际看到的期望准确率）
  >
  > | seed | 整池准确率 | B 条数 / 准确率 / 平衡后 | L 条数 / 准确率 / 平衡后 | c 的 AUROC | 置信度 AUROC |
  > |---|---|---|---|---|---|
  > | 42 | 0.497 | 712 / 0.522 / 0.557 | 287 / 0.690 / 0.723 | 0.593 | 0.554 |
  > | 43 | 0.555 | 2650 / 0.588 / 0.581 | 1473 / 0.602 / 0.687 | 0.546 | 0.586 |
  > | 44 | 0.556 | 1212 / 0.641 / 0.634 | 558 / 0.701 / 0.726 | 0.586 | 0.590 |
  >
  > **结论**
  >
  > - ✅ **B > A**：3 个 seed 都显著，平均 +0.118。半监督有效，B 拿到了 O 上界（平均 +0.213）的一半多。
  > - ❌ **L > B 不成立**：L − B 为 −0.128 / +0.002 / +0.032，平均 −0.031。L 的伪标签平衡后准确率在 3 个 seed 上都高于 B（+0.17 / +0.11 / +0.09），但没有稳定转化为 detector 的提升。
  > - seed 42 的 L 几乎不预测 REFUTES（测试集 2443 条中只预测 39 条，REFUTES F1 0.088），验证 F1 到第 8 个 epoch 仍在上升。这个 L 集合只有 287 条，其中 SUPPORTS 只有 66 条。seed 43/44 的 L 没有出现这个问题（REFUTES F1 0.571 / 0.595）。
  > - 方差很大：B 的标准差 0.066，L 的标准差 0.145，O 只有 0.014。seed 之间的差异主要来自 extractor 的质量（验证 F1 0.473–0.566）：extractor 越好，B 集合越大越准，B − A 越大。
  > - 原因分析和 c 的改进方案见 8.6。
- [x] 按 8.6 的建议实现 Q（L-q）和 K（同规模置信度对照），ratio 0.1 × 3 seeds 跑 Q / K。
  > **执行情况（2026-10-03）**：项目负责人确认采用 L-q。日志 `logs/sci_QK_r0.10_s42-44.log`，约 70 分钟。
  >
  > | 组 | 伪标签条数 | 参数更新 | 全部 | ClimateCheck | Climate-FEVER | SciFact |
  > |---|---|---|---|---|---|---|
  > | B | 1525 | 296 | 0.566 ± 0.066 | 0.557 ± 0.058 | 0.462 ± 0.082 | 0.643 ± 0.114 |
  > | L | 773 | 232 | 0.535 ± 0.145 | 0.517 ± 0.143 | 0.526 ± 0.074 | 0.605 ± 0.199 |
  > | Q（L-q） | 976 | 253 | 0.551 ± 0.050 | 0.539 ± 0.037 | 0.508 ± 0.065 | 0.624 ± 0.142 |
  > | K（同规模对照） | 976 | 253 | 0.544 ± 0.064 | 0.537 ± 0.061 | 0.486 ± 0.068 | 0.595 ± 0.111 |
  >
  > | seed | B | L | Q | K | Q − K | Q − B |
  > |---|---|---|---|---|---|---|
  > | 42 | 0.496 | 0.368 | 0.528 | 0.493 | **+0.035*** | +0.032* |
  > | 43 | 0.627 | 0.629 | 0.609 | 0.616 | −0.007 | −0.019 |
  > | 44 | 0.575 | 0.608 | 0.516 | 0.523 | −0.007 | **−0.059*** |
  >
  > - 伪标签层面，Q 的平衡后准确率比 K 高 0.03 / 0.05 / 0.07（见 8.6），与离线分析一致。
  > - **detector 层面 Q − K 平均只有 +0.007**：只有 seed 42 显著为正，另外两个 seed 无差异。Q 的标准差（0.050）明显小于 L（0.145），说明不再出现 L 在 seed 42 上那种崩溃。
  > - **LogicScore 在 detector 层面的作用仍未得到证明**，原因很可能是 detector 训练本身的随机性太大，见 8.7。
- [ ] 按 8.7 修正 detector 的训练预算，再重跑 A / B / Q / K（待项目负责人决定）。
- [ ] 复合权重改用方向一致性 c、去掉 Discourse，再跑 W。
- [ ] 按 7.3 第三档重新设计 RL 的奖励和状态（使用 c），再跑 R / C。
- [ ] 确定配置后跑完整矩阵。

### 8.6 方向一致性 c 的设计分析

分析脚本 `analysis/c_design_probe.py`：对 train 全部 4289 个句对重新计算 NLI 三类概率（缓存在 `analysis/nli_train_probs.npz`，与已保存的 LogicScore 相关系数 1.0000），在 3 个 seed 的伪标签池上离线比较多种方案。

**1. c 对 SUPPORTS / REFUTES 有效，对 NEI 基本无效**

c 在各伪标签类别内部区分对错的 AUROC（只在类内比较，排除类别构成的影响；3 个 seed）：

| 伪标签类别 | 置信度 | c | NLI 对该类的概率 |
|---|---|---|---|
| SUPPORTS | 0.565 / 0.623 / 0.596 | **0.676 / 0.676 / 0.679** | 0.649 / 0.648 / 0.651 |
| REFUTES | 0.636 / 0.703 / 0.676 | **0.805 / 0.694 / 0.677** | 0.802 / 0.695 / 0.681 |
| NEI | 0.564 / 0.561 / 0.585 | 0.584 / 0.557 / 0.577 | 0.566 / 0.544 / 0.570 |

按来源（3 个 seed 平均，置信度 → c）：ClimateCheck SUP 0.589 → 0.640、REF 0.660 → 0.687；Climate-FEVER SUP 0.652 → 0.667、REF 0.681 → 0.724；SciFact SUP 0.586 → **0.855**、REF 0.692 → **0.822**。在占测试集 78% 的 ClimateCheck 上 c 也优于置信度，只是幅度小。

**2. 问题出在用法，而不是 c 本身**

- **固定阈值 c > 0.3 对 SUPPORTS 太严**。53%–63% 的样本 |LS| < 0.01，SUPPORTS 要求 LS > 0.3，也就是 NLI 判为明显蕴含，只有约 20% 的 SUPPORTS 能通过（seed 42：296 → 66）。L 集合因此变小，类别也偏向 REFUTES（seed 42 中 REFUTES 占 53%）。
- **对 NEI 几乎不起作用**。NEI 的 c = 1 − |LS|，78%–93% 的 NEI 伪标签都能通过。改用 NLI 的 P(中立) 也没有改善（L′ / L″ 方案与 L 几乎相同），因为 NLI 的"中立"和数据集的 NEI 不是一回事，类内 AUROC 只有 0.54–0.57。**NLI 帮不了 NEI，不值得为 NEI 再设计 c。**
- **REFUTES 的错误大多是金标签 NEI**（seed 42/43/44：45/59、109/132、86/108）：证据不足或不相关时，NLI 也会判为矛盾（7.2.0）。
- **L vs B 同时改变了质量和数量**：L 只有 B 的 40%–56%。在同等规模下比较，c 的选择确实优于置信度：

  | seed | L：条数 / 平衡后准确率 | 同规模按置信度取前 N 条 |
  |---|---|---|
  | 42 | 287 / **0.723** | 287 / 0.610 |
  | 43 | 1473 / **0.687** | 1473 / 0.642 |
  | 44 | 558 / **0.726** | 558 / 0.647 |

  所以 L − B 的 detector 结果混合了"挑得更准"和"样本更少"两种效应，不能直接说明 LogicScore 没用。

**3. 候选方案（离线结果，3 个 seed：条数 / 准确率 / 平衡后准确率）**

| 方案 | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| B（现行） | 712 / 0.522 / 0.557 | 2650 / 0.588 / 0.581 | 1212 / 0.641 / 0.634 |
| L（现行，c > 0.3） | 287 / 0.690 / 0.723 | 1473 / 0.602 / 0.687 | 558 / 0.701 / 0.726 |
| **L-q：SUP / REF 各自按 c 保留前 50%，NEI 不用 c** | 401 / 0.651 / 0.653 | 1854 / 0.616 / 0.666 | 673 / 0.722 / 0.704 |
| L-q 的同规模置信度对照 | 401 / 0.496 / 0.625 | 1854 / 0.620 / 0.615 | 673 / 0.629 / 0.634 |
| L-q，保留前 70% | 525 / 0.610 / 0.621 | 2172 / 0.609 / 0.628 | 888 / 0.702 / 0.682 |
| F：extractor 与 NLI 概率融合（α = 0.3），融合置信度 ≥ 0.7 | 875 / 0.626 / 0.674 | 1799 / 0.599 / 0.680 | 917 / 0.678 / 0.706 |
| NLI argmax 与伪标签一致 | 273 / 0.685 / 0.720 | 1420 / 0.600 / 0.687 | 538 / 0.697 / 0.720 |

- **L-q（类内相对阈值）**：SUPPORTS 不会被大量砍掉，集合比 L 大 20%–40%，类别更均衡，平衡后准确率仍比同规模的置信度对照高 0.03–0.07。阈值按每个 run 的分布自适应，不依赖 |LS| 的绝对大小。
- **F（融合）**：会改动伪标签本身（NLI 可以纠正 extractor），集合更大，但 NEI 占比高、精度只有 0.51–0.58。属于 7.3 第三档的方向。
- "NLI argmax 一致"与现行 L 几乎相同，没有额外价值。

**建议**（✅ 2026-10-03 已采用，实现为方法 Q / K，结果见 8.5：伪标签层面有效，detector 层面被训练噪声掩盖，见 8.7）

1. 用 **L-q**（SUP/REF 类内按 c 取前 50%，NEI 只用置信度）替换现行 L 的固定阈值。
2. 同时增加一个**同规模置信度对照组**（按置信度取前 N 条，N 与 L-q 相同），把"挑得更准"和"样本更少"分开。L-q 与这个对照组只差选样规则，是检验 LogicScore 最干净的对比。
3. 伪标签池已经存在，只需要重跑 detector：2 组 × 3 个 seed，每组约 7–10 分钟（主机空闲时），合计约 1 小时。

### 8.7 新问题：小集合的 detector 训练不充分，随机性大于要检验的效果

**现象**

- 几乎相同的伪标签集合，测试 macro-F1 相差很大：
  - seed 42：L 的 287 条全部包含在 Q 的 401 条里，但 L 为 0.368、Q 为 0.528；
  - seed 44：L 有 88% 在 Q 里，但 L 为 0.608、Q 为 0.516。
- 验证集与测试集一致（最佳验证 F1 ≈ 测试 F1），所以不是模型选择的问题，而是训练本身不稳定。

**原因**

- A / B / L / Q / K 的伪标签集合小于"有标签 batch 数 × pseudo_ratio × batch 大小"（ratio 0.1 时约 1344 条），在 `cover_pseudo` 下都只有 224–296 次参数更新。O 有 645 次。
- 小集合组的验证 F1 在第 3–6 个 epoch 之间随机"起飞"，很多组到第 8 个 epoch 仍在上升（seed 42 的 A、L，seed 44 的 A、Q、K）。O 在第 2 个 epoch 就起飞，之后保持平稳。
- seed 43 上各组都已收敛，B / L / Q / K 都在 0.61–0.63，几乎没有差异。

  | seed 42 验证 macro-F1（epoch 1 → 8） | |
  |---|---|
  | L | 0.21 0.21 0.32 0.27 0.27 0.37 0.40 **0.42** |
  | Q | 0.10 0.27 0.38 0.49 0.54 0.54 **0.57** 0.55 |
  | O | 0.31 0.60 0.66 0.67 **0.68** 0.66 0.68 0.66 |

- 因此，现在小集合组之间（包括 B vs A）的差异，有相当一部分取决于"在 224 步内有没有起飞"，而不是伪标签本身。B − A 也受影响，但 3 个 seed 都显著，结论不变。

**方案（待决定）**

1. **统一并加大训练预算**（推荐）：所有组使用相同的总参数更新次数（例如与 O 相同的约 640 次，或增加 `max_epochs` 并配合早停）。这样各组只差伪标签，不再差训练量。需要在 `train_detector.py` 中增加按总步数训练的选项。预计 A / B / Q / K 各 3 个 seed 约 3 小时。
2. **每个伪标签集合重复训练多次**（不同的训练 seed），用平均值降低训练噪声。代价是成倍的计算量，可以与方案 1 结合，只用在关键对比（Q vs K）上。
3. 在训练预算修正之前，不再在小集合组之间做 detector 层面的比较，只比较伪标签质量。

