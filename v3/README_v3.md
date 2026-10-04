# Semi-vast v3：统一训练协议后的消融实验

> **状态（2026-10-05 02:30）**：第一阶段试跑未通过（第 10 节）→ 改为方向一（logic-aware teacher，方法 F）。**方向一主实验完成：主要检验 F − A = +0.049，95% CI (+0.035, +0.062)，5/5 个 seed 为正，成立**；F − B、F − A⊕NLI 也成立（10.9 节）。

v3 来自对 v2 训练逻辑的全面审查（`../v2/README_v2.md` 第 9 节）。v2 每跑一轮都会发现新问题：O ≈ A、LogicScore 下标错误、L 组崩溃、训练噪声大于要检验的效果。v3 的做法是：

1. **先修正审查发现的全部问题，再跑实验**；
2. **方案在跑之前定死**：检验什么、用什么判据，都在第 4 节写好，不事后挑选；
3. **每次运行自动检查**（第 6 节），问题在运行结束时直接报出来，不用再人工翻日志；
4. **先试跑、过关再跑主实验**（第 5 节），试跑的通过标准也事先写好。

v3 是独立目录，没有改动 v2 的任何文件。数据构建、SSL 划分、extractor、伪标签生成与 v2 完全相同，所以 v2 在新数据集上已经生成的伪标签池（seed 42–44）可以直接导入复用。

---

## 1. v2 的问题与 v3 的对应修改

| # | v2 的问题 | 证据（v2 README 第 8.5、8.7、9 节） | v3 的修改 |
|---|---|---|---|
| 1 | detector 从原始 deberta-v3-large 初始化，extractor 从 NLI 模型初始化 | extractor 的验证 F1 在 3 个 seed 上都 ≥ A（seed 43：0.566 vs 0.424） | **M1** detector 与 extractor 同样从 NLI 模型初始化、冻结前 6 层 |
| 2 | 训练预算小且各组不同（224–648 步），LR 与 λ 按各自的总步数衰减 | 小集合组的验证 F1 在随机时间点起飞；L ⊂ Q 的两个集合测试 F1 相差 0.16 | **M2** 所有组固定 640 步，每 40 步评估一次 |
| 3 | logit adjustment 训练时用了 `−τ·log π`（符号相反），又与 focal、类别权重叠加 | 三种校正互相抵消，REFUTES 表现不稳定 | **M3** 监督损失只保留带类别权重的 CE（与 extractor 相同）；符号已修正 |
| 4 | 固定置信度阈值 0.7，extractor 没有校准 | 池中置信度 ≥ 0.7 的比例为 18.5% / 68.7% / 31.4%，B 的规模相差 3.7 倍 | **M4** B = 置信度前 30%；Q、K 都在 B 内且规模相同 |
| 5 | 只有每个 seed 内的 bootstrap | 没有跨 seed 的结论 | **M8** 分层 bootstrap + 事先确定的判据 |
| 6 | 工程问题 | 每次出现新最佳就把 1.7GB 写盘；有标签 loader 最后一个 batch 只有 4 条；无法单独估计训练噪声 | **M5** 最佳权重存内存、`drop_last`；**M6** `--train_seed` |
| 7 | 每跑一轮才人工发现问题；改了配置后旧结果被静默沿用 | — | **M7** 自动检查 S1–S7；配置指纹防止混用 |

---

## 2. 目录结构

```
v3/
├── README_v3.md                    本文档
├── HANDOFF.md                      交付文档（现状、已做的验证、下一步、已知的坑）
├── run_pilot.sh                    第一阶段：试跑（seed 42）
├── run_main.sh                     第二阶段：主实验（seed 42–46）
├── run_all.py / run_all.sh         一键运行（各阶段脚本都调用它）
├── configs/config.yaml             唯一配置；与 v2 config_sci.yaml 的差异标注了 [v3]
├── common/
│   ├── paths.py                    路径、run 目录布局、方法列表、detector 结果的命名（B / B_t1042）
│   ├── fingerprint.py              [新] 配置指纹；决定伪标签池的设置（导入时比对）
│   └── data_utils.py               句对 Dataset、动态补齐、方向一致性 c、[新] 共用的类别权重
├── data_prep/                      与 v2 相同：build_labeled.py、make_ssl_split.py
├── models/
│   ├── detector.py                 [改] NLI 初始化与冻结、logit adjustment 符号、伪标签损失不再做校正
│   └── extractor.py、logic_scorer.py、discourse_scorer.py、rl_selector.py   与 v2 相同
├── training/
│   ├── train_detector.py           [重写] 固定步数训练、按步评估、最佳权重存内存、--train_seed
│   ├── build_baseline_sets.py      [改] B / Q / K 按比例选样；L 被 Q 取代
│   ├── generate_pseudolabels.py    [改] 伪标签先验校正改用自己的 τ（默认关闭，结果不变）
│   └── train_extractor.py、train_rl_selector.py   与 v2 相同
├── evaluation/
│   ├── aggregate_results.py        [重写] 跨 seed 汇总、分层 bootstrap、事先确定的检验、训练噪声、警告汇总
│   ├── sanity_check.py             [新] 每个 run 结束后的自动检查 S1–S7
│   ├── pilot_report.py             [新] 试跑的通过标准 C1–C5
│   └── pseudo_label_quality.py     伪标签质量（B / Q / K 的准确率、平衡后准确率、各信号 AUROC）
├── tools/import_from_v2.py         [新] 导入 v2 的划分与伪标签池（带一致性检查）
└── logs/watch.sh、progress.sh      实时进度 / 进度快照
```

运行时生成（都被 `.gitignore` 的 `*.jsonl` / `*.log` / `processed/` 规则覆盖，json 与 png 会被跟踪）：

```
processed/labeled/{train,dev,test}.jsonl     有标签数据（从 v2 复制，逐字节相同）
runs/protocol.json                           这个 runs 目录使用的配置指纹与完整配置
runs/r0.10_s42/
    imported_from.json                       从 v2 导入时的来源与文件校验和
    data/  pseudo/                           划分、伪标签池、B/Q/K/O 集合（baseline_sets_stats.json）
    outputs/detector/<组>/                   test_results.json、test_predictions.jsonl、train_metrics.json、训练曲线
    outputs/detector/<组>_t1042/             同一划分换训练 seed 的重复训练（只用于估计训练噪声）
    outputs/sanity_check.json                自动检查
    outputs/pilot_report.json                试跑通过标准（只在试跑的 run 中）
results/summary.md                           总表（第 7 节）
logs/*.log                                   运行日志
```

---

## 3. 相对 v2 的改动

### 3.1 detector 的初始化（M1）

- `training.detector_init_from_nli: true`：detector 的编码器从 `cross-encoder/nli-deberta-v3-large` 加载（与 extractor 相同），加载失败直接报错，不会悄悄退回原始模型。
- `training.detector_freeze_layers: 6`：与 extractor 一样冻结编码器前 6 层（词嵌入不冻结）。
- 结果文件记录 `encoder_source` 和 `frozen_layers`，可以核对。
- 与 extractor 仍有的差别：分类头（detector 为两层 MLP，extractor 为单层线性）、训练预算（detector 按步数）、动态补齐。试跑标准 C3 检查 A 的验证 F1 是否不低于 extractor。

### 3.2 固定训练预算（M2）

- 每个组都训练 `training.detector_total_steps = 640` 次参数更新（v2 中 O 的步数），每 `eval_every = 40` 步在 dev 上评估一次，共 16 次，取验证 macro-F1 最高的一步。
- 每一步 = 1 个有标签 batch（16 对）+ `pseudo_ratio = 3` 个伪标签 batch（48 对）；A 只有有标签 batch。两个 loader 都循环使用。
- warmup（10%）与线性衰减、λ 从 0.3 线性升到 0.7，都按同一个总步数计算，**所有组完全相同**。
- 代价：小集合会被反复使用。640 × 48 = 30,720 次抽样：Q / K 约 650–750 条，平均每条约 40–47 次；B 约 1,157 条，约 27 次；O 3,855 条，约 8 次。Q 与 K 规模相同，所以这一点不影响 H2 的比较；dev 上的模型选择会挑在过拟合之前的那一步。

### 3.3 损失（M3）

- 监督损失：带类别权重的 CE。权重 = `(N / (3·n_c))^1.5` 后归一化到均值 1，与 extractor 完全相同（`common.data_utils.class_weights_from_labels`，已与 extractor 日志中的权重逐位核对）。
- `algorithm.logit_adjust_tau: 0`：关闭 logit adjustment。代码中的符号已改为训练时 `logits + τ·log π`（Menon et al., 2021），以后需要时可以打开。
- 伪标签损失：普通 CE，配合平衡采样器（与 v2 相同）；不再做 logit adjustment。
- `imbalance.pseudolabel_prior_tau`：伪标签生成时的先验校正改用独立的键。v2 中它与 detector 损失共用 `algorithm.logit_adjust_tau`，改损失会悄悄改变伪标签。这个校正默认关闭，所以伪标签与 v2 完全相同。

### 3.4 选样（M4）

| 组 | 定义 |
|---|---|
| B | 池中置信度最高的 `confidence_top_fraction = 30%` |
| Q（L-q） | 在 B 内：SUPPORTS、REFUTES 各自按方向一致性 c 降序保留前 `consistency_quantile = 50%`；NEI 全部保留（c 对 NEI 没有区分能力，v2 README 8.6） |
| K | 按置信度取前 \|Q\| 条（因此 K ⊂ B，且 \|K\| = \|Q\|） |

- c 的定义不变：伪标签为 SUPPORTS 时 c = LS，REFUTES 时 c = −LS，NEI 时 c = 1 − \|LS\|，LS = P(蕴含) − P(矛盾)。
- 排序在置信度或 c 相同时按 id 打破平局，结果与文件中的顺序无关（已验证）。
- 在 v2 的伪标签池上离线计算（条数 / 准确率 / 平衡后准确率）：

  | seed | B | Q | K | Q − K（平衡后） |
  |---|---|---|---|---|
  | 42 | 1156 / 0.535 / 0.519 | 712 / 0.622 / 0.619 | 712 / 0.522 / 0.557 | +0.062 |
  | 43 | 1157 / 0.659 / 0.667 | 743 / 0.724 / 0.746 | 743 / 0.664 / 0.682 | +0.064 |
  | 44 | 1157 / 0.644 / 0.637 | 642 / 0.726 / 0.707 | 642 / 0.620 / 0.626 | +0.081 |

  B 的规模在各 seed 间一致（v2 为 712 / 2650 / 1212）；B 中最低的置信度为 0.654 / 0.897 / 0.704。

### 3.5 工程（M5、M6）

- 最佳权重复制到 CPU 内存，训练结束后直接评估测试集，不再反复写 1.7GB 的 checkpoint；需要保留时加 `--keep_checkpoints`。
- 有标签 loader `drop_last=True`。
- `--train_seed`（`run_all.py` 中为 `--extra_train_seeds`）：同一划分换训练 seed 重复训练，结果写到 `<组>_t<seed>`，只用于估计训练噪声，不进入组间比较。

### 3.6 没有改动的部分

数据构建（`data_prep/`）、SSL 划分、extractor 的训练、伪标签生成、LogicScore、Discourse 与 v2 完全相同。W / R / C（复合权重与 RL）的代码也保留原样，但它们仍是 v2 的设计（复合权重用不带方向的 \|LogicScore\|、RL 奖励近似常数），不在默认方法列表中，结果不能用于结论（第 8 节）。

---

## 4. 消融组与事先确定的检验

| 组 | 含义 |
|---|---|
| A | 纯监督（与 extractor 同样的初始化和损失）——最强的纯监督基线 |
| B | 置信度前 30% 的伪标签 |
| Q | B 内按 c 选：SUPPORTS / REFUTES 各留一半，NEI 全留 |
| K | 与 Q 同规模，按置信度选 |
| O | 整个无标签池用金标签——上界 |

| 编号 | 比较 | 回答的问题 |
|---|---|---|
| H1 | B − A | 伪标签在最强的纯监督基线之上是否有效 |
| **H2** | **Q − K** | **同样规模下，LogicScore（c）能否选出更有用的伪标签（主要检验）** |
| H3 | Q − B | 用 c 过滤（更小但更准）是否优于不过滤（次要，混有规模差异） |
| — | O − A | 提升空间 |

- **主指标**：测试集整体 macro-F1。**次要指标**：三个来源 macro-F1 的平均值（ClimateCheck 占测试集 78%，避免它主导结论）。
- **判据**：5 个 seed 的平均 Δ 的分层 bootstrap 95% CI 下界 > 0，且至少 ⌈0.8·n⌉ 个 seed（5 个中至少 4 个）的 Δ > 0，才算"成立"；反方向同理为"显著更差"；其余为"不成立"。
- **分层 bootstrap**：每次先有放回地重抽 seed，再在每个 seed 内对测试样本做配对重抽，算两组 macro-F1 之差的平均值，重复 2000 次。它同时反映划分 / 训练的随机性和测试样本的抽样误差。
- 只用主训练 seed（训练 seed = 划分 seed）的结果；重复训练只进入训练噪声表。

---

## 5. 运行

环境与 v2 相同（`HANDOFF.md` 第 8 节）。**先执行 `export OMP_NUM_THREADS=8`**（当前 shell 的 0 是非法值）。所有命令都在 `v3/` 下运行。

### 5.1 准备：导入 v2 的伪标签池

```bash
cd /root/autodl-tmp/Semi-vast/v3
python tools/import_from_v2.py --seeds 42,43,44
```

- 复制 `processed/labeled/` 与每个 run 的 `data/`、伪标签池、extractor 训练记录；B/Q/K/O 集合和 detector 结果由 v3 重新生成。
- 导入前检查：两边决定伪标签池的设置完全相同（`common/fingerprint.pool_settings`）；已有文件内容相同才跳过、不同则报错；每个 run 内无标签池、金标签、伪标签池的 id 一致，有标签部分与无标签池不重叠。
- `run_pilot.sh` 和 `run_main.sh` 会自动执行导入，可以重复运行。
- 导入的 run 没有 extractor 权重，**不能在这些 run 上跑 C / R**（`run_all.py` 会拒绝）。

### 5.2 第一阶段：试跑（seed 42，约 1.5 小时）

```bash
cd /root/autodl-tmp/Semi-vast/v3
nohup bash run_pilot.sh > logs/pilot.log 2>&1 &
bash logs/watch.sh            # 实时进度，Ctrl+C 退出不影响训练
```

内容：A、B 各用训练 seed 42 和 1042 训练一次，再训练 O；最后由 `evaluation/pilot_report.py` 打印下表并写入 `runs/r0.10_s42/outputs/pilot_report.json`。

| 编号 | 通过标准 | 不通过时怎么办 |
|---|---|---|
| C1 | 所有组的最佳验证点不在最后 10% 的步数内（已收敛） | 增加 `detector_total_steps`（例如 960） |
| C2 | 换训练 seed，A 和 B 的测试 macro-F1 变化都 < 0.02 | 训练噪声仍大：考虑每个集合训练 2 个 seed 取平均（主实验时间翻倍），或降低学习率并加步数 |
| C3 | A 的最佳验证 F1 ≥ extractor（seed 42 为 0.473） | 检查日志中的 `encoder=` 与 `frozen_layers=`；必要时把分类头改为与 extractor 相同的单层线性 |
| C4 | O − A ≥ 0.10 | 提升空间不足，先与项目负责人讨论 |
| C5 | 每类的预测数 ≥ 该类金标签数的 50% | 检查类别权重；考虑 `class_weight_power: 1.0` |

**试跑未通过、需要改配置时**：改动会改变配置指纹，`run_all.py` 会拒绝在 `runs/` 里继续跑。复制一份配置（例如 `configs/config_b.yaml`），把 `paths.runs_dir` / `paths.results_dir` 改成新目录（例如 `runs_b` / `results_b`），然后：

```bash
CONFIG=configs/config_b.yaml nohup bash run_pilot.sh > logs/pilot_b.log 2>&1 &
```

改动 `[pool]` 设置（数据、extractor、伪标签生成）会让导入失败，那样所有 seed 都要从 extractor 重新开始，尽量避免。

### 5.3 第二阶段：主实验（A / B / Q / K / O × seed 42–46，约 7.5–9 小时）

```bash
cd /root/autodl-tmp/Semi-vast/v3
nohup bash run_main.sh > logs/main.log 2>&1 &
```

- `run_main.sh` 先确认试跑已通过（`pilot_report.json` 中 `"passed": true`），否则直接退出。
- seed 42 中试跑已完成的 A / B / O 会被跳过；42–44 用导入的伪标签池；45、46 从 extractor 开始。
- 每个 run 结束后自动做伪标签质量评估和自动检查；全部结束后生成 `results/summary.md`。
- 中断后重新执行同一条命令即可继续：已完成、且配置指纹相同的 detector 会被跳过。

### 5.4 查看进度与停止

```bash
bash logs/watch.sh                    # 实时：关键行逐行打印，进度条原地刷新
watch -n 10 bash logs/progress.sh     # 快照：步骤、当前组的验证记录、自动检查警告、GPU
python run_all.py --dry_run           # 只打印将要执行 / 跳过的步骤，不运行

pgrep -af "[r]un_all.py|[t]rain_detector.py"   # 查 PID 后 kill <PID>；不要用 pkill -f "<命令行>"
```

### 5.5 耗时估计（主机空闲时）

| 步骤 | 耗时 |
|---|---|
| 伪标签组（B / Q / K / O）的 detector，640 步 + 16 次评估 | 约 20 分钟 |
| A 的 detector | 约 7 分钟 |
| 新 seed 的 extractor + 伪标签 | 约 8 分钟（主机忙时可达 25 分钟） |
| 试跑（A×2、B×2、O） | 约 1.5 小时 |
| 主实验 | 约 7.5 小时，主机忙时更久 |

---

## 6. 自动检查与防止混用

### 6.1 配置指纹

- `common/fingerprint.config_fingerprint`：对所有影响结果的设置取哈希。只决定"跑哪些、放在哪"的键（`experiment.label_ratios / seeds / methods / device / cleanup_checkpoints`、`paths`、`training.num_workers / seed`）不参与，所以增加 seed 或方法不会改变指纹。
- `run_all.py` 第一次运行时把指纹写进 `runs/protocol.json`；之后配置的指纹不同就拒绝运行（换目录，或 `--force_protocol`）。
- 每个 detector 结果记录指纹：已有结果的指纹与当前相同才跳过，否则重跑。
- `aggregate_results.py` 发现同一目录里有不同指纹的结果时报错（`--allow_mixed` 可跳过）。

### 6.2 每个 run 结束后的检查（`evaluation/sanity_check.py`）

| 编号 | 检查 | 对应 v2 中人工才发现的问题 |
|---|---|---|
| S1 | 同一 run 内各 detector 的参数更新次数相同，且等于配置值 | 组间训练量不同 |
| S2 | 最佳验证点不在最后 10% 的步数内 | L、Q、A 到最后一个 epoch 还在上升 |
| S3 | 测试集上每类的预测数 ≥ 该类金标签数的 50% | L 只预测出 39 条 REFUTES |
| S4 | \|Q\| = \|K\| | 同规模对照失效 |
| S5 | 结果的配置指纹与当前配置一致 | 改配置后沿用旧结果 |
| S6 | \|B\| = round(30% × 池大小) | 选样规则没有生效 |
| S7 | 结果中的伪标签条数与当前集合文件一致 | 集合重建后 detector 结果过期 |

只警告、不中断；结果写入 `outputs/sanity_check.json`，所有警告汇总在 `results/summary.md` 第 5 节。

---

## 7. 结果文件

`results/summary.md`（同时有对应的 csv）：

1. 各组测试集 macro-F1 的 mean ± std：整体、来源平均、三个来源，以及 accuracy、AUC（`summary.csv`；逐个 run 见 `per_seed.csv`）；
2. 事先确定的检验 H1–H3 与 O − A：逐 seed Δ、均值 ± 标准差、Δ > 0 的 seed 数、95% CI、结论；来源平均指标同样给出（`hypotheses.csv`）；
3. 训练噪声：同一划分换训练 seed 的差异（`training_noise.csv`）；
4. 伪标签集合质量：B / Q / K 的条数、准确率、平衡后准确率，c 与置信度的 AUROC（`pseudo_quality.csv`）；
5. 自动检查的警告。

每个 detector 的 `test_results.json` 包含：配置指纹、编码器来源与冻结层数、实际与计划的步数、最佳步数与最佳验证指标、伪标签集合大小与类别分布、测试集各类的金标签数与预测数、测试指标、训练耗时。

---

## 8. 已知限制

- **W / R / C 未修订**：复合权重仍用不带方向的 \|LogicScore\| 和没有信号的 Discourse；RL 奖励中 0.3·mean\|LS\| 近似常数（0.224 ± 0.006，ΔF1 项只有 ±0.007）。在第二阶段的 H2 结论出来之前不修订、不使用。
- **小集合重复使用**：见 3.2 节。Q 与 K 规模相同，H2 不受影响；H3（Q vs B）混有规模差异。
- **A 与 extractor 并不完全相同**：分类头、训练预算、补齐方式不同（3.1 节）。
- **extractor 过度自信**：没有改（它是方法的一部分），由按比例选样绕开。
- **dev 同时用于 extractor 和 detector 的模型选择**；测试集只在最后评估一次。
- **非完全确定性**：`cudnn.benchmark` 等使同一设置的重复结果有微小差异，这部分包含在训练噪声里。
- **5 个 seed 的 CI 仍然较宽**：判据同时要求 CI 与方向一致性；结论不成立时，要区分"效果为 0"和"功效不足"（看 CI 的宽度和均值）。

---

## 9. 执行清单

- [x] 0. 导入 v2 的伪标签池（`python tools/import_from_v2.py --seeds 42,43,44`）
  > 执行情况（2026-10-04）：seed 42 已导入（`run_pilot.sh` 自动执行）：设置一致性检查通过，有标签 436 条，无标签池 3853 条，复制 14 个文件。方向一中 seed 42–44 都导入了 `runs_f/`（`run_main_f.sh`），一致性检查全部通过。
- [ ] 1. 第一阶段试跑（`run_pilot.sh`），C1–C5 全部通过
  > 执行情况（2026-10-04 15:56–17:23，日志 `logs/pilot.log`，报告 `runs/r0.10_s42/outputs/pilot_report.json`）：**未通过**。C1 ✅ 全部收敛（最佳步 120–480 / 640）；C2 ✅ A |Δ| = 0.0040、B |Δ| = 0.0118；C3 ✅ A 最佳验证 F1 0.5426 ≥ extractor 0.4727；**C4 ❌ O − A = +0.0986**（阈值 0.10，95% CI (+0.077, +0.119)）；**C5 ❌ A_t1042 的 REFUTES 只预测 143 条 / 金标签 372 条**。耗时：A 7.4 分钟，B / O 约 23 分钟（每步约 1.95 秒），显存约 8.3GB（A）。详见第 10 节。
  > 方向一（2026-10-04）：项目负责人选择先做方向一（10.5 节）。离线验证通过（10.6 节），18:09–19:37 方向一试跑（`run_pilot_f.sh`，日志 `logs/pilot_f.log`）：**未通过**。G1 ❌：F − A = +0.046 / +0.013（阈值 0.02）；G2 ✅：F − A⊕NLI = +0.074 / +0.058。详见 10.7.1 节。项目负责人决定仍进入方向一主实验（选项乙，10.8 节）。
- [x] 2. 第二阶段主实验（`run_main.sh`），`results/summary.md` 生成且没有未解释的警告
  > 执行情况：原方案（`run_main.sh`，A/B/Q/K/O）没有运行。改为方向一的主实验（`run_main_f.sh`，A/B/F/O × seed 42–46，2026-10-04 19:43 – 10-05 02:22，日志 `logs/main_f.log`，汇总 `results_f/summary.md`）。2 条警告都已解释（10.9 节，限制第 4 条）。
- [x] 3. 解读：H1（伪标签是否有效）、H2（LogicScore 是否有效）、H3
  > 执行情况（2026-10-05）：按方向一的检验解读（10.9 节）。**F1（F − A）+0.049，CI (+0.035, +0.062)，5/5，成立**；F2（F − B）+0.041 成立；F3（F − A⊕NLI）+0.053 成立；H1（B − A）+0.008，不成立。H2 / H3（Q、K）没有运行：LogicScore 已经以融合 teacher 的形式用在 F 中。
- [ ] 4. 第三阶段（根据 H2）：
  - H2 成立：把 c 用进复合权重（W：SUPPORTS / REFUTES 用 c，NEI 只用置信度），再决定 RL 的改法（奖励去掉常数项，状态加入类别和 c）；
  - H2 不成立：LogicScore 只作为伪标签质量分析的结论，论文方法部分需要调整，RL 大概率不再保留。
- [ ] 5. 结论稳定后，用同一协议跑 ratio 0.25 / 0.5（`experiment.label_ratios`）

每完成一步，在这里打勾，并写"执行情况"（实际数字、日志位置、遇到的问题）。

---

## 10. 试跑结果与问题（2026-10-04）

### 10.1 结果（seed 42，ratio 0.1）

| 组 | 测试 macro-F1 | 来源平均 | 最佳步 / 640 | 最佳验证 F1 | 测试预测数 SUP / REF / NEI（金标签 983 / 372 / 1088） | 每类 F1 SUP / REF / NEI |
|---|---|---|---|---|---|---|
| A | 0.5168 | 0.5218 | 200 | 0.5426 | 836 / 306 / 1301 | 0.522 / 0.469 / 0.559 |
| A_t1042 | 0.5128 | — | 480 | 0.5436 | 1181 / **143** / 1119 | 0.609 / 0.377 / 0.553 |
| B | 0.5096 | 0.4902 | 120 | 0.5319 | 857 / 413 / 1173 | 0.520 / 0.471 / 0.538 |
| B_t1042 | 0.5213 | — | 280 | 0.5263 | 1285 / 380 / 778 | 0.617 / 0.460 / 0.487 |
| O | 0.6154 | 0.6388 | 280 | 0.6097 | 1184 / 358 / 901 | — |

- 通过标准：C1 ✅、C2 ✅（A 0.0040 / B 0.0118）、C3 ✅（0.5426 vs 0.4727）、**C4 ❌（O − A = +0.0986）**、**C5 ❌（A_t1042 REFUTES 143/372）**。
- 自动检查：S1、S2、S5、S6、S7 通过；S3 警告与 C5 相同；S4 不适用（试跑不含 Q/K）。
- 伪标签池（3853 条）准确率 0.497；B（1156 条）准确率 0.535，平衡后 0.519；各伪标签类别精度 SUP 0.608 / REF 0.328 / NEI 0.452。
- 判断伪标签对错的 AUROC：方向一致性 c 为 0.593，置信度为 0.554。按来源，c 在 SciFact 上为 0.748、在 Climate-FEVER 上为 0.620、在 ClimateCheck 上为 0.536（测试集 78% 来自 ClimateCheck）。

### 10.2 C4：提升空间刚好低于阈值

- O − A = +0.0986，只比阈值低 0.0014，小于训练噪声（换训练 seed 后 |Δ| 为 0.004–0.012）；95% CI (+0.077, +0.119) 包含 0.10；来源平均为 +0.117 (+0.087, +0.148)。
- 与 v2 seed 42 相比：A 从 0.431 升到 0.517（+0.086），O 从 0.648 降到 0.615（−0.033）。提升空间缩小主要是因为 A 变强了，而让 A 成为公平的基线正是 M1–M3 的目的。v2 中 B − A = +0.065 也基本来自 A 太弱：v3 中两个训练 seed 下 B − A 分别为 −0.007 和 +0.009。
- **根本原因是 teacher 比 student 弱**：extractor 的最佳验证 F1 为 0.473，A 为 0.543；伪标签池准确率 0.497，B 为 0.535，与 A 自己在测试集上的准确率（0.533）相当；REFUTES 伪标签的精度只有 0.328。用和 student 一样准（或更差）的伪标签训练，B 很难超过 A。B/Q/K 的效果都会受这个限制。
- seed 42 的 extractor 是 v2 三个 seed 中最弱的（验证 F1：42 为 0.473、43 为 0.566、44 为 0.512），所以试跑可能低估了 B − A。

### 10.3 C5：换训练 seed 后类别边界不稳定，不是系统性塌缩

- 只有 A_t1042 不通过：REFUTES 预测 143 条，占金标签的 38%；同一划分下主训练 seed 的 A 是 306 条（82%）。B 两个训练 seed 都正常，O 也正常。
- 换训练 seed 后 macro-F1 几乎不变（A 为 0.517 和 0.513），但预测分布变化很大：A 的 REFUTES F1 从 0.469 降到 0.377，SUPPORTS F1 从 0.522 升到 0.609；B 的 NEI 预测从 1173 条降到 778 条。可见训练噪声主要体现在决策边界偏向哪个类别，macro-F1 只是碰巧被平均掉了。
- **README 5.2 给出的处理办法不适用**：`class_weight_power` 是 `[pool]` 设置（extractor 与 detector 共用），改动后 v2 的伪标签池不能导入，所有 seed 都要从 extractor 重跑；而且从 1.5 降到 1.0 会降低 REFUTES 的权重，只会让 REFUTES 预测更少。如果要单独调整 detector 的类别权重，需要新增一个不属于 `[pool]` 的配置项。

### 10.4 待项目负责人决定

1. **C4**：
   - 甲：当作边界通过（差距在噪声范围内），按原计划跑主实验，用 5 个 seed 的 O − A 与 B − A 回答问题；
   - 乙：先提升 teacher（例如 extractor 改用与 A 相同的训练设置，或用 A 生成伪标签），伪标签池重新生成，所有 seed 不再导入 v2 的伪标签池（每个 seed 约 8–25 分钟），然后重新试跑；
   - 丙：讨论研究方向（例如换数据集或换 ratio）。
2. **C5**：
   - 甲：C5 只对主训练 seed 判定，重复训练 seed 的类别分布变化记作训练噪声的一部分（这是事后修改通过标准，需要明确记录）；
   - 乙：主实验中每个集合训练 2 个 seed 取平均（主实验时间约翻倍）；
   - 丙：新增 detector 专用的类别权重配置（不属于 `[pool]`），重新试跑。

### 10.5 研究方向讨论（2026-10-04，项目负责人选择 C4 的"丙"，待决定）

**试跑后可以确定的事实**

1. 在公平的纯监督基线之上，现有伪标签没有带来提升：两个训练 seed 下 B − A 为 −0.007 和 +0.009。v2 中 B − A 为 +0.065 到 +0.206，主要来自当时的 A 太弱。
2. 提升空间小：即使无标签池 3853 条全部用金标签，O − A 也只有 +0.10（来源平均 +0.12）。现实的伪标签只能拿到其中一部分，很容易被训练噪声（0.004–0.012）盖住。
3. teacher 比 student 弱：extractor 验证 F1 0.473，A 为 0.543；伪标签池准确率 0.497，平衡后 0.463。
4. 稳定成立的正面结果：c 区分伪标签对错的能力优于置信度。v2 三个 seed 中，SUP/REF 类内 c 的 AUROC 为 0.68–0.81，置信度为 0.57–0.70；在 SciFact 上 c 达到 0.82–0.86。同规模下，按 c 选出的集合平衡后准确率高 0.03–0.07（v2 README 8.6）。
5. 三个来源差别很大（A、O 为 seed 42 的测试 macro-F1；NLI 零样本为 v2 测得；c 的 AUROC 为 seed 42 的伪标签池）：

   | 来源 | 测试条数 | A | O | 提升空间 | NLI 零样本 | c 的 AUROC |
   |---|---|---|---|---|---|---|
   | ClimateCheck | 1904（78%） | 0.508 | 0.598 | +0.09 | 0.408 | 0.536 |
   | Climate-FEVER | 200 | 0.566 | 0.612 | +0.05 | 0.421 | 0.620 |
   | SciFact | 339 | 0.491 | 0.706 | **+0.21** | **0.621** | **0.748** |

   在 SciFact 上，NLI 零样本（0.621）比训练出来的 A（0.491）还高，LogicScore 的前提成立得最好。但测试集由 ClimateCheck 主导，那里 NLI 几乎没有信号。
6. RL 奖励中 ΔF1 项只有 ±0.007，小于训练噪声，在当前设置下 RL 学不到有意义的策略（第 8 节）。

**可选方向**

| 方向 | 做法 | 依据 | 代价 | 风险 |
|---|---|---|---|---|
| 一：logic-aware teacher（保留方法框架，修正根本原因） | 伪标签由 extractor 与 NLI 的概率融合产生（可以按来源 / 类别设权重），c 继续用于选样；或者两者一致才保留（co-training） | v2 8.6 中融合方案 F 的平衡后准确率为 0.67–0.71，B 为 0.52–0.63；SciFact 上 NLI 零样本强于 A | 改 `[pool]`，所有 seed 重新生成伪标签池（每个 seed 约 10–25 分钟），然后重新试跑（1.5 小时）和主实验（约 9 小时） | ClimateCheck 上 NLI 零样本只有 0.408，整体提升可能仍被噪声盖住 |
| 二：把贡献改为"LogicScore 是更好的伪标签可靠性信号" | 主要结果放在伪标签层面：类内 AUROC、精度–覆盖率曲线、跨 seed、跨来源；detector 结果如实报告为次要结果 | 事实 4，已经在 3 个 seed 上稳定成立；数据都已有 | 几乎不需要新的训练 | 贡献偏分析性，缺少端任务上的提升，投稿定位要调整 |
| 三：评估重点放在 LogicScore 前提成立的数据上 | 以来源平均或分来源结果为主；SciFact（加 Climate-FEVER）为主要评估，ClimateCheck 作为困难 / 域外设置 | SciFact 的提升空间（+0.21）、c 和 NLI 的信号都最强 | 汇总方式要改，训练不变 | SciFact 测试只有 339 条，CI 宽；有挑数据的嫌疑，必须在跑之前写定判据 |

不论选哪个方向，RL 都建议降为可选或放进附录；等伪标签本身确实有效之后，再考虑重新设计 RL。

**建议**：以方向一为主，方向二作为保底；在全量重跑之前，先花约 1 小时做离线验证（不训练、不改配置）：

- 在 seed 42–44 已有的伪标签池上，计算 extractor 与 NLI 融合后伪标签的准确率与平衡后准确率（整体、分来源、分类别），与 A 在测试集上的水平比较（seed 42：平衡后准确率 0.508；ClimateCheck 0.496、SciFact 0.531、Climate-FEVER 0.579）；
- 如果融合后的伪标签明显比 A 准（特别是在 ClimateCheck 上不比 extractor 差），就走方向一；否则走方向二，并加上方向三的分来源报告。

### 10.6 方向一的离线验证（2026-10-04，项目负责人选择先做方向一）

脚本 `analysis/teacher_fusion_probe.py`，结果 `analysis/teacher_fusion_probe.md`。不训练模型、不改配置。NLI 概率：train 复用 v2 的缓存（与 v3 数据完全相同），dev / test 新算并缓存到 `analysis/`。融合方式：p = (1 − α)·p_extractor + α·p_NLI（NLI 的蕴含 / 矛盾 / 中立对应 SUP / REF / NEI），取 argmax 作为伪标签，最大概率作为置信度。伪标签池的金标签只用于评估。

**1. teacher 的伪标签质量（v2 seed 42–44 的伪标签池，3 个 seed 平均）**

| teacher | 整池准确率 | 整池平衡后 | 前 30%：准确率 | 前 30%：平衡后 |
|---|---|---|---|---|
| extractor（现行） | 0.536 | 0.515 | 0.613 | 0.608 |
| 融合 α = 0.2 | 0.535 | 0.533 | 0.619 | 0.665 |
| 融合 α = 0.3 | 0.521 | 0.545 | 0.624 | 0.681 |
| 融合 α = 0.5 | 0.485 | 0.574 | 0.622 | 0.689 |
| NLI 零样本 | 0.474 | 0.563 | 0.516 | 0.673 |

- 前 30% 的平衡后准确率（平衡采样器下 detector 实际看到的期望准确率）从 0.608 升到 0.68–0.69；seed 42 上从 0.519 升到 0.66–0.68，三个 seed 中提升最大。
- 提升主要来自 SUPPORTS 的精度（约 0.71 升到 0.89）和 REFUTES 的精度（约 0.51 升到 0.63）；NEI 的精度仍只有 0.53 左右，而且 NEI 在选中的集合中占到 50%–67%（NLI 帮不了 NEI，与 v2 8.6 一致）。
- 前 30% 中分来源的平衡后准确率（extractor → 融合 α = 0.5）：ClimateCheck 0.594 → 0.647，Climate-FEVER 0.666 → 0.660，**SciFact 0.670 → 0.816**。ClimateCheck 的原始准确率下降（0.614 → 0.559），因为 NEI 变多，但平衡后准确率上升。
- 参照：A 在 seed 42 测试集上的平衡后准确率为 0.508（注意两者的数据分布不同：伪标签池来自 train，ClimateCheck 占 60%；测试集中 ClimateCheck 占 78%）。

**2. 对照：不用伪标签，只在测试时把 detector 与 NLI 融合（seed 42）**

| detector | α = 0 | α = 0.3 | α = 0.5 | NLI 零样本 |
|---|---|---|---|---|
| A：测试 macro-F1 | 0.517 | 0.508 | 0.461 | 0.461 |
| A：SciFact | 0.491 | 0.551 | 0.608 | 0.626 |
| A：ClimateCheck | 0.508 | 0.495 | 0.426 | 0.420 |
| O：测试 macro-F1 | 0.615 | 0.624 | 0.490 | — |

- 测试时直接融合 NLI，能大幅提升 SciFact，但会拉低 ClimateCheck，整体没有提升。所以方向一如果在 SciFact 上有提升，必须与这个"不用伪标签"的对照比较，才能说明提升来自半监督，而不只是 NLI 本身的知识。
- 这里的 α 网格是在测试集上看的，只用于诊断，不能用来选 α。

**3. 结论**

- 按 10.5 节事先写下的判断标准：融合后被选中的伪标签明显比 A 准（平衡后 0.68–0.69 vs 0.51）；在 ClimateCheck 上按平衡后准确率不比 extractor 差（0.647 vs 0.594），只是原始准确率下降。**方向一可以继续。**
- **成本比 10.5 节估计的低**：融合发生在伪标签池之后（选样阶段），`[pool]` 设置不变，v2 的伪标签池仍然可以导入，seed 42–44 不需要重新训练 extractor。只需要为伪标签池补上 NLI 的三类概率。但新增的融合设置会改变配置指纹，需要新的 runs 目录（A、O 也要在新目录中重跑）。
- 风险：融合后的集合以 NEI 为主，而 NEI 的精度没有改善；ClimateCheck（测试集 78%）上的提升有限。

### 10.7 方向一的实现与试跑（2026-10-04，α = 0.5 由项目负责人确定）

**改动**（`configs/config.yaml` 与 `runs/` 中已有的结果都不受影响）

| 文件 | 改动 |
|---|---|
| `configs/config_f.yaml` | **新**：与 `config.yaml` 只差两处：`teacher.nli_fusion_alpha: 0.5`，以及 `runs_f/`、`results_f/` 两个新目录。`[pool]` 设置不变，仍然导入 v2 的伪标签池 |
| `models/logic_scorer.py` | 加 `probs_batch`：NLI 三类概率，顺序为 SUP / REF / NEI；`score_batch` 改为由它计算（结果相同） |
| `training/compute_nli_probs.py` | **新**：计算伪标签池的 NLI 概率，写入 `<run>/pseudo/nli_probs.jsonl`，并核对 P(蕴含) − P(矛盾) 与池中 LogicScore 是否一致；也计算测试集的 NLI 概率（`processed/nli/test.jsonl`），供 A⊕NLI 对照使用 |
| `training/build_baseline_sets.py` | 加方法 **F**：按融合后的置信度取前 30%，标签取融合后的 argmax；同时记录融合改动了多少标签 |
| `run_all.py` | 请求 F 时，增加 `NLI probs (test)`（全局一次）与 `NLI probs (pool)`（每个 run）两个步骤 |
| `common/paths.py` | 方法列表加 F，加 `nli_probs` 和 `nli_split_path`，`nli_fusion_alpha()` 读取配置 |
| `evaluation/pilot_report_f.py` | **新**：方向一试跑的通过标准（见下），A⊕NLI 对照不训练，直接由 A 的测试概率与 NLI 概率融合得到 |
| `evaluation/{pseudo_label_quality,sanity_check,aggregate_results}.py` | 识别 F：集合质量、S6 检查 \|F\|、汇总中加入 F1（F − A）与 F2（F − B） |
| `run_pilot_f.sh` | **新**：导入 seed 42 → A、F × 训练 seed 42 / 1042 → O → 通过标准 |

**离线检查**（2026-10-04）：
- 全部文件编译通过。
- `--dry_run` 计划正确。
- seed 42 池的 NLI 概率与池中 LogicScore 的差异最大为 9.6e-7。
- F 集合与 10.6 节离线分析完全一致：1156 条，准确率 0.601，平衡后准确率 0.675，SUP / REF / NEI 为 189 / 196 / 771。
- 用试跑的 B 结果冒充 F，测试了 `pilot_report_f.py` 和 `aggregate_results.py`：A⊕NLI 的数值与 10.6 节一致。

**观察**：融合改动了伪标签池中 2020 / 3853 条的标签，但被选进 F 的 1156 条**没有一条被改动**：融合后置信度高的样本，正好是 extractor 与 NLI 一致、而且都有把握的样本。所以 F 的效果全部来自"选哪些样本"，而不是"纠正标签"。

**通过标准**（2026-10-04 与项目负责人确定）：

| 编号 | 标准 | 作用 |
|---|---|---|
| G1 | 两个训练 seed 下 F − A 都 > 0.02 | 提升大于训练噪声 |
| G2 | 两个训练 seed 下 F 都优于 A⊕NLI（A 的测试概率与 NLI 按同一个 α = 0.5 融合，不用伪标签） | 排除"提升只来自 NLI 本身"的解释 |

只作参考、不决定是否通过：收敛、训练噪声、类别塌缩、O − A、提升空间比例 (F − A) / (O − A)、来源平均与分来源结果。

```bash
cd /root/autodl-tmp/Semi-vast/v3
nohup bash run_pilot_f.sh > logs/pilot_f.log 2>&1 &
bash logs/watch.sh logs/pilot_f.log
```

#### 10.7.1 方向一试跑结果（2026-10-04 18:09–19:37，日志 `logs/pilot_f.log`，报告 `runs_f/r0.10_s42/outputs/pilot_report_f.json`）

**结论：未通过。G1 ❌，G2 ✅。**

| 编号 | 结果 | 说明 |
|---|---|---|
| G1 | ❌ | F − A = **+0.0460**（训练 seed 42）/ **+0.0126**（训练 seed 1042），阈值 > 0.02 |
| G2 | ✅ | F − A⊕NLI = +0.0735 / +0.0575 |

| 组 | 测试 macro-F1 | 来源平均 | ClimateCheck | Climate-FEVER | SciFact | 最佳步 | 最佳验证 F1 |
|---|---|---|---|---|---|---|---|
| A | 0.5042 | 0.5292 | 0.4809 | 0.5361 | 0.5707 | 280 | 0.5599 |
| A⊕NLI | 0.4768 | 0.4974 | 0.4453 | 0.4387 | 0.6082 | — | — |
| A_t1042 | 0.5317 | 0.5366 | 0.5167 | 0.5090 | 0.5841 | 320 | 0.5368 |
| A⊕NLI_t1042 | 0.4868 | 0.5043 | 0.4579 | 0.4403 | 0.6147 | — | — |
| **F** | **0.5502** | **0.5788** | 0.5225 | 0.5763 | 0.6378 | 480 | 0.6083 |
| **F_t1042** | **0.5443** | **0.5712** | 0.5138 | 0.5502 | 0.6497 | 360 | 0.5788 |
| O | 0.6082 | 0.6341 | 0.5872 | 0.5950 | 0.7200 | 400 | 0.6105 |

- 自动检查 S1–S7 全部通过，没有类别塌缩；所有组都在训练结束前收敛。
- O − A = +0.104；按训练 seed 42 这一对，F 拿到了 44% 的提升空间。
- **F 比 A 稳定**：换训练 seed 后，F 的测试 F1 只变化 0.0059，A 变化了 0.0275。
- **G1 第二对没通过，主要是 A 的噪声造成的**：同一划分下 A 一共跑了 4 次（第一次试跑 2 次、这次 2 次），结果为 0.5168 / 0.5128 / 0.5042 / 0.5317，平均 0.5164，标准差 0.0115。A_t1042（0.5317）正好是其中最高的一次。2 个 F（0.5502 / 0.5443）都高于这 4 个 A；F 的平均值比 4 个 A 的平均值高 +0.031。
- **同一配置、同一 seed 也不能复现**：`runs/` 和 `runs_f/` 中的 A 设置完全相同，测试 F1 却分别为 0.5168 和 0.5042（训练 seed 42），0.5128 和 0.5317（训练 seed 1042）。`cudnn.benchmark` 和 bf16 带来的非确定性，与换训练 seed 的影响一样大。所以比较必须在同一个 runs 目录内进行。
- **分来源**：
  - SciFact：F（0.638 / 0.650）不但高于 A（0.571 / 0.584），也高于 A⊕NLI（0.608 / 0.615）。说明在 SciFact 上，半监督比"测试时直接用 NLI"更好。
  - Climate-FEVER：F 两次都提升（+0.040 / +0.041）。
  - ClimateCheck（测试集的 78%）：第一对 +0.042，第二对 −0.003，不稳定。
- 参考：上一轮试跑（`runs/`）中，B（只用 extractor 作为 teacher）为 0.5096 / 0.5213，F 比它高约 +0.03。两者不在同一个 runs 目录，只能粗略比较。

### 10.8 方向一主实验：事先确定的检验（2026-10-04，写于运行之前）

**决定**：方向一试跑的 G1 未通过（10.7.1 节）。项目负责人在 2026-10-04 仍决定进入主实验，理由是：
- 两对差值都为正；G2 稳定通过；
- 效果估计约 +0.03，大约是 A 单次运行标准差（0.0115）的 3 倍；
- G1 第二对没过主要是 A 的噪声造成的。

**这是对事先确定的试跑标准的放宽，在报告中必须如实说明。**

**设置**：`configs/config_f.yaml`（指纹 `4342c1690f11`），组为 A / B / F / O × 划分 seed 42–46，ratio 0.1，每个集合 1 个训练 seed（训练 seed = 划分 seed）。A⊕NLI 对照在汇总时由 A 的测试概率与 NLI 按 α = 0.5 融合得到，不训练。seed 42–44 导入 v2 的伪标签池，45、46 从 extractor 开始。seed 42 的 A / F / O 沿用试跑的结果（同一个 runs 目录、同一个指纹）。

**检验**（汇总在 `results_f/summary.md` 第 2 节）：

| 编号 | 比较 | 地位 | 回答的问题 |
|---|---|---|---|
| **F1** | **F − A** | **主要检验** | 融合 teacher 的伪标签在最强的纯监督基线之上是否有效 |
| F2 | F − B | 次要 | 融合 teacher 是否优于只用 extractor 的 teacher（LogicScore / NLI 的贡献） |
| F3 | F − A⊕NLI | 次要 | 提升是否来自半监督，而不只是 NLI 本身 |
| H1 | B − A | 参考 | 只用 extractor 的 teacher 是否有效（第一次试跑中约为 0） |
| — | O − A | 参考 | 提升空间 |

- **判据与 v3 第 4 节相同**：5 个 seed 的分层 bootstrap（2000 次）95% CI 下界 > 0，且至少 4 / 5 个 seed 的 Δ > 0，才算"成立"。
- **主指标**：整体测试 macro-F1。**次要指标**：来源平均 macro-F1；ClimateCheck、Climate-FEVER、SciFact 分别报告，但不作为判据。
- **不在结果出来之后改**：α（0.5）、选样比例（30%）、训练步数（640）、组和 seed。

```bash
cd /root/autodl-tmp/Semi-vast/v3
nohup bash run_main_f.sh > logs/main_f.log 2>&1 &
bash logs/watch.sh logs/main_f.log
```

**改动**：
- `evaluation/aggregate_results.py`：加入 A⊕NLI 对照（只有配置中有 `teacher.nli_fusion_alpha` 时才生成）和 F3 检验；用 `config.yaml` 汇总 `runs/` 的结果不受影响（已核对）。
- `configs/config_f.yaml`：`experiment.methods` 改为 `[A, B, F, O]`（不影响指纹）。
- 新增 `run_main_f.sh`。

### 10.9 方向一主实验结果（2026-10-04 19:43 – 10-05 02:22，日志 `logs/main_f.log`，汇总 `results_f/summary.md`）

**结论：事先确定的检验 F1、F2、F3 全部成立；H1（只用 extractor 的 teacher）不成立。**

| 编号 | 比较 | 逐 seed Δ（42–46） | 均值 ± 标准差 | Δ > 0 | 95% CI（分层 bootstrap） | 结论 | 来源平均 Δ（95% CI） |
|---|---|---|---|---|---|---|---|
| **F1（主要）** | **F − A** | +0.046 / +0.041 / +0.033 / +0.065 / +0.060 | **+0.049 ± 0.013** | 5/5 | **(+0.035, +0.062)** | **成立** | +0.054 (+0.035, +0.074) 成立 |
| F2 | F − B | +0.046 / +0.032 / +0.024 / +0.042 / +0.059 | +0.041 ± 0.013 | 5/5 | (+0.028, +0.054) | 成立 | +0.049 (+0.030, +0.070) 成立 |
| F3 | F − A⊕NLI | +0.074 / +0.034 / +0.097 / +0.034 / +0.024 | +0.053 ± 0.031 | 5/5 | (+0.028, +0.081) | 成立 | +0.062 (+0.038, +0.090) 成立 |
| H1 | B − A | +0.000 / +0.009 / +0.009 / +0.023 / +0.001 | +0.008 ± 0.009 | 5/5 | (−0.003, +0.020) | 不成立 | +0.005 (−0.020, +0.032) 不成立 |
| — | O − A | +0.104 / +0.078 / +0.078 / +0.080 / +0.110 | +0.090 ± 0.016 | 5/5 | (+0.075, +0.105) | — | +0.093 |

| 组（5 个 seed，mean ± std） | 测试 macro-F1 | 来源平均 | ClimateCheck | Climate-FEVER | SciFact |
|---|---|---|---|---|---|
| A | 0.518 ± 0.014 | 0.540 | 0.499 | 0.538 | 0.583 |
| A⊕NLI（不训练） | 0.515 ± 0.040 | 0.531 | 0.486 | 0.463 | 0.645 |
| B | 0.527 ± 0.020 | 0.545 | 0.508 | 0.530 | 0.597 |
| **F** | **0.567 ± 0.012** | **0.594** | **0.543** | **0.575** | **0.663** |
| O | 0.608 ± 0.007 | 0.634 | 0.586 | 0.593 | 0.721 |

**解读**

- **主要结论**：用 extractor 与 NLI 融合的 teacher（α = 0.5，事先固定）选出的伪标签，在最强的纯监督基线之上平均提升 +0.049 macro-F1，5 个 seed 全部为正。平均拿到了 O − A 提升空间的 54%（各 seed 为 44% / 52% / 42% / 81% / 54%）。
- **只用 extractor 的 teacher 不够**：B − A 不显著（+0.008）。F − B（+0.041，两者伪标签条数相同）说明提升来自 teacher 中的 NLI / 逻辑信号。
- **不只是 NLI 本身的知识**：在测试时直接融合 NLI（A⊕NLI）整体上没有提升（0.515 vs 0.518），只在 SciFact 上有效；F 在三个来源上都提升，并且在 SciFact 上也高于 A⊕NLI（0.663 vs 0.645）。
- **三个来源都提升**（F − A）：ClimateCheck +0.044，Climate-FEVER +0.037，SciFact +0.080。之前担心的 ClimateCheck（测试集的 78%）在 5 个 seed 的平均上也有提升。
- **F 是最稳定的组**：跨 seed 的标准差为 0.012（A 为 0.014，B 为 0.020）；同一划分换训练 seed 后只变化 0.006（A 为 0.028）。
- **teacher 越弱，融合的作用越大**：extractor 最弱的两个 seed（46：0.445；42：0.473）上 B − A ≈ 0，F − B 最大（+0.059 / +0.046）；最强的 seed 43（0.566）上 F − B 为 +0.032。
- **机制**：进入 F 的样本没有一条被融合改动标签（5 个 seed 都是 0），融合只是改变了"选哪些样本"——选的是 extractor 与 NLI 一致且都有把握的样本。F 的平衡后准确率为 0.656–0.697，B 为 0.519–0.667。

**需要如实说明的限制**

1. 方向一试跑的 G1 未通过，进入主实验是项目负责人的决定（10.8 节）。
2. α = 0.5 是在看过 seed 42–44 伪标签池的离线分析（10.6 节，用了池中的金标签做评估）之后才确定的，不过选择的理由是"两个 teacher 等权、不调参"，而不是在金标签上取最优（α = 0.3 与 0.5 的离线结果几乎相同）。
3. 每个集合只用 1 个训练 seed；同一配置、同一 seed 的重复训练也有 0.013–0.019 的差异（`cudnn.benchmark` 与 bf16）。分层 bootstrap 把这部分噪声包含在 seed 之间的差异里。
4. 自动检查的警告：seed 43 的 O 最佳点在第 640 步（可能未收敛，只影响提升空间这一参照）；seed 43 的 A 预测 REFUTES 181 条，占金标签的 48.7%（略低于 50%）。两者都不影响 F1–F3。
5. 只做了 ratio 0.1。
