# Semi-vast v3：统一训练协议后的消融实验

> **状态（2026-10-04）**：代码已完成并通过离线检查（见 `HANDOFF.md` 第 3 节），**还没有运行任何训练**。下一步是第一阶段试跑（第 5.2 节）。

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

- [ ] 0. 导入 v2 的伪标签池（`python tools/import_from_v2.py --seeds 42,43,44`）
- [ ] 1. 第一阶段试跑（`run_pilot.sh`），C1–C5 全部通过
- [ ] 2. 第二阶段主实验（`run_main.sh`），`results/summary.md` 生成且没有未解释的警告
- [ ] 3. 解读：H1（伪标签是否有效）、H2（LogicScore 是否有效）、H3
- [ ] 4. 第三阶段（根据 H2）：
  - H2 成立：把 c 用进复合权重（W：SUPPORTS / REFUTES 用 c，NEI 只用置信度），再决定 RL 的改法（奖励去掉常数项，状态加入类别和 c）；
  - H2 不成立：LogicScore 只作为伪标签质量分析的结论，论文方法部分需要调整，RL 大概率不再保留。
- [ ] 5. 结论稳定后，用同一协议跑 ratio 0.25 / 0.5（`experiment.label_ratios`）

每完成一步，在这里打勾，并写"执行情况"（实际数字、日志位置、遇到的问题）。
