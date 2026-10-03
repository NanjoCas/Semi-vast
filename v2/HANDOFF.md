# Semi-vast v2 交接文档

> 更新于 2026-10-03。上一个工作会话已经关闭，本文档加上 `README_v2.md` 的第 4、7、8 节就是全部上下文。
> 代码已审查并提交到分支 `v2-data-design`（2026-10-03，见第 4 节和 `git log`），未推送到远程。

## 0. 现状

- 训练数据已从 Climate-FEVER + PUBHEALTH 换成 **Climate-FEVER + ClimateCheck + SciFact**，配置文件为 `configs/config_sci.yaml`。PUBHEALTH 不再使用。
- 修复了一个从 v1 起就存在的 bug：LogicScore 实际算的是 P(中立) − P(矛盾)。
- 在新数据上做了低成本验证（ratio 0.1、seed 42、A/B/O 三组）：
  - O − A = **+0.217**，B − A = **+0.065**，两者都显著；
  - 修正后的 LogicScore（带方向）能区分伪标签的对错。
- **B+L 组（方法 L）已实现并跑完 ratio 0.1 × 3 seed 的 A / B / L / O（2026-10-03，README 8.5）**：
  - B − A 在 3 个 seed 上都显著，平均 +0.118；O − A 平均 +0.213；
  - L − B 为 −0.128 / +0.002 / +0.032，**L > B 不成立**。L 的伪标签质量更高，但集合只有 B 的 40%–56%。
- c 的设计分析（README 8.6）：c 在 SUPPORTS/REFUTES 类内有效（优于置信度），对 NEI 无效；问题出在固定阈值 c > 0.3 的用法上。
- 方法 Q（L-q）和 K（同规模置信度对照）已跑完 3 个 seed（README 8.5）：Q − K 为 +0.035* / −0.007 / −0.007，平均 +0.007。伪标签层面 Q 比 K 准 0.03–0.07，但没有转化为 detector 的提升。
- **新问题（README 8.7）**：小集合组只有 224–296 次参数更新，验证 F1 起飞的时间随机，几乎相同的伪标签集合测试 F1 可以相差 0.1–0.16（seed 42：L ⊂ Q，0.368 vs 0.528）。detector 层面的组间比较被训练噪声掩盖。
- **下一步（待项目负责人决定）**：统一并加大 detector 的训练预算（所有组相同的总参数更新次数），再重跑 A / B / Q / K。

## 1. 项目与环境

| 项 | 内容 |
|---|---|
| 代码 | `/root/autodl-tmp/Semi-vast`，git 分支 `v2-data-design`，最新提交 `4aa8664`；v2 在 `v2/` 下，主文档 `v2/README_v2.md` |
| 任务 | 半监督虚假信息检测：extractor（DeBERTa-v3-large，从 NLI 模型初始化）给无标签句对打伪标签 → 按置信度、复合权重或 RL 筛选 → detector 用有标签数据加伪标签训练。消融组：A 纯监督、B 置信度 ≥ 0.7、W 复合权重、R 随机、C RL、O 金标签（上界） |
| 硬件 | AutoDL，1 张 RTX 5090（32GB）；数据盘 `/root/autodl-tmp` 约 47GB 可用 |
| Python | `/root/miniconda3/bin/python`（3.12），torch 2.12 + cu130，transformers 5.5 |
| 模型缓存 | `Semi-vast/model_cache`：microsoft/deberta-v3-large、cross-encoder/nli-deberta-v3-large。HuggingFace 可以直连，"unauthenticated requests" 警告可以忽略 |
| 原始数据 | `Semi-vast/Data/`：`Climate Fever Dataset/`、`ClimateCheck/`（HF `rabuahmad/climatecheck`，MIT 许可）、`SciFact/data/`（AllenAI S3 release）、`PUBHEALTH-DATASET/`（已弃用）、`Environment News Dataset/`（Guardian，未使用） |

⚠️ 当前 shell 的 `OMP_NUM_THREADS=0` 是非法值，运行任何脚本前先执行 `export OMP_NUM_THREADS=8`。

## 2. 工作经过

1. **第一步（README 4）**：用 `build_labeled.py` 重建 PUBHEALTH 版数据，并用 `process_labeled.py` 生成 v1 数据做对比，确认了 v1 的三处泄漏，还发现 v1 的证据条数本身就暴露了标签。
2. **第二步**：`run_all.py --ratios 0.1 --seeds 42` 跑出来 O ≈ A。原因是 detector 每个 epoch 只遍历一遍有标签数据，8 个 epoch 一共只有 136 次参数更新。
3. 新增 `epoch_mode: cover_pseudo`（每个 epoch 覆盖一遍伪标签集）和动态补齐，在 `runs/r0.10_s42_fix` 重跑 6 组：O − A = +0.071；B/W/R 与 A 相比没有显著差异；C 显著更差。
4. 分析发现瓶颈在伪标签质量：类别先验校正和平衡采样都拉低了准确率，LogicScore 和 Discourse 没有信号，RL 没有在学习。这些写进了 README 第 7 节。
5. 数据质量检查：PUBHEALTH 的证据不包含判定依据，存在来源捷径（新闻 vs 核查文章），两个数据集的 mixture/DISPUTED 映射不一致，Climate-FEVER 有同一 claim 对应多个 id 的情况。
6. **发现 LogicScore bug**：NLI 模型的标签顺序是 `{0: 矛盾, 1: 蕴含, 2: 中立}`，代码却把下标 2 当成了蕴含。已改为从模型配置读取。
7. 下载并评估 ClimateCheck 和 SciFact。项目负责人决定换数据集，不再使用 PUBHEALTH。
8. 低成本验证（`runs_sci/r0.10_s42`）。第一次尝试时 extractor 崩溃（验证 F1 0.221），原因是梯度累积 4 加上平衡采样器重复校正。修正后完成验证（README 第 8 节）。
9. **（2026-10-03）** README 中基于错误 LogicScore 的数字和 W/R/C/RL 结果全部删除，第 7 节用修正后的数字重写。
10. 实现方法 L（B+L），跑完 A/B/L/O × seed 42/43/44（README 8.5）：B 稳定优于 A，L 不优于 B。
11. c 的设计分析（`analysis/c_design_probe.py`，README 8.6）：c 在 SUP/REF 类内有效、对 NEI 无效；固定阈值 c > 0.3 砍掉约 80% 的 SUPPORTS，导致 L 集合小且偏 REFUTES。提出 L-q 方案和同规模对照组，等待决定。

## 3. 关键结果

**新数据 3 个 seed（ratio 0.1，seed 42/43/44，README 8.5）**

| 组 | 测试 macro-F1（mean ± std） | 各 seed |
|---|---|---|
| A | 0.448 ± 0.039 | 0.431 / 0.421 / 0.492 |
| B | 0.566 ± 0.066 | 0.496 / 0.627 / 0.575 |
| L（置信度 ≥ 0.7 且 c > 0.3） | 0.535 ± 0.145 | 0.368 / 0.629 / 0.608 |
| O | 0.661 ± 0.014 | 0.648 / 0.661 / 0.675 |

- seed 之间的差异主要来自 extractor（验证 F1 为 0.473 / 0.566 / 0.512）。
- seed 42 的 L 几乎不预测 REFUTES（只有 287 条伪标签，其中 SUPPORTS 66 条）。
- 同等规模下，c 选出的样本平衡后准确率比按置信度选高 0.05–0.11（README 8.6）。

**新数据低成本验证（ratio 0.1，seed 42，第一次）**

| 组 | 伪标签条数 | 参数更新次数 | 测试 macro-F1 | ClimateCheck | Climate-FEVER | SciFact |
|---|---|---|---|---|---|---|
| A | 0 | 224 | 0.431 | 0.419 | 0.427 | 0.446 |
| B | 712 | 224 | 0.496 | 0.493 | 0.404 | 0.531 |
| O | 3853 | 640 | 0.648 | 0.620 | 0.591 | 0.804 |

- O − A = +0.217（95% CI +0.189, +0.246）；B − A = +0.065（+0.041, +0.091）。
- 只看 claim 的基线（TF-IDF + LR）：用同样 10% 的标签为 0.421，用全部 train 标签为 0.500。A 组只比它略高。
- extractor 验证 macro-F1 为 0.473；整池伪标签准确率 0.497。
- 区分伪标签对错的 AUROC：置信度 0.554；**方向一致性 c 为 0.593**（SciFact 0.748，Climate-FEVER 0.620，ClimateCheck 0.536）；|LogicScore| 0.519；Discourse 0.505。
- **B+L 过滤**（置信度 ≥ 0.7 且 c > 0.3）：712 条 / 准确率 0.522 → 287 条 / 0.690。Climate-FEVER 上从 0.329 提升到 0.606。

方向一致性 c 的定义：伪标签为 SUPPORTS 时 c = LS；为 REFUTES 时 c = −LS；为 NEI 时 c = 1 − |LS|。其中 LS = P(蕴含) − P(矛盾)。

**旧数据（PUBHEALTH）的结论，仅作参考**：训练预算修正后 O − A = +0.071；修正后的 LogicScore 在 PUBHEALTH 上也没有信号（方向一致性 AUROC 0.507），NLI 零样本 macro-F1 只有 0.309。

## 4. 代码改动（2026-10-03 审查后提交）

| 文件 | 改动 |
|---|---|
| `v2/models/logic_scorer.py` | **bug 修复**：蕴含/矛盾的下标改为从 `model.config.id2label` 读取 |
| `v2/training/train_detector.py` | `epoch_mode`（`labeled` 为原做法，`cover_pseudo` 为新做法）；动态补齐；`gradient_checkpointing` 可配置；伪标签 loss 只对带伪标签的步求平均；结果中新增 `epoch_mode`、`total_optimizer_steps` |
| `v2/common/data_utils.py` | `dynamic_padding` 选项、`DynamicPaddingCollator`、`pad_sequences`；`max_evidences` 默认值 3 → 5 |
| `v2/data_prep/build_labeled.py` | `data.sources`；ClimateCheck 和 SciFact 的构建；`cf_disputed`；`dedup_claims`；跨数据集去重和跨 split 去重；记录新增 `group` 字段 |
| `v2/data_prep/make_ssl_split.py` | 按 claim 分组划分。旧数据上与原结果完全一致（已验证） |
| `v2/training/train_extractor.py` | 可选的 `training.extractor_max_epochs` / `extractor_gradient_accumulation` |
| `v2/training/train_rl_selector.py`、`v2/models/extractor.py` | `max_evidences` 默认值 3 → 5 |
| `v2/configs/config.yaml` | 新增 `epoch_mode`、`dynamic_padding`、`gradient_checkpointing` |
| `v2/configs/config_sci.yaml` | **新增**：新数据集的配置，与 config.yaml 的差异都标注了 `[sci]` |
| `v2/README_v2.md` | 第 4 节第一、二步打勾并写执行情况；第 2.5 节；新增第 7 节（问题与方案）和第 8 节（更换数据集） |
| `v2/analysis/` | **新增**：离线分析脚本（见第 5 节） |
| B+L（方法 L，2026-10-03） | `build_baseline_sets.py`（集合 `set_L_conf_logic.jsonl`；`baseline_sets_stats.json` 改为合并写入）、`paths.py`、`run_all.py`、`train_detector.py`、`aggregate_results.py`（L vs A、L vs B）、`pseudo_label_quality.py`（L、c 的 AUROC、按来源拆分）、两个配置文件（`experiment.consistency_threshold: 0.3`，`methods` 加入 L） |
| `v2/common/data_utils.py` | 新增 `direction_consistency()`；`label_to_id` 改为接受 numpy 整数（原来只认 `int`，numpy 整数会被静默映射成 NEI。现有调用都传 JSON 里的 int，不受影响） |
| `v2/logs/watch.sh` | **新增**：实时打印训练进度（关键行逐行打印，进度条原地刷新） |
| Q / K（2026-10-03） | 方法 Q（L-q：SUP/REF 类内按 c 保留前 `experiment.consistency_quantile`，NEI 不用 c）和 K（按置信度取前 \|Q\| 条）；改动位置同 L；`pseudo_label_quality.py` 新增 `balanced_precision`；`sci_multiseed_report.py` 加入 Q/K |
| `v2/logs/` | 运行日志，以及 `progress.sh`、`run_fix_*.sh` 两个辅助脚本 |

- `configs/config.1epoch.yaml`（v1 的配置）的修改在这次工作之前就已经存在，不是这次改的。
- `build_labeled.py` 中 PUBHEALTH 那条路径的代码结构有调整，但**没有重跑过**：重跑会覆盖 `processed/labeled`，而且耗时较长。
- 2026-10-03 审查了全部改动，没有发现影响结果的问题（`label_to_id` 不接受 numpy 整数的隐患已修复）。代码和文档一次提交；`runs*/`、`results*/` 的结果（json、png）在 Q/K 跑完后单独提交。`.gitignore` 忽略 `*.jsonl`、`*.csv`、`*.log`、`*.npz`、`processed/`。
- `configs/config.1epoch.yaml`（v1 配置，仓库根目录）的修改不是这次工作做的，**没有提交**。

## 5. 目录与产物

| 路径 | 内容 |
|---|---|
| `v2/processed/labeled/` | PUBHEALTH 版数据（旧） |
| `v2/processed/sci/labeled/` | 新数据：train 4289 / dev 759 / test 2443，`build_stats.json` |
| `v2/runs/r0.10_s42/` | PUBHEALTH、旧训练预算、6 组 |
| `v2/runs/r0.10_s42_fix/` | PUBHEALTH、新训练预算、6 组（`outputs/significance.csv`） |
| `v2/runs_sci/r0.10_s{42,43,44}/` | 新数据、A/B/L/O 四组 |
| `v2/results/`、`v2/results_sci/` | `aggregate_results.py` 的汇总结果。`runs/r0.10_s42_fix` 的目录名不符合汇总脚本的命名规则，不在汇总结果中 |
| `v2/logs/step2_r0.10_s42.log` | 第二步（旧训练预算） |
| `v2/logs/fix_AO_*.log`、`fix_BWRC_*.log` | 新训练预算下的 PUBHEALTH 复测 |
| `v2/logs/sci_validate_r0.10_s42.log` | 新数据的验证（`*.accum4_aborted.log` 为中止的第一次尝试） |
| `v2/logs/sci_ABLO_r0.10_s42-44.log` | A/B/L/O × 3 seed |

checkpoint 在每个 run 结束后会自动删除（`cleanup_checkpoints: true`），只保留指标和预测结果。

**离线分析脚本 `v2/analysis/`**（README 第 7、8 节中的不少数字由它们算出）：

| 脚本 | 用途 | 运行方式 |
|---|---|---|
| `sci_validation_report.py` | 单个 run 的验证报告：伪标签、各信号 AUROC、B+L、分来源 F1、bootstrap、只看 claim 的基线 | `python analysis/sci_validation_report.py [runs_sci/r0.10_s43]` |
| `sci_multiseed_report.py` | 多 seed 汇总：A/B/L/O 的 mean ± std（整体和分来源）、每个 seed 的伪标签集合质量和 bootstrap | `python analysis/sci_multiseed_report.py [0.10] [42,43,44]` |
| `c_design_probe.py` | c 的设计分析：类内 AUROC、各种过滤/融合方案的离线比较；NLI 概率缓存在 `nli_train_probs.npz` | 任意目录运行；第一次需要 GPU 约 1 分钟 |
| `nli_probe.py` | 四个数据集的 NLI 零样本检验，以及修正后 LogicScore 的 AUROC | 在 `Semi-vast/` 下运行 |
| `compare_v1_v2.py`、`leakage_probe.py` | v1 与 v2 数据对比、TF-IDF 泄漏探针 | 在 `Semi-vast/` 下运行 |
| `memprobe.py` | detector 显存和速度探针 | 在 `v2/` 下运行 |

## 6. 运行与查看进度

```bash
cd /root/autodl-tmp/Semi-vast/v2
export OMP_NUM_THREADS=8

# 后台运行（例：新数据，ratio 0.1，seed 42，A/B/O）
PYTHONUNBUFFERED=1 nohup python run_all.py --config configs/config_sci.yaml \
    --ratios 0.1 --seeds 42 --methods A,B,O > logs/<名字>.log 2>&1 &

# 查看进度（默认看最新的日志）
bash logs/watch.sh                  # 实时打印
bash logs/progress.sh
watch -n 10 bash logs/progress.sh

# 生成验证报告
python analysis/sci_validation_report.py

# 停止：不要用 pkill -f "<命令行>"，它会匹配到执行这条命令的 shell 自己
pgrep -af "[r]un_all.py|[t]rain_detector.py"   # 查 PID
kill <PID>                                      # 数据加载子进程可能要单独 kill
```

耗时参考（新数据，ratio 0.1）：extractor 约 5 分钟，伪标签约 2.5 分钟，A 约 4.5 分钟，B 约 7 分钟，O 约 24 分钟。

## 7. 下一步（按优先级）

1. **修正 detector 训练预算（待决定，README 8.7）**：在 `train_detector.py` 中增加按总参数更新次数训练的选项（例如所有组都用约 640 次，与 O 相同），或增加 `max_epochs` 并配合早停；然后重跑 A / B / Q / K × 3 seed（约 3 小时）。可选：对 Q vs K 每个集合用多个训练 seed 重复。
2. **（已完成）改进 L，README 8.5、8.6**：
   - 新方法 L-q：置信度 ≥ 0.7；SUPPORTS / REFUTES 各自在类内按 c 保留前 50%；NEI 不用 c；
   - 新增同规模置信度对照组：按置信度取前 N 条，N 与 L-q 相同；
   - 实现方式与 L 相同：`build_baseline_sets.py`、`paths.py`、`run_all.py`、`train_detector.py`、`aggregate_results.py`、`pseudo_label_quality.py`；
   - 伪标签池已有，只需重跑 detector：2 组 × 3 个 seed，约 1 小时。
   - 备选方案：extractor 与 NLI 概率融合（α = 0.3），会改动伪标签本身。
3. **复合权重**：把 |LogicScore| 换成 c（c 可能为负，需要截断到 [0, 1]），去掉 Discourse（`hyperparameters.beta3: 0`），重新设定 `experiment.weight_threshold`，再跑 W。根据 8.6，c 对 NEI 无效，NEI 的权重可能也应只用置信度。
4. **RL 重新设计**：奖励中的 0.3·mean|LogicScore| 修正后仍几乎是常数（0.224 ± 0.006，ΔF1 项只有 ±0.007），应去掉或换成 c；状态中加入伪标签类别和 c；加大 `episode_size`；取消 `min_keep_ratio` 补齐。
5. 确定配置后跑完整矩阵（`label_ratios` × 至少 3 个 seed）。seed 间方差大（B 的标准差 0.066），可能需要 5 个 seed。

## 8. 待决问题（需要项目负责人决定）

- Climate-FEVER 的 DISPUTED 目前是丢弃。可选做法：`refutes`（v2 原做法）或 `nei`，对应配置项 `data.cf_disputed`。
- RL（C 组）是否保留，取决于第 7 节第 3 步的结果。
- extractor 从第 3 个 epoch 起验证 loss 回升（F1 到第 6 个 epoch 仍在提高），可以考虑缩短训练，或者按 loss 做早停。
- 论文中是否报告"只看 claim"的基线：10% 标签时 A 组只比它略高。
- 是否增加其他测试集（HealthFC、AVeriTeC 的健康/气候子集），以及是否做方案二（Guardian 新闻 + 检索，可以用 ClimateCheck 的 39.4 万篇文献库，894MB，尚未下载）。

## 9. 已知的坑

- `run_all.py` 遇到已有产物会**跳过**。改了配置后要用 `--force`，或者换一个新的 run 目录。
- `run_all.py` 每个 detector 单独启动一个进程，并且在启动时读取配置。run 进行中改配置会影响后面的组。
- SciFact 官方 test 没有标签，所以用官方 dev 当 test，dev 从官方 train 中切出。
- ClimateCheck 的标签针对 (claim, 摘要) 对，同一个 claim 可能有多种标签，所以划分必须按 claim 分组（`group` 字段）。
- NLI 模型对**不相关**的 claim 也会判为矛盾（−0.998），LogicScore 为负不一定表示被反驳。
- 平衡采样器是有放回抽样，少数类会被重复抽到（旧数据中 B 组的 REFUTES 每个 epoch 约被抽 6.8 次）。
- `compute_joint_loss` 会把伪标签权重截断到 [0, 1]，所以不能靠把权重放大到均值 1 来对齐各组的损失强度。
- 显存：开梯度检查点时，64 条 × 512 token 的峰值约 12.5GB；**不能关梯度检查点**（关掉后 64 条 × 256 token 就会 OOM）。
- README 中基于错误 LogicScore 的数字和 W / R / C / RL 结果已全部删除（2026-10-03），README 第 7 节现在只包含修正后的数字。PUBHEALTH 版的旧数字如需查看，在 `runs/` 下的结果文件中。
- 日志里的进度条用 `\r` 刷新，用 `tr '\r' '\n' < log` 处理后再 grep。
