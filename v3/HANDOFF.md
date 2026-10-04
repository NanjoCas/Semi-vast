# Semi-vast v3 交接文档

> 更新于 2026-10-04 17:30（试跑完成）。本文档加上 `README_v3.md` 就是 v3 的全部上下文；v3 的由来见 `../v2/README_v2.md` 第 9 节，v2 的历史见 `../v2/HANDOFF.md`。

## 0. 现状

- **v3 代码已完成，并通过离线检查（第 3 节）；第一阶段试跑已运行（结果见下）。**
- v3 按项目负责人确认的推荐方案（v2 README 9.6）实现了 M1–M8：
  - detector 与 extractor 同样从 NLI 模型初始化、冻结前 6 层；
  - 所有组固定 640 步，每 40 步评估一次；
  - 监督损失只保留带类别权重的 CE；
  - B 取置信度前 30%，Q / K 在 B 内且规模相同；
  - 最佳权重存内存，可以换训练 seed 重复训练；
  - 自动检查与配置指纹；
  - 跨 seed 分层 bootstrap 与事先确定的判据。
- **试跑已完成（2026-10-04 15:56–17:23，`logs/pilot.log`）：未通过**。C1–C3 通过；C4 O − A = +0.0986（阈值 0.10）；C5 A_t1042 的 REFUTES 预测 143 / 372。分析与待决定事项见 README_v3 第 10 节，根本问题是 teacher（extractor 验证 F1 0.473）比 A（0.543）弱。
- **方向一（2026-10-04）**：项目负责人选择重新讨论方向（README_v3 10.5），然后先做方向一（logic-aware teacher：extractor 与 NLI 融合，α = 0.5）。离线验证通过（10.6）；实现见 10.7（`configs/config_f.yaml`、方法 F、`runs_f/`、`results_f/`）。
- **方向一试跑（18:09–19:37，`logs/pilot_f.log`）：未通过**。G1 ❌ F − A = +0.046 / +0.013（阈值 0.02）；G2 ✅ F − A⊕NLI = +0.074 / +0.058。F（0.550 / 0.544）高于同一划分下全部 4 个 A（0.504–0.532）。同一配置、同一 seed 的 A 也不能复现（差 0.013–0.019）。详见 README_v3 10.7.1。
- **方向一主实验完成（2026-10-04 19:43 – 10-05 02:22，`logs/main_f.log`，`results_f/summary.md`）**：事先确定的检验全部成立。F1：F − A = +0.049，95% CI (+0.035, +0.062)，5/5 个 seed 为正；F2：F − B = +0.041；F3：F − A⊕NLI = +0.053。H1（B − A）+0.008 不成立。F 在三个来源上都提升，是最稳定的组。结果与限制见 README_v3 10.9。
- **下一步（待项目负责人决定）**：ratio 0.25 / 0.5 的推广、α 敏感性分析、机制消融（例如"extractor 与 NLI 一致"过滤）；RL 是否保留。
- v3 已提交到 git 分支 **`v3-training-protocol`**（基于 `v2-data-design` 的 `69a4519`，包含 v2 的代码与结果两次提交）；远程仓库 `github.com/NanjoCas/Semi-vast`。2026-10-05 又按 v2 的做法分两次提交：方向一的代码（"v3: direction 1 …"）和试跑 / 主实验的结果（"v3: add pilot and direction-1 results"，`runs/`、`runs_f/`、`results/`、`results_f/` 下的 json、png 与 md）。服务器上没有 GitHub 凭据，推送需要在自己的终端完成（第 8 节）。

## 1. 项目与环境

| 项 | 内容 |
|---|---|
| 代码 | `/root/autodl-tmp/Semi-vast/v3`（独立目录，没有改动 v2）；主文档 `README_v3.md` |
| 任务 | 半监督虚假信息检测：extractor（DeBERTa-v3-large，从 NLI 模型初始化）给无标签句对打伪标签 → 按置信度 / LogicScore 方向一致性 c 选样 → detector 用有标签数据加伪标签训练 |
| 数据 | Climate-FEVER + ClimateCheck + SciFact（与 v2 `config_sci.yaml` 相同）：train 4289 / dev 759 / test 2443（test 中 ClimateCheck 占 78%） |
| 硬件 | AutoDL，1 张 RTX 5090（32GB），内存充足（约 750GB） |
| Python | `/root/miniconda3/bin/python`（3.12），torch 2.12 + cu130，transformers 5.5 |
| 模型缓存 | `Semi-vast/model_cache`：microsoft/deberta-v3-large、cross-encoder/nli-deberta-v3-large |

⚠️ 当前 shell 的 `OMP_NUM_THREADS=0` 是非法值，运行任何脚本前先 `export OMP_NUM_THREADS=8`（`run_pilot.sh` / `run_main.sh` 已经设置）。

## 2. 交付内容

| 文件 | 相对 v2 |
|---|---|
| `configs/config.yaml` | 由 v2 `config_sci.yaml` 改来；差异都标注了 `[v3]`，决定伪标签池的设置标注了 `[pool]`（与 v2 完全相同） |
| `common/fingerprint.py` | **新**：配置指纹；决定伪标签池的设置（导入时比对） |
| `common/paths.py` | 改：方法列表 A/B/Q/K/W/R/C/O（去掉 L）；detector 结果命名 `B` / `B_t1042`；`detector_results()` |
| `common/data_utils.py` | 加 `class_weights_from_labels`（与 extractor 相同的公式） |
| `models/detector.py` | 改：`encoder_name` + `freeze_layers`（M1）；`_adjust_logits` 符号改为 `+τ·log π`；伪标签损失不再做 logit adjustment；删除按文件计算的旧 `compute_class_weights` |
| `training/train_detector.py` | **重写**：固定步数训练、按步评估、最佳权重存内存、`--train_seed`、结果记录更多诊断信息 |
| `training/build_baseline_sets.py` | 改：B 按比例、Q = B 内 SUP/REF 按 c 取一半 + 全部 NEI、K = 置信度前 \|Q\|；排序确定 |
| `training/generate_pseudolabels.py` | 改：伪标签先验校正改用 `imbalance.pseudolabel_prior_tau`（默认关闭，与 v2 结果相同） |
| `evaluation/aggregate_results.py` | **重写**：跨 seed 汇总、分层 bootstrap、H1–H3 判据、来源平均指标、训练噪声、警告汇总、混用配置时报错 |
| `evaluation/sanity_check.py` | **新**：每个 run 结束后的检查 S1–S7 |
| `evaluation/pilot_report.py` | **新**：试跑通过标准 C1–C5，退出码 0 / 1 |
| `evaluation/pseudo_label_quality.py` | 方法列表改为 B/Q/K/W/R/C |
| `run_all.py` | 改：配置指纹检查（`runs/protocol.json`）、按指纹跳过、`--extra_train_seeds`、`--dry_run`、每个 run 后自动检查、C/R 与导入池不一致时拒绝 |
| `tools/import_from_v2.py` | **新**：导入 v2 的划分与伪标签池，带一致性检查，可重复运行 |
| `run_pilot.sh` / `run_main.sh` | **新**：两个阶段的入口；`run_main.sh` 要求试跑已通过；都支持 `CONFIG=...` |
| `logs/watch.sh` / `logs/progress.sh` | 按 v3 的日志格式（按步训练）重写 |
| `data_prep/`、`models/extractor.py` 等其余文件 | 与 v2 相同 |

## 3. 已做的验证（2026-10-04，全部在 CPU 上、临时目录中，没有训练真实模型）

| 检查 | 结果 |
|---|---|
| 全部 `.py` 编译、18 个模块导入；所有 `.sh` 语法检查 | 通过 |
| 共用类别权重 vs extractor 日志中的权重（seed 42/43/44） | 3 个 seed 都逐位一致 |
| logit adjustment 符号 | 训练时少数类得到负偏移（`+τ·log π`）；τ = 0 时不改变 logits |
| 评估点 | `eval_steps(640, 40)` 共 16 次，最后一次在 640 步；缺少 `detector_total_steps` 时报错 |
| B / Q / K 选样（v2 的真实伪标签池，只读） | \|B\| = round(0.3 × 池)；Q ⊂ B、K ⊂ B、\|Q\| = \|K\|；打乱输入顺序结果不变；数字见 README_v3 3.4 |
| 训练循环（用微型模型代替 DeBERTa，其余为真实代码：数据、动态补齐、平衡采样器、损失、λ、学习率调度） | 12 步、评估点 5/10/12；λ 线性上升；warmup 后学习率线性降到 0；B 有伪标签损失、A 为 0；最佳权重在 CPU 上，之后的训练不会改动它 |
| `import_from_v2.py`（导入到临时目录） | 3 个 seed 导入成功；再次运行全部识别为相同；把学习率改成 2e-5 后拒绝导入并列出差异 |
| `sanity_check.py`（用 v2 的测试预测充当 detector 结果） | 发现了 v2 中人工才发现的问题：L 的 REFUTES 塌缩（S3）、最佳点在最后（S2）；也发现了充当结果与新集合不一致（S7）。测试中修复了一个 bug：A 组记录了集合路径时 S7 会崩溃 |
| `aggregate_results.py`（同上） | 逐 seed 的 B − A、Q − K、O − A 与 v2 已经算出的数字完全一致（例如 Q − K = +0.0345 / −0.0072 / −0.0069）；不同指纹的结果混在一起时报错 |
| `pilot_report.py` | 正确判定 C1–C5（用 v2 结果时 C1/C2/C3/C5 不通过，符合预期），退出码 1 |
| `run_all.py --dry_run` | 导入的 seed 跳过 extractor 与伪标签生成、只重建集合；已有且指纹相同的 detector 跳过；新 seed 完整执行；导入的 run 上请求 C 时拒绝 |
| 配置指纹 | 改训练设置后拒绝运行；只增加 seed / 方法时指纹不变 |
| `run_main.sh` 的前置检查 | 试跑未通过时退出 |

**没有验证的**（要靠试跑回答）：真实 DeBERTa 在新设置下的收敛速度、训练噪声、显存（冻结 6 层后应略低于 v2 的约 12.5GB）、每步耗时（估计约 1.8 秒）。

## 4. 怎么开始

```bash
cd /root/autodl-tmp/Semi-vast/v3

# 可选：先导入 seed 42（会复制文件，可重复运行；run_pilot.sh 也会做这一步），再看计划（--dry_run 不运行任何步骤）
python tools/import_from_v2.py --seeds 42
python run_all.py --ratios 0.1 --seeds 42 --methods A,B,O --extra_train_seeds 1042 --dry_run

# 第一阶段：试跑（约 1.5 小时）
nohup bash run_pilot.sh > logs/pilot.log 2>&1 &
bash logs/watch.sh

# 试跑通过后，第二阶段：主实验（约 7.5–9 小时）
nohup bash run_main.sh > logs/main.log 2>&1 &
bash logs/watch.sh
```

试跑结束时，日志最后一段就是 C1–C5 的表格和结论；也可以单独运行 `python evaluation/pilot_report.py`。

## 5. 试跑之后的决定

- **全部通过**：直接运行 `run_main.sh`。
- **有不通过的**：按 README_v3 5.2 的表格调整。注意：
  - 任何影响结果的配置改动都会改变配置指纹：复制一份配置，把 `paths.runs_dir` / `paths.results_dir` 换成新目录，再用 `CONFIG=configs/<新配置>.yaml bash run_pilot.sh` 重试；
  - 尽量不要改 `[pool]` 设置（数据、extractor、伪标签生成）：改了以后 v2 的伪标签池不能导入，所有 seed 都要从 extractor 重新开始；
  - C4（O − A < 0.10）意味着提升空间不足，需要项目负责人决定方向，不要自行调参绕过。

## 6. 主实验之后

- 看 `results/summary.md` 第 2 节的 H1–H3 结论，和第 5 节的警告（应当没有，或每条都能解释）。
- 第三阶段由 H2 决定（README_v3 第 9 节第 4 步）。
- 把结果写进 README_v3 第 9 节的执行清单（打勾 + 执行情况）。

## 7. 待决问题（需要项目负责人决定）

- 试跑不通过时的调整方向（特别是 C2 训练噪声仍大时：加训练 seed 还是改学习率和步数）。
- v2 留下的问题仍然有效：
  - Climate-FEVER 的 DISPUTED 处理（目前丢弃）；
  - 论文中是否报告只看 claim 的基线；
  - 是否增加其他测试集；
  - RL（C 组）是否保留，等 H2 的结论再定。

## 8. 已知的坑

- `run_all.py` 现在按配置指纹判断能否跳过：指纹相同的 detector 结果会被跳过，不同的会重跑；整个 runs 目录的指纹记录在 `runs/protocol.json`，与当前配置不同时直接拒绝运行。
- 导入的 run（seed 42–44）没有 extractor 权重，不能跑 C / R；`run_all.py` 会拒绝并说明原因。
- 伪标签集合（B/Q/K/O）每次运行 `run_all.py` 都会重建（几秒钟，结果确定）。
- 停止运行：先 `pgrep -af "[r]un_all.py|[t]rain_detector.py"` 查 PID 再 `kill`，不要用 `pkill -f "<命令行>"`（会匹配到执行它的 shell 自己）；数据加载子进程可能要单独 kill。
- 日志里的进度条用 `\r` 刷新：要 grep 时先 `tr '\r' '\n' < log`；或直接用 `logs/watch.sh` / `logs/progress.sh`。
- 显存：梯度检查点不能关（关闭后 64 条 × 256 token 即 OOM）。
- 主机负载高时（load average 25–33）每步耗时会翻倍，checkpoint 写盘也会变慢；v3 已不再反复写盘。
- v2 的分析脚本（`../v2/analysis/`）硬编码了 v2 的路径，没有复制到 v3；v3 的汇总都在 `evaluation/aggregate_results.py` 中。
- git：服务器上没有 GitHub 凭据（没有 token、SSH 私钥或凭据助手），`git push` 会报 `could not read Username`。在 VS Code 的终端里执行 `git push -u origin v3-training-protocol`（密码处填 GitHub Personal Access Token），或用 VS Code 源代码管理面板的"发布分支"。服务器上 git 没有配置作者身份，之前的提交都用 `git -c user.name=... -c user.email=...` 沿用上一个提交的作者。
