# Semi-vast v3 交接文档

> 更新于 2026-11-14 18:00（标准半监督基线 FixMatch / FreeMatch 跑完，U1、U2 成立，结果写入 README_v3 14.7；之前（15:00）：换卡后 AT 跑完，T1 成立，结果写入 README_v3 13.7.1，第 13 节全部完成；之前（14:20）：AT 两次没跑成，11-07 被关机命令误杀，11-14 GPU 显存有坏区，见第 0 节、第 8 节；之前（2026-11-07 23:30）：TAPT 可行性测试完成、设置选定；之前：第 13 节的 AC / F25 / F75 / F100 / E 跑完，结果写入 README_v3 13.7；之前：全量实验完成，结果写入 README_v3 12.4、12.6；2026-11-06 换卡后重新配置训练、改用 torch.compile，见 README_v3 第 11、12 节；之前的内容更新于 2026-10-04 17:30）。本文档加上 `README_v3.md` 就是 v3 的全部上下文；v3 的由来见 `../v2/README_v2.md` 第 9 节，v2 的历史见 `../v2/HANDOFF.md`。v1 → v3 的演变汇总见 `../history.md`（供写论文参考）。

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
- **换卡后的训练配置（2026-11-05）**：服务器换成 RTX PRO 6000（96GB）。detector 前 18 层、extractor 全部 24 层不做梯度检查点：detector 每步 1.76 → 1.39 秒，extractor 每个 epoch 20 → 14 秒；数值与已有结果等价，配置指纹不变，可以直接在 `runs_f/` 里接着跑。多任务并发没有加速，不要同时跑。torch.compile 能再快一倍，但训练动态有系统性差异（2026-11-06 项目负责人决定采用，见下一条）。详见 README_v3 第 11 节。
- **改用 torch.compile（2026-11-06，项目负责人决定）**：新配置 `configs/config_fc.yaml`（指纹 `df8ea323b54a`，目录 `runs_fc/`、`results_fc/`）。detector 的 encoder 层编译（`fallback_random`，保留与 eager 相同的 dropout），不做检查点；每个 detector 22.5 → 8.8 分钟。extractor 不编译，`[pool]` 不变，所以 `run_main_fc.sh` 沿用 `runs_f` 的伪标签池，结果可以和 `runs_f` 逐 seed 配对比较（`evaluation/compare_runs.py`）。注意 dynamo 默认重编译上限 8 会让部分层静默退回 eager，代码已调到 64，并新增自动检查 S8。主实验 2026-11-06 00:20 启动（`logs/main_fc.log`，约 2.7 小时）。详见 README_v3 第 12 节。
- **全量实验完成（2026-11-06 00:20 – 09:14，`run_full_fc.sh`，已自动关机）**：ratio 0.1 / 0.25 / 0.5 × A / B / F / O × seed 42–46，没有失败的阶段，60 个 detector 的 `uncompiled_frames` 都为 0。汇总 `results_fc/summary.md`，配对比较 `results_fc/compare_runs.md`。
  - **ratio 0.1：编译与 eager 一致**。F − A = +0.053，95% CI (+0.038, +0.069)，5/5（eager 为 +0.049）；F2、F3 成立，H1 不成立。各方法编译 − eager 的平均差在 −0.007 到 +0.000 之间，没有系统性偏移（README_v3 12.4.1）。
  - **随 ratio**：F − A 为 +0.053 / +0.031 / +0.023（0.5 的 CI 下界只有 +0.002），全部"成立"；**F − B 为 +0.040 / +0.005 / +0.006，只在 ratio 0.1 成立**。标注变多后 extractor 变强（验证 F1 0.507 → 0.580 → 0.629），B 本身已经很准，融合的额外作用消失；ratio 0.5 时 O − A 也只有 +0.020（不成立）。F 在每个 ratio 上都是最稳定的组。c 的 AUROC 随 ratio 从 0.583 降到 0.521，低于置信度（README_v3 12.6）。
  - **对论文的含义**：主要结论限定在低标注比例（10%）；0.25 / 0.5 作为"效果随标注增多而收敛"的分析，如实报告 F − B 不成立。
  - 更正：README_v3 10.9 原来写"进入 F 的样本 5 个 seed 都没有被融合改动标签"，实际 seed 46 有 76 / 1164 条（已更正；`../history.md` 同步更正）。
- **文献检索（2026-11-06）**：`../LITERATURE_REVIEW.md`（在线文档 https://claude.ai/artifact/J7HyzwS86S2Cwnc7td551N ）。方法 F 有很接近的前人工作（Folino & Pontieri 2026、VerifyMatch 2024、MultiMatch 2025），论文需要正面比较，并补上只看 claim 的基线和 TAPT 等外部基线。
- **第 5 节已全部完成；下一步见第 6 节。**
- **外部基线与 α 敏感性（2026-11-07，项目负责人决定）**：不补 ratio 0.05；做第 6 节第 1–4 项（只看 claim 的基线 AC、α 敏感性 F25 / F75 / F100、多 NLI 集成 E、TAPT 的 AT）。方法、检验与判据在运行之前写进 README_v3 第 13 节并提交。实现见提交 `50e479f`（离线检查：重建的 B / F / O 集合与 `runs_fc` 逐字节相同；`hypotheses.csv` 不变；3 个 NLI 模型验证集零样本 macro-F1 0.462 / 0.511 / 0.433；seed 42 的池中 2363 / 3853 条三个模型一致，|E| = 1156）。**AC、F25 / F75 / F100、E 已完成**（2026-11-07 14:40 – 17:52，`run_ext_fc.sh`，日志 `logs/ext_fc.log`；25 个 detector 没有失败，`uncompiled_frames` 都为 0；汇总 `results_fc/summary.md` 第 2b 节；结果与解读见 README_v3 13.7）：
  - **X1 A − AC = +0.083，CI (+0.064, +0.106)，5/5，成立**。AC 为 0.428，高于随机；claim 本身的线索集中在 ClimateCheck（AC 0.447，A 只高 0.049），SciFact 上 AC 接近随机。
  - **R1 / R2 成立**：F25 − A = +0.050、F75 − A = +0.052。α 在 0.25–0.75 之间测试 F1 为 0.561–0.564；只按验证集选出的 α 是 0.5。
  - **R3 不成立**：F − F100 = +0.016，5/5 为正，但 CI 下界 −0.0008（来源平均成立）；差异主要来自 seed 45。
  - **E1 不成立**：F − E = −0.001，2/5。均值相同（0.564 / 0.565），E 的跨 seed 标准差 0.034，F 为 0.009。E − F100 = +0.017 成立。
  - 机制（事后分析）：只用 NLI 的 teacher 几乎不给 SUPPORTS（F100 集合中 7–8 条，E 中约 85 条，F 中 149–258 条，池中金标签占 45%），detector 随之少预测 SUPPORTS；seed 45 最明显。
  - 新增 2 条 S2 警告（F25 在 s43 / s45 最佳步靠后），对 R1 是保守的。
  - **TAPT（README_v3 13.5、13.5.1）**：已实现（`training/tapt.py`；AT 接入 `train_detector.py`、`run_all.py`、`common/paths.py`，设置写在 `TAPT_SETTINGS`，不进配置，指纹不变）。冒烟测试：流程本身不改变结果（对照 0.5305 vs A 0.5252）；但学习率 3e-5 训练 1 个 epoch 后验证 F1 跌到 0.434（A − 0.091），MLM 很快破坏 NLI 能力。**可行性测试**：第一轮（18:55 – 20:35，`tools/tapt_probe.sh`，seed 42，12 个候选，只看验证集，日志 `logs/tapt_probe.log`）全部低于 A，最好的是 3e-6、1 个 epoch（0.5203 vs A 0.5252）；学习率 3e-5 的验证 F1 在第 10 个 epoch 回升，触发事先写定的补跑条件。**补跑已完成**（20:36 – 23:15，3e-5、100 个 epoch，日志 `logs/tapt_probe_e100.log`）：第 20 个 epoch 之后验证 F1 停在 0.46–0.47，全部比 A 低 0.05 以上。17 个候选按规则**选定 3e-6、遮盖 15%、1 个 epoch**（0.5203），已填进 `TAPT_SETTINGS`、补进 README_v3 13.5.1。
  - **AT 已完成（2026-11-14 14:15 – 14:47，换卡后，日志 `logs/at_fc.log`；结果与解读见 README_v3 13.7.1）**：
    - **T1 F − AT = +0.075，CI (+0.055, +0.099)，5/5，成立**；来源平均 +0.092。
    - AT 0.489 ± 0.019，比 A 低 0.022（1/5，CI (−0.051, +0.001)，不显著；来源平均 −0.037 显著更差）：从 NLI 模型出发的 TAPT 没有帮助，主要损失在 REFUTES（F1 0.433 → 0.385）。
    - 新增 1 条 S3 警告（s44 的 AT 只预测 152 / 372 条 REFUTES），去掉这个 seed，T1 仍成立。T1 加入后，`summary.md` 第 2b 节排在它后面的参考比较 CI 在第三位小数上变动，结论不变（README_v3 13.7.1 末尾）。
    - **第 13 节的 6 项主要检验全部完成：X1、R1、R2、T1 成立；R3、E1 不成立。**
  - **AT 之前两次没跑成，都没有产生结果，失败与代码和设置无关**：
    - 2026-11-07 23:26 启动，23:28 在 seed 42 的 TAPT（epoch 1 第 81 / 120 步）被关机：等待任务结束再关机的命令是在终端里临时写的，`pgrep` 返回了两个 PID，变量里带换行，判断进程是否存在时失效，于是当场关机（日志 `logs/at_fc_killed.log`、`logs/at_fc_killed_shutdown.log`）。
    - 2026-11-14 13:51 重跑，seed 42 的 TAPT 在只训头的阶段结束后、全参数训练的第一步报 `CUDA error: device-side assert`（DeBERTa 相对位置注意力的 `torch.gather` 下标越界，日志 `logs/at_fc_badgpu.log`）。只训头阶段的 loss 与 11-07 的可行性测试逐位相同（train 7.4784、dev 6.8662），代码、库、驱动（590.44.01）都没变。排查结论：**这张 GPU 的显存有坏区**。`tools/gpu_memtest.py` 把 88 GiB 写满已知数值再读回，第 15–24 个 GiB 共约 190 万个 int64 读回的值不对，3 次结果一致；ECC 是关闭的，所以不报错。全参数训练时显存用到了坏区，DeBERTa 共用的相对位置下标（经 expand 读取）读回乱码，gather 越界；只训头时显存用得少，所以没事。同样的 gather、bmm 在空 GPU 上单独测试结果正确。
    - **已有结果不受影响的依据**：11-07 的可行性测试用完全相同的代码跑了 17 次全参数 TAPT 都没出错，而现在第一步就稳定复现；全量实验和第 13 节的 detector 显存峰值 69.4GB，必然用到 15–25 GiB 区域，如果那时显存就有坏区，也会这样崩溃。所以坏区是 11-14 这次开机才有的（分到了另一张卡，或者原来的卡坏了）。没有记录过 GPU 的 UUID，无法直接比对；今天这张卡是 `GPU-a672ea40-00a5-24d1-8270-1a14080c690c`（PCI `A8:00.0`）。
    - `runs_fc` 里没有留下不完整的 TAPT 文件（`checkpoints/tapt` 不存在），换卡后直接重跑即可。
  - **换卡**（2026-11-14）：新卡 `GPU-d0ece34f-8412-512a-d613-0d8e79eb4a5e`（PCI `C8:00.0`，驱动同为 590.44.01），`tools/gpu_memtest.py` 通过；seed 42 的 TAPT 与 11-07 的可行性测试一致（验证集 MLM loss 6.149 vs 6.150）。下一步见第 6 节（第 1–4 项已全部完成）。
- **标准半监督基线（2026-11-14，项目负责人决定；README_v3 第 14 节）**：FixMatch（FIX）与 FreeMatch（FREE），按 USB 的实现：弱增强 = 原文，强增强 = 德语 / 俄语回译之一（`tools/backtranslate.py`，缓存在 `processed/backtranslation/`）；其余与 F 相同（640 步、每步 16 + 48、λ 0.3 → 0.7）。方案先提交（`f5fb558`），回译检查按事先的规则选定 V1（claim 和证据都回译，方向翻转率 2.2% / 3.5%），冒烟测试不改设置。正式运行 15:51 – 17:45（`run_ssl_fc.sh`，日志 `logs/ssl_fc.log`），10 个 detector 没有失败：
  - **U1 F − FIX = +0.057，CI (+0.038, +0.073)，5/5，成立；U2 F − FREE = +0.079，CI (+0.047, +0.111)，5/5，成立**。
  - FIX − A = −0.004、FREE − A = −0.026，都不显著：一致性训练（和只用 extractor 的伪标签 B 一样）没有超过 A，有效的是外部的 NLI 信号。
  - FREE 的阈值在 640 步内只升到 0.55，mask 比例全程约 1（事先写明的后果，S9 警告 × 5）；新增 S2（s43 FREE）、S3（s44 FIX / FREE 少预测 NEI）各一类警告，去掉对应 seed 结论不变。
  - **第 13、14 节的 8 项主要检验全部完成：X1、R1、R2、T1、U1、U2 成立；R3、E1 不成立。**
  - USB 源码"下载到本机执行、逐元素比对"被权限设置拦下，改为按公式手算核对（README_v3 14.4 第 4 项）。
- git：v3 在分支 **`v3-training-protocol`**（远程 `github.com/NanjoCas/Semi-vast`）。2026-10-05 的两次提交（方向一的代码；试跑与主实验的结果）已推送：2026-11-05 核对，远程分支与本地 `b37b0c6` 相同。**2026-11-07 已提交并推送**（远程与本地 `05511ce` 相同）：换卡与 torch.compile 的代码（`v3: faster training on the 96GB card; torch.compile protocol`），全量实验结果与文档（`v3: full experiment under config_fc`）。有两处有意没有提交：仓库根目录 `configs/config.1epoch.yaml` 是项目负责人自己的改动（v1 文件）；`.github/copilot-instructions.md` 在工作区被删除、来源不明，等项目负责人确认是否有意删除，不是的话用 `git checkout -- .github/copilot-instructions.md` 恢复。服务器现在可以直接推送（deploy key，第 8 节）。之后的提交 `50e479f`、`6043e9b`（第 13 节的实现与结果）已推送；**`390be69`（TAPT 与可行性测试）还没有推送**（2026-11-14 核对，本地领先远程 1 个提交）。2026-11-14 的改动已提交：`a753e70`（`tools/gpu_memtest.py`）、`5838463`（AT 的结果、README_v3 13.7.1、HANDOFF、`../history.md`）、`f5fb558`（第 14 节的事先方案）、`fd2a4c4`（FIX / FREE 的实现与回译）、`3563897`（冒烟测试），以及 FIX / FREE 的结果。**`390be69` 起的提交还没有推送**。

## 1. 项目与环境

| 项 | 内容 |
|---|---|
| 代码 | `/root/autodl-tmp/Semi-vast/v3`（独立目录，没有改动 v2）；主文档 `README_v3.md` |
| 任务 | 半监督虚假信息检测：extractor（DeBERTa-v3-large，从 NLI 模型初始化）给无标签句对打伪标签 → 方法 F：extractor 与 NLI 交叉编码器的概率融合（α = 0.5），取置信度前 30% → detector 用有标签数据加伪标签训练 |
| 数据 | Climate-FEVER + ClimateCheck + SciFact（与 v2 `config_sci.yaml` 相同）：train 4289 / dev 759 / test 2443（test 中 ClimateCheck 占 78%）。ClimateCheck 是 2026-06 的新版：训练 3023 对（2025 版 1144 + 2026 版新增 1879），已全部用上（训练 + 验证 2996 对）；测试 1904 对是 2025 版的测试 claim（2026 版沿用），金标签 2026-06 公开。没有更多 ClimateCheck 数据可加（2026-11-06 核对） |
| 硬件 | AutoDL，1 张 RTX PRO 6000 Blackwell Server Edition（96GB，600W）；容器 16 个 CPU 核、内存上限约 110GB。2026-11-05 之前是 RTX 5090（32GB），换卡后的训练配置见 README_v3 第 11 节 |
| Python | `/root/miniconda3/bin/python`（3.12），torch 2.12 + cu130，transformers 5.5 |
| 模型缓存 | `Semi-vast/model_cache`：microsoft/deberta-v3-large、cross-encoder/nli-deberta-v3-large |

⚠️ 旧机器 shell 里的 `OMP_NUM_THREADS=0` 是非法值；现在的机器上是 16（有效）。各入口脚本都会设成 8，单独运行脚本时也建议先 `export OMP_NUM_THREADS=8`。

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

### 2.1 之后新增或修改的文件（2026-10-04 至 11-06）

| 文件 | 内容 |
|---|---|
| `configs/config_f.yaml`、`run_pilot_f.sh`、`run_main_f.sh`、`evaluation/pilot_report_f.py` | 方向一（方法 F，eager 训练）的配置与入口；结果在 `runs_f/`、`results_f/`（README_v3 10.7–10.9） |
| `training/compute_nli_probs.py`、`analysis/teacher_fusion_probe.*` | NLI 三类概率（方法 F 与 A⊕NLI 对照用）；融合 teacher 的离线验证（10.6） |
| `models/encoder_runtime.py` | **新**（第 11、12 节）：选择性梯度检查点；torch.compile（`fallback_random`、重编译上限 64、`uncompiled_frames`） |
| `training/train_detector.py`、`training/train_extractor.py` | 改：`*_gradient_checkpointing_skip_layers`、`torch_compile`；结果新增 `checkpointed_layers`、`gpu_peak_gb`、`torch_compile`、`uncompiled_frames` |
| `common/fingerprint.py` | 改：两个 `*_skip_layers` 键不进指纹（已有指纹不变） |
| `configs/config_fc.yaml` | **新**：方向一 + torch.compile（指纹 `df8ea323b54a`，`runs_fc/`、`results_fc/`） |
| `run_main_fc.sh`、`run_full_fc.sh` | **新**：ratio 0.1（沿用 `runs_f` 的伪标签池）；全量实验 + 自动关机 |
| `evaluation/compare_runs.py` | **新**：两套配置在相同划分上的配对比较 |
| `evaluation/sanity_check.py` | 加 S8（编译是否覆盖所有层） |
| `logs/watch.sh` | 显示编译与检查点相关的日志行 |

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

2026-11-05 / 06 的验证（选择性检查点的数值等价性、编译后的单步梯度与完整训练对照、重编译上限）见 README_v3 11.2、11.3、12.2、12.3。

## 4. 常用命令

```bash
cd /root/autodl-tmp/Semi-vast/v3
export OMP_NUM_THREADS=8

# 全量实验的进度（Ctrl+C 退出，不影响训练）
cat logs/full_fc_status.txt               # 每个阶段的开始 / 完成 / 失败
bash logs/watch.sh logs/main_fc.log       # ratio 0.1；ratio 0.25 / 0.5 的日志在 logs/full_fc.log
bash logs/progress.sh logs/full_fc.log    # 进度快照

# 全量实验中断或失败后续跑（已完成且指纹相同的会跳过）；不想关机加 NO_SHUTDOWN=1
NO_SHUTDOWN=1 setsid nohup bash run_full_fc.sh > logs/full_fc.log 2>&1 < /dev/null &

# 第 13 节的实验（AC、F25 / F75 / F100、E）：进度与中断后续跑（已完成的会跳过）
bash logs/watch.sh logs/ext_fc.log
setsid nohup bash run_ext_fc.sh > logs/ext_fc.log 2>&1 < /dev/null &

# TAPT 可行性测试（seed 42，只看验证集）：进度、中断后续跑、汇总
grep -E "STEP|DONE|epoch [0-9]+/|Training complete" logs/tapt_probe.log | tail
setsid nohup bash tools/tapt_probe.sh > logs/tapt_probe.log 2>&1 < /dev/null &
python tools/tapt_probe_report.py --run_dir runs_fc/r0.10_s42      # -> runs_fc/r0.10_s42/outputs/tapt_probe/report.md

# GPU 显存自检（换卡或开机后、跑实验之前；约 1 分钟，退出码 0 才算通过）
python tools/gpu_memtest.py

# 第 14 节：回译（已缓存的不重新翻译，并重做 14.4 的检查）与 FixMatch / FreeMatch（中断后重跑同一条命令即可续跑）
python tools/backtranslate.py --config configs/config_fc.yaml --device cuda
setsid nohup bash run_ssl_fc.sh > logs/ssl_fc.log 2>&1 < /dev/null &

# 汇总与配对比较（只读结果，几十秒）
python run_all.py --config configs/config_fc.yaml --aggregate_only   # -> results_fc/summary.md
python evaluation/compare_runs.py --ratio 0.1                        # -> results_fc/compare_runs.md
```

最初试跑 / 主实验方案的命令见 README_v3 第 5 节。

## 5. 全量实验之后的收尾

1. ✅ `logs/full_fc_status.txt`：所有阶段完成，没有失败（2026-11-07 核对）。
2. ✅ `results_fc/summary.md`：结果与 7 条自动检查警告（6 条 S2、1 条 S3）都已解释，写在 README_v3 12.6 末尾。
3. ✅ `results_fc/compare_runs.md`：编译与 eager 没有系统性偏移，F − A 一致（+0.053 vs +0.049），见 README_v3 12.4.1。
4. ✅ 结果已写入 README_v3 12.4.1、12.5（执行情况）、新增的 12.6（F − A 随 ratio 的变化）和第 9 节执行清单第 5 步（2026-11-07）。
5. ✅ 提交 git（2026-11-07，两次提交，见第 0 节最后一条；有两处有意没有提交）。✅ 已推送（2026-11-07）。

## 6. 下一步（按优先级；改代码前先确认没有任务在跑）

第 1–4 项的设计与事先确定的检验见 README_v3 第 13 节（2026-11-07）。第 1、2、4 项已完成（2026-11-07，结果见 README_v3 13.7）；第 3 项也已完成（2026-11-14，T1 成立，README_v3 13.7.1）。**第 1–4 项全部完成**；第 7 项（标准半监督基线）也已完成（2026-11-14，U1、U2 成立，README_v3 14.7）。剩下的是建议项 5（HealthFC）、8（方法改进）和 9（写作），做哪些需要项目负责人决定；6（ratio 0.05）已决定不做。

下面的实验都和 `runs_fc` 的 A / F 比较，所以都用 `config_fc.yaml` 的训练协议，ratio 0.1、seed 42–46。全量实验表明 F 相对 B 的优势只在 ratio 0.1 成立（README_v3 12.6），所以这些实验都放在 ratio 0.1 上做，不需要在 0.25 / 0.5 上重复。

| 优先级 | 任务 | 做法 | 代价 | 依据 |
|---|---|---|---|---|
| ✅ 1 必做 | 只看 claim 的基线（已完成：方法 AC，X1 成立，README_v3 13.7） | 新方法（例如 `A_claim`）：detector 训练和测试都只输入 claim，其余与 A 相同；要改 Dataset、方法列表与 `common/paths.py` | 约 20 分钟（A 每个约 3.3 分钟） | Schuster 2019；Usmanova 2026。评审一定会问 |
| ✅ 2 必做 | α 敏感性（已完成：F25 / F75 / F100，R1 / R2 成立、R3 不成立；没有用单独的配置，见 README_v3 13.1） | α = 0 就是 B，α = 0.5 已有；补 α = 0.25、0.75、1.0（1.0 = 只用 NLI 的 teacher）。`teacher.nli_fusion_alpha` 在指纹里，每个 α 要单独的配置和 runs 目录，像 `run_main_fc.sh` 一样沿用 `runs_f` 的伪标签池 | 约 2.3 小时（3 × 5 × 9 分钟） | README_v3 10.9 限制 2：α 是看过金标签后定的 |
| ✅ 3 必做 | TAPT 外部基线（已完成：方法 AT，T1 成立，AT − A = −0.022 不显著，README_v3 13.7.1；设计已定在 README_v3 13.5：从 `cross-encoder/nli-deberta-v3-large` 起、新初始化 MLM 头、每个 seed 只用自己的无标签池、不用 39 万篇摘要） | 在无标签池（可加上 ClimateCheck 的 39 万篇摘要）上继续预训练 encoder，再按 A 的设置微调。**注意**：DeBERTa-v3 用 RTD 预训练，官方权重没有 MLM 头，要新初始化 MLM 头或改用 RTD 目标，先做小规模可行性验证 | 预训练时间待测，微调约 20 分钟 | Shi 2023；Gururangan 2020 |
| ✅ 4 建议 | 多 NLI 模型集成 + 拒识（已完成：方法 E，E1 不成立，README_v3 13.7） | 2–3 个现成 NLI 模型零样本标注无标签池，一致才保留，作为新方法与 F 比较 | 低：推理加选样规则，约 1 小时 | Folino & Pontieri 2026（最接近的工作） |
| 5 建议 | 泛化测试（HealthFC） | ① 转换 HealthFC，核对标签能否映射到三分类、证据格式（不占 GPU）；② 在单独目录重训 ratio 0.1 的 A、F 并保存权重（`run_all.py --keep_checkpoints`，10 个模型约 17GB，数据盘剩约 46GB）；③ 写一个"加载保存的权重、在任意测试集上评估"的脚本 | 约 1 小时 GPU | 回应"测试集被 ClimateCheck 主导" |
| 6 建议 | 更低的标注比例（ratio 0.05） | 加到 `config_fc.yaml` 的 `experiment.label_ratios`（不影响指纹），A / B / F / O × seed 42–46，从 SSL 划分和 extractor 开始 | 约 3 小时（按 ratio 0.25 的耗时估计） | 12.6：F 的贡献随标注减少而增大，补一个更低的点能支撑"低资源"的定位；需要项目负责人决定，且要在跑之前写定判据 |
| ✅ 7 可选 | 标准半监督基线（已完成：FixMatch 与 FreeMatch，方法 FIX / FREE，U1、U2 成立，FIX / FREE − A 不显著；README_v3 第 14 节） | USB 的 FixMatch / FreeMatch（或 MultiMatch），需要回译等文本增强 | 中到高 | USB 2022；Sirbu 2025 |
| 8 可选 | 方法改进 | 不一致的样本降权或 MixUp，而不是丢弃；LLM 作第三个 teacher | 中 | VerifyMatch 2024；LG-CoTrain 2025 |
| 9 写作 | 相关工作 | 按 `../LITERATURE_REVIEW.md` 组织；补上缺失的引用：ClimateCheck 2025 / 2026、Oliver 2018、Shi 2023、Schuster 2019、UPS、Folino & Pontieri、VerifyMatch、MultiMatch、DecomposeRL、Usmanova 2026 | — | — |

## 7. 待决问题（需要项目负责人决定，建议和导师确认）

- ~~论文定位~~：**2026-11-14 项目负责人告知导师已通过**，按下面的建议写：主要贡献是证据感知的融合 teacher（方法 F），主要结论限定在低标注比例（ratio 0.1）；RL 选择器与 DiscourseScore 作为负面结果；R3、E1 如实报告为未显示出显著差异（描述性地报告 F 更稳定）；外部基线的结论分别限定为"从 NLI 模型出发的 TAPT""相同训练预算下的 FixMatch / FreeMatch"。原建议：建议写成"证据感知的逻辑 teacher 用于伪标签选择"，RL 作为负面结果或放进附录（与开题报告的差异要确认）。全量实验后，建议主要结论限定在低标注比例（ratio 0.1），ratio 0.25 / 0.5 作为"teacher 变强后融合不再有额外作用"的分析（README_v3 12.6）。
- ~~是否补 ratio 0.05~~：2026-11-07 决定不补。
- 外部基线的范围：2026-11-07 决定做第 6 节 1–4 项（README_v3 第 13 节）；7（标准半监督基线）2026-11-14 决定做 FixMatch 与 FreeMatch，已完成（README_v3 14.7）。
- R3、E1 不成立之后的论文定位（2026-11-07 新增）：方法 F 不再能声称优于只用 NLI 的 teacher 或多 NLI 集成。建议把贡献写成"证据感知的融合 teacher 在低标注比例下稳定有效、对 α 不敏感"，与 E 的比较如实报告为"均值相当、F 跨 seed 更稳定、类别覆盖更完整"（描述性，README_v3 13.7 第 5 条）。是否另加一个"按类别配额选样"的 E 变体来检验这个机制，需要项目负责人决定，且要在跑之前写定判据。2026-11-14：T1 成立，可以在贡献里加上"优于 TAPT"（限定为 NLI 初始化的设置，README_v3 13.7.1 第 4 条）；U1、U2 成立，可以再加上"在相同训练预算下优于 FixMatch / FreeMatch"（README_v3 14.7 第 5 条）。
- 泛化测试集：是否用 HealthFC（健康领域，标签映射待核对）。
- α 是否按来源分别设定：会多一个超参数，选择规则要事先写定，避免在测试集上挑。
- Climate-FEVER 的 DISPUTED 处理（目前丢弃）。

## 8. 已知的坑

- `run_all.py` 现在按配置指纹判断能否跳过：指纹相同的 detector 结果会被跳过，不同的会重跑；整个 runs 目录的指纹记录在 `runs/protocol.json`，与当前配置不同时直接拒绝运行。
- 导入的 run（seed 42–44）没有 extractor 权重，不能跑 C / R；`run_all.py` 会拒绝并说明原因。
- 伪标签集合（B/Q/K/O）每次运行 `run_all.py` 都会重建（几秒钟，结果确定）。
- 停止运行：先 `pgrep -af "[r]un_all.py|[t]rain_detector.py"` 查 PID 再 `kill`，不要用 `pkill -f "<命令行>"`（会匹配到执行它的 shell 自己）；数据加载子进程可能要单独 kill。
- 日志里的进度条用 `\r` 刷新：要 grep 时先 `tr '\r' '\n' < log`；或直接用 `logs/watch.sh` / `logs/progress.sh`。
- 显存：eager 配置（`config_f.yaml`）下 detector 前 18 层不做检查点，最坏情况 84GB；编译配置（`config_fc.yaml`）不做检查点，最坏情况 69GB（README_v3 第 11、12 节）。**不要在 GPU 上同时跑别的任务**，并发本身也没有加速（实测比串行慢约 9%）；`train_detector.py` 发现显存已被占用时会警告。换回 32GB 的卡：`config_f` 把 `gradient_checkpointing_skip_layers` 改为 0；`config_fc` 需要打开 `gradient_checkpointing`，而它在指纹里，改了要换 runs 目录。
- **改代码前确认没有任务在跑**：每个 detector 都是新进程，启动时重新读代码；串联脚本运行期间改 `training/`、`models/`、`common/` 会影响后面还没跑的 run。
- torch.compile：dynamo 默认每段代码最多 8 个编译版本，超出的部分静默退回 eager；`models/encoder_runtime.py` 已调到 64，结果里的 `uncompiled_frames` 应为 0，S8 会检查。每个 detector 的第一个 40 步窗口含约 15 秒编译。
- `run_full_fc.sh` 最后会执行 `shutdown`（AutoDL 实例停机、停止计费，数据盘保留）。取消：`pgrep -af "[r]un_full_fc.sh"` 查 PID 后 `kill`，不影响正在训练的 run。
- **临时写"等任务结束再关机"的命令要小心**：`pgrep -f` 可能返回多个 PID（包括执行它的 shell），整个变量传给 `kill -0` 会失效，被当成任务已结束而立刻关机（2026-11-07 的 AT 就是这样被关掉的）。用 `while pgrep -f "[r]un_ext_fc.sh" > /dev/null; do sleep 60; done; shutdown`，或者只等一个确定的 PID。
- **会话重启后可能有两个 Claude 进程同时在干活**：2026-11-14 VS Code 重连两次，旧的 Claude 进程没有退出，一次继续改代码、提交、启动冒烟测试，一次挂着后台等待、运行结束时会被唤醒。重启后先 `ps -eo pid,lstart,cmd | grep "[c]laude --output"`，多于一个时先结束旧的（后台的训练是独立进程，不受影响），再继续。
- 等某个后台任务结束时，用 `pgrep -f "^bash run_xxx.sh"` 这样锚定开头的匹配拿 PID；不锚定的 `pgrep -f` 会匹配到执行它的那个 shell（2026-11-14 回译时就等错了进程）。
- **GPU 显存可能有坏区而不报错**：RTX PRO 6000 的 ECC 默认关闭，坏区只会悄悄改写数据。2026-11-14 分到的 `GPU-a672ea40-…`（PCI `A8:00.0`）在第 15–24 个 GiB 有坏区（换成 `GPU-d0ece34f-…` 后自检通过），表现为 DeBERTa 全参数训练时 `torch.gather` 报 device-side assert（下标越界），不求梯度、显存用得少时一切正常。换卡或开机后先跑 `python tools/gpu_memtest.py`；遇到无法解释的 CUDA assert，也先跑它。
- detector 的权重默认不保存（`cleanup_checkpoints: true`，最佳权重只在内存里）。全部保存约 100GB，数据盘放不下；要在新测试集上评估时只保存关键模型（第 6 节第 5 项）。
- 服务器可以访问 huggingface.co 和 arxiv.org（curl 可用）；模型缓存在 `../model_cache`。
- 主机负载高时（load average 25–33）每步耗时会翻倍，checkpoint 写盘也会变慢；v3 已不再反复写盘。
- v2 的分析脚本（`../v2/analysis/`）硬编码了 v2 的路径，没有复制到 v3；v3 的汇总都在 `evaluation/aggregate_results.py` 中。
- git：2026-11-07 起服务器用 deploy key 推送：私钥 `~/.ssh/semi_vast_deploy`，公钥加在 GitHub 仓库 Settings → Deploy keys（有写权限，只对这个仓库有效，不用时在那里删除）。仓库设置：`origin` 拉取仍走 HTTPS，推送走 SSH（`remote.origin.pushurl`），`core.sshCommand` 指定这把密钥；fetch 规则原来只有 `v2-data-design`，已追加 `v3-training-protocol`，所以直接 `git push` / `git fetch` 即可。私钥在系统盘上（不在 `autodl-tmp`），迁移实例或重置系统盘后会丢失，需要重新生成并在 GitHub 上替换公钥。VS Code 的 GitHub 登录弹窗在这台服务器上不出现；用 token 推送时在终端执行 `GIT_ASKPASS= git push`。服务器上 git 没有配置作者身份，之前的提交都用 `git -c user.name=... -c user.email=...` 沿用上一个提交的作者。
