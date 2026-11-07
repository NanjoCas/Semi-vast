# 文献检索：相关工作、2024–2026 新进展与可选方法

> 检索日期 2026-11-06。检索范围：ACL Anthology、arXiv、会议与期刊页面，共约 35 次检索，重点是 2024–2026 年。
> 本文只读了摘要和检索摘要，**没有读全文**。标 ✓ 的文献，出处（会议 / 期刊 / arXiv 编号）在本次检索中核对过；标 * 的是领域经典文献，出处凭记忆写出，写进论文前请逐篇核对。
> 与本项目的对应关系以 `v3/README_v3.md` 为准：方法 F = extractor 与 NLI 交叉编码器的概率融合（α = 0.5）后按置信度取前 30% 作为伪标签；主结果 F − A = +0.049（10.9 节）。

---

## 1. 要点

1. **本项目在文献中的位置**：半监督（伪标签）+ 证据感知的科学 / 气候事实核查 + 用 NLI 模型作为第二个 teacher。三者分别都有大量工作，组合起来的直接先例很少。
2. **方法 F 已有非常接近的工作，论文必须正面比较**：
   - Folino & Pontieri（2026）：多个预训练 NLI 模型做零样本标注，按一致性和拒识阈值筛选伪标签，再微调编码器，实验数据包括假新闻检测；
   - VerifyMatch（EMNLP 2024）：LLM 给伪标签，半监督模型充当核验者，两者不一致时特殊处理；
   - LG-CoTrain（2025）、MultiMatch（EMNLP 2025）：按模型间一致性加置信度筛选伪标签。

   方法 F 的新颖性不在"融合两个 teacher"这个操作本身，而在于：（a）任务是证据感知的三分类核查（claim + 摘要），而不是单句文本分类；（b）NLI 模型和任务模型看的是同一对 claim–evidence，NLI 的判断天然对应 SUPPORTS / REFUTES；（c）在严格协议下（固定预算、强基线、5 个 seed、分层 bootstrap）证明了提升，并分析了提升来自哪里（F − B、F − A⊕NLI、分来源结果）。
3. **强化学习选样**：2024–2026 年，RL 用于数据选择的工作主要转向了 LLM 后训练（GRPO 等）。与本项目直接相关的只有 DecomposeRL（2026）：它用 RL 做半监督的 claim 核验，在只有 10% 有标签数据时有效，但它是让 RL 学习"怎么分解 claim"，而不是"挑哪些伪标签"。开题所依据的 WeFEND（AAAI 2020）之后，同类的"RL 伪标签选择器"在 NLP 中没有成为主流。本项目的负面结果（RL 奖励中的 ΔF1 项只有 ±0.007，小于训练噪声）可以作为一条分析结论来写。
4. **ClimateCheck 上的最新发现和本项目一致**：Usmanova et al.（2026）发现，在 ClimateCheck 上，微调模型和**只看 claim** 的模型反而优于零样本 LLM；在 SciFact 上最好的系统换到 ClimateCheck 后成绩大幅下降。这与本项目"NLI 零样本在 SciFact 上强（0.621）、在 ClimateCheck 上弱（0.408）"的观察一致，可以直接引用来支撑分来源分析。同时说明**论文必须报告只看 claim 的基线**（Schuster et al. 2019 的经典问题），这也是 v2 留下的待决问题。
5. **评测方法论有现成文献支撑**：v3 发现"v2 的提升主要来自基线 A 太弱"，这正是 Oliver et al.（2018）指出的问题；Shi et al.（2023）进一步发现，任务自适应预训练（TAPT）常常比复杂的自训练方法更强、更稳。两者都应该引用，TAPT 还应该作为外部基线。

---

## 2. 分主题文献

### 2.1 任务与数据集

| 文献 | 出处 | 与本项目的关系 |
|---|---|---|
| Thorne et al. FEVER | NAACL 2018 * | 三分类核查（SUPPORTS / REFUTES / NEI）的源头 |
| Diggelmann et al. CLIMATE-FEVER | NeurIPS 2020 Workshop（Tackling Climate Change with ML）✓ | 本项目数据之一（测试集 200 条） |
| Wadden et al. Fact or Fiction: Verifying Scientific Claims（SciFact） | EMNLP 2020 ✓ | 本项目数据之一（测试集 339 条），NLI 信号最强的来源 |
| Kotonya & Toni. Explainable Automated Fact-Checking for Public Health Claims（PUBHEALTH） | EMNLP 2020 ✓ | 开题时的数据，v2 起不再使用 |
| Abu Ahmad, Usmanova & Rehm. The ClimateCheck Shared Task: Scientific Fact-Checking of Social Media Claims about Climate Change | SDP Workshop @ ACL 2025（2025.sdp-1.24）✓ | **本项目测试集 78% 的来源，目前文档里没有引用，必须补上**。435 条原子化的推文风格 claim，配有标注为 supports / refutes / NEI 的论文摘要；子任务二（分类）冠军 F1 约 0.73（与检索结果联合计分，不能和本项目的数字直接比较） |
| ClimateCheck 2026: Scientific Fact-Checking and Disinformation Narrative Classification of Climate-related Claims | NSLP Workshop @ LREC 2026，arXiv 2603.26449 ✓ | 第二届：在 2025 版训练集上新增人工标注数据，合计 3023 对；测试 claim 不变；新增虚假叙事分类（CARDS 分类体系）。**本项目已经在用 2026 版的全部训练数据**（训练 + 验证 2996 对，2026-11-06 核对） |
| Kiepura & Lam. ClimateCheck2025: Multi-Stage Retrieval Meets LLMs for Automated Scientific Fact-Checking | SDP 2025（2025.sdp-1.28）✓ | 参赛系统（总分第二），可用来说明这个数据上的性能水平 |
| Vladika, Schneider & Matthes. HealthFC | LREC-COLING 2024 ✓ | 750 条健康 claim，带医学专家标注的证据；可以作为额外测试集（v2 HANDOFF 提过） |
| Schlichtkrull et al. The Automated Verification of Textual Claims (AVeriTeC) Shared Task | FEVER Workshop 2024（2024.fever-1.1）✓ | 真实世界 claim 加网络证据，四分类；2025 版要求只用开源 LLM |
| Leippold et al. Automated fact-checking of climate claims with large language models（Climinator） | npj Climate Action 4, 17（2025）✓ | 气候领域的 LLM 核查系统（多个 LLM 辩论，依据 IPCC 等权威来源） |
| Climate-Eval: A Comprehensive Benchmark for NLP Tasks | arXiv 2505.18653（2025）✓ | 气候 NLP 基准，包含 claim 核查 |
| **Usmanova, Iklassov, Leippold & Usbeck. How Robust Are Automated Fact-Checking Systems? A Cross-Benchmark Evaluation** | arXiv 2608.25934（2026-08）✓ | **与本项目数据最相关的新工作**，第一作者是 ClimateCheck 的组织者之一；见要点 4。另一个发现：检索是主要瓶颈，用金标准证据时准确率高 14–22 个百分点。本项目用的是金标准证据，论文里要说明这个范围限定 |

### 2.2 综述

| 文献 | 出处 | 用途 |
|---|---|---|
| Vladika & Matthes. Scientific Fact-Checking: A Survey of Resources and Approaches | Findings of ACL 2023 ✓ | 科学事实核查的资源与方法，写相关工作第一节的骨架 |
| Dmonte et al. Claim Verification in the Age of Large Language Models: A Survey | ACL 2026（arXiv 2408.14317）✓ | LLM 时代的检索、提示、微调方法 |
| Generative Large Language Models in Automated Fact-Checking: A Survey | arXiv 2407.02351（2024）✓ | 生成式 LLM 在事实核查中的用法 |

### 2.3 半监督学习与伪标签选择

| 文献 | 出处 | 与本项目的关系 |
|---|---|---|
| Blum & Mitchell. Combining labeled and unlabeled data with co-training | COLT 1998 * | 两个"视角"相互提供伪标签。方法 F 中 extractor 与 NLI 的一致性筛选可以看作它的变体 |
| Lee. Pseudo-Label | ICML 2013 Workshop * | 伪标签的基本做法（基线 B 的来源） |
| Arazo et al. Pseudo-labeling and confirmation bias in deep semi-supervised learning | IJCNN 2020 * | 确认偏差：teacher 的错误会在学生中被放大。可以用来解释 B − A 不显著 |
| Sohn et al. FixMatch | NeurIPS 2020 * | 置信度阈值加一致性正则，标准的半监督基线 |
| Xie et al. Unsupervised Data Augmentation（UDA） | NeurIPS 2020 * | NLP 中常用的一致性正则方法（回译增强） |
| Xie et al. Self-training with Noisy Student | CVPR 2020 * | teacher–student 迭代自训练 |
| Rizve et al. In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework（UPS） | ICLR 2021 ✓（arXiv 2101.06329） | **"只看置信度不够"已有的代表工作**。开题的核心创新点必须和它区分：UPS 用的是模型自身的不确定性，本项目用的是外部的逻辑信号（NLI） |
| Han et al. Co-teaching | NeurIPS 2018 * | 两个网络互相挑选"小损失"样本，处理噪声标签 |
| Wang et al. USB: A Unified Semi-supervised Learning Benchmark for Classification | NeurIPS 2022 Datasets & Benchmarks ✓ | 统一实现了 FixMatch、FreeMatch、SoftMatch 等，**包含 NLP 任务，可以直接用来跑外部基线** |
| Zou & Caragea. JointMatch | EMNLP 2023 ✓ | 半监督文本分类，两个网络协同生成伪标签 |
| Sirbu et al. MultiMatch: Multihead Consistency Regularization Matching for Semi-Supervised Text Classification | EMNLP 2025 ✓ | 按"多头是否一致 + 置信度"筛选和加权伪标签，在类别不平衡时尤其稳。**与方法 F 的筛选机制最像的半监督方法** |
| Sadat & Caragea. Learning to Infer from Unlabeled Data: A Semi-supervised Learning Approach for Robust NLI | Findings of EMNLP 2022 *（arXiv 2211.02971 ✓） | NLI 任务本身的半监督 |

### 2.4 LLM / NLI 作 teacher：弱监督、蒸馏、协同训练（与方法 F 最接近）

| 文献 | 出处 | 与方法 F 的异同 |
|---|---|---|
| Yin, Hay & Roth. Benchmarking Zero-shot Text Classification | EMNLP 2019 * | 用 NLI 做零样本分类的起点 |
| **Folino & Pontieri. Sustainable hybrid text classification: Enhancing encoder-only language models with NLI-derived pseudo-examples** | Elsevier 期刊（ScienceDirect，2026）✓ | **最接近的工作**：多个 NLI 模型（BART-MNLI、RoBERTa-MNLI）零样本标注，按一致性和风险阈值拒识，再微调编码器；实验包括假新闻检测。区别：它是单句分类、NLI 只充当零样本标注器；方法 F 是 claim–evidence 对、NLI 与任务模型融合 |
| **Park & Caragea. VerifyMatch** | EMNLP 2024 ✓ | LLM 伪标签 + 半监督模型核验；两者不一致或置信度低时用 MixUp，而不是丢弃。可以借鉴来处理"不一致"的样本（方法 F 现在直接丢掉） |
| Rahman & Caragea. LLM-Guided Co-Training for Text Classification（LG-CoTrain） | arXiv 2509.16516（2025）✓ | LLM 伪标签 + 两个小模型协同训练，相互交换对伪标签质量的权重 |
| Li et al. Learning with Less: Knowledge Distillation from LLMs via Unlabeled Data（LLKD） | Findings of NAACL 2025 ✓ | 选 teacher 有把握、学生不确定的样本。**可以作为选样策略的对照**：方法 F 选的是两个 teacher 都有把握的样本 |
| Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation（CanDist） | ACL 2025（2025.acl-long.139）✓ | 用 LLM 的候选标签集合而不是单一标签来蒸馏 |
| Lang et al. Co-training Improves Prompt-based Learning for Large Language Models | ICML 2022 *（arXiv 2202.00828 ✓） | LLM 与小模型协同训练的早期工作 |

### 2.5 强化学习用于样本选择

| 文献 | 出处 | 与本项目的关系 |
|---|---|---|
| **Wang et al. Weak Supervision for Fake News Detection via Reinforcement Learning（WeFEND）** | AAAI 2020, 34(01): 516–523 ✓ | **开题框架的直接模板**：标注器 → 强化选择器 → 检测器。注意 WeFEND 的弱标签来自用户举报，信号比伪标签强 |
| Schulman et al. PPO | arXiv 1707.06347 * | 选择器的优化算法 |
| Selective Pseudo-Labeling with Reinforcement Learning for Semi-Supervised Domain Adaptation | arXiv 2012.03438 ✓ | RL 挑伪标签（视觉领域） |
| Learning to Clean: Reinforcement Learning for Noisy Label Correction | arXiv 2511.19808（2025）✓ | RL 纠正噪声标签 |
| RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment | arXiv 2506.21037（2025）✓ | RL 做数据选择 |
| **DecomposeRL: Learning to Ask Useful, Informative, and Diverse Questions for Semi-Supervised, Traceable Claim Verification** | arXiv 2605.27858（2026-05）✓ | **与"RL + 半监督 + 核查"唯一直接重合的新工作**：用 GRPO 学习分解 claim，多数投票得到伪标签，在 10% 有标签数据时优于基线 |

### 2.6 与"逻辑"相关的核查工作（支撑开题的"逻辑特征"动机）

| 文献 | 出处 | 关系 |
|---|---|---|
| Glockner et al. MISSCI: Reconstructing Fallacies in Misrepresented Science | ACL 2024 ✓ | 健康错误信息常常歪曲引用的论文，需要识别其中的逻辑谬误 |
| Glockner et al. Grounding Fallacies Misrepresenting Scientific Publications in Evidence（MissciPlus） | NAACL 2025 ✓ | 发现现有核查模型难以用被歪曲的论文段落来反驳 claim |
| Strong, Aly & Vlachos. Zero-Shot Fact Verification via Natural Logic and Large Language Models（Zero-NatVer） | Findings of EMNLP 2024 ✓ | 用自然逻辑做零样本核查，可解释 |

### 2.7 合成数据与小模型核查

| 文献 | 出处 | 关系 |
|---|---|---|
| Tang, Laban & Durrett. MiniCheck | EMNLP 2024 ✓ | 用 GPT-4 生成有难度的合成数据，训练出达到 GPT-4 水平的小核查模型 |
| Zhang et al. Enhancing Health Fact-Checking with LLM-Generated Synthetic Data | arXiv 2508.20525（2025）✓ | 用 LLM 生成合成 claim 增强训练，PubHealth 上 F1 提高 0.019，SciFact 上提高 0.049。**与伪标签互为替代方案，数据集与本项目重合** |

### 2.8 评测方法论

| 文献 | 出处 | 关系 |
|---|---|---|
| **Oliver et al. Realistic Evaluation of Deep Semi-Supervised Learning Algorithms** | NeurIPS 2018 ✓ | 基线调得够强时，半监督的收益比报告的小。**正是 v2 → v3 改协议的理由** |
| **Shi et al. Rethinking Semi-supervised Learning with Language Models** | Findings of ACL 2023 ✓ | TAPT 比 5 种自训练方法更强、更稳，即使只有几百条无标签数据 |
| Schuster et al. Towards Debiasing Fact Verification Models | EMNLP-IJCNLP 2019 ✓ | 只看 claim 的模型在 FEVER 上就能接近有证据的模型。**必须报告只看 claim 的基线** |
| Dror et al. The Hitchhiker's Guide to Testing Statistical Significance in NLP | ACL 2018 * | 显著性检验的依据（本项目用分层 bootstrap） |
| Menon et al. Long-tail learning via logit adjustment | ICLR 2021 * | v3 已关闭，但代码与配置里仍引用 |
| Gururangan et al. Don't Stop Pretraining | ACL 2020 * | 领域 / 任务自适应预训练（DAPT / TAPT） |

---

## 3. 其他可选方法

按"对论文的必要性"排序。代价是相对于现在每个 detector 约 9 分钟（编译后）的粗略估计。

| # | 方法 | 依据 | 在本项目中怎么做 | 代价 | 作用 |
|---|---|---|---|---|---|
| 1 | **只看 claim 的基线** | Schuster 2019；Usmanova 2026 | detector 输入只给 claim，其余设置与 A 相同 | 低：改输入即可，每个 seed 一个 run | 评审一定会问；如果它在 ClimateCheck 上接近 A，就需要讨论数据偏差 |
| 2 | **TAPT**（在无标签池上继续做掩码语言模型预训练，再按 A 的设置微调） | Shi 2023；Gururangan 2020 | 用无标签池中的 claim + evidence（也可加上 ClimateCheck 的 39 万篇摘要）做预训练 | 中：一次预训练，之后和 A 一样 | **最重要的外部基线**：代表"不用伪标签也能利用无标签数据"。如果 TAPT ≥ F，结论要改写；如果 TAPT + F 更好，则是加分项 |
| 3 | **标准半监督方法**（FixMatch / FreeMatch / SoftMatch，或 MultiMatch） | USB 2022；Sirbu 2025 | 用 USB 的实现，或在现有训练循环里加一致性损失；需要文本增强（回译等） | 中到高：要接入增强与新的损失 | 回答"为什么不用现成的半监督方法" |
| 4 | **多 NLI 模型集成 + 拒识** | Folino & Pontieri 2026 | 用 2–3 个现成 NLI 模型零样本标注无标签池，一致才保留 | 低：只需推理和改选样规则 | 和最接近的工作直接比较，说明 extractor 参与融合的价值 |
| 5 | **LLM 作第三个 teacher** | VerifyMatch 2024；LG-CoTrain 2025；LLKD 2025 | 用开源 LLM 零 / 少样本标注无标签池，与 extractor、NLI 三方融合或按一致性筛选 | 中：LLM 推理（约 4000 条 × 每个 seed） | 自然的延伸；也可以检验"逻辑信号"能否被 LLM 替代 |
| 6 | **不一致样本不丢弃，而是做 MixUp 或降权** | VerifyMatch 2024 | 方法 F 现在只用两者一致的样本；可以把不一致的样本以低权重加入 | 低到中 | 可能提高覆盖率，也是方法上的小改进 |
| 7 | **用 LLM 生成合成数据** | MiniCheck 2024；Zhang 2025 | 由摘要生成合成的 claim–evidence 对加入训练 | 高：生成与质量控制都要花时间 | 伪标签的替代方案；作为对照组或未来工作 |
| 8 | **α 敏感性与按来源设置权重** | — | α ∈ {0, 0.25, 0.5, 0.75, 1}，以及按来源分别设定 α | 低到中：每个 α 一组 run | **必须做**：α = 0.5 是看过金标签后定的（README_v3 10.9 节限制 2） |
| 9 | **重新设计 RL**（奖励去掉常数项、状态加入类别和 c；或者参考 DecomposeRL） | WeFEND 2020；DecomposeRL 2026 | 只有在 F 有效之后才值得做 | 高 | 建议作为负面结果写进论文或放附录，不作为主线 |

---

## 4. 对论文的建议

1. **相关工作章节可以按 2.1–2.8 的顺序组织**：任务与数据 → 半监督与伪标签 → LLM / NLI 作 teacher → RL 选样 → 逻辑与谬误 → 评测方法论。
2. **必须引用、目前缺失的**：ClimateCheck（2025 和 2026 两版）、Oliver 2018、Shi 2023、Schuster 2019、Rizve 2021（UPS）、Folino & Pontieri 2026、VerifyMatch 2024、MultiMatch 2025、DecomposeRL 2026、Usmanova 2026。
3. **至少补两个外部基线**：只看 claim 的模型，和 TAPT。如果时间允许，再加一个标准半监督方法（USB 中的 FixMatch 或 FreeMatch）。
4. **方法 F 的定位**：建议写成"证据感知的逻辑 teacher（logic-aware teacher）用于科学 / 气候事实核查的伪标签选择"，并与 Folino & Pontieri、VerifyMatch、MultiMatch 逐条对比差异（输入形式、teacher 的来源、如何处理不一致、领域）。
5. **数据**：本项目已经包含 ClimateCheck 2026 新增的全部训练数据，2026 版的测试 claim 与 2025 版相同，没有更多 ClimateCheck 数据可加；可以加入 HealthFC 作为额外测试集，以回应"测试集被 ClimateCheck 主导"的问题。

---

## 5. 来源

- ClimateCheck 2025：https://aclanthology.org/2025.sdp-1.24/ ；数据集 https://huggingface.co/datasets/rabuahmad/climatecheck
- ClimateCheck 2026：https://arxiv.org/abs/2603.26449
- ClimateCheck2025 参赛系统（Kiepura & Lam）：https://aclanthology.org/2025.sdp-1.28
- How Robust Are Automated Fact-Checking Systems?：https://arxiv.org/abs/2608.25934
- HealthFC：https://aclanthology.org/2024.lrec-main.709
- AVeriTeC 2024：https://aclanthology.org/2024.fever-1.1
- Climinator：https://www.zora.uzh.ch/entities/publication/b9be4b4f-b90e-42f9-959c-8b35c4df7106 （npj Climate Action 4, 17, 2025）
- Climate-Eval：https://arxiv.org/abs/2505.18653
- CLIMATE-FEVER：https://www.climatechange.ai/papers/neurips2020/67
- Scientific Fact-Checking survey：https://aclanthology.org/2023.findings-acl.387
- Claim Verification in the Age of LLMs：https://arxiv.org/abs/2408.14317
- Generative LLMs in Automated Fact-Checking：https://arxiv.org/abs/2407.02351
- UPS：https://arxiv.org/abs/2101.06329
- USB：https://arxiv.org/abs/2208.07204
- JointMatch：https://aclanthology.org/2023.emnlp-main.451
- MultiMatch：https://aclanthology.org/2025.emnlp-main.139
- SSL for robust NLI：https://arxiv.org/abs/2211.02971
- Folino & Pontieri 2026：https://www.sciencedirect.com/science/article/pii/S2590005626000287
- VerifyMatch：https://aclanthology.org/2024.emnlp-main.1076
- LG-CoTrain：https://arxiv.org/abs/2509.16516
- LLKD：https://aclanthology.org/2025.findings-naacl.142/
- CanDist：https://aclanthology.org/2025.acl-long.139
- Co-training for prompt-based learning：https://arxiv.org/abs/2202.00828
- WeFEND：https://ojs.aaai.org/index.php/AAAI/article/view/5389
- Selective Pseudo-Labeling with RL：https://arxiv.org/abs/2012.03438
- Learning to Clean：https://arxiv.org/abs/2511.19808
- RL-Selector：https://arxiv.org/abs/2506.21037
- DecomposeRL：https://arxiv.org/abs/2605.27858
- MISSCI：https://aclanthology.org/2024.acl-long.240
- MissciPlus：https://arxiv.org/abs/2408.12812
- Zero-NatVer：https://aclanthology.org/2024.findings-emnlp.991/
- MiniCheck：https://arxiv.org/abs/2404.10774
- Health synthetic data：https://arxiv.org/abs/2508.20525
- Oliver et al. 2018：https://arxiv.org/abs/1804.09170
- Shi et al. 2023：https://aclanthology.org/2023.findings-acl.347
- Schuster et al. 2019：https://aclanthology.org/D19-1341
