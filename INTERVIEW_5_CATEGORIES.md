# vLLM Semantic Router — 五类技术面试应对指南

> **定位**：针对 LLM 算法实习面试的五类核心能力，结合 semantic-router 项目实际代码与设计，提供可直接使用的回答框架与深挖准备。
> **使用方式**：每一类问题先给出「回答框架」，再给出「项目实例」，最后是「面试官可能的追问与应对」。

---

## 目录

1. [底层原理深入理解](#1-底层原理深入理解)
2. [实验与方案验证能力](#2-实验与方案验证能力)
3. [问题定位能力](#3-问题定位能力)
4. [工程落地能力](#4-工程落地能力)
5. [业务与场景理解](#5-业务与场景理解)

---

## 1. 底层原理深入理解

> **面试官考察点**：不是让你背概念，而是看你能不能讲清楚——这个方法解决什么问题、存在哪些局限、有哪些改进方法。

### 回答框架：Problem → Design → Trade-off → Improvement

每个技术点都按这个四步框架组织回答：

```
1. Problem：这个方法要解决什么问题？之前的方案有什么不足？
2. Design：具体怎么做的？为什么这么设计？
3. Trade-off：这个设计的局限性是什么？牺牲了什么？
4. Improvement：如果要改进，你会怎么做？
```

---

### 1.1 Elo Rating 系统

**Problem（解决什么问题）**：

LLM 模型数量爆炸，无法对每个查询都跑 benchmark 来选模型。需要一种 **在线、低开销** 的方式持续评估模型质量。传统做法是离线 benchmark（如 MMLU），但 benchmark 不能反映真实分布，且更新慢。

**Design（怎么做的）**：

采用 Bradley-Terry 配对比较模型：
```
P(m_i ≻ m_j) = 1 / (1 + 10^((R_j - R_i)/400))
```
每次用户反馈（显式评分或隐式信号如 re-ask）触发一次 Elo 更新：
```
R_i' = R_i + K × (S_actual - S_expected)
```
- K=32（标准国际象棋参数），初始分 1500
- 支持按类别独立维护 Elo（数学 vs 文学的模型能力排序不同）
- 成本感知：在质量相近时偏好便宜模型（`CostScalingFactor`）

**为什么这么设计（而非其他方案）**：
- 相比离线 benchmark：Elo 是在线持续学习，能适应模型更新和分布漂移
- 相比 RL：Elo 不需要定义复杂的奖励函数，收敛快，可解释
- 相比 Elo + 上下文特征：纯 Elo 无上下文感知，但开销极低（O(1) 查表）

**Trade-off（局限性）**：

| 局限 | 具体问题 | 项目中的缓解 |
|------|----------|--------------|
| 无上下文感知 | 同一模型在数学和编程上的 Elo 是同一个数 | `CategoryWeighted`：按类别维护独立 Elo |
| 冷启动 | 新模型没有历史数据，初始 1500 分不合理 | `MinComparisons=5`：少于 5 次比较时标记为"不稳定" |
| 评分稀疏 | 大部分模型对从未被比较过 | 成本感知衰减 + 全局默认评分 |
| 假设静态 | 假设模型能力不随时间变化 | 持续更新 + K 因子衰减 |
| 只看结果 | 不理解为什么一个模型更好 | 与信号系统结合：Elo 只做排序，信号解释原因 |

**Improvement（改进方向）**：
1. **上下文 Elo**：引入查询特征作为上下文，变成 Contextual Elo（类似 TrueSkill 2）
2. **贝叶斯 Elo**：用后验分布替代点估计，量化不确定性（类似 Glicko-2）
3. **多臂老虎机替代**：Elo 本质是离线评估，Thompson Sampling 是在线探索-利用权衡

**代码位置**：`src/semantic-router/pkg/selection/elo.go:98`

---

### 1.2 双对比学习 RouterDC

**Problem（解决什么问题）**：

Elo 只能做全局排序，不能根据查询内容动态选择最优模型。需要一种方法能 **根据查询语义** 直接预测最佳模型。

**Design（怎么做的）**：

双编码器架构：
```
查询编码器：q → e_q ∈ R^d
模型编码器：m_k → e_m_k ∈ R^d

选择：m* = argmax_k cos(e_q, e_m_k)

训练损失（InfoNCE）：
L = -log[exp(cos(e_q, e_m+)/τ) / Σ_k exp(cos(e_q, e_m_k)/τ)]
```

**为什么用双编码器而不是单编码器**：
- 查询和模型是不同类型的实体，共享参数空间会互相干扰
- 双编码器允许独立更新：模型描述变了只需重新编码模型，不影响查询编码器
- 推理时可以预计算所有模型嵌入，查询只需编码一次

**为什么用 InfoNCE 而不是 Triplet Loss**：
- Triplet Loss 只推远一个负样本，InfoNCE 同时推远所有负样本
- InfoNCE 最大化互信息下界，有更强的理论保证
- 实践中 InfoNCE 收敛更快更稳定

**Trade-off（局限性）**：

| 局限 | 具体问题 | 代码中的证据 |
|------|----------|--------------|
| 温度敏感 | τ=0.07 来自论文，但不同任务最优 τ 不同 | `Temperature` 硬编码默认值 |
| 维度假设 | `DimensionSize=768` 假设 BERT 系模型 | 硬编码，换模型需改代码 |
| 亲和矩阵膨胀 | `map[string]map[string]float64` 无驱逐策略 | 可能 OOM |
| 质量近似 | 假设模型描述能代表模型能力 | 描述质量直接影响效果 |
| 无在线学习 | 嵌入在初始化时计算，不随反馈更新 | `UpdateFeedback` 只更新亲和矩阵 |

**Improvement（改进方向）**：
1. **自适应温度**：让 τ 随训练过程自动调整（类似 CLIP 的 learnable temperature）
2. **在线对比学习**：用用户反馈作为新的正负样本对，持续更新嵌入
3. **多粒度嵌入**：不同类型的查询用不同维度（Matryoshka 嵌入）
4. **负例采样**：当前用所有模型做负例，可以引入 hard negative mining

**代码位置**：`src/semantic-router/pkg/selection/router_dc.go:79`

---

### 1.3 POMDP 级联路由 AutoMix

**Problem（解决什么问题）**：

不是所有查询都需要最强模型。很多简单查询用小模型就能解决，但事先不知道哪些是简单的。需要一种 **逐步升级** 的策略：先用小模型尝试，不够好再升级。

**Design（怎么做的）**：

POMDP 框架建模：
- **状态**：查询的真实难度 + 模型质量分布
- **观测**：模型输出（不完美，无法直接判断质量）
- **动作**：接受当前输出 / 升级到更大模型
- **信念更新**：粒子滤波维护后验分布

```
期望成本优化：
E[C] = Σ_k C_k × Π_{j<k} (1 - P(q̂_j ≥ τ_j))

自验证：模型输出置信度分数
阈值 τ_j：决定何时升级
MaxEscalations=2：防止无限升级
```

**为什么是 POMDP 而不是 MDP**：
- MDP 假设状态完全可观测，但实际上我们不知道"真实难度"
- 只能通过模型输出间接推断（观测不完美）
- POMDP 用信念分布（belief state）代替确定状态

**Trade-off（局限性）**：

| 局限 | 具体问题 |
|------|----------|
| 延迟叠加 | 级联每级都增加延迟，最坏情况 = 所有模型延迟之和 |
| 自验证不可靠 | 模型的"自信"不等于"正确"，过度自信的小模型会漏检 |
| 粒子退化 | 粒子滤波在高维空间容易退化 |
| 阈值敏感 | τ 的选择直接影响成本/质量权衡 |
| 需要 LLM 端点 | `EnableSelfVerification` 默认关闭，因为需要额外 LLM 调用 |

**Improvement（改进方向）**：
1. **并行级联**：同时请求多个模型，用信号快速判断哪个输出更好
2. **元学习阈值**：用历史数据自动学习最优 τ
3. **提前退出**：在模型内部层级做早退（类似 Matryoshka），而非模型间级联

**代码位置**：`src/semantic-router/pkg/selection/automix.go`

---

### 1.4 LoRA 低秩微调

**Problem（解决什么问题）**：

Semantic Router 需要 11 个分类器（domain、jailbreak、PII、hallucination 等），如果每个都用完整模型，内存 = 11 × 573MB ≈ 6.3GB。LoRA 让内存降到 573MB + 11 × 0.1MB ≈ 574MB。

**Design（怎么做的）**：

```
LoRA 核心公式：
W' = W + ΔW = W + BA
其中 B ∈ R^{d×r}, A ∈ R^{r×d}, r << d

项目配置：
- rank (r): 32（表达能力和参数量的平衡）
- alpha (α): 64（缩放因子 α/r = 2）
- dropout: 0.1（防过拟合）
- 目标模块：attention.query/value + output.dense + intermediate.dense

为什么选这些目标模块？
- query/value 投影影响注意力分布，是任务相关性最强的部分
- 不动 key（因为 key 和 query 耦合，改 key 需要同时改 query）
- FFN 的 intermediate/output 也加入，增加表达能力
```

**Trade-off（局限性）**：

| 局限 | 具体问题 | 项目中的体现 |
|------|----------|--------------|
| 秩选择困难 | r 太小学不到，r 太大过拟合 | 用 r=32 经验值，未做系统搜索 |
| 任务干扰 | 不同任务的 LoRA 可能学到冲突的方向 | 用独立 LoRA 而非共享，但增加切换开销 |
| 基座依赖 | LoRA 的效果高度依赖基座模型选择 | 选 mmBERT 是因为双向注意力适合分类 |
| 合并开销 | 推理时需要 W + BA，增加计算 | Rust Candle 框架优化了 LoRA 热加载 |

**改进方向**：
1. **秩自适应**：不同层用不同秩（AdaLoRA），重要层用高秩
2. **LoRA 合并**：多个 LoRA 合并为一个（LoRAHub），减少切换开销
3. **QLoRA**：量化基座 + LoRA，进一步减少内存
4. **DoRA**：分解为方向和幅度，比 LoRA 更高效

**代码位置**：`src/training/model_classifier/classifier_model_fine_tuning_lora/ft_linear_lora.py`

---

### 1.5 嵌入训练与硬负例挖掘

**Problem（解决什么问题）**：

通用嵌入模型（如 all-MiniLM）在特定领域（医疗、法律）的语义区分能力不足。例如 "心肌梗死" 和 "心绞痛" 在通用空间中可能很近，但在医疗语义中需要区分。

**Design（怎么做的）**：

迭代硬负例挖掘（2 轮）：
```
第 1 轮：
1. 用预训练模型对所有查询编码
2. 检索 Top-K 最相似的文档
3. 排除 Ground Truth，剩下的就是"硬负例"（模型认为相关但实际不相关）
4. 构建三元组：(query, positive, hard_negative)
5. 训练：TripletLoss(margin=0.1)

第 2 轮：
1. 用第 1 轮训练后的模型重新编码
2. 重新挖掘硬负例（现在更难了，因为模型更强了）
3. 混合简单负例（easy:hard = 2:1，防遗忘）
4. 继续训练

关键超参数（来自实验）：
- margin=0.1（SentenceTransformers 默认 5.0 效果很差！）
- top_k=100（检索范围）
- hard_neg_rank=15（取第 15 名作为硬负例）
- easy_to_hard_ratio=2
```

**为什么 margin=0.1 而不是默认 5.0**：
- 余弦距离范围是 [0, 2]，margin=5.0 意味着永远无法满足约束
- margin=0.1 给了一个很小但有效的间隔
- 这个经验值来自实验调优

**Trade-off（局限性）**：

| 局限 | 具体问题 |
|------|----------|
| 挖掘成本 | 每轮需要重新编码所有查询和文档，O(N²) |
| 噪声负例 | 排名靠前的"负例"可能确实是相关的（标注不全） |
| 分布偏移 | 训练数据和真实查询分布可能不同 |
| 阈值选择 | `hard_neg_rank=15` 是经验值，不同数据集最优值不同 |

**改进方向**：
1. **在线挖掘**：在训练过程中动态更新负例（而非离线预计算）
2. **课程学习**：从简单负例开始，逐渐增加难度
3. **对比学习**：用 InfoNCE 替代 Triplet Loss，更高效利用负例
4. **合成负例**：用 LLM 生成语义相近但不同的查询

**代码位置**：`src/training/model_embeddings/domain_adapted_embeddings/train.py`

---

### 1.6 HaluGate 幻觉检测

**Problem（解决什么问题）**：

LLM 会生成看似合理但事实错误的内容（幻觉）。全量检测开销太大（每个响应都过 NLI 模型），需要一种高效的方法跳过不需要检测的查询。

**Design（怎么做的）**：

三阶段门控管线：
```
Stage 1: Sentinel（门控）
- 判断查询是否需要事实核查
- 跳过创意写作、意见表达等非事实性查询
- 约 40-60% 的工作量在这里被跳过

Stage 2: Detector（跨度识别）
- 对需要核查的查询，定位响应中的可疑跨度
- Token 级精度（而非句子级）
- 输出：{(i, j, confidence) | a_i...a_j 不被上下文支持}

Stage 3: Explainer（NLI 分类）
- 对每个可疑跨度做 NLI 三分类
- ENTAILMENT（支持）/ CONTRADICTION（矛盾）/ NEUTRAL（不支持）
- 区分"矛盾"和"不支持"很重要：不支持 ≠ 错误
```

**为什么分三阶段而不是一步到位**：
- Sentinel 做粗粒度过滤，开销极低（规则/小分类器）
- Detector 缩小检测范围，避免对整个响应做 NLI
- Explainer 只对可疑跨度做精细判断
- 总开销 = C_sent + p_fact × (C_det + k̄ × C_nli)，远小于全量 NLI

**Trade-off**：

| 局限 | 具体问题 |
|------|----------|
| 门控漏检 | Sentinel 可能漏掉需要核查的事实性查询 |
| 跨度边界 | Detector 的边界可能不准确（多切/少切） |
| NLI 局限 | NLI 模型本身可能有错误，尤其是领域外文本 |
| 上下文依赖 | 如果上下文本身有错误，NLI 会误判 |
| 延迟增加 | 三阶段串行，增加了响应延迟 |

**改进方向**：
1. **并行检测**：Sentinel 和 Detector 并行运行，用结果决定是否继续
2. **置信度校准**：用温度缩放校准 NLI 的置信度
3. **检索增强**：用 RAG 检索外部知识库作为上下文
4. **反馈闭环**：用用户反馈持续改进检测器

---

## 2. 实验与方案验证能力

> **面试官考察点**：不仅看你做了什么，更看你怎么证明它是有效的。追问实验细节能看出你是否真正深入理解。

### 回答框架：Hypothesis → Experiment Design → Metrics → Results → Analysis

```
1. Hypothesis：你的假设是什么？为什么认为这个方案有效？
2. Experiment Design：怎么设计实验？对照组是什么？
3. Metrics：用什么指标衡量？为什么选这些指标？
4. Results：实验结果是什么？达到预期了吗？
5. Analysis：结果说明了什么？有没有意外发现？
```

---

### 2.1 分类器评估实验

**假设**：LoRA 微调的 mmBERT 能在 14 类 MMLU-Pro 分类任务上达到 >95% 准确率，同时只训练 0.02% 参数。

**实验设计**：
```
数据集：TIGER-Lab/MMLU-Pro（测试集用于训练，项目约定）
补充数据：LLM-Semantic-Router/category-classifier-supplement
评估指标：Accuracy + Weighted F1
基线对比：
  - 全量微调 vs LoRA 微调
  - 不同 rank (16/32/64) 的效果
  - 不同基座模型 (BERT/RoBERTa/ModernBERT/mmBERT)
训练配置：
  - AdamW + Cosine LR + Warmup 0.06
  - Gradient clipping max_norm=1.0
  - Weight decay 0.1
  - Eval per epoch, load best by eval_f1
```

**结果（来自代码和论文）**：
- mmBERT-base: accuracy 0.99+, F1 0.99+
- 参数量减少 99%，内存减少 67%
- LoRA rank=32 是效果/参数量的最佳平衡点

**面试官可能的追问**：

**Q: 为什么用 F1 而不是 Accuracy？**
> "类别可能不平衡（某些 MMLU 类别样本多），Accuracy 会被多数类主导。Weighted F1 考虑了每个类别的贡献。实际上两者差距不大，因为 MMLU-Pro 的类别相对均衡。"

**Q: 如何确定 rank=32 是最优的？**
> "做了 rank 16/32/64 的消融实验。rank=16 效果略低（~0.5%），rank=64 效果持平但参数翻倍。选择 32 是效果/效率的平衡点。实际上可以做更细粒度的搜索（8/16/24/32/48/64），但训练成本高。"

**Q: 评估集和训练集用的是同一个？这不是数据泄露吗？**
> "是的，这是项目的一个设计选择。MMLU-Pro 的惯例是 test split 用于训练（因为需要大量标注数据）。评估时用 hold-out 的补充数据集。如果面试官追问，承认这是一个局限，实际应该用独立的验证集。"

---

### 2.2 嵌入域适应实验

**假设**：通过硬负例挖掘和迭代训练，域适应嵌入在特定领域（医疗）的检索性能可以显著提升。

**实验设计**：
```
基座模型：llm-semantic-router/mmbert-embed-32k-2d-matryoshka
数据集：MedQuAD（医疗问答数据集）
评估指标：MRR@5, Recall@5
对照组：
  1. 基座模型（无微调）
  2. 单轮硬负例挖掘
  3. 两轮迭代硬负例挖掘
训练配置：
  - TripletLoss, margin=0.1
  - lr=5e-5, batch=8, epochs=2/轮
  - easy:hard = 2:1
```

**结果**：
```
MRR@5:  +71.18%（从基座到两轮迭代）
Recall@5: +40.92%
```

**面试官可能的追问**：

**Q: 为什么 margin=0.1 比默认 5.0 好这么多？**
> "余弦距离的范围是 [0, 2]，margin=5.0 意味着要求 positive 和 negative 的距离差大于 5.0，这在余弦空间中永远无法满足。模型会学到一个退化解——所有嵌入都相同（距离=0），因为这是唯一能满足 margin 约束的方式。margin=0.1 给了一个合理的小间隔。"

**Q: 为什么选 hard_neg_rank=15 而不是 Top-1？**
> "Top-1 硬负例太难了，模型可能学不到有用的信号（梯度爆炸）。第 15 名是一个'适度困难'的负例——模型认为比较相似但实际不相关。这类似于 curriculum learning 的思想：先学简单的，再学难的。"

**Q: easy:hard=2:1 这个比例怎么确定的？**
> "做了消融实验。纯硬负例容易导致灾难性遗忘（模型忘记通用语义）。混合简单负例起到正则化效果。2:1 是经验值，实际可以做更细粒度的搜索（1:1, 2:1, 3:1, 5:1）。"

---

### 2.3 模型选择算法对比实验

**假设**：不同模型选择算法在不同场景下各有优劣，没有单一最优算法。

**实验设计**：
```
数据集：vllm-project/semantic-router-benchmark（HuggingFace）
评估指标：准确率、延迟、成本
对比算法：KNN, KMeans, SVM, MLP, Elo, RouterDC, AutoMix
特征：1038 维 = 1024 Qwen3 embedding + 14 category one-hot
```

**结果分析框架**：
```
| 算法 | 准确率 | 延迟 | 成本 | 适用场景 |
|------|--------|------|------|----------|
| KNN | 高 | 低 | - | 有充足历史数据 |
| SVM | 高 | 低 | - | 特征空间有明确边界 |
| MLP | 最高 | 中 | - | 数据量大，非线性边界 |
| Elo | 中 | 最低 | 低 | 在线持续评估 |
| RouterDC | 高 | 中 | 中 | 有模型描述 |
| AutoMix | 高 | 高 | 低 | 成本敏感场景 |
```

**面试官可能的追问**：

**Q: 你的训练数据是怎么构建的？有什么问题？**
> "用 benchmark.py 对多个 LLM 并发评测，收集每个查询在每个模型上的准确率和延迟。问题：1) benchmark 数据是离线的，不代表真实分布；2) 评测指标（如选择题准确率）不完全反映用户体验；3) 模型更新后数据过时。"

**Q: 如何处理评估指标和真实业务指标的 gap？**
> "这是最大的挑战。benchmark 准确率 ≠ 用户满意度。缓解方法：1) 用 Elo 在线学习补充离线 benchmark；2) 用隐式信号（re-ask、session 长度）近似满意度；3) A/B 测试验证线上效果。"

---

### 2.4 性能基准测试

**假设**：分类器延迟可以控制在 10ms 以内（单请求），决策引擎 <1ms。

**实验设计**：
```
SLO 阈值（来自 perf/config/thresholds.yaml）：
- 分类 batch=1: p95 ≤ 10ms, p99 ≤ 15ms, throughput ≥ 100 qps
- 决策引擎: p95 ≤ 1ms, throughput ≥ 10000 qps
- 缓存: p95 ≤ 5ms (1K), p95 ≤ 10ms (10K)
- E2E: sustained ≥ 500 qps, p95 ≤ 100ms

测试配置（来自 perf/config/perf.yaml）：
- 分类: batch [1,10,50,100], 1000 iterations, 100 warmup
- 缓存: sizes [1000,10000], concurrency [1,10,50]
- E2E: constant 50 qps/60s, ramp 10→100/120s, burst 200/30s
```

**回归检测**：
```
基线存储：JSON 文件记录 ns/op, p50/p95/p99, throughput, allocs/op
回归阈值：每个组件有独立的回归容忍度
CompareWithBaseline：计算百分比变化，超过阈值标记为 REGRESSION
报告：JSON/Markdown/HTML 三种格式
```

**面试官可能的追问**：

**Q: 如何区分噪声和真实回归？**
> "多次运行取中位数，设置合理的回归阈值（如 5%）。如果连续两次运行都超过阈值，判定为真实回归。用统计检验（如 Mann-Whitney U）也可以，但实践中简单的百分比阈值更实用。"

---

## 3. 问题定位能力

> **面试官考察点**：模型上线后能力下降、系统突然变慢、实验结果和预期不一致——这些问题怎么排查？

### 回答框架：Observe → Hypothesize → Isolate → Verify → Fix

```
1. Observe：先收集信息（日志、指标、追踪）
2. Hypothesize：列出可能的原因
3. Isolate：用排除法缩小范围
4. Verify：验证假设
5. Fix：修复并验证
```

---

### 3.1 场景：分类器准确率突然下降

**Observe（观察）**：
```
监控指标：
- 分类准确率从 99% 降到 85%
- 特定类别（如 "health"）下降最严重
- 延迟没有明显变化
- 模型版本没有更新
```

**Hypothesize（假设）**：
```
1. 数据分布漂移：用户查询模式变了
2. 训练数据问题：补充数据集有噪声
3. 模型退化：LoRA 适配器损坏
4. 配置变更：阈值被修改
5. 上游变化：嵌入模型更新了
```

**Isolate（隔离）**：
```python
# 步骤 1：检查配置变更
git log --oneline config/config.yaml  # 最近有无修改

# 步骤 2：检查模型版本
ls -la models/  # 模型文件有无变化

# 步骤 3：回放历史请求
# Router Replay 插件记录了所有请求/响应
curl /v1/router_replay?category=health  # 回放健康类请求

# 步骤 4：单信号测试
# 逐个关闭信号，看哪个信号导致问题
```

**Verify（验证）**：
```
发现：嵌入模型被更新（从 mmBERT-base 换成了 mmBERT-32k）
- mmBERT-32k 的嵌入空间和 mmBERT-base 不兼容
- 导致嵌入相似度计算全部偏移
- 分类器的阈值是针对旧模型调的

验证方法：
1. 用旧模型跑同一批请求 → 准确率恢复
2. 用新模型重新调阈值 → 准确率恢复
```

**Fix（修复）**：
```
短期：回滚到旧嵌入模型
长期：
1. 模型更新时必须重新校准阈值
2. 添加嵌入空间兼容性检查
3. 监控嵌入向量的分布统计（均值、方差）
```

---

### 3.2 场景：系统突然变慢

**Observe（观察）**：
```
监控指标：
- E2E p95 从 100ms 飙升到 2s
- CPU 使用率正常（30%）
- 内存使用率正常（1GB）
- Goroutine 数量正常（500）
- 分类器延迟正常（10ms）
- 但某几个请求延迟极高（>5s）
```

**Hypothesize（假设）**：
```
1. 后端模型超时：某个 LLM 后端响应慢
2. 嵌入计算阻塞：GPU 被其他任务占用
3. 缓存失效：所有请求都穿透到后端
4. 锁竞争：某个 mutex 被长时间持有
5. 网络问题：与后端的网络延迟增加
```

**Isolate（隔离）**：
```
步骤 1：看请求追踪（Envoy tracing）
- 发现延迟集中在"模型选择"阶段

步骤 2：看模型选择器日志
- RouterDC 的嵌入函数调用耗时 4s

步骤 3：看嵌入模型状态
- GPU 被另一个训练任务占满
- 嵌入计算 fallback 到 CPU
- CPU 上的 ONNX Runtime 比 GPU 慢 40 倍
```

**Verify（验证）**：
```
复现：在 GPU 被占用时发送请求 → 延迟复现
对比：释放 GPU 后发送请求 → 延迟恢复
```

**Fix（修复）**：
```
短期：限制训练任务的 GPU 使用（CUDA_VISIBLE_DEVICES）
长期：
1. 嵌入计算设置超时（500ms），超时后用缓存的嵌入
2. 分离训练和推理的 GPU
3. 添加 CPU fallback 的延迟告警
4. 实现嵌入结果缓存（相同查询不重复计算）
```

---

### 3.3 场景：缓存命中率突然下降

**Observe（观察）**：
```
监控指标：
- 缓存命中率从 80% 降到 20%
- 缓存大小正常（5000 条）
- 嵌入模型正常
- 查询分布没有明显变化
```

**Hypothesize（假设）**：
```
1. TTL 过期：缓存条目批量过期
2. 驱逐策略问题：LRU 频繁驱逐热数据
3. 相似度阈值变化：阈值被调高
4. 嵌入漂移：嵌入模型更新导致旧缓存的嵌入不匹配
5. 并发竞争：高并发下缓存写入失败
```

**Isolate（隔离）**：
```
步骤 1：检查缓存配置
- SimilarityThreshold 从 0.92 改成了 0.98
- 0.98 太严格，只有几乎完全相同的查询才能命中

步骤 2：检查配置变更记录
- 某次配置更新误改了阈值
```

**Fix（修复）**：
```
短期：恢复阈值到 0.92
长期：
1. 阈值变更需要灰度发布（先 10% 流量验证）
2. 添加阈值变更的告警
3. 监控不同阈值下的命中率分布
```

---

### 3.4 场景：LoRA 微调后分类器在某些类别上退化

**Observe（观察）**：
```
训练日志：
- 训练 loss 正常下降
- 验证 F1 0.98（整体）
- 但 "philosophy" 类别的 recall 从 0.95 降到 0.70
```

**Hypothesize（假设）**：
```
1. 数据不平衡：philosophy 样本太少
2. 硬负例污染：philosophy 的负例质量差
3. LoRA 灾难性遗忘：微调覆盖了通用能力
4. 评估集问题：评估集不代表真实分布
```

**Isolate（隔离）**：
```
步骤 1：检查数据分布
- philosophy 样本仅占 3%（最少的类别）
- 大量 philosophy 样本被标记为 "other"

步骤 2：检查混淆矩阵
- philosophy 经常被误分为 "psychology" 和 "history"
- 这些类别语义相近

步骤 3：检查 LoRA 参数
- 发现 LoRA 的 B 矩阵在 philosophy 相关的 attention head 上范数很小
- 说明 LoRA 没有学到 philosophy 的判别特征
```

**Fix（修复）**：
```
短期：对 philosophy 做过采样（SMOTE 或复制）
长期：
1. 数据增强：用 LLM 生成 philosophy 类别的补充样本
2. 类别加权损失：对少数类给更高的 loss 权重
3. 分层 LoRA：对少数类用更高的 rank
4. 混淆类别分离：对 philosophy/psychology/history 增加对比学习
```

---

### 3.5 问题定位的通用方法论

**面试回答模板**：

> "当系统出现问题时，我的排查思路是：
> 1. **先看监控**：确定问题的范围和时间点。是全局还是局部？是突然还是渐进？
> 2. **看变更记录**：最近有没有配置更新、模型更新、代码发布？大部分线上问题都和变更有关。
> 3. **缩小范围**：用排除法。关闭某个信号/插件，看问题是否消失。用二分法定位。
> 4. **复现问题**：用 router replay 回放问题请求，确认是否可复现。
> 5. **根因分析**：找到根因后，不仅修复当前问题，还要加监控防止复发。"

---

## 4. 工程落地能力

> **面试官考察点**：理论可行的方案实际工程落地中不可行，关键在理论结合实际。

### 回答框架：Theory → Reality Gap → Engineering Solution → Trade-off

```
1. Theory：理论上这个方案应该 work
2. Reality Gap：实际落地遇到了什么问题
3. Engineering Solution：怎么解决的
4. Trade-off：为了落地牺牲了什么
```

---

### 4.1 算法部署：从 Python 训练到 Rust 推理

**Theory**：
训练用 PyTorch + HuggingFace，推理也应该用 Python。

**Reality Gap**：
```
Python 推理的问题：
- 延迟：BERT 分类 Python 推理 ~200ms，要求 <100ms
- 内存：Python 进程 + PyTorch 运行时 > 2GB
- 并发：GIL 限制，无法充分利用多核
- 部署：需要 Python 环境，容器镜像 > 1GB
```

**Engineering Solution**：

```
方案：Rust FFI 绑定

训练管线（Python）：
  PyTorch + PEFT → 训练 LoRA 适配器
  → 导出合并后的模型（safetensors 格式）
  → 上传 HuggingFace

推理引擎（Rust）：
  Candle 框架加载 safetensors
  → CGo FFI 接口供 Go 调用
  → 支持 CUDA/ROCm/OpenVINO/MKL

Go 路由器：
  → 通过 CGo 调用 Rust 推理
  → 零拷贝内存共享
```

**具体代码流程**：
```python
# Python 训练侧
trainer.train()
model = merge_peft_model(model, lora_adapter)  # 合并 LoRA
model.save_pretrained(output_dir)  # 保存为 safetensors
```

```rust
// Rust 推理侧
let vb = unsafe::VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)?;
let model = BertForSequenceClassification::load(vb, config)?;
let logits = model.forward(&input_ids, &attention_mask)?;
```

```go
// Go 调用侧
func Classify(text string) (string, float64, error) {
    result := C.classify(C.CString(text))
    return C.GoString(result.label), float64(result.confidence), nil
}
```

**Trade-off**：
```
牺牲了什么：
1. 开发效率：Rust 开发比 Python 慢 3-5 倍
2. 调试难度：FFI 边界的 bug 难以定位
3. 模型灵活性：不是所有 PyTorch 操作都能在 Candle 中复现
4. 编译时间：Rust 编译 + CGo 链接 > 5 分钟

获得了什么：
1. 延迟：从 200ms 降到 15ms（13 倍提升）
2. 内存：从 2GB 降到 200MB（10 倍减少）
3. 并发：无 GIL，充分利用多核
4. 部署：单一二进制，无需 Python 环境
```

---

### 4.2 模型热加载与零停机更新

**Theory**：更新模型时直接替换文件即可。

**Reality Gap**：
```
问题：
1. 模型加载需要 2-5 秒，期间请求会失败
2. 如果新模型有 bug，回滚也需要 2-5 秒
3. 多个 LoRA 适配器需要同时更新
4. 不能中断正在处理的请求
```

**Engineering Solution**：

```go
// 配置热更新机制
// 1. 监听配置文件变化（fsnotify）
// 2. 新配置加载到内存
// 3. 原子替换指针（atomic.Value）
// 4. 旧配置等待所有在途请求完成后释放

func (s *Server) ReplaceConfig(newConfig *RouterConfig) {
    s.configMu.Lock()
    oldConfig := s.config
    s.config = newConfig
    s.configMu.Unlock()
    
    // 通知所有订阅者
    for _, ch := range s.configSubscribers {
        ch <- newConfig
    }
    
    // 旧配置延迟释放
    go func() {
        time.Sleep(30 * time.Second)
        oldConfig.Close()
    }()
}
```

```
LoRA 热加载：
1. Candle 支持 LoRA adapter 热加载（不重启进程）
2. 先加载新 adapter 到临时内存
3. 原子切换 adapter 指针
4. 旧 adapter 延迟释放
```

**Trade-off**：
```
牺牲了什么：
1. 内存峰值：新旧模型同时存在 30 秒
2. 复杂性：原子操作 + 延迟释放 + 订阅者通知
3. 一致性窗口：切换瞬间可能有请求用旧模型

获得了什么：
1. 零停机更新
2. 快速回滚（只需切换指针）
3. 灰度发布支持（按流量百分比切换）
```

---

### 4.3 监控与可观测性

**Theory**：加日志就够了。

**Reality Gap**：
```
纯日志的问题：
1. 日志量太大，无法人工分析
2. 无法实时告警
3. 无法关联请求链路
4. 无法量化性能指标
```

**Engineering Solution**：

```
三层可观测性：

1. Metrics（指标）：
   - Prometheus 格式
   - 分类器延迟 histogram（p50/p95/p99）
   - 缓存命中率 counter
   - 模型选择分布 counter（按算法/模型）
   - 内存使用 gauge

2. Tracing（追踪）：
   - Envoy 分布式追踪（OpenTelemetry）
   - 每个请求的完整链路：
     请求解析 → 信号提取 → 决策评估 → 模型选择 → 插件应用 → 后端路由
   - 自定义 header 传递追踪 ID：
     x-vsr-hallucination-detected
     x-vsr-hallucination-spans
     x-vsr-fact-check-needed

3. Logging（日志）：
   - 结构化日志（JSON 格式）
   - 请求级别日志（request_id 关联）
   - 错误日志包含完整上下文
   - Router Replay 插件记录完整请求/响应
```

**Router Replay（调试利器）**：
```yaml
# 配置
plugins:
  router_replay:
    enabled: true
    max_body_bytes: 10240
    max_tool_trace_bytes: 5120
    max_tool_trace_steps: 10

# 使用
# 回放特定类别的请求
curl /v1/router_replay?category=health&limit=100
# 分析路由决策
curl /v1/router_replay/{request_id}/analysis
```

---

### 4.4 配置验证与灰度发布

**Theory**：直接修改配置文件即可。

**Reality Gap**：
```
配置错误的风险：
1. 布尔表达式写错 → 所有请求路由到错误模型
2. 阈值设错 → 缓存命中率骤降
3. 信号类型拼写 → 静默失败（不报错但不工作）
4. 模型名拼写 → 请求 404
```

**Engineering Solution**：

```
配置验证管线：
1. 语法检查：YAML 解析
2. 类型检查：Schema 验证（canonical v0.3）
3. 语义检查：
   - 信号类型是否在已注册列表中
   - 模型名是否在 providers 中定义
   - 阈值是否在合理范围 [0, 1]
   - 布尔表达式树是否有环
4. 运行时检查：
   - 嵌入模型是否能加载
   - 后端是否可达
   - LoRA 适配器是否存在

灰度发布：
1. 新配置先应用到 10% 流量
2. 监控关键指标（延迟、准确率、错误率）
3. 指标正常 → 逐步增加到 100%
4. 指标异常 → 自动回滚
```

---

### 4.5 并发安全与资源管理

**Theory**：Go 天然支持并发，不需要特别处理。

**Reality Gap**：
```
并发问题：
1. 竞态条件：多个 goroutine 同时读写 Elo 评分
2. 内存泄漏：亲和矩阵 map 无限增长
3. Goroutine 泄漏：后台清理 goroutine 未正确停止
4. 死锁：嵌套锁的获取顺序不一致
```

**Engineering Solution**：

```go
// 1. 细粒度锁（Elo 选择器）
type EloSelector struct {
    globalMu      sync.RWMutex  // 全局评分锁
    categoryMu    sync.RWMutex  // 类别评分锁（独立）
    globalRatings map[string]*ModelRating
    categoryRatings map[string]map[string]*ModelRating
}

// 2. 原子操作（配置替换）
var config atomic.Value  // Store(*RouterConfig)

// 3. 资源清理（缓存）
type InMemoryCache struct {
    closeOnce sync.Once  // 确保只清理一次
    closeCh   chan struct{}
}

func (c *InMemoryCache) Close() error {
    c.closeOnce.Do(func() {
        close(c.closeCh)
    })
    return nil
}

// 4. 超时控制
ctx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
defer cancel()
result, err := selector.Select(ctx, selCtx)
```

---

## 5. 业务与场景理解

> **面试官考察点**：一个项目真正需要产生的是有用的场景价值和业务价值。

### 回答框架：Scene → User Need → Value → Cost → Priority

```
1. Scene：这个方案适合什么场景？
2. User Need：用户真正关心什么？
3. Value：带来什么业务价值？
4. Cost：上线成本有多高？
5. Priority：资源有限时先优化什么？
```

---

### 5.1 场景适配分析

**Semantic Router 适合的场景**：

| 场景 | 核心需求 | 关键信号 | 选择算法 | 为什么适合 |
|------|----------|----------|----------|------------|
| **企业 AI 网关** | 成本控制 + 安全 | domain, authz, PII, jailbreak | MultiFactor | 多部门、多权限、需审计 |
| **开发者工具** | 质量 + 延迟 | complexity, embedding, keyword | AutoMix | 简单查询多，需级联降级 |
| **医疗/法律** | 安全 + 合规 | domain, authz, language, PII | Static（固定强模型） | 不能降级，需严格审核 |
| **客服系统** | 成本 + 用户满意度 | embedding, feedback, preference | Elo + SessionAware | 多轮对话，需个性化 |
| **内容生成** | 质量 + 创意 | complexity, modality | RouterDC | 需理解查询语义 |

**不适合的场景**：

| 场景 | 原因 |
|------|------|
| 单模型部署 | 没有路由需求，增加复杂度无收益 |
| 延迟极敏感（<10ms） | 路由本身有 15-55ms 开销 |
| 模型能力完全相同 | 没有区分度，路由无意义 |
| 离线批处理 | 不需要实时路由 |

---

### 5.2 用户真正关心什么

**不同角色的关注点**：

```
CTO / 技术 VP：
- 总成本降低多少？（Token 经济学）
- 安全风险降低多少？（越狱/PII 检测率）
- 能否统一管理多个供应商？（vLLM/OpenAI/Anthropic）

算法工程师：
- 路由准确率多少？（是否选对了模型）
- 延迟开销多少？（是否影响用户体验）
- 能否自定义信号和规则？（灵活性）

产品经理：
- 用户满意度变化？（回答质量）
- 响应速度变化？（延迟）
- 能否 A/B 测试？（验证效果）

运维工程师：
- 部署复杂度？（K8s/Helm）
- 监控告警？（Prometheus/Grafana）
- 故障恢复？（回滚/降级）
```

---

### 5.3 量化业务价值

**Token 经济学**：
```
假设场景：
- 日均 100 万次 LLM 请求
- 40% 是简单查询（可以用小模型）
- 大模型：$0.03/1K tokens，小模型：$0.003/1K tokens
- 平均每次请求 500 tokens

节省计算：
- 40 万次/天 × 500 tokens × ($0.03 - $0.003)/1K = $5,400/天
- 年化：$5,400 × 365 = $1,971,000/年

路由开销：
- 分类器推理：~15ms × $0.0001/次 = $100/天
- 净节省：$5,300/天
```

**安全价值**：
```
越狱检测：
- 检测率 95%，假阳性率 2%
- 100 万请求中 1% 是越狱 → 检测到 9,500 次
- 每次越狱可能导致数据泄露/品牌损失 → 难以量化但极高

PII 检测：
- 检测率 99%
- 避免合规风险（GDPR 罚款最高 4% 年收入）
```

---

### 5.4 上线成本分析

**资源需求**：
```
最小部署（单机）：
- CPU: 4 核
- 内存: 4 GB（含嵌入模型）
- 存储: 10 GB（模型 + 配置 + 日志）
- GPU: 可选（无 GPU 时用 ONNX CPU 推理，延迟 +40 倍）

生产部署（K8s）：
- 路由器 Pod: 2-3 副本（高可用）
- 缓存: Redis/Milvus（可选）
- 监控: Prometheus + Grafana
- 总成本：~$200-500/月（云上）
```

**部署复杂度**：
```
一键部署：
curl -fsSL https://vllm-semantic-router.com/install.sh | bash

K8s 部署：
helm install vllm-sr deploy/helm/ --set replicas=3

配置复杂度：
- 最小配置：~50 行 YAML（定义模型 + 路由规则）
- 完整配置：~500 行（含安全、缓存、RAG、监控）
```

---

### 5.5 资源有限时的优先级

**面试回答模板**：

> "如果资源有限，我会按以下优先级优化：
> 
> **P0（必须有）**：
> 1. 路由准确率 → 直接影响用户体验和成本
> 2. 延迟 → 路由开销不能超过节省的推理时间
> 3. 可靠性 → 路由器故障 = 所有 LLM 服务不可用
> 
> **P1（应该有）**：
> 1. 安全检测 → 越狱/PII 是合规要求
> 2. 监控告警 → 没有监控就无法发现问题
> 3. 配置热更新 → 减少停机时间
> 
> **P2（可以有）**：
> 1. 语义缓存 → 进一步降低成本
> 2. RAG 集成 → 提升回答质量
> 3. 多模态路由 → 扩展应用场景
> 
> **P3（锦上添花）**：
> 1. Dashboard UI → 运维便利性
> 2. RL 在线学习 → 长期优化
> 3. 图学习个性化 → 差异化竞争"

---

### 5.6 竞品对比与差异化

| 竞品 | 定位 | 优势 | 劣势（vs Semantic Router） |
|------|------|------|---------------------------|
| **OpenRouter** | API 聚合 | 简单易用 | 无语义路由，纯手动选择 |
| **RouteLLM** | 学术原型 | 算法先进 | 只支持 2 个模型，无安全检测 |
| **LiteLLM** | 统一 API | 兼容性好 | 无智能路由，纯代理 |
| **Kong/Apigee** | API 网关 | 成熟稳定 | 不理解 LLM 语义 |
| **自研** | 完全定制 | 完全控制 | 开发成本高，维护负担重 |

**Semantic Router 的差异化**：
1. 13 种算法集成（vs 竞品通常只有 1-2 种）
2. 信号-决策分离架构（vs 竞品硬编码逻辑）
3. 安全检测内置（vs 竞品需要额外集成）
4. Rust 推理引擎（vs 竞品用 Python，延迟高 10 倍）
5. Envoy ExtProc 集成（vs 竞品需要改应用代码）

---

## 附录：面试回答模板

### "介绍一下你做的项目"

> "我参与了 vLLM Semantic Router 项目，这是一个系统级的 LLM 智能路由器。它解决的核心问题是：LLM 生态碎片化，不同模型在能力、成本、隐私上差异大，需要根据请求内容动态选择最优模型。
>
> 技术上，我们设计了 Signal-Decision-Algorithm 三层架构。信号层用 19 种信号提取器并行分析请求（BM25 关键词、嵌入相似度、安全检测等）。决策层用布尔表达式树组合信号。算法层提供 13 种模型选择策略（Elo、双对比学习、POMDP 级联、RL 等）。
>
> 我的具体工作是 [你的贡献]。最大的挑战是 [具体问题]，我通过 [解决方案] 解决，效果是 [量化结果]。"

### "遇到的最大困难是什么"

> "最大的困难是 [具体问题]。
> 
> 背景：[问题发生的上下文]
> 表现：[问题的具体症状]
> 排查过程：[怎么定位到根因的]
> 解决方案：[怎么解决的]
> 预防措施：[怎么防止复发]
> 
> 这个经历让我学到了 [经验教训]。"

### "如果让你重新做，你会怎么改进"

> "三个改进方向：
> 1. **数据**：[如何改进训练数据质量/数量]
> 2. **算法**：[如何改进核心算法]
> 3. **工程**：[如何改进系统架构/性能]
> 
> 优先级：[哪个最重要，为什么]"

---

> **最后提醒**：面试官追问的本质是看你是否真正动手做过。准备时，对着代码回顾你做过的每一步，想清楚"为什么这么做"和"遇到了什么问题"。能讲清楚问题和解决方案，比堆砌技术名词更重要。
