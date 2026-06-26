# vLLM Semantic Router 面试准备文档

> **目标读者**：找 LLM / Agent 应用或后训练方向算法实习的 MS 在读学生
> **项目定位**：系统级 LLM 智能路由器 (Infra + 应用层)，但涵盖大量可迁移到算法面试的核心概念
> **使用方式**：先通读第一、二部分建立全局理解，再按第三部分的面试题深挖准备

---

## 目录

1. [项目全局概览](#1-项目全局概览)
2. [核心算法与技术深度解析](#2-核心算法与技术深度解析)
3. [面试高频考点与深挖准备](#3-面试高频考点与深挖准备)
4. [如何向面试官介绍这个项目](#4-如何向面试官介绍这个项目)
5. [延伸阅读与论文索引](#5-延伸阅读与论文索引)

---

## 1. 项目全局概览

### 1.1 一句话定义

**vLLM Semantic Router** 是一个 **信号驱动 (signal-driven)** 的 LLM 智能路由器，部署在 Envoy 代理层，拦截所有 LLM API 请求，通过语义分析决定将请求路由到哪个模型后端。

### 1.2 解决什么问题

LLM 生态存在模型碎片化问题：不同模型在能力、规模、成本、隐私边界上差异巨大。选择和连接正确的模型是一个 **系统级优化问题**。

核心价值主张：
- **Token 经济学**：简单查询路由到小模型，节省成本
- **LLM 安全**：检测越狱、PII 泄露、幻觉
- **全网智能**：协调本地/私有/前沿模型

### 1.3 技术栈

| 层次 | 语言 | 用途 |
|------|------|------|
| 核心路由器 | **Go** | Envoy ExtProc gRPC 服务、决策引擎、配置管理 |
| ML 推理 | **Rust** (Candle/ONNX/Linfa) | BERT 分类、LoRA 推理、KNN/KMeans/SVM、BM25/N-gram |
| 训练管线 | **Python** (PyTorch/PEFT/HuggingFace) | LoRA 微调、ML 模型训练、RL 训练 |
| CLI / 模拟 | **Python** | 开发工具、舰队模拟器 |
| 前端 | **TypeScript** (React) | Dashboard UI |

### 1.4 核心架构：Signal → Decision → Algorithm → Plugin

```
请求流:
  Envoy Proxy
    → gRPC ExtProc
      → 解析请求头/体 (OpenAI/Anthropic 格式)
      → 信号提取 (并行 19 种信号)
      → 决策评估 (布尔表达式树)
      → 模型选择 (13 种算法)
      → 插件应用 (缓存/RAG/记忆/工具)
      → 路由到后端模型
```

**为什么这个分层重要（面试角度）**：
- **信号层** = 特征工程，将非结构化请求转化为结构化特征向量
- **决策层** = 规则引擎 + 逻辑推理，布尔表达式树 = 可解释的分类器
- **算法层** = 排序/选择问题，从候选集中选最优模型
- **插件层** = 后处理管线（缓存、RAG、安全检查等）

---

## 2. 核心算法与技术深度解析

### 2.1 信号提取引擎 (Signal Extraction Engine)

**定义**：信号规则 ρ = (τ, n, f)，其中 f: R → {0,1} × [0,1]，输出匹配与否 + 置信度。

#### 2.1.1 启发式信号 (<1ms)

| 信号类型 | 方法 | 面试关联 |
|----------|------|----------|
| **Keyword** | BM25 (Okapi) + N-gram (Jaccard) + Regex | 信息检索基础，TF-IDF 理解 |
| **Context Length** | Token 计数区间 [l, u] | 简单但实用的特征 |
| **Language** | 统计 n-gram 检测 (lingua-go)，100+ 语言 | 多语言处理 |
| **Authorization** | RBAC 权限检查 | 系统设计 |

**BM25 详解（面试高频）**：
```
BM25(q, d) = Σ IDF(qi) × [f(qi,d) × (k1+1)] / [f(qi,d) + k1 × (1 - b + b × |d|/avgdl)]
```
- `f(qi,d)`: 词频
- `IDF(qi)`: 逆文档频率，log[(N-n(qi)+0.5)/(n(qi)+0.5)+1]
- `k1`: 词频饱和参数 (通常 1.2-2.0)
- `b`: 文档长度归一化参数 (通常 0.75)
- **关键理解**：BM25 是 TF-IDF 的改进，引入了词频饱和和文档长度归一化

#### 2.1.2 学习型信号 (10-100ms)

| 信号类型 | 方法 | 面试关联 |
|----------|------|----------|
| **Embedding** | 余弦相似度 + 原型银行 (Prototype Bank) | 表示学习、相似度检索 |
| **Domain** | LoRA 微调的 mmBERT 分类器 (MMLU 类别) | 微调技术、多任务学习 |
| **Complexity** | 对比学习：δ = max_h sim(q,h) - max_e sim(q,e) | 难度估计 |
| **Jailbreak** | BERT 分类器 OR 对比嵌入 + 最大对比链聚合 | 安全对齐、对抗鲁棒性 |
| **PII** | Token 级 NER + 按规则类型过滤 | 序列标注、隐私保护 |
| **Hallucination** | 3 阶段门控：Sentinel → Detector → Explainer (NLI) | 幻觉检测、NLI |

**Embedding 分类详解（面试可深挖）**：

```
原型银行 (Prototype Bank) 系统：
1. 初始化：将候选示例聚类为原型
2. 推理：计算查询与所有原型的余弦相似度
3. 评分：使用 best_weight（最佳原型权重）和 top_m（考虑的顶部原型数）
4. 匹配：硬匹配（阈值）或软匹配（min_score_threshold）
```

**关键设计选择**：
- 为什么用原型银行而不是直接和所有候选比较？→ 减少计算量，提高泛化性
- 硬匹配 vs 软匹配的区别？→ 硬匹配是二值决策，软匹配输出连续置信度
- 多模态支持：文本和图像嵌入在同一空间

**Complexity 信号的对比学习思路**：
```
δ = max_h sim(q, h) - max_e sim(q, e)
- q: 查询向量
- h: 难题原型向量
- e: 简单题原型向量
- δ > 0 → 查询更像难题 → 路由到强模型
- δ < 0 → 查询更像简单题 → 路由到弱模型
```

#### 2.1.3 信号评估的信息论分析

**路由熵**：
```
H(M|r_raw) ≈ log2 K  (K 为候选模型数)
```
信号的作用是降低路由不确定性：
```
H(M|S(r)) → 0  (理想情况)
```
这与 **互信息** 直接相关：`I(M; S(r)) = H(M) - H(M|S(r))`

**面试角度**：这本质上是特征选择问题，选择信息量最大的信号子集来最大化 `I(M; S(r))`。

---

### 2.2 决策引擎 (Decision Engine)

#### 2.2.1 布尔表达式树

每个决策 d = (n, φ, M_d, Π_d, p)，其中 φ 是布尔表达式树：

```
评估函数（递归）：
eval(φ, S(r)) = 
  if leaf: 1[∃(ρ,1,c) ∈ S(r)]     # 叶子节点：检查信号是否匹配
  if AND:  ∧_i eval(φ_i)            # 合取：所有子节点都匹配
  if OR:   ∨_i eval(φ_i)            # 析取：任一子节点匹配
  if NOT:  ¬eval(φ_1)               # 取反：子节点不匹配
```

**置信度传播**：
```
conf(d, S(r)) = (1/|Γ_sat|) × Σ c_j(r)  (Γ_sat 为满足的叶子节点集合)
```

**复杂度**：O(M × L_max)，实践中 <0.1ms

#### 2.2.2 理论保证

- **单决策完备性**：任何 f:{0,1}^N → {0,1} 都可以用一个规则节点表示
- **路由策略完备性**：任何 π:{0,1}^N → M 都可以用决策集 + 优先级实现
- **PLA 对应**：深度-1 树 ↔ 可编程逻辑阵列；递归树 ↔ 通用电路

**模糊推广**（面试可提及）：
```
模糊评估（用 t-norm 替代经典逻辑）：
AND → min
OR  → max
NOT → 1 - x
```

#### 2.2.3 与 MoE 的类比

- **MoE (Mixture of Experts)**：token 级、模型内部、学习的 softmax 门控
- **MoM (Mixture of Models)**：请求级、模型之间、可配置的符号门控
- 两者是互补的：MoM 决定用哪个模型，MoE 决定模型内的专家

---

### 2.3 模型选择算法 (Model Selection)

这是面试中最容易被深挖的部分。

#### 2.3.1 Elo Rating（RouteLLM 论文）

```
Bradley-Terry 模型：
P(m_i ≻ m_j) = 1 / (1 + 10^((R_j - R_i)/400))

更新规则（当 m_i 胜出时）：
R_i' = R_i + K × (1 - P(m_i ≻ m_j))
R_j' = R_j + K × (0 - P(m_j ≻ m_i))
```

**面试深挖点**：
- K 因子的选择：K 太大 → 振荡，K 太小 → 收敛慢
- 类别加权：不同类别（数学 vs 文学）的 Elo 独立维护
- 成本感知：在质量相近时偏好便宜模型
- 冷启动问题：新模型没有历史数据时如何处理？

**代码位置**：`src/semantic-router/pkg/selection/elo.go:98`

#### 2.3.2 RouterDC（双对比学习）

论文：arXiv:2409.19886

```
核心思想：
1. 学习查询嵌入 e_q 和模型嵌入 e_m
2. 选择：m* = argmax cos(e_q, e_m_k)
3. 训练：对比损失拉近好模型对，推远差模型对

损失函数：
L = -log[exp(cos(e_q, e_m+)/τ) / Σ_k exp(cos(e_q, e_m_k)/τ)]
```

**面试深挖点**：
- 为什么需要双编码器而不是单编码器？→ 查询和模型是不同类型的实体
- 温度参数 τ 的作用？→ 控制分布的尖锐程度
- 负样本采样策略？→ 所有其他模型作为负样本
- 如何处理同一查询在不同模型上质量相近的情况？

**代码位置**：`src/semantic-router/pkg/selection/router_dc.go:79`

#### 2.3.3 AutoMix（POMDP 级联路由）

论文：arXiv:2310.12963

```
核心思想：级联多个模型，用小模型先尝试，不够好再升级

期望成本：
E[C] = Σ_k C_k × Π_{j<k} (1 - P(q̂_j ≥ τ_j))

关键组件：
- 自验证：模型输出置信度
- 信念粒子过滤：维护模型质量的后验分布
- 成本-质量权衡：优化 E[Quality] - λ × E[Cost]
```

**面试深挖点**：
- POMDP vs MDP 的区别？→ 部分可观测（不确定模型输出质量）
- 粒子滤波如何工作？→ 用采样近似后验分布
- 阈值 τ 的选择？→ 交叉验证或在线学习
- 级联的顺序如何决定？→ 按成本从低到高

**代码位置**：`src/semantic-router/pkg/selection/automix.go`

#### 2.3.4 RL-Driven（Router-R1）

论文：arXiv:2506.09033

```
核心思想：用强化学习训练路由器

奖励函数：
R = 0.1 × R_format + 0.7 × R_outcome + 0.2 × R_cost

- R_format：输出格式正确性（<think>/<route> 标签）
- R_outcome：选择的模型是否最优
- R_cost：成本效率

算法：REINFORCE 策略梯度
∇J(θ) = E[Σ_t ∇log π_θ(a_t|s_t) × R_t]
```

**面试深挖点**：
- 为什么用 REINFORCE 而不是 PPO？→ 离线训练，不需要在线交互
- 奖励权重如何调？→ 业务需求决定（质量优先 vs 成本优先）
- 基座模型选择？→ 小型指令模型（Phi-3-mini）
- Think→Route 模式的优势？→ 可解释性，先推理再决策

**代码位置**：`src/training/model_selection/rl_model_selection/train_router_r1.py`

#### 2.3.5 GMTRouter（异构图学习）

论文：arXiv:2511.08590

```
核心思想：用异构图 Transformer 学习多轮对话的个性化路由

节点类型：User, LLM, Query, Response, Turn
边类型：10 种异构边
GNN：SimplifiedHGTConv + 多头注意力

输入：用户-LLM 交互历史图
输出：预测用户对各模型的评分
```

**面试深挖点**：
- 为什么用图而不是序列？→ 捕获用户-模型-查询的多对多关系
- 异构图 vs 同构图的区别？→ 不同类型的节点和边有不同的参数
- 冷启动用户如何处理？→ 基于查询相似度的默认路由

**代码位置**：`src/training/model_selection/rl_model_selection/train_gmtrouter.py`

#### 2.3.6 其他算法速览

| 算法 | 核心思想 | 适用场景 |
|------|----------|----------|
| **KNN** | 质量加权 K 近邻投票 | 有历史数据时的快速路由 |
| **KMeans** | 聚类分配 + 效率加权 | 模型能力边界清晰时 |
| **SVM** | RBF 核 + 一对多分类 | 特征空间有明确决策边界 |
| **MLP** | 两层神经网络 | 复杂的非线性决策边界 |
| **Latency-Aware** | TPOT/TTFT 百分位选择 | 延迟敏感场景 |
| **Multi-Factor** | 加权质量/延迟/成本/负载 | 多目标优化 |
| **Session-Aware** | 工具循环锁定 + 前缀缓存亲和 | Agent 多轮对话 |
| **ReMoM** | 多轮并行推理 + 宽度调度 | 复杂推理任务 |

---

### 2.4 LoRA 多任务分类

#### 2.4.1 核心思想

```
问题：n 个分类任务需要 n 份完整模型副本
  M_indep = n × |θ_base|

解决：单份冻结基座 + n 个 LoRA 适配器
  M_LoRA = |θ_base| + n × 2rd

LoRA 公式：
  W' = W + ΔW = W + BA
  其中 B ∈ R^{d×r}, A ∈ R^{r×d}
```

**内存效率**：当 n=6, r=32, d=768 时，每个适配器仅 49,152 参数（~0.02% 基座）。

#### 2.4.2 MoM 模型家族（11 个模型）

| 模型 | 任务 | 训练数据 |
|------|------|----------|
| mom-domain | 领域分类 | MMLU |
| mom-pii | PII 实体识别 | Presidio |
| mom-jailbreak | 提示注入检测 | 对抗数据集 |
| mom-sentinel | 事实核查门控 | 事实 vs 创意查询 |
| mom-detector | 幻觉检测 | 标注 LLM 输出 |
| mom-explainer | NLI 解释 | NLI 基准 |
| mom-feedback | 用户反馈 | 对话标注 |
| mom-modality | 模态分类 | DiffusionDB + text |
| mom-embedding | 语义嵌入 | 对比预训练 |
| mom-toolcall | 工具选择 | 函数调用数据集 |
| mom-intent | 意图分类 | 客服对话 |

#### 2.4.3 LoRA 训练细节（面试深挖）

**目标模块选择**：
```python
# BERT/RoBERTa
target_modules = ["attention.self.query", "attention.self.value", 
                  "attention.output.dense", "intermediate.dense", "output.dense"]

# ModernBERT/mmBERT
target_modules = ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]
```

**超参数**：
- rank (r): 16/32/64，越大表达能力越强但参数越多
- alpha (α): 通常设为 2r，缩放因子 α/r
- dropout: 0.1，防止过拟合

**一致性训练（创新点）**：
```python
loss = clean_loss + typo_loss + λ × KL_div(clean_logits, typo_logits)
```
- 对输入施加人为 typo（键盘相邻、交换、删除、元音替换）
- 鲁棒性提升：干净 +3%，typo +11%

**代码位置**：`src/training/model_classifier/classifier_model_fine_tuning_lora/ft_linear_lora_consistency.py`

---

### 2.5 HaluGate 幻觉检测

#### 2.5.1 三阶段门控管线

```
Sentinel (门控) → Detector (跨度识别) → Explainer (NLI 分类)

1. Sentinel: g_sent(q) ∈ {NEEDS_FACT_CHECK, NO_FACT_CHECK}
   - 跳过非事实性查询（40-60% 工作量）
   
2. Detector: g_det(q, c, a) = {(i,j,c_ij) | a_i...a_j 不被上下文 c 支持}
   - Token 级跨度识别
   
3. Explainer: g_nli(a_{i:j}, c) ∈ {ENTAILMENT, CONTRADICTION, NEUTRAL}
   - NLI 分类：支持/矛盾/中性
```

**期望成本**：
```
E[Cost] = C_sent + p_fact × (C_det + k̄ × C_nli)
```

#### 2.5.2 与现有方法对比

| 方法 | 局限性 | HaluGate 优势 |
|------|--------|---------------|
| SelfCheckGPT | 统一开销 | Sentinel 门控跳过非事实查询 |
| FActScore | 句子级评分 | Token 级跨度识别 |
| - | 不区分矛盾和中性 | NLI 区分矛盾 vs 不支持 |

---

### 2.6 语义缓存 (Semantic Cache)

```
核心思想：嵌入相似度匹配 → 命中缓存 → 直接返回

相似度阈值：θ = 0.92
- 精确匹配：100% 命中率，<5ms
- 改写匹配：60-80% 命中率

后端选择：
- InMemory: HNSW 索引 + SIMD 距离计算（单机高性能）
- Milvus: 分布式向量数据库
- Redis/Valkey: 内存数据库
- Qdrant: 向量数据库
```

**HNSW 算法面试要点**：
- 层级可导航小世界图
- 构建：随机层分配 + 贪心连接
- 搜索：从顶层开始贪心下降，逐层细化
- 复杂度：O(log N) 查询，O(N log N) 构建

---

### 2.7 ML 推理架构

#### Rust FFI 绑定

```
Go (核心路由器)
  ↓ CGo FFI
Rust (ML 推理)
  ├── Candle: BERT/ModernBERT/DeBERTa 分类 + LoRA + MLP
  ├── Linfa: KNN/KMeans/SVM
  ├── ONNX Runtime: 嵌入模型
  └── NLP Binding: BM25/N-gram
```

**为什么用 Rust 而不是 Python？**
- 延迟要求：分类 <100ms，Python 推理通常 >200ms
- 内存效率：无 GIL，并发安全
- 硬件支持：CUDA/ROCm/OpenVINO/MKL/Accelerate
- 部署简化：无需 Python 运行时

**Flash Attention 2 加速**：
```
序列长度 | SDPA | FA | 加速比
512      | 19ms | 19ms | 1.0×
4,096    | 167ms | 51ms | 3.3×
8,192    | OOM | 105ms | -
32,768   | OOM | 756ms | -
```

---

### 2.8 嵌入训练

#### 2.8.1 语义缓存嵌入

```
基座模型：all-MiniLM-L12-v2 (384-dim, 90MB)
LoRA：r=8, α=32, 目标=["query", "value"]
损失：Multiple Negatives Ranking (MNR)，温度=0.05
适配器大小：~600KB
效果：多域 LoRA 医疗 +24.9%，法律 +27.3%，编程 +12.4%
```

#### 2.8.2 域适应嵌入

```
方法：迭代硬负例挖掘（2 轮）
损失：TripletLoss，margin=0.1（余弦距离）
训练：lr=5e-5, batch=8, epoch=2/轮
比例：简单:硬 = 2:1（防遗忘）
效果：MedQuAD MRR@5 +71.18%，Recall@5 +40.92%
```

**硬负例挖掘面试要点**：
- 什么是硬负例？→ 模型认为相似但实际不相似的样本
- 为什么要挖掘？→ 提高判别能力，学到更细粒度的区别
- 防遗忘策略？→ 混合简单负例（easy:hard = 2:1）

---

## 3. 面试高频考点与深挖准备

### 3.1 Embedding 与表示学习

#### 考点 1：余弦相似度 vs 欧氏距离

```
余弦相似度：cos(a,b) = (a·b) / (||a|| × ||b||)
- 关注方向，忽略大小
- 适合高维稀疏向量
- 范围 [-1, 1]

欧氏距离：d(a,b) = √Σ(a_i - b_i)²
- 关注绝对差异
- 适合低维稠密向量
- 范围 [0, ∞)

为什么 Semantic Router 用余弦？→ 嵌入向量已归一化，关注语义方向而非模长
```

#### 考点 2：原型银行 vs 全量比较

```
全量比较：O(N) 计算，N 为候选数
原型银行：先聚类为 K 个原型，O(K) 计算

权衡：
- 精度：全量 > 原型（原型有信息损失）
- 速度：原型 > 全量（K << N）
- 泛化：原型更好（聚类有正则化效果）
```

#### 考点 3：HNSW 算法

```
构建过程：
1. 每个节点随机分配到 [0, max_layer] 层
2. 对于每一层，找到最近的 M 个邻居并建立连接
3. 使用贪心搜索 + 优先队列

查询过程：
1. 从最高层的入口点开始
2. 在当前层贪心搜索最近邻
3. 下降到下一层，以当前最近邻为入口继续
4. 直到第 0 层

参数：
- M：每层连接数（影响精度/速度权衡）
- ef_construction：构建时的搜索宽度
- ef_search：查询时的搜索宽度
```

#### 考点 4：Matryoshka 嵌入

```
核心思想：嵌入向量的前 d 维也包含有用信息

二维 Matryoshka：
1. 层级早退：不同层输出不同精度的嵌入
2. 维度截断：只用前 d 维（如 256/512/1024）

优势：
- 粗排用低维（快），精排用高维（准）
- 节省存储和计算
```

---

### 3.2 分类与微调

#### 考点 1：LoRA 原理

```
低秩分解：ΔW = BA，其中 B ∈ R^{d×r}, A ∈ R^{r×d}
- r << d，所以参数量从 d² 降到 2dr
- 初始化：A ~ N(0, σ²), B = 0（训练开始时 ΔW = 0）
- 缩放：ΔW = (α/r) × BA

为什么有效？
- 大模型的权重矩阵是低秩的（内在维度低）
- 微调只需更新一个低秩子空间
- 类似于 PCA 降维的思想

面试追问：LoRA vs Adapter vs Prefix Tuning vs Full Fine-tuning？
- LoRA：修改权重矩阵，无推理延迟
- Adapter：插入额外层，有推理延迟
- Prefix Tuning：添加可学习前缀，限制在注意力层
- Full FT：更新所有参数，最高质量但最贵
```

#### 考点 2：多任务学习 vs 单任务学习

```
Semantic Router 的选择：共享基座 + 独立 LoRA（而非硬参数共享）

为什么？
- 任务差异大：PII 是序列标注，Domain 是序列分类
- 灵活性：可以独立添加/删除任务
- 干扰避免：不同任务的梯度可能冲突

权衡：
- 多任务共享：参数效率高，可能有正迁移
- 独立 LoRA：灵活性高，无负迁移
```

#### 考点 3：一致性训练 (Consistency Training)

```
核心思想：模型对原始输入和扰动输入的输出应该一致

实现：
1. 对输入施加扰动（如 typo）
2. 计算原始输出和扰动输出的 KL 敦度
3. 总损失 = 原始损失 + 扰动损失 + λ × KL

为什么有效？
- 正则化效果，防止过拟合
- 提高对输入扰动的鲁棒性
- 类似于数据增强但更精细
```

---

### 3.3 对比学习

#### 考点 1：对比损失 (Contrastive Loss)

```
InfoNCE Loss（Semantic Router 使用）：
L = -log[exp(sim(q, k+)/τ) / Σ_k exp(sim(q, k_k)/τ)]

- q: 查询向量
- k+: 正样本向量
- k_k: 所有样本向量（包含正样本）
- τ: 温度参数

温度参数的作用：
- τ → 0：分布趋近 one-hot，只关注最难的负样本
- τ → ∞：分布趋近均匀，所有样本等权重
- 通常 τ = 0.05-0.1

面试追问：为什么用 InfoNCE 而不是 Triplet Loss？
- Triplet Loss：只推远单个负样本，效率低
- InfoNCE：同时推远所有负样本，更高效
- 数学上，InfoNCE 最大化互信息的下界
```

#### 考点 2：硬负例挖掘

```
什么是硬负例？
- 相似但不匹配的样本（如 "机器学习" vs "深度学习"，但标签不同）

为什么要挖掘？
- 随机负例太简单，模型学不到有用信息
- 硬负例提供更强的梯度信号

如何挖掘？
1. 离线挖掘：用当前模型检索 Top-K 最相似的负样本
2. 在线挖掘：在训练过程中动态更新负样本
3. 迭代挖掘：每轮训练后重新挖掘（Semantic Router 用 2 轮）

防遗忘策略：
- 混合简单负例（easy:hard = 2:1）
- 防止模型只关注硬负例而忘记简单模式
```

---

### 3.4 模型选择与排序

#### 考点 1：Elo Rating 系统

```
核心假设：每个模型有真实能力 R，比赛结果由 Bradley-Terry 模型决定

P(i beats j) = 1 / (1 + 10^((R_j - R_i)/400))

更新规则：
- 胜者加分，败者减分
- K 因子控制学习率
- 弱者赢强者 → 大幅更新
- 强者赢弱者 → 小幅更新

面试追问：
1. 为什么是 400 分？→ 历史惯例，使得 2:1 胜率对应 200 分差距
2. K 因子如何选择？→ 新模型 K 大（快速学习），成熟模型 K 小（稳定）
3. 如何处理多人同时比赛？→ 逐一更新或批量更新
4. 冷启动问题？→ 用先验分布（如所有模型初始 1500 分）
```

#### 考点 2：Contextual Bandit

```
Semantic Router 的视角：
- 上下文：信号向量 S(r)
- 动作：选择模型 m
- 奖励：输出质量 Q(r, m)

遗憾界：O(√T) 收敛到最优固定策略

算法选择：
- Thompson Sampling：贝叶斯方法，维护后验分布
- UCB：乐观面对不确定性
- ε-greedy：简单探索-利用权衡
```

#### 考点 3：POMDP 级联路由

```
为什么是 POMDP 而不是 MDP？
- MDP：完全可观测状态
- POMDP：部分可观测（不知道小模型输出质量）

状态：查询特征 + 模型质量分布
观测：模型输出（不完美）
动作：升级到更大模型 / 当前输出足够好

信念更新：粒子滤波
- 用采样近似后验分布
- 每个粒子表示一种可能的模型质量组合
```

---

### 3.5 安全与对齐

#### 考点 1：越狱检测

```
方法 1：分类器
- 训练 BERT 分类器区分安全/不安全提示
- 优势：快速，可解释
- 劣势：对抗样本可能绕过

方法 2：对比嵌入
- 计算与越狱模式库的相似度
- 最大对比链聚合（多个原型取最大值）
- 优势：泛化性好
- 劣势：需要维护模式库

方法 3：混合
- 分类器 + 嵌入相似度 + 模式匹配
- 任意一个触发则拒绝

面试追问：如何评估越狱检测器？
- 真阳性率 (TPR)：正确识别越狱
- 假阳性率 (FPR)：误报正常请求
- 权衡：FPR 太高影响用户体验，TPR 太低有安全风险
```

#### 考点 2：幻觉检测

```
三阶段管线的创新点：
1. Sentinel 门控：跳过非事实性查询（40-60%）
   - 为什么？→ 事实性查询才需要核查，创意查询不需要
   
2. Detector 跨度识别：Token 级而非句子级
   - 为什么？→ 更精确的定位，减少误报
   
3. Explainer NLI：区分矛盾 vs 不支持
   - 为什么？→ "不支持"不等于"错误"，可能是缺少证据

面试追问：如何构建训练数据？
- 正样本：人工标注的事实/非事实查询
- 负样本：LLM 生成的幻觉内容
- NLI：使用 NLI 基准数据集（如 SNLI, MNLI）
```

---

### 3.6 信息检索

#### 考点 1：BM25 vs TF-IDF

```
TF-IDF：
- TF(t,d) = 词频
- IDF(t) = log(N/df(t))
- 问题：词频无上限，长文档有利

BM25 改进：
1. 词频饱和：f(t,d) / (f(t,d) + k1)
   - 词频增加，收益递减
2. 文档长度归一化：(1 - b + b × |d|/avgdl)
   - 长文档惩罚
3. 参数调优：k1 (1.2-2.0), b (0.75)

面试追问：BM25 的变体？
- BM25L：改进长文档处理
- BM25F：多字段加权（标题 vs 正文）
- BM25+：改进短查询处理
```

#### 考点 2：语义缓存

```
为什么需要语义缓存？
- 精确缓存（hash）：无法处理改写
- 语义缓存（embedding）：捕获语义等价

相似度阈值选择：
- θ = 0.92：非常严格的匹配，假阳性少
- θ = 0.80：较宽松，命中率高但可能有误匹配
- 需要根据业务场景调优

HNSW 索引选择：
- 为什么不用暴力搜索？→ O(N) 太慢
- 为什么不用 LSH？→ 精度不够
- HNSW：O(log N) 查询，高精度
```

---

### 3.7 RL 与在线学习

#### 考点 1：RL 训练路由器

```
奖励设计：
R = w1 × R_format + w2 × R_outcome + w3 × R_cost

- R_format：格式正确性（防止模型输出垃圾）
- R_outcome：任务完成质量（核心目标）
- R_cost：成本效率（辅助目标）

权重调优：
- w2 > w3：质量优先于成本
- w1 较小但必要：确保输出可用

算法选择：
- REINFORCE：简单，方差大
- PPO：稳定，需要在线交互
- A3C：异步，适合分布式

为什么选 REINFORCE？
- 离线训练（不需要真实模型推理）
- 简单实现
- 足够用于推理路由任务
```

#### 考点 2：Thompson Sampling

```
核心思想：维护每个模型的后验分布，按后验采样选择

Beta 分布：
- 先验：Beta(α, β) = Beta(1, 1)（均匀分布）
- 更新：成功 → α += 1，失败 → β += 1
- 采样：θ ~ Beta(α, β)，选择 argmax θ

优势：
- 自动探索-利用权衡
- 收敛到最优策略
- 计算简单

局限：
- 假设独立（不考虑上下文）
- 需要足够的数据
```

---

### 3.8 Agent 与工具使用

#### 考点 1：工具选择

```
Semantic Router 的方案：嵌入相似度
- 工具描述嵌入库
- 查询与工具描述的余弦相似度
- Top-K 最相似的工具

替代方案：
- 基于规则：关键词匹配
- 基于 LLM：让 LLM 选择工具
- 混合：先规则过滤，再嵌入排序

权衡：
- 嵌入方法：泛化性好，但需要训练
- LLM 方法：最灵活，但延迟高
- 规则方法：最快，但不泛化
```

#### 考点 2：多轮对话管理

```
Session-Aware 选择：
- 工具循环锁定：避免频繁切换工具
- 前缀缓存亲和：偏好有缓存的模型
- 切换惩罚：避免不必要的模型切换

为什么需要？
- Agent 通常有多轮工具调用
- 频繁切换模型会破坏上下文
- 前缀缓存可以显著降低延迟
```

---

## 4. 如何向面试官介绍这个项目

### 4.1 30 秒电梯演讲

> "我参与了 vLLM Semantic Router 项目，这是一个系统级的 LLM 智能路由器。它部署在代理层，通过信号提取（如嵌入相似度、关键词匹配、安全检测）将 LLM API 请求路由到最合适的模型后端。核心创新是 Signal-Decision-Algorithm 三层架构，以及 13 种模型选择算法（包括 Elo、双对比学习、POMDP 级联、RL 训练等）。我的工作集中在 [你的具体贡献]，涉及 [具体技术]。"

### 4.2 2 分钟详细介绍

> **问题定义**：LLM 生态碎片化，不同模型在能力、成本、隐私上差异大。选择正确的模型是一个系统级优化问题。
>
> **技术方案**：我们设计了 Signal-Decision-Algorithm 三层架构：
> - **信号层**：19 种信号提取器并行运行，从启发式（BM25、关键词）到学习型（BERT 分类、嵌入相似度）
> - **决策层**：布尔表达式树组合信号，支持 AND/OR/NOT 逻辑
> - **算法层**：13 种模型选择算法，从经典 ML（KNN/SVM）到 RL（Router-R1）到图学习（GMTRouter）
>
> **创新点**：
> 1. 信号-决策分离：信号提取和逻辑组合正交，便于扩展
> 2. 多算法支持：同一框架支持 Elo、对比学习、POMDP、RL 等多种选择策略
> 3. LoRA 多任务：单份基座 + n 个 LoRA 适配器，内存效率提升 6 倍
> 4. HaluGate：三阶段幻觉检测，门控跳过非事实性查询
>
> **工程实现**：Go 核心路由器 + Rust ML 推理（Candle/ONNX/Linfa）+ Python 训练管线

### 4.3 技术亮点突出点

根据面试官背景选择重点：

| 面试官背景 | 重点突出 |
|------------|----------|
| ML/算法 | Elo、RouterDC、AutoMix、RL 训练、对比学习 |
| 系统/Infra | Envoy ExtProc、Rust FFI、HNSW、语义缓存 |
| Agent/应用 | 工具选择、多轮对话、Session-Aware、HaluGate |
| NLP | BM25、嵌入模型、LoRA 微调、一致性训练 |

### 4.4 面试中可能的追问

**Q: 这个项目偏 Infra，你为什么选它？**

> "虽然项目定位是 Infra，但算法含量很高。比如 13 种模型选择算法涵盖了 Elo（推荐系统）、对比学习（表示学习）、POMDP（强化学习）、图神经网络等多个算法方向。LoRA 多任务微调、HaluGate 幻觉检测、硬负例挖掘等都是纯算法工作。这些技术在 LLM 后训练和 Agent 应用中都很关键。"

**Q: 你从这个项目学到了什么？**

> "1) 算法落地需要系统思维——一个好的算法如果延迟太高就没法用，所以 Rust 推理很重要。2) 工程和算法的交叉——LoRA 不仅是训练技术，还是内存优化技术。3) 评估指标的选择——幻觉检测需要平衡精确率和召回率，FActScore 和 NLI 评估方式各有优劣。"

**Q: 如果让你改进这个项目，你会怎么做？**

> "1) 训练数据质量——目前训练数据主要是 MMLU，可以加入更多真实场景数据。2) 在线学习——目前主要是离线训练，可以加入在线反馈循环。3) 多模态扩展——目前主要是文本，可以加强图像/音频模态的路由。4) 端到端优化——目前信号提取和模型选择是分开优化的，可以端到端联合训练。"

---

## 5. 延伸阅读与论文索引

### 5.1 直接相关论文

| 论文 | 关键技术 | 项目中的应用 |
|------|----------|--------------|
| RouteLLM (arXiv:2406.18665) | Elo Rating, Bradley-Terry 模型 | Elo 选择器 |
| RouterDC (arXiv:2409.19886) | 双对比学习 | RouterDC 选择器 |
| AutoMix (arXiv:2310.12963) | POMDP 级联路由 | AutoMix 选择器 |
| Router-R1 (arXiv:2506.09033) | RL 训练路由器 | RLDriven 选择器 |
| GMTRouter (arXiv:2511.08590) | 异构图 Transformer | GMTRouter 选择器 |
| FusionFactory (arXiv:2507.10540) | MLP 路由 | MLP 选择器 |
| Avengers-Pro (arXiv:2508.12631) | KMeans 路由 | KMeans 选择器 |
| HaluGate (arXiv:2510.08731) | 幻觉检测门控 | HaluGate 插件 |
| Category-Aware Cache (arXiv:2510.26835) | 语义缓存 | 缓存插件 |
| Distilling Wisdom (arXiv:2512.08088) | 域适应嵌入 | 嵌入训练 |

### 5.2 基础知识论文

| 主题 | 论文 | 重要性 |
|------|------|--------|
| Attention | "Attention Is All You Need" (2017) | 必读 |
| BERT | "BERT: Pre-training of Deep Bidirectional Transformers" (2019) | 必读 |
| LoRA | "LoRA: Low-Rank Adaptation of Large Language Models" (2022) | 必读 |
| RLHF | "Training language models to follow instructions with human feedback" (2022) | 重要 |
| MoE | "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer" (2017) | 重要 |
| Contrastive Learning | "SimCLR" / "MoCo" | 重要 |
| InfoNCE | "Representation Learning with Contrastive Predictive Coding" (2018) | 重要 |
| BM25 | "The Probabilistic Relevance Framework: BM25 and Beyond" (2009) | 基础 |
| HNSW | "Efficient and robust approximate nearest neighbor search using HNSW graphs" (2018) | 基础 |

### 5.3 面试准备路径

```
Week 1: 基础巩固
  - 复习 Attention、BERT、LoRA 原理
  - 理解 BM25、余弦相似度、HNSW
  - 掌握对比学习基础（InfoNCE）

Week 2: 算法深入
  - 精读 RouteLLM、RouterDC、AutoMix 论文
  - 实现一个简单的 Elo Rating 系统
  - 理解 POMDP 和 Thompson Sampling

Week 3: 项目理解
  - 通读论文各节
  - 理解 Signal-Decision-Algorithm 架构
  - 掌握 LoRA 多任务微调细节

Week 4: 模拟面试
  - 练习 30 秒/2 分钟项目介绍
  - 准备技术追问的回答
  - 总结项目创新点和局限性
```

---

## 附录：关键代码位置索引

| 组件 | 文件路径 | 说明 |
|------|----------|------|
| 路由器核心 | `src/semantic-router/pkg/extproc/router.go:29` | OpenAIRouter 定义 |
| 信号分类器 | `src/semantic-router/pkg/classification/classifier.go:10` | Classifier 编排器 |
| 嵌入分类器 | `src/semantic-router/pkg/classification/embedding_classifier.go:15` | 原型银行系统 |
| 决策引擎 | `src/semantic-router/pkg/decision/engine.go:32` | 布尔表达式树评估 |
| Elo 选择器 | `src/semantic-router/pkg/selection/elo.go:98` | Bradley-Terry 模型 |
| RouterDC | `src/semantic-router/pkg/selection/router_dc.go:79` | 双对比学习 |
| AutoMix | `src/semantic-router/pkg/selection/automix.go` | POMDP 级联 |
| RL 训练 | `src/training/model_selection/rl_model_selection/train_router_r1.py` | REINFORCE 训练 |
| LoRA 微调 | `src/training/model_classifier/classifier_model_fine_tuning_lora/ft_linear_lora.py` | 标准 LoRA |
| 一致性训练 | `src/training/model_classifier/classifier_model_fine_tuning_lora/ft_linear_lora_consistency.py` | Typo 鲁棒性 |
| 缓存嵌入 | `src/training/model_embeddings/cache_embeddings/lora_trainer.py` | MNR 损失训练 |
| 域适应嵌入 | `src/training/model_embeddings/domain_adapted_embeddings/train.py` | 硬负例挖掘 |
| ML 模型训练 | `src/training/model_selection/ml_model_selection/train.py` | KNN/KMeans/SVM/MLP |
| 论文 | `paper/main.tex` | 完整技术论文 |

---

> **最后提醒**：面试时最重要的是 **理解为什么** 而不是 **记住是什么**。每个设计选择背后都有权衡（trade-off），理解这些权衡比记住具体算法更重要。祝面试顺利！
