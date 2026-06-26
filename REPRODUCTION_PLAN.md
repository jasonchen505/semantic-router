# vLLM Semantic Router — 8×4090 全流程复现计划

> **硬件资源**：8 × NVIDIA RTX 4090 (24GB VRAM each)
> **目标**：完整复现项目核心训练管线 + 推理部署 + 端到端验证
> **预计总时长**：3-4 周（全职）

---

## 目录

1. [资源评估与可行性分析](#1-资源评估与可行性分析)
2. [环境准备](#2-环境准备)
3. [Phase 1：数据准备与 Benchmark 生成](#3-phase-1数据准备与-benchmark-生成)
4. [Phase 2：分类器 LoRA 微调](#4-phase-2分类器-lora-微调)
5. [Phase 3：嵌入模型训练](#5-phase-3嵌入模型训练)
6. [Phase 4：ML 模型选择训练](#6-phase-4ml-模型选择训练)
7. [Phase 5：RL 模型选择训练](#7-phase-5rl-模型选择训练)
8. [Phase 6：Rust 推理引擎验证](#8-phase-6rust-推理引擎验证)
9. [Phase 7：端到端集成与部署](#9-phase-7端到端集成与部署)
10. [Phase 8：性能基准测试](#10-phase-8性能基准测试)
11. [GPU 调度策略](#11-gpu-调度策略)
12. [学习检查清单](#12-学习检查清单)

---

## 1. 资源评估与可行性分析

### 1.1 各模块资源需求汇总

| 模块 | 模型 | 参数量 | 估算 VRAM | 4090 可行性 | 预计时长 |
|------|------|--------|-----------|-------------|----------|
| **LLM 推理 (benchmark 数据生成)** | Qwen3-8B | 8B | ~16 GB (FP16) | ✅ 1 卡 | 6-12h |
| | Qwen3-0.6B | 0.6B | ~1.5 GB | ✅ 1 卡 | 1-2h |
| | Qwen3-32B | 32B | ~64 GB | ⚠️ 4 卡或量化 | 12-24h |
| **分类器 LoRA 微调** | mmBERT-32K | 149M | ~4 GB | ✅ 1 卡 | 每任务 30-60min |
| | BERT-base | 110M | ~3 GB | ✅ 1 卡 | 每任务 45-60min |
| **生成式分类器** | Qwen3-0.6B | 752M | ~8 GB | ✅ 1 卡 | 2-4h |
| **一致性训练** | BERT-base | 110M | ~6 GB (BF16) | ✅ 1 卡 | 2-3h |
| **缓存嵌入 LoRA** | all-MiniLM-L12-v2 | 33M | ~1 GB | ✅ 1 卡 | 10-30min |
| **域适应嵌入** | mmBERT-embed-32K | 110M | ~4 GB | ✅ 1 卡 | 1-2h/轮 × 2轮 |
| **ML 模型选择** | Qwen3 (嵌入) | - | ~2 GB | ✅ 1 卡 | 1-2h |
| | KNN/KMeans/SVM/MLP | <1M | CPU | ✅ CPU | 10-30min |
| **Router-R1 RL** | Phi-3-mini | 3.8B | ~12 GB (FP16) | ✅ 1 卡 | 6-12h |
| **GMTRouter GNN** | all-MiniLM-L6-v2 | 22M | ~1 GB | ✅ 1 卡 | 1-2h |
| **Rust 推理** | 各分类器 | 110-149M | CPU | ✅ CPU | 验证即可 |
| **E2E 部署** | 路由器 + 1个小模型 | - | ~18 GB | ✅ 1 卡 | 配置为主 |

### 1.2 关键结论

```
✅ 完全可行（8×4090 绰绰有余）：
   - 所有分类器 LoRA 微调（6 个任务，每个 <4 GB）
   - 嵌入模型训练（<4 GB）
   - ML 模型选择（CPU）
   - GMTRouter GNN（<1 GB）
   - 缓存嵌入 LoRA（<1 GB）

⚠️ 需要优化（单卡 24GB 有压力）：
   - Router-R1 RL 训练（Phi-3 3.8B，~12 GB，batch_size 需调小）
   - Qwen3-32B 推理（需 4-bit 量化或 4 卡并行）

✅ 数据生成方案：
   - 用 Qwen3-8B 在 1 卡上生成 benchmark 数据
   - 或用 Ollama 本地部署多个小模型
   - API 调用 OpenAI/Anthropic 作为对照

🔄 需要替代方案：
   - 32B+ 模型：用 AWQ/GPTQ 量化后在 1-2 卡运行
   - 或用 Qwen3-8B 替代 32B 作为 benchmark 数据源
```

### 1.3 8 卡分配策略

```
最优分配（并行训练）：
┌─────────────┬─────────────────────────────────────────────┐
│ GPU 0-1     │ LLM 推理服务 (Qwen3-8B) — benchmark 数据生成 │
│ GPU 2-3     │ 分类器 LoRA 微调 (mmBERT/BERT) — 6 个任务    │
│ GPU 4       │ 嵌入训练 (域适应 + 缓存嵌入)                  │
│ GPU 5       │ Router-R1 RL 训练 (Phi-3-mini)              │
│ GPU 6       │ Qwen3-0.6B 生成式分类器                      │
│ GPU 7       │ 备用 / E2E 测试                              │
└─────────────┴─────────────────────────────────────────────┘

串行训练（最大化利用）：
  Week 1: 数据生成 (GPU 0) + 分类器训练 (GPU 1-7)
  Week 2: 嵌入训练 (GPU 0-1) + ML 选择 (CPU) + RL 训练 (GPU 2-3)
  Week 3: E2E 部署 (GPU 0) + 性能测试 (GPU 1-7)
```

---

## 2. 环境准备

### 2.1 基础环境

```bash
# 1. CUDA 环境
# 确认 CUDA 版本 (4090 需要 CUDA 11.8+)
nvidia-smi
nvcc --version

# 2. Python 环境 (训练管线)
conda create -n sr-train python=3.11
conda activate sr-train
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 训练依赖
pip install transformers peft accelerate datasets scikit-learn
pip install sentence-transformers  # 嵌入训练
pip install bitsandbytes  # 量化支持
pip install vllm  # LLM 推理服务
pip install ollama  # 可选：本地模型管理

# 4. Go 环境 (路由器)
wget https://go.dev/dl/go1.22.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.0.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# 5. Rust 环境 (ML 推理)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 6. 项目克隆
cd /data/home/yizhou
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router
```

### 2.2 验证环境

```bash
# 验证 GPU
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_mem/1e9:.1f}GB') for i in range(torch.cuda.device_count())]"

# 验证项目结构
ls src/training/
ls src/semantic-router/pkg/selection/
```

---

## 3. Phase 1：数据准备与 Benchmark 生成

> **目标**：生成 ML 模型选择所需的训练数据
> **GPU 需求**：1-2 卡 (Qwen3-8B 推理)
> **预计时长**：1-2 天

### 3.1 启动 LLM 推理服务

```bash
# 方案 A：用 vLLM 部署 Qwen3-8B
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --port 8000

# 方案 B：用 Ollama 部署多个小模型
ollama serve &
ollama pull qwen3:8b
ollama pull qwen3:0.6b
ollama pull llama3.2:1b
```

### 3.2 生成 Benchmark 数据

```bash
cd src/training/model_selection/ml_model_selection

# 查看 benchmark 配置
cat benchmark.py | head -50

# 运行 benchmark（生成训练数据）
python benchmark.py \
    --config benchmark_config.yaml \
    --concurrency 4 \
    --output benchmark_results.jsonl

# benchmark_config.yaml 示例：
# models:
#   - name: qwen3-8b
#     endpoint: http://localhost:8000/v1
#     api_key: dummy
# datasets:
#   - mmlu_pro
#   - gsm8k
#   - math
```

### 3.3 数据格式验证

```bash
# 检查生成的数据格式
head -5 benchmark_results.jsonl
# 预期格式：{"query", "category", "model_name", "performance", "response_time"}

# 数据统计
python -c "
import json
data = [json.loads(l) for l in open('benchmark_results.jsonl')]
print(f'Total samples: {len(data)}')
print(f'Categories: {set(d[\"category\"] for d in data)}')
print(f'Models: {set(d[\"model_name\"] for d in data)}')
print(f'Avg performance: {sum(d[\"performance\"] for d in data)/len(data):.3f}')
"
```

### 3.4 学习要点

```
[ ] 理解 benchmark 数据的构建方式
[ ] 理解不同评估指标（em_mc, GSM8K, MATH, f1_score）的含义
[ ] 理解为什么需要多模型对比数据
[ ] 理解 API 并发调用的线程池设计
```

---

## 4. Phase 2：分类器 LoRA 微调

> **目标**：复现 6 个分类器的 LoRA 微调
> **GPU 需求**：1-4 卡（每个任务 <4 GB，可并行）
> **预计时长**：3-5 天

### 4.1 任务清单

| 序号 | 任务 | 脚本 | 数据集 | 目标模型 | 预计时长 |
|------|------|------|--------|----------|----------|
| 2.1 | 领域分类 (14类) | `ft_linear_lora.py` | MMLU-Pro | mmBERT-32K | 1h |
| 2.2 | 越狱检测 (2类) | `ft_linear_lora.py` | Adversarial | mmBERT-32K | 1h |
| 2.3 | PII 检测 (35标签) | `ft_linear_lora.py` | Presidio | mmBERT-32K | 1h |
| 2.4 | 事实核查 (2类) | `ft_linear_lora.py` | Fact-check | mmBERT-32K | 1h |
| 2.5 | 用户反馈 (4类) | `ft_linear_lora.py` | Feedback | mmBERT-32K | 1h |
| 2.6 | 一致性训练 | `ft_linear_lora_consistency.py` | MMLU + typo | BERT-base | 3h |
| 2.7 | 生成式分类 | `ft_qwen3_generative_lora.py` | MMLU-Pro | Qwen3-0.6B | 4h |

### 4.2 并行训练方案

```bash
# GPU 2: 领域分类
CUDA_VISIBLE_DEVICES=2 python ft_linear_lora.py \
    --model mmbert-32k \
    --task domain \
    --rank 32 \
    --epochs 3 \
    --batch-size 8 \
    --max-samples 5000 \
    --output-dir ./output/domain

# GPU 3: 越狱检测（同时启动）
CUDA_VISIBLE_DEVICES=3 python ft_linear_lora.py \
    --model mmbert-32k \
    --task jailbreak \
    --rank 32 \
    --epochs 3 \
    --batch-size 8 \
    --max-samples 5000 \
    --output-dir ./output/jailbreak

# GPU 4: PII 检测
CUDA_VISIBLE_DEVICES=4 python ft_linear_lora.py \
    --model mmbert-32k \
    --task pii \
    --rank 32 \
    --epochs 3 \
    --batch-size 8 \
    --max-samples 5000 \
    --output-dir ./output/pii

# ... 依此类推
```

### 4.3 关键参数调优（4090 适配）

```bash
# 如果 OOM，调整以下参数：
# 1. 减小 batch_size: 8 → 4（配合增大 gradient_accumulation_steps: 2 → 4）
# 2. 减小 max_length: 512 → 256
# 3. 使用 bf16 而非 fp32: --bf16
# 4. 使用梯度检查点: --gradient-checkpointing
# 5. 减少 max_samples: 5000 → 2000
```

### 4.4 结果验证

```bash
# 检查训练日志
cat output/domain/training_log.json | python -m json.tool

# 验证模型加载
python -c "
from peft import PeftModel
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('llm-semantic-router/mmbert-32k-yarn')
model = PeftModel.from_pretrained(model, './output/domain/checkpoint-best')
print('Model loaded successfully')
print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
"

# 合并 LoRA 权重（用于 Rust 推理）
python -c "
from peft import PeftModel
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('llm-semantic-router/mmbert-32k-yarn')
model = PeftModel.from_pretrained(model, './output/domain/checkpoint-best')
model = model.merge_and_unload()
model.save_pretrained('./output/domain-merged')
print('Merged model saved')
"
```

### 4.5 学习要点

```
[ ] 理解 LoRA 的低秩分解原理 (W' = W + BA)
[ ] 理解 rank/alpha/dropout 对训练的影响
[ ] 理解为什么选择 attention.query/value 作为目标模块
[ ] 理解合并 LoRA 权重的原因（推理时无额外开销）
[ ] 理解不同任务类型（序列分类 vs token 分类）的区别
[ ] 理解一致性训练的 typo 鲁棒性原理
[ ] 理解生成式分类 vs 判别式分类的 trade-off
```

---

## 5. Phase 3：嵌入模型训练

> **目标**：复现域适应嵌入和缓存嵌入训练
> **GPU 需求**：1-2 卡
> **预计时长**：2-3 天

### 5.1 缓存嵌入 LoRA 训练

```bash
cd src/training/model_embeddings/cache_embeddings

# 训练缓存嵌入（非常快，~10-30 分钟）
CUDA_VISIBLE_DEVICES=4 python lora_trainer.py \
    --base-model sentence-transformers/all-MiniLM-L12-v2 \
    --rank 8 \
    --alpha 32 \
    --batch-size 32 \
    --epochs 1 \
    --output-dir ./output/cache-embeddings

# 验证
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
# 加载 LoRA adapter
# 测试相似度计算
queries = ['What is machine learning?', 'How does ML work?']
embeddings = model.encode(queries)
from sklearn.metrics.pairwise import cosine_similarity
print(f'Similarity: {cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]:.3f}')
"
```

### 5.2 域适应嵌入训练（迭代硬负例挖掘）

```bash
cd src/training/model_embeddings/domain_adapted_embeddings

# 准备数据（需要 MedQuAD 或类似数据集）
# 如果没有现成数据，先下载
python prepare_data.py --dataset medquad --output ./data

# 第 1 轮训练
CUDA_VISIBLE_DEVICES=4 python train.py \
    --model llm-semantic-router/mmbert-embed-32k-2d-matryoshka \
    --data-dir ./data \
    --iterations 1 \
    --epochs 2 \
    --batch-size 8 \
    --lr 5e-5 \
    --margin 0.1 \
    --output-dir ./output/domain-embed-iter1

# 第 2 轮训练（用第 1 轮的模型）
CUDA_VISIBLE_DEVICES=4 python train.py \
    --model ./output/domain-embed-iter1 \
    --data-dir ./data \
    --iterations 1 \
    --epochs 2 \
    --batch-size 8 \
    --lr 5e-5 \
    --margin 0.1 \
    --output-dir ./output/domain-embed-iter2

# 评估
python evaluate.py \
    --model ./output/domain-embed-iter2 \
    --test-data ./data/test_queries.pkl \
    --output eval_results.json
```

### 5.3 学习要点

```
[ ] 理解 MNR (Multiple Negatives Ranking) 损失的原理
[ ] 理解为什么 margin=0.1 而非默认 5.0
[ ] 理解迭代硬负例挖掘的过程和意义
[ ] 理解 easy:hard=2:1 比例的防遗忘作用
[ ] 理解 Matryoshka 嵌入的多粒度检索思想
[ ] 理解域适应 vs 通用嵌入的 trade-off
```

---

## 6. Phase 4：ML 模型选择训练

> **目标**：复现 KNN/KMeans/SVM/MLP 四种模型选择算法
> **GPU 需求**：1 卡（仅嵌入生成）+ CPU（模型训练）
> **预计时长**：1-2 天

### 6.1 生成嵌入特征

```bash
cd src/training/model_selection/ml_model_selection

# 生成 Qwen3 嵌入
CUDA_VISIBLE_DEVICES=4 python embeddings.py \
    --model qwen3 \
    --input benchmark_results.jsonl \
    --output embeddings.jsonl

# 验证嵌入维度
python -c "
import json
data = [json.loads(l) for l in open('embeddings.jsonl')]
print(f'Embedding dim: {len(data[0][\"embedding\"])}')  # 应为 1024
"
```

### 6.2 训练 ML 模型

```bash
# 训练所有模型
python train.py \
    --input embeddings.jsonl \
    --algorithm all \
    --output-dir ./output/ml-models

# 或单独训练
python train.py --input embeddings.jsonl --algorithm knn --output-dir ./output/ml-models
python train.py --input embeddings.jsonl --algorithm kmeans --output-dir ./output/ml-models
python train.py --input embeddings.jsonl --algorithm svm --output-dir ./output/ml-models
python train.py --input embeddings.jsonl --algorithm mlp --output-dir ./output/ml-models

# 检查输出
ls -la ./output/ml-models/
# 预期：knn.json, kmeans.json, svm.json, mlp.json
```

### 6.3 模型验证

```python
# 验证模型加载和推理
import json
import numpy as np

# 加载 KNN 模型
with open('./output/ml-models/knn.json') as f:
    knn_model = json.load(f)

print(f"KNN samples: {len(knn_model['samples'])}")
print(f"KNN classes: {set(s['label'] for s in knn_model['samples'])}")

# 模拟推理
def predict_knn(query_embedding, model, k=5):
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity([query_embedding], [s['embedding'] for s in model['samples']])
    top_k = np.argsort(sims[0])[-k:][::-1]
    # 质量加权投票
    votes = {}
    for idx in top_k:
        label = model['samples'][idx]['label']
        weight = model['samples'][idx].get('quality', 1.0)
        votes[label] = votes.get(label, 0) + weight
    return max(votes, key=votes.get)
```

### 6.4 学习要点

```
[ ] 理解 KNN 质量加权投票的设计
[ ] 理解 KMeans 聚类分配的路由逻辑
[ ] 理解 SVM RBF 核的决策边界
[ ] 理解 MLP 的特征组合能力
[ ] 理解 1038 维特征向量 = 1024 embedding + 14 category one-hot
[ ] 理解为什么模型导出为 JSON（Rust 推理兼容）
```

---

## 7. Phase 5：RL 模型选择训练

> **目标**：复现 Router-R1 和 GMTRouter
> **GPU 需求**：1-2 卡
> **预计时长**：2-3 天

### 7.1 Router-R1 RL 训练

```bash
cd src/training/model_selection/rl_model_selection

# 准备训练数据
# 格式：{"query", "query_type", "candidate_models", "optimal_model", "ground_truth"}
# 可以从 benchmark 数据转换
python -c "
import json
# 将 benchmark 数据转换为 Router-R1 格式
benchmark = [json.loads(l) for l in open('../../ml_model_selection/benchmark_results.jsonl')]
# ... 转换逻辑
"

# 训练 Router-R1
CUDA_VISIBLE_DEVICES=5 python train_router_r1.py \
    --model microsoft/Phi-3-mini-4k-instruct \
    --data router_r1_data.jsonl \
    --batch-size 4 \
    --gradient-accumulation 8 \
    --epochs 10 \
    --lr 1e-5 \
    --max-length 2048 \
    --output-dir ./output/router-r1

# 关键参数说明：
# batch_size=4 (4090 24GB 适配)
# gradient_accumulation=8 (effective batch = 32)
# max_length=2048 (Phi-3 context)
```

### 7.2 GMTRouter 图神经网络训练

```bash
# GMTRouter 训练（轻量级，~1-2 小时）
CUDA_VISIBLE_DEVICES=6 python train_gmtrouter.py \
    --plm sentence-transformers/all-MiniLM-L6-v2 \
    --hidden-dim 256 \
    --gnn-layers 2 \
    --attention-heads 8 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --output-dir ./output/gmtrouter

# 验证
python -c "
import torch
model = torch.load('./output/gmtrouter/model.pt')
print(f'GNN layers: {model[\"gnn_layers\"]}')
print(f'Node types: {model[\"node_types\"]}')
"
```

### 7.3 学习要点

```
[ ] 理解 REINFORCE 策略梯度的原理
[ ] 理解奖励函数设计（format=0.1, outcome=0.7, cost=0.2）
[ ] 理解 Think→Route 模式的可解释性价值
[ ] 理解 PPO clip 的作用（虽然 Router-R1 用 REINFORCE，但可以对比）
[ ] 理解异构图 Transformer (HGT) 的消息传递机制
[ ] 理解 Thompson Sampling 的贝叶斯探索-利用权衡
```

---

## 8. Phase 6：Rust 推理引擎验证

> **目标**：验证 Rust 推理引擎能正确加载和运行训练好的模型
> **GPU 需求**：CPU 即可（Rust 推理默认 CPU）
> **预计时长**：1-2 天

### 8.1 编译 Rust 绑定

```bash
cd candle-binding

# 编译（CUDA 版本）
cargo build --release --features cuda

# 或 CPU 版本
cargo build --release --no-default-features --features mkl

# 编译 Go FFI 测试
cd ..
go test ./src/semantic-router/pkg/classification/ -run TestEmbedding -v
```

### 8.2 模型格式转换

```python
# 将训练好的 LoRA 模型转换为 Rust 可用格式
# 步骤 1：合并 LoRA
from peft import PeftModel
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    'llm-semantic-router/mmbert-32k-yarn',
    num_labels=14
)
model = PeftModel.from_pretrained(model, './output/domain/checkpoint-best')
model = model.merge_and_unload()

# 步骤 2：保存为 safetensors
model.save_pretrained('./output/domain-merged', safe_serialization=True)

# 步骤 3：验证文件
import os
for f in os.listdir('./output/domain-merged'):
    print(f)
# 预期：config.json, model.safetensors, tokenizer.json, ...
```

### 8.3 集成测试

```bash
# 复制模型到 Rust 推理目录
cp -r ./output/domain-merged candle-binding/models/mom-domain

# 运行 Rust 测试
cd candle-binding
cargo test --features cuda

# Go FFI 集成测试
cd ..
go test ./src/semantic-router/pkg/classification/ -v -run TestClassify
```

### 8.4 学习要点

```
[ ] 理解 Rust FFI 的 CGo 绑定机制
[ ] 理解 safetensors 格式的优势（安全、零拷贝）
[ ] 理解 Candle 框架的推理流程
[ ] 理解 CPU vs GPU 推理的性能差异
[ ] 理解 LoRA 热加载的实现
```

---

## 9. Phase 7：端到端集成与部署

> **目标**：完整运行 Semantic Router 的请求处理管线
> **GPU 需求**：1-2 卡（路由器 + 小模型）
> **预计时长**：2-3 天

### 9.1 最小化 E2E 部署

```bash
# 方案 A：本地开发模式
make vllm-sr-dev
vllm-sr serve --image-pull-policy never

# 方案 B：直接 Go 运行
cd src/semantic-router
go run cmd/main.go --config config/config-minimal.yaml
```

### 9.2 最小配置文件

```yaml
# config-minimal.yaml
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 300s

providers:
  models:
    - name: qwen3-8b
      provider: openai
      base_url: http://localhost:8000/v1
      api_key: dummy
      max_tokens: 2048

routing:
  signals:
    embedding_rules:
      - name: math
        candidates:
          - "mathematics"
          - "algebra"
          - "calculus"
        similarity_threshold: 0.8
    keyword_rules:
      - name: code
        keywords: ["python", "code", "programming", "function"]
        method: bm25
  decisions:
    - name: math-route
      priority: 10
      rules:
        type: embedding
        name: math
      modelRefs:
        - model: qwen3-8b
    - name: default
      priority: 0
      modelRefs:
        - model: qwen3-8b
```

### 9.3 发送测试请求

```bash
# 测试路由
curl http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is the derivative of x^2?"}]
  }'

# 测试信号提取
curl http://localhost:8899/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"content": "Write a Python function to sort a list"}'

# 检查路由决策
curl http://localhost:8899/v1/router_replay?limit=10
```

### 9.4 学习要点

```
[ ] 理解 Envoy ExtProc 的工作原理
[ ] 理解信号-决策-算法的完整管线
[ ] 理解配置热更新机制
[ ] 理解请求/响应的完整生命周期
[ ] 理解插件系统（缓存、RAG、记忆）的集成方式
```

---

## 10. Phase 8：性能基准测试

> **目标**：验证系统性能是否满足 SLO
> **GPU 需求**：1-2 卡
> **预计时长**：1-2 天

### 10.1 运行性能测试

```bash
# Go 性能测试
cd perf
make run-benchmarks

# 或手动运行
go test ./benchmarks/ -bench=. -benchmem -count=3

# Python 性能测试
cd bench/hallucination
python evaluate.py \
    --router-url http://localhost:8899 \
    --dataset hallucination_test.jsonl \
    --output results.json
```

### 10.2 SLO 验证

```yaml
# 检查 SLO 阈值 (perf/config/thresholds.yaml)
classification:
  batch_size_1:
    p95_ms: 10
    p99_ms: 15
    throughput_qps: 100
decision_engine:
  p95_ms: 1
  throughput_qps: 10000
cache:
  p95_ms: 5  # 1K entries
  hit_rate: 0.8
e2e:
  sustained_qps: 500
  success_rate: 0.99
  p95_ms: 100
```

### 10.3 学习要点

```
[ ] 理解性能测试的方法论（warmup, iterations, baseline）
[ ] 理解 p50/p95/p99 延迟的含义和用途
[ ] 理解回归检测机制（baseline 对比）
[ ] 理解 CPU/内存/goroutine profiling
```

---

## 11. GPU 调度策略

### 11.1 时间线甘特图

```
Week 1:
Day 1-2:  [GPU 0] vLLM 部署 Qwen3-8B
          [GPU 1-3] 分类器 LoRA 微调 (并行 3 个任务)
          [GPU 4] 缓存嵌入训练
          [GPU 5] GMTRouter 训练
Day 3-4:  [GPU 0] 继续 benchmark 数据生成
          [GPU 1-3] 分类器微调 (剩余 3 个任务)
          [GPU 4] 域适应嵌入训练
          [GPU 5] Router-R1 训练 (开始)
Day 5:    [GPU 0-7] 验证所有训练结果

Week 2:
Day 1-2:  [GPU 0] E2E 部署
          [GPU 1] Router-R1 训练 (继续)
          [GPU 2-7] 性能测试
Day 3-4:  [GPU 0-7] 端到端测试 + 性能优化
Day 5:    文档整理 + 学习总结
```

### 11.2 并行训练注意事项

```bash
# 监控所有 GPU 使用
watch -n 1 nvidia-smi

# 如果某个任务 OOM，调整：
# 1. 减小 batch_size
# 2. 减小 max_length
# 3. 使用 gradient_checkpointing
# 4. 使用 bf16/fp16

# 设置 GPU 可见性
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 或单任务
CUDA_VISIBLE_DEVICES=0 python train.py ...
```

---

## 12. 学习检查清单

### 12.1 算法理解

```
[ ] Elo Rating: Bradley-Terry 模型、K 因子、类别加权
[ ] RouterDC: 双对比学习、InfoNCE 损失、温度参数
[ ] AutoMix: POMDP、粒子滤波、级联路由
[ ] Router-R1: REINFORCE、奖励设计、Think→Route
[ ] GMTRouter: 异构图、HGT、消息传递
[ ] LoRA: 低秩分解、目标模块选择、rank/alpha 调优
[ ] 硬负例挖掘: 迭代挖掘、防遗忘、margin 选择
[ ] HaluGate: 三阶段门控、NLI、跨度识别
```

### 12.2 工程理解

```
[ ] Go + Rust + Python 多语言架构
[ ] CGo FFI 绑定机制
[ ] Envoy ExtProc 协议
[ ] 配置热更新
[ ] HNSW 索引
[ ] 语义缓存
[ ] 性能测试方法论
```

### 12.3 实验能力

```
[ ] 能独立设计消融实验
[ ] 理解评估指标的选择
[ ] 能分析实验结果和局限性
[ ] 能复现论文中的关键结果
[ ] 能提出改进方案并验证
```

---

## 附录：常见问题

### Q1: 4090 训练 OOM 怎么办？

```bash
# 方案 1: 减小 batch_size
--batch-size 4 --gradient-accumulation 4

# 方案 2: 使用 bf16
--bf16

# 方案 3: 梯度检查点
--gradient-checkpointing

# 方案 4: 减少序列长度
--max-length 256

# 方案 5: 使用 QLoRA (4-bit 量化基座)
--load-in-4bit --bnb-4bit-quant-type nf4
```

### Q2: 没有 MedQuAD 数据集怎么办？

```bash
# 替代方案 1: 用 MMLU 子集
# 替代方案 2: 用 SQuAD
# 替代方案 3: 合成数据（用 LLM 生成问答对）
```

### Q3: Rust 编译失败怎么办？

```bash
# 检查 CUDA 版本
nvcc --version

# 使用 CPU 编译
cargo build --release --no-default-features --features mkl

# 或跳过 Rust，用 Python 推理验证
```

### Q4: benchmark 数据生成太慢怎么办？

```bash
# 方案 1: 增加并发
--concurrency 8

# 方案 2: 使用更小的模型
ollama pull qwen3:0.6b

# 方案 3: 减少数据量
--max-samples 1000

# 方案 4: 使用现成的 benchmark 数据集
# HuggingFace: vllm-project/semantic-router-benchmark
```

---

> **最后提醒**：复现的核心目的是 **学习**，不是追求完全一致的结果。重点关注：
> 1. 每个模块 **为什么这么设计**
> 2. 训练过程中 **遇到了什么问题**
> 3. 结果和论文 **有什么差异，为什么**
> 4. 如果要改进 **你会怎么做**
