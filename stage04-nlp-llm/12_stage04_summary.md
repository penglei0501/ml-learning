# 12：stage04 NLP / LLM 总复盘

> 所属阶段：stage04-nlp-llm
> 学习主题：NLP、Transformer、LLM、Embedding、RAG 前置知识总复盘
> 当前状态：stage04 总结
> 前置基础：stage03 PyTorch 与深度学习基础

---

## 0. 本阶段目录

```text
stage04-nlp-llm/
├── 01_text_to_number.ipynb        
├── 02_embedding.ipynb             
├── 03_text_classification.ipynb   
├── 04_attention_basic.ipynb       
├── 05_transformer_basic.ipynb     
├── 06_gpt_bert_basic.md           
├── 06_gpt_bert_basic.ipynb        
├── 07_prompt_basic.md             
├── 08_tokenizer_basic.ipynb       
├── 09_pretrained_model_basic.ipynb 
├── 10_sentence_embedding.ipynb    
├── 11_vector_similarity.ipynb     
└── 12_stage04_summary.md          
```

---

# 1. stage04 学习目标

这个阶段的核心目标不是直接造一个大模型，而是理解现代 NLP / LLM 的基础链路。

完整链路是：

```text
自然语言文本
    ↓
文本数字化
    ↓
Token / Tokenizer
    ↓
Embedding
    ↓
Attention
    ↓
Transformer
    ↓
BERT / GPT
    ↓
预训练模型调用
    ↓
Sentence Embedding
    ↓
Vector Similarity
    ↓
RAG / Agent 记忆 / 知识库问答
```

也就是说，stage04 解决的问题是：

> 文本如何被模型理解，以及大模型应用的底层技术是怎么一步步搭起来的。

---

# 2. 01：Text to Number

## 2.1 学了什么？

本节主要学习：

```text
文本不能直接进入模型；
模型只能处理数字；
所以必须先把文本转换成数字。
```

核心内容包括：

* 文本切分；
* 词表 vocabulary；
* one-hot 编码；
* 词袋模型 Bag of Words；
* TF-IDF；
* 文本向量化的基本思想。

---

## 2.2 核心理解

机器学习模型不能直接理解：

```text
我喜欢机器学习
```

它真正接收的是类似这样的数字：

```text
[0, 1, 0, 3, 2, ...]
```

或者：

```text
[0.12, 0.38, 0.00, 0.91, ...]
```

所以 NLP 的第一步永远是：

```text
文本 → 数字
```

---

## 2.3 关键词

```text
文本数字化
词表
one-hot
Bag of Words
TF-IDF
特征向量
```

---

# 3. 02：Embedding

## 3.1 学了什么？

本节学的是：

> 不再用稀疏的 one-hot，而是用低维连续向量表示词语。

例如：

```text
机器学习 → [0.21, -0.35, 0.72, ...]
人工智能 → [0.24, -0.31, 0.69, ...]
火锅     → [-0.48, 0.12, 0.05, ...]
```

如果两个词语语义接近，它们的向量也应该更接近。

---

## 3.2 one-hot 和 embedding 的区别

| 表示方式      | 特点        | 问题            |
| --------- | --------- | ------------- |
| one-hot   | 简单、直观     | 维度高、稀疏、无法表达语义 |
| embedding | 低维、稠密、可学习 | 需要训练或使用预训练模型  |

---

## 3.3 核心理解

Embedding 的本质是：

> 把离散的词语压缩成连续向量，让模型可以用数学方式处理语义。

---

## 3.4 关键词

```text
Embedding
词向量
稠密向量
语义空间
向量距离
```

---

# 4. 03：Text Classification

## 4.1 学了什么？

本节学的是文本分类。

文本分类任务就是：

```text
输入一段文本
    ↓
模型判断它属于哪个类别
```

例如：

```text
“这个电影太好看了” → 正面
“这个产品太差了”   → 负面
```

---

## 4.2 文本分类基本流程

```text
文本数据
    ↓
文本向量化
    ↓
训练分类模型
    ↓
预测类别
    ↓
评估准确率
```

常见模型包括：

* 朴素贝叶斯；
* 逻辑回归；
* SVM；
* 神经网络；
* BERT 分类模型。

---

## 4.3 核心理解

文本分类本质上还是监督学习：

```text
输入 X：文本特征
输出 y：类别标签
```

只不过这里的 X 原本是自然语言，需要先转成数字向量。

---

## 4.4 关键词

```text
文本分类
情感分析
监督学习
标签
分类器
accuracy
```

---

# 5. 04：Attention Basic

## 5.1 学了什么？

本节学的是 Attention 注意力机制。

Attention 解决的问题是：

> 模型在处理一句话时，应该重点关注哪些词？

例如：

```text
我昨天买的手机很好用，它的屏幕非常清晰。
```

当模型看到“它”时，应该知道“它”主要指的是“手机”。

---

## 5.2 Attention 的直觉

Attention 可以理解为：

```text
当前词
    ↓
去看句子中的其他词
    ↓
判断哪些词更重要
    ↓
给重要词更高权重
```

也就是说：

```text
不是所有词都同等重要。
```

---

## 5.3 Q、K、V 是什么？

Attention 里最重要的是三个向量：

```text
Q：Query，查询
K：Key，键
V：Value，值
```

可以直观理解为：

```text
Q：我现在想找什么信息？
K：每个词能提供什么线索？
V：每个词真正携带的信息是什么？
```

计算过程：

```text
Q 和 K 做相似度计算
    ↓
得到注意力分数
    ↓
softmax 变成权重
    ↓
用权重加权 V
    ↓
得到新的表示
```

---

## 5.4 Attention 公式

经典公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

含义：

```text
QK^T：计算 Query 和 Key 的相似度
sqrt(d_k)：缩放，防止数值过大
softmax：转成注意力权重
乘以 V：根据权重汇总信息
```

---

## 5.5 关键词

```text
Attention
Self-Attention
Q
K
V
softmax
注意力权重
上下文表示
```

---

# 6. 05：Transformer Basic

## 6.1 学了什么？

本节学的是 Transformer。

Transformer 是现代大模型的核心结构。

BERT、GPT、T5、LLaMA、Qwen、DeepSeek 等模型，底层都和 Transformer 密切相关。

---

## 6.2 Transformer 为什么重要？

以前处理序列常用：

```text
RNN
LSTM
GRU
```

它们有一个问题：

```text
按顺序处理文本，难以并行，长距离依赖处理困难。
```

Transformer 的核心优势：

```text
用 Self-Attention 同时建模所有位置之间的关系。
```

---

## 6.3 Transformer 基本结构

Transformer 原始结构包含：

```text
Encoder
Decoder
```

Encoder 适合理解文本；

Decoder 适合生成文本。

---

## 6.4 Encoder 和 Decoder 区别

| 结构              | 主要作用        | 代表模型 |
| --------------- | ----------- | ---- |
| Encoder         | 理解输入文本      | BERT |
| Decoder         | 自回归生成文本     | GPT  |
| Encoder-Decoder | 输入理解 + 输出生成 | T5   |

---

## 6.5 Transformer 核心组件

Transformer 主要包括：

```text
Token Embedding
Position Embedding
Multi-Head Self-Attention
Feed Forward Network
Residual Connection
Layer Normalization
```

---

## 6.6 核心理解

Transformer 的本质是：

> 用注意力机制让文本中每个位置都能直接关注其他位置，从而获得上下文语义表示。

---

## 6.7 关键词

```text
Transformer
Encoder
Decoder
Self-Attention
Multi-Head Attention
Position Embedding
Feed Forward
Residual Connection
LayerNorm
```

---

# 7. 06：GPT / BERT Basic

## 7.1 学了什么？

本节学的是两个经典大模型方向：

```text
BERT
GPT
```

它们都基于 Transformer，但结构和任务目标不同。

---

## 7.2 BERT 是什么？

BERT 是：

```text
基于 Transformer Encoder 的预训练语言模型。
```

BERT 更适合：

```text
文本理解任务
```

例如：

* 文本分类；
* 情感分析；
* 命名实体识别；
* 文本匹配；
* 阅读理解。

---

## 7.3 GPT 是什么？

GPT 是：

```text
基于 Transformer Decoder 的生成式语言模型。
```

GPT 更适合：

```text
文本生成任务
```

例如：

* 对话；
* 写作；
* 代码生成；
* 摘要；
* 问答；
* Agent 规划。

---

## 7.4 BERT 和 GPT 的区别

| 对比项  | BERT                | GPT                  |
| ---- | ------------------- | -------------------- |
| 结构   | Transformer Encoder | Transformer Decoder  |
| 目标   | 理解文本                | 生成文本                 |
| 注意力  | 双向注意力               | 单向因果注意力              |
| 典型任务 | 分类、匹配、抽取            | 对话、写作、生成             |
| 输出方式 | 通常输出标签或表示           | 一个 token 一个 token 生成 |

---

## 7.5 核心理解

BERT 更像：

```text
阅读理解型模型
```

GPT 更像：

```text
语言生成型模型
```

现在的 ChatGPT、DeepSeek、Qwen、Claude 等大模型，大多属于 GPT 这一类生成式模型思想的延伸。

---

## 7.6 关键词

```text
BERT
GPT
Encoder-only
Decoder-only
Masked Language Modeling
Causal Language Modeling
预训练
微调
```

---

# 8. 07：Prompt Basic

## 8.1 学了什么？

本节学的是 Prompt。

Prompt 是：

> 用户给大模型的输入指令。

例如：

```text
请你用通俗语言解释什么是 Transformer。
```

这就是一个 prompt。

---

## 8.2 Prompt 为什么重要？

因为大模型不是传统意义上的固定函数调用。

同一个模型，不同提示词，输出可能完全不同。

所以 Prompt 的作用是：

```text
约束任务
提供上下文
规定格式
指定角色
给出示例
降低歧义
```

---

## 8.3 常见 Prompt 结构

一个比较好的 Prompt 通常包括：

```text
角色
任务
背景
约束
输出格式
示例
```

例如：

```text
你是一名机器学习老师。
请用研一学生能理解的方式解释 Attention。
要求：
1. 先讲直觉；
2. 再讲公式；
3. 最后给一个例子。
```

---

## 8.4 Prompt 和 Agent 的关系

Agent 本质上也是基于 Prompt 组织起来的复杂流程。

例如：

```text
任务规划 prompt
工具调用 prompt
反思 prompt
总结 prompt
记忆检索 prompt
```

所以 Prompt 是 Agent 的基础能力之一。

---

## 8.5 关键词

```text
Prompt
Prompt Engineering
角色设定
上下文
输出格式
Few-shot
Chain-of-Thought
Agent Prompt
```

---

# 9. 08：Tokenizer Basic

## 9.1 学了什么？

本节学的是 Tokenizer。

Tokenizer 的作用是：

```text
把自然语言文本切成 token，并转换成 token id。
```

例如：

```text
我喜欢机器学习
    ↓
["我", "喜欢", "机器", "学习"]
    ↓
[101, 2769, 1599, 3322, 102]
```

---

## 9.2 Token 是什么？

Token 可以是：

* 一个字；
* 一个词；
* 一个子词；
* 一个符号；
* 一段英文单词的一部分。

例如英文：

```text
unbelievable
```

可能被切成：

```text
un
##believ
##able
```

---

## 9.3 Tokenizer 输出什么？

常见输出包括：

```text
input_ids
attention_mask
token_type_ids
```

其中：

```text
input_ids：token 对应的数字编号
attention_mask：哪些是真实 token，哪些是 padding
token_type_ids：区分句子 A 和句子 B，常见于 BERT
```

---

## 9.4 Tokenizer 和模型的关系

模型不能直接读文本。

真正进入模型的是：

```text
input_ids
attention_mask
```

所以完整流程是：

```text
文本
    ↓
Tokenizer
    ↓
input_ids / attention_mask
    ↓
Model
    ↓
logits / embedding / generated tokens
```

---

## 9.5 关键词

```text
Tokenizer
Token
input_ids
attention_mask
padding
truncation
vocabulary
subword
BPE
WordPiece
```

---

# 10. 09：Pretrained Model Basic

## 10.1 学了什么？

本节学的是预训练模型。

预训练模型是：

> 已经在大量文本上训练过的模型，我们可以直接加载使用。

例如：

```text
bert-base-chinese
distilbert-base-uncased
gpt2
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

---

## 10.2 为什么不用从零训练？

从零训练大模型需要：

```text
大量数据
大量 GPU
大量时间
大量工程经验
```

普通学习者和应用开发者通常不会从零训练大模型。

更常见的方式是：

```text
直接调用预训练模型
或
在预训练模型基础上微调
```

---

## 10.3 pipeline 是什么？

`pipeline` 是一种高层封装。

它帮我们自动完成：

```text
加载 tokenizer
加载 model
文本预处理
模型推理
结果后处理
```

例如：

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I love machine learning.")
```

---

## 10.4 手动加载模型流程

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

完整流程：

```text
文本
    ↓
tokenizer
    ↓
input_ids / attention_mask
    ↓
model
    ↓
logits
    ↓
softmax
    ↓
预测类别
```

---

## 10.5 logits 是什么？

logits 是模型最后输出的原始分数。

它还不是概率。

需要经过 softmax：

```text
logits → softmax → probability
```

例如：

```text
logits: [1.2, 4.8]
probability: [0.026, 0.974]
```

说明模型更倾向第二个类别。

---

## 10.6 预训练模型和微调模型区别

| 类型    | 含义           | 举例                |
| ----- | ------------ | ----------------- |
| 预训练模型 | 在大规模通用数据上训练  | bert-base-chinese |
| 微调模型  | 在具体任务数据上继续训练 | 中文情感分类 BERT       |

核心理解：

```text
预训练模型提供基础语言能力；
微调模型提供具体任务能力。
```

---

## 10.7 关键词

```text
Pretrained Model
Fine-tuning
pipeline
AutoTokenizer
AutoModel
logits
softmax
inference
```

---

# 11. 10：Sentence Embedding

## 11.1 学了什么？

本节学的是 Sentence Embedding。

Sentence Embedding 是：

> 把一句话变成一个向量。

例如：

```text
我正在学习机器学习。
    ↓
[0.21, -0.14, 0.38, ..., 0.09]
```

这个向量表达的是整句话的语义。

---

## 11.2 Word Embedding 和 Sentence Embedding 区别

| 类型                 | 表示对象    | 例子             |
| ------------------ | ------- | -------------- |
| Word Embedding     | 词       | 机器学习 → 向量      |
| Sentence Embedding | 句子 / 段落 | 我正在学习机器学习 → 向量 |

---

## 11.3 Sentence Embedding 有什么用？

它可以用于：

```text
句子相似度
语义搜索
文本聚类
推荐系统
RAG 检索
Agent 记忆
知识库问答
```

---

## 11.4 基本代码

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

sentence = "我正在学习机器学习。"
embedding = model.encode(sentence)

print(embedding.shape)
```

---

## 11.5 核心理解

Sentence Embedding 的本质是：

> 把文本语义压缩到一个固定长度的向量中。

---

## 11.6 关键词

```text
Sentence Embedding
SentenceTransformer
model.encode
semantic representation
dense vector
语义向量
```

---

# 12. 11：Vector Similarity

## 12.1 学了什么？

本节学的是向量相似度。

有了句子向量以后，我们还要判断：

```text
两个句子向量有多接近？
```

这就是向量相似度。

---

## 12.2 常见相似度方法

常见方法包括：

```text
余弦相似度
欧氏距离
点积
```

---

## 12.3 余弦相似度

余弦相似度关注的是：

```text
两个向量方向是否接近。
```

数值越大，越相似。

常用于文本语义相似度和 RAG 检索。

---

## 12.4 欧氏距离

欧氏距离关注的是：

```text
两个向量在空间中的距离。
```

距离越小，越相似。

---

## 12.5 点积

点积也是一种相似度计算方式。

Transformer 的 Attention 中也大量使用点积。

例如：

```text
QK^T
```

就是 Query 和 Key 之间的点积相似度计算。

---

## 12.6 Top-K 检索

Top-K 检索是：

> 从所有文档中找出和用户问题最相似的前 K 个。

流程：

```text
用户问题
    ↓
问题向量
    ↓
和所有文档向量计算相似度
    ↓
按相似度排序
    ↓
返回前 K 个文档
```

---

## 12.7 threshold 阈值

Top-K 会强行返回前 K 个结果。

但是有时用户问题和知识库完全无关。

所以可以设置阈值：

```text
只有相似度超过 threshold 的结果才返回。
```

---

## 12.8 关键词

```text
cosine similarity
euclidean distance
dot product
Top-K
threshold
vector search
vector database
semantic search
```

---

# 13. stage04 总体知识链路

现在可以把整个 stage04 串成一条完整路线：

```text
1. 文本不能直接进入模型
    ↓
2. 文本要先变成数字
    ↓
3. one-hot / TF-IDF 是早期文本表示方法
    ↓
4. embedding 可以表达语义
    ↓
5. attention 让模型知道重点关注哪些词
    ↓
6. transformer 用 self-attention 建模上下文
    ↓
7. BERT 用 encoder 理解文本
    ↓
8. GPT 用 decoder 生成文本
    ↓
9. prompt 用自然语言控制大模型
    ↓
10. tokenizer 把文本切成 token id
    ↓
11. pretrained model 可以直接调用
    ↓
12. sentence embedding 把句子变成语义向量
    ↓
13. vector similarity 判断语义是否接近
    ↓
14. RAG 和 Agent 记忆基于 embedding + 检索实现
```

---

# 14. NLP、LLM、RAG、Agent 的关系

## 14.1 NLP 是什么？

NLP 是 Natural Language Processing，自然语言处理。

它研究的是：

```text
如何让计算机处理人类语言。
```

任务包括：

```text
文本分类
情感分析
机器翻译
命名实体识别
文本摘要
问答系统
信息抽取
文本生成
```

---

## 14.2 LLM 是什么？

LLM 是 Large Language Model，大语言模型。

它是 NLP 发展到大规模预训练阶段后的代表。

LLM 具有：

```text
文本理解能力
文本生成能力
上下文学习能力
指令跟随能力
一定的推理能力
工具调用潜力
```

---

## 14.3 RAG 是什么？

RAG 是 Retrieval-Augmented Generation，检索增强生成。

它解决的问题是：

```text
大模型自身知识可能过时、不完整、容易幻觉。
```

RAG 的做法是：

```text
先检索外部知识
再让大模型基于检索内容生成答案
```

核心流程：

```text
文档
    ↓
切分 chunk
    ↓
embedding
    ↓
存入向量库

用户问题
    ↓
embedding
    ↓
向量检索
    ↓
Top-K 文档
    ↓
拼接 prompt
    ↓
LLM 生成答案
```

---

## 14.4 Agent 是什么？

AI Agent 可以理解为：

> 能够围绕目标进行规划、调用工具、使用记忆，并根据反馈调整行为的 AI 系统。

Agent 通常包含：

```text
LLM
Prompt
Memory
Tools
Planning
Reflection
Workflow
```

---

## 14.5 它们之间的关系

```text
NLP 是大领域
    ↓
Transformer 是核心模型结构
    ↓
LLM 是基于 Transformer 的大规模语言模型
    ↓
RAG 是增强 LLM 知识能力的方法
    ↓
Agent 是基于 LLM + 工具 + 记忆 + 规划构建的应用形态
```

---

# 15. stage04 中最重要的几个概念

## 15.1 Token

Token 是模型处理文本的基本单位。

```text
文本 → token → token id → 模型输入
```

---

## 15.2 Embedding

Embedding 是把离散对象变成连续向量。

```text
token id → embedding vector
```

---

## 15.3 Attention

Attention 让模型知道：

```text
当前 token 应该关注哪些其他 token。
```

---

## 15.4 Transformer

Transformer 是现代大模型的核心架构。

```text
Embedding + Attention + FFN + 残差连接 + LayerNorm
```

---

## 15.5 Pretrained Model

预训练模型是已经在大量数据上训练过的模型。

```text
不用从零训练，直接加载使用。
```

---

## 15.6 Sentence Embedding

Sentence Embedding 把一句话变成向量。

```text
句子 → 语义向量
```

---

## 15.7 Vector Similarity

Vector Similarity 判断两个向量是否相似。

```text
语义是否接近 → 数学距离
```

---

## 15.8 RAG

RAG 用检索增强大模型回答。

```text
外部知识库 + 向量检索 + LLM 生成
```

---

## 15.9 Agent Memory

Agent 记忆可以通过 embedding + vector search 实现。

```text
历史信息 → 向量 → 检索 → 辅助回答
```

---

# 16. 一个完整 RAG 小系统的底层流程

假设我们要做一个“机器学习知识库问答助手”。

## 16.1 离线阶段

```text
准备文档
    ↓
切分成 chunk
    ↓
用 embedding 模型转向量
    ↓
存入向量数据库
```

示例：

```python
documents = [
    "机器学习是一种让计算机从数据中学习规律的方法。",
    "Transformer 是一种基于自注意力机制的模型结构。",
    "RAG 可以让大模型结合外部知识库回答问题。"
]
```

---

## 16.2 在线阶段

```text
用户提问
    ↓
问题转向量
    ↓
向量检索 Top-K
    ↓
取回相关文档
    ↓
拼接 Prompt
    ↓
LLM 生成回答
```

示例 Prompt：

```text
请根据以下资料回答用户问题。

资料：
{retrieved_documents}

用户问题：
{query}

要求：
1. 只能根据资料回答；
2. 如果资料中没有答案，请说不知道；
3. 回答要简洁清楚。
```

---

## 16.3 这和 stage04 的对应关系

| RAG 步骤    | stage04 对应知识  |
| --------- | ------------- |
| 文档文本处理    | 01         |
| 文本转向量     | 02 / 10 |
| 模型理解文本    | 04 / 05 |
| 使用预训练模型   | 09         |
| 问题和文档相似度  | 11         |
| prompt 拼接 | 07         |
| LLM 生成答案  | 06         |

---

# 17. 一个 Agent 的底层流程

一个简单 Agent 可以这样理解：

```text
用户提出目标
    ↓
LLM 理解目标
    ↓
生成计划
    ↓
判断是否需要调用工具
    ↓
调用工具
    ↓
观察工具结果
    ↓
继续思考
    ↓
输出最终答案
```

---

## 17.1 Agent 里面用到了哪些 stage04 知识？

| Agent 能力 | stage04 对应知识                           |
| -------- | -------------------------------------- |
| 理解用户输入   | Tokenizer / Transformer / LLM          |
| 生成计划     | GPT / Prompt                           |
| 调用工具     | Prompt / Function Calling 思想           |
| 使用记忆     | Sentence Embedding / Vector Similarity |
| 检索知识     | RAG / Vector Search                    |
| 总结结果     | LLM 生成                                 |

---

# 18. 重点概念对比表

## 18.1 Tokenizer 和 Embedding

| 概念        | 作用              |
| --------- | --------------- |
| Tokenizer | 把文本切成 token id  |
| Embedding | 把 token id 变成向量 |

流程：

```text
文本 → Tokenizer → token ids → Embedding → vectors
```

---

## 18.2 Attention 和 Transformer

| 概念          | 作用                    |
| ----------- | --------------------- |
| Attention   | 计算 token 之间的重要性       |
| Transformer | 基于 Attention 搭建完整模型结构 |

---

## 18.3 BERT 和 GPT

| 概念   | 结构           | 适合任务 |
| ---- | ------------ | ---- |
| BERT | Encoder-only | 文本理解 |
| GPT  | Decoder-only | 文本生成 |

---

## 18.4 Embedding 和 Sentence Embedding

| 概念                 | 表示对象    |
| ------------------ | ------- |
| Word Embedding     | 词       |
| Token Embedding    | token   |
| Sentence Embedding | 句子 / 段落 |

---

## 18.5 关键词检索和向量检索

| 检索方式  | 特点      |
| ----- | ------- |
| 关键词检索 | 看字面是否匹配 |
| 向量检索  | 看语义是否接近 |

---

# 19. stage04 代码能力总结

学完 stage04 后，应该具备以下代码能力：

## 19.1 会做基础文本向量化

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
```

能够理解：

```text
文本 → 稀疏向量
```

---

## 19.2 会使用 embedding

能够理解：

```text
词 / 句子 → 稠密向量
```

---

## 19.3 会调用 Hugging Face 模型

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I love machine learning.")
```

---

## 19.4 会手动加载 tokenizer 和 model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

---

## 19.5 会使用 SentenceTransformer

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embedding = model.encode("我正在学习机器学习。")
```

---

## 19.6 会计算余弦相似度

```python
from sklearn.metrics.pairwise import cosine_similarity

score = cosine_similarity(emb1, emb2)
```

---

## 19.7 会写简单 Top-K 检索

```python
import numpy as np

sorted_indices = np.argsort(scores)[::-1]
top_k_indices = sorted_indices[:3]
```

---

## 19.8 会写简单向量检索器

```python
class SimpleVectorStore:
    def __init__(self, documents, model):
        self.documents = documents
        self.model = model
        self.embeddings = model.encode(documents)

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        sorted_indices = np.argsort(scores)[::-1]

        results = []

        for idx in sorted_indices[:top_k]:
            results.append({
                "document": self.documents[idx],
                "score": scores[idx]
            })

        return results
```

---

# 20. stage04 常见误区

## 20.1 误区一：模型能直接理解文字

错误理解：

```text
模型直接读中文、英文。
```

正确理解：

```text
模型只能处理数字。
文本必须先经过 tokenizer 变成 token id。
```

---

## 20.2 误区二：Tokenizer 等于 Embedding

错误理解：

```text
Tokenizer 就是 embedding。
```

正确理解：

```text
Tokenizer：文本 → token id
Embedding：token id → 向量
```

---

## 20.3 误区三：Attention 就是看哪个词重要

这个说法有一定道理，但不完整。

更准确地说：

```text
Attention 是根据 Q 和 K 的相似度，计算不同 token 对当前 token 表示的贡献权重。
```

---

## 20.4 误区四：BERT 和 GPT 是一回事

错误理解：

```text
BERT 和 GPT 都是大模型，所以差不多。
```

正确理解：

```text
BERT 偏理解；
GPT 偏生成。
```

---

## 20.5 误区五：Embedding 相似就一定语义完全相同

错误理解：

```text
向量相似度高，就说明两个句子完全一样。
```

正确理解：

```text
向量相似度只是模型估计的语义接近程度，不是绝对真理。
```

---

## 20.6 误区六：RAG 就是简单搜索

错误理解：

```text
RAG = 搜索一下资料。
```

正确理解：

```text
RAG = 文档处理 + embedding + 向量检索 + prompt 拼接 + LLM 生成 + 结果评估。
```

---

# 21. 自测问题

学完 stage04 后，应该能回答下面这些问题。

## 21.1 基础 NLP

```text
1. 为什么文本需要转成数字？
2. one-hot 有什么缺点？
3. TF-IDF 的基本思想是什么？
4. embedding 为什么比 one-hot 更适合表示语义？
5. 文本分类的基本流程是什么？
```

---

## 21.2 Attention / Transformer

```text
1. Attention 解决什么问题？
2. Q、K、V 分别是什么意思？
3. Self-Attention 和普通 Attention 有什么区别？
4. Transformer 由哪些核心模块组成？
5. 为什么 Transformer 比 RNN / LSTM 更适合大规模并行训练？
```

---

## 21.3 BERT / GPT / LLM

```text
1. BERT 的结构是什么？
2. GPT 的结构是什么？
3. BERT 和 GPT 有什么区别？
4. 什么是预训练？
5. 什么是微调？
6. 为什么现在很少从零训练大模型？
```

---

## 21.4 Tokenizer / Pretrained Model

```text
1. Token 是什么？
2. Tokenizer 的作用是什么？
3. input_ids 是什么？
4. attention_mask 是什么？
5. pipeline 有什么用？
6. logits 和 probability 有什么区别？
```

---

## 21.5 Embedding / Vector Search / RAG

```text
1. Sentence Embedding 是什么？
2. 余弦相似度是什么？
3. 欧氏距离和余弦相似度有什么区别？
4. Top-K 检索是什么意思？
5. threshold 有什么作用？
6. 向量数据库解决什么问题？
7. RAG 的检索阶段是怎么工作的？
8. Agent 记忆为什么可以用向量检索实现？
```

---

# 22. stage04 最重要的一句话

```text
stage04 的核心是理解：自然语言如何一步步变成模型可以处理的向量，并进一步被 Transformer / LLM 用来完成理解、生成、检索和 Agent 记忆。
```

---

# 23. 进入下一阶段前，我应该掌握什么？

进入 stage05 或 stage07 之前，至少应该掌握：

```text
1. 文本必须先变成数字；
2. Tokenizer 负责切 token 和转 id；
3. Embedding 负责把 id 变成向量；
4. Attention 负责计算 token 之间的关系；
5. Transformer 是现代 LLM 的核心结构；
6. BERT 偏理解，GPT 偏生成；
7. Prompt 是控制大模型输出的重要方式；
8. 预训练模型可以直接调用；
9. Sentence Embedding 可以把句子变成语义向量；
10. Vector Similarity 可以做语义检索；
11. RAG = 检索 + 生成；
12. Agent = LLM + 工具 + 记忆 + 规划。
```

---

# 24. stage04 和后续阶段的关系

## 24.1 和 RAG 的关系

stage04 是 RAG 的前置基础。

因为 RAG 需要：

```text
文本切分
Embedding
向量检索
Prompt
LLM 生成
```

这些内容在 stage04 已经全部接触到了。

---

## 24.2 和 Agent 的关系

stage04 也是 Agent 的前置基础。

因为 Agent 需要：

```text
LLM 理解用户意图
Prompt 组织任务
工具调用
Memory 记忆
RAG 检索
多轮规划
```

其中 LLM、Prompt、Embedding、Vector Search 都来自 stage04。

---

## 24.3 和深度学习的关系

stage03 学的是：

```text
神经网络怎么训练
PyTorch 怎么写模型
CNN / RNN / LSTM 等基础结构
```

stage04 学的是：

```text
文本怎么进模型
Transformer 怎么工作
大模型怎么调用
RAG / Agent 的底层基础
```

两者关系：

```text
stage03：深度学习通用基础
stage04：NLP / LLM 专门方向
```

---

# 25. 我的学习路线总结

到这里，我已经完成：

```text
stage02：传统机器学习
stage03：PyTorch 与深度学习基础
stage04：NLP / Transformer / LLM 基础
```

目前能力从：

```text
传统机器学习建模
```

逐步扩展到了：

```text
深度学习
NLP
Transformer
LLM
Embedding
RAG 前置知识
Agent 前置知识
```

---

# 26. 下一步建议

学完 stage04 后，建议下一步进入：

```text
stage05-time-series-deep-learning
```

或者：

```text
stage07-rag-agent
```

如果目标是继续打基础：

```text
先学 stage05：时间序列深度学习
再学 stage06：图学习
最后学 stage07：RAG / Agent
```

如果目标是尽快进入 AI Agent 应用：

```text
可以先进入 stage07：RAG / Agent
后面再补 stage05 和 stage06
```

---

# 27. 本阶段最终总结

```text
stage04 是我从传统机器学习和普通深度学习，正式进入 NLP、LLM、RAG 和 Agent 的关键阶段。

我已经理解了文本如何变成数字，tokenizer 如何切分文本，embedding 如何表达语义，attention 如何建模上下文，transformer 如何成为大模型的基础架构。

同时，我也理解了 BERT 和 GPT 的区别，知道了如何调用预训练模型，如何把句子转成向量，如何计算向量相似度，并用 Top-K 检索模拟 RAG 的核心流程。

这为后续学习 RAG、AI Agent、向量数据库、工具调用、多智能体系统打下了基础。
```

---

# 28. 一句话收尾

```text
NLP / LLM 的底层主线就是：文本 → token → 向量 → 注意力 → Transformer → 预训练模型 → 语义检索 → RAG / Agent。
```
