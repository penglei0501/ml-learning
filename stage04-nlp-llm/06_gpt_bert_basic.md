# Day06：GPT 与 BERT 基础

> 所属阶段：Stage04 - NLP / Transformer / LLM 基础
> 当前主题：从 Transformer 过渡到预训练语言模型
> 学习目标：理解 GPT 与 BERT 的核心思想、结构差异、训练任务和应用场景

---

## 1. 本节学习目标

学完本节后，需要掌握：

1. GPT 和 BERT 都是基于 Transformer 的预训练语言模型。
2. GPT 更偏向文本生成，BERT 更偏向文本理解。
3. GPT 主要使用 Transformer Decoder。
4. BERT 主要使用 Transformer Encoder。
5. GPT 的核心训练任务是预测下一个 token。
6. BERT 的核心训练任务是预测被遮住的 token。
7. 理解 Causal Mask 和 Masked Language Model 的作用。
8. 知道 GPT、BERT 在实际任务中的典型应用场景。

---

## 2. 前置知识回顾

在进入 GPT 和 BERT 之前，前面已经学习了：

```text
01_text_to_number.ipynb           文本如何转成数字
02_embedding.ipynb                词向量 / Embedding
03_text_classification.ipynb      文本分类
04_attention_basic.ipynb          Attention 注意力机制
05_transformer_basic.ipynb        Transformer 基础结构
```

这些内容可以串成一条线：

```text
文本
↓
数字化
↓
Embedding
↓
Attention
↓
Transformer
↓
GPT / BERT
↓
LLM / RAG / Agent
```

GPT 和 BERT 可以看作是 Transformer 在 NLP 领域的两个经典发展方向。

---

## 3. GPT 和 BERT 是什么？

GPT 和 BERT 都属于：

```text
预训练语言模型
Pre-trained Language Model
```

它们都基于 Transformer，但是使用 Transformer 的方式不同。

Transformer 原始结构包括两个核心部分：

```text
Transformer
├── Encoder：更适合理解输入文本
└── Decoder：更适合生成输出文本
```

因此可以先这样记：

```text
BERT ≈ Transformer Encoder
GPT  ≈ Transformer Decoder
```

更直观地说：

```text
BERT：偏理解
GPT ：偏生成
```

---

## 4. GPT 的核心思想

GPT 的核心任务是：

> 根据前面的内容，预测下一个 token。

例如：

```text
输入：我 今天 想 去
预测：图书馆 / 学校 / 吃饭 / 广州塔
```

也就是说，GPT 是一步一步往后生成文本的。

例如完整生成过程可以理解为：

```text
输入：我
预测：今天

输入：我 今天
预测：想

输入：我 今天 想
预测：去

输入：我 今天 想 去
预测：图书馆
```

这类模型叫：

```text
自回归语言模型
Autoregressive Language Model
```

---

## 5. GPT 为什么适合生成？

因为 GPT 训练时学的就是：

```text
给定前文，预测后文
```

所以它天然适合做生成类任务，例如：

```text
聊天对话
文章续写
摘要生成
代码生成
问答系统
翻译
写作助手
AI Agent
```

现在很多主流大语言模型，例如 ChatGPT、Claude、DeepSeek、Qwen 等，都更接近 GPT 这一类 Decoder-only 路线。

---

## 6. GPT 的结构直觉

GPT 使用的是 Transformer Decoder。

它有一个非常重要的特点：

> 当前 token 只能看到它前面的 token，不能看到后面的 token。

例如句子：

```text
我 今天 想 去 图书馆
```

当模型预测“去”的时候，它只能看到：

```text
我 今天 想
```

它不能提前看到：

```text
图书馆
```

否则模型就相当于作弊了。

这个机制叫：

```text
Causal Mask
因果掩码
```

---

## 7. 什么是 Causal Mask？

Causal Mask 的作用是：

> 防止模型在预测当前位置时偷看未来的信息。

假设一句话有 5 个 token：

```text
我 / 今天 / 想 / 去 / 图书馆
```

那么 GPT 在不同位置能看到的内容大概是：

| 当前预测位置 | 可以看到的内容  |
| ------ | -------- |
| 今天     | 我        |
| 想      | 我、今天     |
| 去      | 我、今天、想   |
| 图书馆    | 我、今天、想、去 |

它不能看右边，只能从左往右生成。

因此 GPT 的生成方式是：

```text
从左到右
一个 token 一个 token 地生成
```

---

## 8. GPT 的输入输出形式

GPT 的输入是一段文本，输出是下一个 token 的概率分布。

例如输入：

```text
机器学习是一门
```

模型可能输出：

| 候选 token | 概率   |
| -------- | ---- |
| 学科       | 0.42 |
| 技术       | 0.25 |
| 方法       | 0.15 |
| 课程       | 0.08 |

如果选择概率最高的 token，就会得到：

```text
机器学习是一门学科
```

然后继续预测：

```text
机器学习是一门学科，它
```

再继续：

```text
机器学习是一门学科，它研究如何让计算机从数据中学习。
```

这就是 GPT 生成文本的基本过程。

---

## 9. BERT 的核心思想

BERT 和 GPT 不一样。

BERT 的核心任务不是预测下一个 token，而是：

> 把句子中的某些 token 遮住，让模型猜被遮住的 token 是什么。

例如：

```text
我 今天 去 [MASK] 学习
```

BERT 需要预测：

```text
图书馆
学校
教室
```

这种训练方式叫：

```text
Masked Language Model
掩码语言模型
```

简称：

```text
MLM
```

---

## 10. BERT 为什么适合理解？

BERT 在预测 `[MASK]` 的时候，可以同时看到左边和右边的上下文。

例如：

```text
我 今天 去 [MASK] 学习
```

BERT 可以看到：

```text
左边：我 今天 去
右边：学习
```

因此它可以结合完整上下文来理解句子。

而 GPT 只能看到左边：

```text
我 今天 去
```

这就是二者的关键区别：

```text
GPT：只能看前文
BERT：可以同时看前文和后文
```

所以 BERT 更适合做文本理解任务。

---

## 11. GPT 和 BERT 的核心区别

| 对比维度  | GPT                 | BERT                  |
| ----- | ------------------- | --------------------- |
| 基础结构  | Transformer Decoder | Transformer Encoder   |
| 模型方向  | 生成式模型               | 理解式模型                 |
| 训练任务  | 预测下一个 token         | 预测被遮住的 token          |
| 看文本方式 | 从左到右，只看前文           | 双向看上下文                |
| 关键机制  | Causal Mask         | Masked Language Model |
| 典型能力  | 生成文本                | 理解文本                  |
| 常见任务  | 聊天、写作、代码生成、问答       | 分类、匹配、抽取、搜索排序         |

一句话总结：

```text
GPT：会接着往下说
BERT：会理解一句话
```

---

## 12. 例子对比

假设原句是：

```text
苹果发布了新款手机。
```

### 12.1 GPT 的方式

GPT 的任务是根据前面的内容预测后面的内容：

```text
输入：苹果发布了新款
预测：手机
```

它是从左到右生成文本。

---

### 12.2 BERT 的方式

BERT 的任务是根据上下文预测被遮住的词：

```text
输入：苹果发布了新款 [MASK]。
预测：手机
```

它是通过理解完整句子来完成预测。

---

## 13. GPT 的常见应用

GPT 更适合生成类任务。

常见应用包括：

```text
1. 聊天机器人
2. 智能问答
3. 文章生成
4. 代码生成
5. 文本摘要
6. 翻译
7. 写作助手
8. AI Agent
9. 自动报告生成
10. 多轮对话系统
```

例如：

```text
用户输入：帮我总结一下这篇论文
GPT 输出：这篇论文主要研究了……
```

GPT 的特点是能够生成新的文本内容。

---

## 14. BERT 的常见应用

BERT 更适合理解类任务。

常见应用包括：

```text
1. 文本分类
2. 情感分析
3. 命名实体识别
4. 句子相似度计算
5. 搜索排序
6. 信息抽取
7. 文本匹配
8. 医学文本理解
9. 法律文本理解
10. 问答中的文本理解模块
```

例如：

```text
输入：这个电影真的很好看
任务：判断情感倾向
输出：正向
```

BERT 的特点是能够理解文本含义，并做出判断。

---

## 15. 预训练和微调

GPT 和 BERT 都是预训练模型。

可以这样理解：

```text
预训练：先在海量文本中学习通用语言知识
微调：再在具体任务数据上进一步训练
```

### 15.1 BERT 的使用流程

例如做医学文本分类：

```text
第一步：使用已经预训练好的 BERT
第二步：准备医学文本分类数据
第三步：在医学数据上微调 BERT
第四步：用于疾病分类、病历理解、实体识别等任务
```

### 15.2 GPT 的使用流程

例如做智能问答助手：

```text
第一步：使用已经预训练好的 GPT 类模型
第二步：通过指令微调或提示词让模型学会回答问题
第三步：接入知识库、工具、工作流
第四步：形成问答助手或 AI Agent
```

---

## 16. 为什么 GPT 和 BERT 很重要？

GPT 和 BERT 是理解现代大模型的两个重要入口。

它们代表了两条经典路线：

```text
BERT 路线：
Encoder-only
偏文本理解
适合分类、匹配、检索、抽取
```

```text
GPT 路线：
Decoder-only
偏文本生成
适合聊天、写作、代码、Agent
```

现代 LLM 大多更接近 GPT 路线，因为它们需要强大的生成能力。

但在搜索、分类、信息抽取、语义匹配等任务中，BERT 类模型仍然非常重要。

---

## 17. 重点概念整理

| 概念             | 含义                                 |
| -------------- | ---------------------------------- |
| GPT            | 基于 Transformer Decoder 的生成式语言模型    |
| BERT           | 基于 Transformer Encoder 的理解式语言模型    |
| Encoder        | 负责理解输入信息                           |
| Decoder        | 负责生成输出信息                           |
| Autoregressive | 自回归，表示根据前文预测后文                     |
| Causal Mask    | 因果掩码，防止模型看到未来信息                    |
| MLM            | Masked Language Model，预测被遮住的 token |
| Pre-training   | 预训练，在大规模语料上学习通用知识                  |
| Fine-tuning    | 微调，在具体任务数据上继续训练                    |
| Token          | 模型处理文本的基本单位                        |

---

## 18. 记忆口诀

可以用下面几句话记住 GPT 和 BERT：

```text
GPT 看左边，负责往后写。
BERT 看两边，负责读懂话。
GPT 偏生成，BERT 偏理解。
GPT 用 Decoder，BERT 用 Encoder。
```

再压缩成一句：

```text
GPT：接着说
BERT：认真读
```

---

## 19. 自测问题

学完本节后，尝试回答以下问题：

1. GPT 和 BERT 都是基于什么结构发展而来的？
2. GPT 使用 Transformer 的 Encoder 还是 Decoder？
3. BERT 使用 Transformer 的 Encoder 还是 Decoder？
4. GPT 的训练目标是什么？
5. BERT 的训练目标是什么？
6. 什么是 Causal Mask？
7. 为什么 GPT 不能看到未来 token？
8. 什么是 Masked Language Model？
9. 为什么 BERT 更适合文本理解？
10. 为什么 GPT 更适合文本生成？
11. 如果要做文本分类，优先想到 GPT 还是 BERT？
12. 如果要做聊天机器人，优先想到 GPT 还是 BERT？

---

## 20. 本节总结

本节学习了 GPT 和 BERT 的基础区别。

核心结论是：

```text
GPT 和 BERT 都是基于 Transformer 的预训练语言模型。
GPT 使用 Decoder，更适合生成。
BERT 使用 Encoder，更适合理解。
GPT 通过预测下一个 token 学习语言。
BERT 通过预测被遮住的 token 学习语言。
```

从整体学习路线看，GPT 和 BERT 是从 Transformer 走向 LLM 的关键桥梁。

下一节将进入：

```text
07_prompt_basic.md
```

也就是 Prompt 基础。

Prompt 是后续学习 LLM 应用、RAG、AI Agent 的重要入口。
