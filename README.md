# ML Learning Journey

这是一个个人机器学习与深度学习学习项目，用来记录从 Python 数据分析、传统机器学习、深度学习、NLP / LLM 到时间序列深度学习的完整练习过程。

项目以 Jupyter Notebook 为主，每个阶段围绕一个主题展开，包含基础概念、代码实验、模型训练、结果分析和阶段复盘笔记。

## 项目结构

```text
ml-learning/
├── stage01-python-data-analysis/          # Python 与数据分析基础
├── stage02-machine-learning/              # 传统机器学习算法
├── stage03-deep-learning/                 # PyTorch 与深度学习基础
├── stage04-nlp-llm/                       # NLP、Transformer、LLM、RAG 基础
└── stage05-time-series-deep-learning/     # 时间序列与深度学习预测
```

## 学习阶段

### Stage 01：Python Data Analysis

主要内容：

- Python 基础语法
- CSV 数据读取与分析
- pandas 基础操作
- 简单数据统计与可视化

入口文件：

- [stage01-python-data-analysis/README.md](stage01-python-data-analysis/README.md)
- [stage01-python-data-analysis/01_python.ipynb](stage01-python-data-analysis/01_python.ipynb)

### Stage 02：Machine Learning

主要内容：

- 线性回归、逻辑回归
- 决策树、随机森林、KNN、SVM
- 朴素贝叶斯、KMeans、PCA
- XGBoost、LightGBM
- 异常检测、时间序列机器学习
- 交叉验证与网格搜索
- Titanic 项目练习

复盘笔记：

- [stage02-machine-learning/stage02_summary.md](stage02-machine-learning/stage02_summary.md)

### Stage 03：Deep Learning

主要内容：

- PyTorch Tensor
- 自动求导 Autograd
- 神经网络基础
- 分类与回归任务
- MNIST 手写数字分类
- CNN 基础
- 模型保存与加载
- 过拟合与正则化

复盘笔记：

- [stage03-deep-learning/stage03_summary.md](stage03-deep-learning/stage03_summary.md)

### Stage 04：NLP / LLM

主要内容：

- 文本数字化
- Embedding
- 文本分类
- Attention 与 Transformer
- GPT / BERT 基础
- Prompt 基础
- Tokenizer
- 预训练模型调用
- Sentence Embedding
- 向量相似度、Top-K 检索、RAG 基础

复盘笔记：

- [stage04-nlp-llm/12_stage04_summary.md](stage04-nlp-llm/12_stage04_summary.md)

### Stage 05：Time Series Deep Learning

主要内容：

- 时间序列基础
- 滑动窗口数据集
- MLP 时间序列预测
- RNN、LSTM、GRU 基础

代表文件：

- [stage05-time-series-deep-learning/01_time_series_basic.ipynb](stage05-time-series-deep-learning/01_time_series_basic.ipynb)
- [stage05-time-series-deep-learning/02_sliding_window_dataset.ipynb](stage05-time-series-deep-learning/02_sliding_window_dataset.ipynb)
- [stage05-time-series-deep-learning/03_mlp_time_series_forecasting.ipynb](stage05-time-series-deep-learning/03_mlp_time_series_forecasting.ipynb)
- [stage05-time-series-deep-learning/04_rnn_basic.ipynb](stage05-time-series-deep-learning/04_rnn_basic.ipynb)

## 技术栈

- Python
- Jupyter Notebook
- NumPy
- pandas
- Matplotlib / Seaborn
- scikit-learn
- PyTorch
- Transformers
- SentenceTransformers
- XGBoost / LightGBM

## 项目说明

这个仓库主要用于个人学习记录和阶段性复盘，不是完整的生产级机器学习项目。

每个 notebook 更关注：

- 概念理解
- 代码流程
- 输入输出含义
- 模型训练过程
- 结果解释

部分模型文件、虚拟环境和大体积生成文件没有上传到仓库中。

## 后续计划

- 补充 Stage 05 的阶段复盘笔记
- 增加更多真实数据集项目练习
- 整理每个阶段的 README
- 继续学习 RAG、Agent 和模型部署相关内容
