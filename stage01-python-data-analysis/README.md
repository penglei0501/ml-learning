# Stage 01：Python 与数据分析基础

这是我的 Python、NumPy、pandas 和 Matplotlib 入门学习记录。

本阶段通过学生成绩数据练习 Python 基础语法、表格数据处理、CSV 文件读写和简单的数据可视化，为后续机器学习学习打基础。

## 学习目标

- 掌握 Python 变量、列表、循环和条件判断
- 学会编写和调用简单函数
- 理解字典与列表组合的数据结构
- 使用 NumPy 进行基础数值计算
- 使用 pandas 创建和分析 DataFrame
- 读取和保存 CSV 文件
- 计算平均分、总分和分组统计结果
- 使用 Matplotlib 绘制基础图表

## 文件说明

```text
stage01-python-data-analysis/
├── 01_python.ipynb  # Python 与数据分析练习
├── students.csv     # 学生成绩示例数据
└── README.md        # 本阶段说明
```

## Notebook 内容

`01_python.ipynb` 主要包含以下内容：

1. Python 变量与基本输出
2. 列表、索引和长度
3. `for` 循环
4. `if / elif / else` 条件判断
5. 函数定义与平均分计算
6. 字典形式的学生成绩数据
7. NumPy 数组与统计计算
8. pandas DataFrame 的创建和筛选
9. CSV 文件读取与成绩分析
10. Matplotlib 柱状图绘制

## CSV 数据说明

`students.csv` 包含以下字段：

| 字段 | 含义 |
|---|---|
| `name` | 学生姓名 |
| `gender` | 性别 |
| `math` | 数学成绩 |
| `english` | 英语成绩 |
| `computer` | 计算机成绩 |

## 运行方式

建议使用 VS Code 或 Jupyter Notebook 打开：

```text
01_python.ipynb
```

Notebook 和 `students.csv` 需要保持在同一目录，因为代码使用相对路径读取数据：

```python
pd.read_csv("students.csv")
```

## 使用的主要工具

- Python
- NumPy
- pandas
- Matplotlib
- Jupyter Notebook

## 阶段总结

完成本阶段后，我能够使用 Python 处理简单的结构化数据，并完成以下任务：

- 读取和查看 CSV 数据
- 筛选满足条件的学生
- 计算各科平均分和学生总分
- 找出成绩最高的学生
- 按性别进行分组统计
- 绘制成绩柱状图

这些内容是后续使用 scikit-learn 学习机器学习算法的数据处理基础。

> 本目录是个人学习记录，示例数据主要用于编程练习，分析结果不代表正式研究结论。
