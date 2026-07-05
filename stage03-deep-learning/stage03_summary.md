# 第三阶段复盘：PyTorch 与神经网络基础

## 1. 本阶段学习目标

第三阶段的核心目标是：

> 学会使用 PyTorch 搭建、训练和评估一个基础神经网络模型。

在这个阶段，我主要完成了以下内容：

1. 理解 Tensor 是什么
2. 理解 PyTorch 的自动求导机制
3. 学会定义简单神经网络
4. 学会写完整的训练循环
5. 完成一个二分类任务
6. 完成一个手写数字多分类任务
7. 完成一个 MLP 回归任务
8. 理解 CNN 卷积神经网络的基本结构
9. 使用 CNN 完成 MNIST 图像分类
10. 初步理解过拟合、正则化、Dropout、weight decay
11. 学会保存和加载 PyTorch 模型
12. 初步理解深度学习模型训练、评估、保存、复用的完整流程

---

## 1.1 本阶段文件说明

本阶段对应的 notebook 文件如下：

| 文件 | 主题 | 核心内容 |
| --- | --- | --- |
| `01_pytorch_tensor.ipynb` | Tensor 基础 | Tensor 创建、shape、dtype、基本运算 |
| `02_autograd.ipynb` | 自动求导 | `requires_grad`、`backward()`、梯度 |
| `03_neural_network_basic.ipynb` | 神经网络基础 | `nn.Module`、`nn.Linear`、前向传播 |
| `04_pytorch_classification.ipynb` | PyTorch 二分类 | Sigmoid、BCELoss、训练循环 |
| `05_mnist_classification.ipynb` | MLP 手写数字分类 | 多分类、CrossEntropyLoss、准确率 |
| `06_mlp_regression.ipynb` | MLP 回归 | 连续数值预测、MSELoss、真实值 vs 预测值 |
| `07_cnn_basic.ipynb` | CNN 基础 | Conv2d、padding、ReLU、MaxPool、flatten |
| `08_cnn_image_classification.ipynb` | CNN 图像分类 | MNIST、DataLoader、CNN 训练、测试准确率 |
| `09_overfitting_regularization.ipynb` | 过拟合与正则化 | 训练集/测试集差距、Dropout、weight decay |
| `10_model_save_load.ipynb` | 模型保存与加载 | `.pth`、checkpoint、`.pkl`、模型复用 |

其中：

```text
data/MNIST/
```

保存的是 MNIST 数据集文件。

```text
saved_models/
```

保存的是训练好的模型参数、checkpoint 和标准化器。

---

## 2. Tensor 是什么？

Tensor 是 PyTorch 中最基础的数据结构。

可以把 Tensor 理解为：

> PyTorch 里的数组、向量、矩阵或更高维的数据。

在深度学习中，下面这些内容本质上都是 Tensor：

* 输入数据
* 标签数据
* 模型参数
* 模型输出
* 损失值
* 梯度

常见 Tensor 示例：

```python
import torch

x = torch.tensor([1, 2, 3])
print(x)
```

二维 Tensor 示例：

```python
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print(x)
print(x.shape)
```

输出的 `shape` 表示 Tensor 的形状。

例如：

```text
torch.Size([2, 3])
```

表示这个 Tensor 有 2 行 3 列。

---

## 3. Tensor 的常见操作

### 3.1 创建随机 Tensor

```python
x = torch.rand(3, 4)
print(x)
```

表示创建一个 3 行 4 列的随机 Tensor。

### 3.2 创建全 0 Tensor

```python
x = torch.zeros(2, 3)
print(x)
```

### 3.3 创建全 1 Tensor

```python
x = torch.ones(2, 3)
print(x)
```

### 3.4 Tensor 加减乘除

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20, 30])

print(a + b)
print(b - a)
print(a * b)
print(b / a)
```

### 3.5 矩阵乘法

```python
A = torch.tensor([
    [1, 2],
    [3, 4]
], dtype=torch.float32)

B = torch.tensor([
    [5, 6],
    [7, 8]
], dtype=torch.float32)

C = torch.matmul(A, B)

print(C)
```

神经网络中的很多计算，本质上就是矩阵运算。

---

## 4. Autograd 自动求导是什么？

Autograd 是 PyTorch 的自动求导机制。

深度学习训练模型时，需要不断计算：

> 损失函数对模型参数的梯度。

然后根据梯度更新模型参数，让模型预测得越来越准。

PyTorch 可以自动完成求导过程。

---

## 5. 自动求导的基本例子

例如：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)
```

这里：

```python
requires_grad=True
```

表示告诉 PyTorch：

> 这个变量需要计算梯度。

数学上：

```text
y = x²
dy/dx = 2x
```

当：

```text
x = 2
```

所以：

```text
dy/dx = 4
```

因此输出结果是：

```text
tensor(4.)
```

---

## 6. backward() 的作用

```python
loss.backward()
```

表示：

> 从损失值出发，反向计算每个参数的梯度。

在神经网络训练中，通常会反复执行：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

含义分别是：

| 代码                      | 含义        |
| ----------------------- | --------- |
| `optimizer.zero_grad()` | 清空上一轮梯度   |
| `loss.backward()`       | 反向传播，计算梯度 |
| `optimizer.step()`      | 根据梯度更新参数  |

---

## 7. 神经网络是什么？

神经网络可以理解为：

> 多层线性变换和非线性激活函数组合起来的模型。

一个简单神经网络通常包括：

```text
输入层 → 隐藏层 → 输出层
```

例如学生是否通过考试的任务：

```text
输入：
学习时间、出勤率、作业完成率

经过神经网络计算

输出：
是否通过考试
```

---

## 8. PyTorch 中如何定义神经网络？

PyTorch 中通常通过继承 `nn.Module` 来定义神经网络。

示例：

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

创建模型：

```python
model = SimpleNet()
print(model)
```

---

## 9. 神经网络结构解释

```python
nn.Linear(3, 4)
```

表示：

> 输入 3 个特征，输出 4 个隐藏神经元。

```python
nn.ReLU()
```

表示：

> 激活函数，让模型具备非线性表达能力。

```python
nn.Linear(4, 1)
```

表示：

> 4 个隐藏神经元，输出 1 个结果。

```python
nn.Sigmoid()
```

表示：

> 把输出压缩到 0 到 1 之间，适合二分类任务。

---

## 10. 激活函数的作用

如果神经网络中只有线性层，那么无论叠多少层，本质上仍然只是线性模型。

加入激活函数后，神经网络才能学习更复杂的非线性关系。

常见激活函数包括：

| 激活函数    | 作用         |
| ------- | ---------- |
| ReLU    | 常用于隐藏层     |
| Sigmoid | 常用于二分类输出层  |
| Softmax | 常用于多分类输出解释 |

在前面的二分类任务中，输出层用了：

```python
nn.Sigmoid()
```

因为二分类需要输出一个 0 到 1 之间的概率。

---

## 11. 损失函数是什么？

损失函数用来衡量：

> 模型预测结果和真实结果之间差得有多远。

损失越小，说明模型预测越接近真实结果。

常见损失函数：

| 任务类型  | 损失函数             |
| ----- | ---------------- |
| 回归任务  | MSELoss          |
| 二分类任务 | BCELoss          |
| 多分类任务 | CrossEntropyLoss |

---

## 12. 优化器是什么？

优化器的作用是：

> 根据梯度更新模型参数。

常见优化器包括：

| 优化器  | 特点          |
| ---- | ----------- |
| SGD  | 基础梯度下降方法    |
| Adam | 更常用，训练通常更稳定 |

示例：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

其中：

```python
model.parameters()
```

表示模型中需要学习的参数。

```python
lr
```

表示学习率，控制每次参数更新的步子大小。

---

## 13. 神经网络训练流程

神经网络训练的标准流程是：

```text
1. 准备数据 X 和 y
2. 定义模型 model
3. 定义损失函数 criterion
4. 定义优化器 optimizer
5. 前向传播：outputs = model(X)
6. 计算损失：loss = criterion(outputs, y)
7. 清空梯度：optimizer.zero_grad()
8. 反向传播：loss.backward()
9. 更新参数：optimizer.step()
10. 重复多个 epoch
```

对应代码结构：

```python
for epoch in range(100):
    outputs = model(X)

    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())
```

---

## 14. 二分类任务总结

二分类任务是指：

> 预测结果只有两个类别。

例如：

* 是否通过考试
* 是否生存
* 是否患病
* 是否违约
* 是否点击

二分类模型常见结构：

```python
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

二分类常用损失函数：

```python
criterion = nn.BCELoss()
```

模型输出：

```text
0 到 1 之间的概率
```

例如：

```text
0.82
```

可以理解为：

```text
模型认为该样本属于正类的概率是 82%
```

通常判断规则是：

```text
概率 >= 0.5 → 类别 1
概率 < 0.5 → 类别 0
```

---

## 15. 多分类任务总结

多分类任务是指：

> 预测结果有多个类别。

例如手写数字识别：

```text
输入：一张手写数字图片
输出：0、1、2、3、4、5、6、7、8、9 中的一个类别
```

手写数字识别模型示例：

```python
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)
```

这里：

```python
nn.Linear(64, 128)
```

表示：

> 每张图片有 64 个输入特征。

因为 sklearn 自带的 digits 数据集图片大小是：

```text
8 × 8 = 64
```

最后一层：

```python
nn.Linear(64, 10)
```

表示：

> 输出 10 个类别分数，对应数字 0 到 9。

---

## 16. CrossEntropyLoss 是什么？

多分类任务中通常使用：

```python
criterion = nn.CrossEntropyLoss()
```

它适合处理多个类别的分类问题。

注意：

使用 `CrossEntropyLoss` 时，模型最后一层通常不需要手动加 Softmax。

因为 `CrossEntropyLoss` 内部已经包含了相关计算。

---

## 17. torch.argmax 的作用

多分类模型会输出多个类别的分数。

例如：

```text
[0.1, 0.3, 2.5, 0.4, 0.2, ...]
```

我们需要选出分数最高的类别作为预测结果。

代码：

```python
predictions = torch.argmax(outputs, dim=1)
```

含义是：

> 在每一行中找到最大值所在的位置，这个位置就是预测类别。

---

## 18. DataLoader 是什么？

DataLoader 用于分批读取数据。

在真实训练中，通常不会一次性把所有数据全部放进模型，而是每次取一小批数据进行训练。

示例：

```python
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

其中：

| 参数              | 含义         |
| --------------- | ---------- |
| `batch_size=32` | 每次取 32 个样本 |
| `shuffle=True`  | 每轮训练前打乱数据  |

---

## 19. 使用 DataLoader 的训练循环

```python
for epoch in range(20):
    total_loss = 0

    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"第 {epoch+1} 轮，loss = {total_loss:.4f}")
```

这说明训练流程变成了：

```text
每一轮 epoch 中，模型会分多个 batch 进行训练。
```

---

## 20. 二分类和多分类的区别

| 对比项    | 二分类                       | 多分类                   |
| ------ | ------------------------- | --------------------- |
| 输出类别数量 | 2 个                       | 多个                    |
| 典型例子   | 是否通过考试                    | 手写数字 0-9              |
| 输出层    | 1 个神经元 + Sigmoid          | 类别数个神经元               |
| 标签格式   | float 类型，形状通常是 `[样本数, 1]` | long 类型，形状通常是 `[样本数]` |
| 损失函数   | BCELoss                   | CrossEntropyLoss      |
| 预测方式   | 概率是否大于 0.5                | `torch.argmax`        |

---

## 21. 本阶段完整训练代码结构

```python
import torch
import torch.nn as nn

# 1. 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(输入特征数, 隐藏层神经元数),
            nn.ReLU(),
            nn.Linear(隐藏层神经元数, 输出类别数)
        )

    def forward(self, x):
        return self.net(x)


# 2. 创建模型
model = MyModel()

# 3. 定义损失函数
criterion = nn.CrossEntropyLoss()

# 4. 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 训练模型
for epoch in range(训练轮数):
    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())
```

---

## 22. 本阶段我应该真正理解的核心

这一阶段最重要的不是记住所有代码，而是理解下面这条主线：

```text
数据进入模型
→ 模型给出预测
→ 损失函数衡量预测错误
→ backward 计算梯度
→ optimizer 更新参数
→ 模型下一次预测更准
```

这就是深度学习模型训练的本质。

---

## 23. 本阶段常见关键词总结

| 关键词              | 含义             |
| ---------------- | -------------- |
| Tensor           | PyTorch 中的数据结构 |
| Autograd         | 自动求导机制         |
| requires_grad    | 是否需要计算梯度       |
| backward         | 反向传播，计算梯度      |
| grad             | 梯度             |
| nn.Module        | 定义神经网络的基类      |
| nn.Linear        | 全连接层           |
| ReLU             | 激活函数           |
| Sigmoid          | 二分类概率输出        |
| Loss             | 损失函数           |
| Optimizer        | 优化器            |
| SGD              | 随机梯度下降         |
| Adam             | 常用优化器          |
| epoch            | 完整训练一轮数据       |
| batch            | 一小批训练数据        |
| DataLoader       | 分批加载数据         |
| CrossEntropyLoss | 多分类损失函数        |
| BCELoss          | 二分类损失函数        |
| argmax           | 取最大值对应的类别      |

---

## 24. 我现在已经掌握了什么？

通过第三阶段学习，我已经能够：

1. 使用 PyTorch 创建和操作 Tensor
2. 理解 Tensor 的 shape 和 dtype
3. 使用 Autograd 自动计算梯度
4. 理解 `requires_grad=True`
5. 理解 `loss.backward()` 的作用
6. 手动模拟参数更新过程
7. 使用 `nn.Module` 定义神经网络
8. 使用 `nn.Sequential` 快速搭建模型
9. 理解 `nn.Linear`、`ReLU`、`Sigmoid`
10. 使用 `BCELoss` 完成二分类任务
11. 使用 `CrossEntropyLoss` 完成多分类任务
12. 使用 `optimizer.step()` 更新模型参数
13. 写出完整训练循环
14. 使用 DataLoader 进行分批训练
15. 完成手写数字识别任务
16. 使用 MLP 完成回归预测任务
17. 理解 CNN 中卷积层、池化层和全连接层的作用
18. 使用 CNN 完成 MNIST 图像分类任务
19. 初步判断模型是否过拟合
20. 使用 Dropout 和 weight decay 缓解过拟合
21. 保存和加载 PyTorch 模型参数
22. 保存和加载标准化器等预处理对象

---

## 25. MLP 回归任务总结

在 `06_mlp_regression.ipynb` 中，我学习了如何用 MLP 解决回归任务。

回归任务和分类任务不同。

分类任务预测的是类别：

```text
0 / 1
猫 / 狗
数字 0-9
```

回归任务预测的是连续数值：

```text
分数
价格
销量
温度
```

本节中构造的数据规律是：

```text
y = 3*x1 - 2*x2 + 5 + noise
```

模型需要从两个输入特征 `x1`、`x2` 中学习这个规律，并预测连续值 `y`。

回归模型的特点：

| 项目 | 回归任务 |
| --- | --- |
| 输出 | 一个连续数值 |
| 输出层 | 通常是 `nn.Linear(..., 1)` |
| 损失函数 | 常用 `nn.MSELoss()` |
| 评估方式 | 看 MSE、MAE、真实值和预测值差距 |

回归任务最后一层通常不加 `Sigmoid` 或 `Softmax`，因为输出不需要限制为概率。

---

## 26. CNN 卷积神经网络总结

在 `07_cnn_basic.ipynb` 和 `08_cnn_image_classification.ipynb` 中，我学习了 CNN。

CNN 更适合处理图像数据。

图像在 PyTorch 中常见 shape 是：

```text
[batch_size, channels, height, width]
```

例如 MNIST 灰度图片：

```text
[64, 1, 28, 28]
```

表示：

```text
一次输入 64 张图片
每张图片 1 个通道
高度 28
宽度 28
```

CNN 的典型结构是：

```text
输入图片
→ Conv2d 卷积层
→ ReLU 激活函数
→ MaxPool2d 池化层
→ Conv2d 卷积层
→ ReLU 激活函数
→ MaxPool2d 池化层
→ flatten 展平
→ Linear 全连接层
→ 输出分类结果
```

关键概念：

| 概念 | 含义 |
| --- | --- |
| `Conv2d` | 提取图像局部特征 |
| `in_channels` | 输入通道数 |
| `out_channels` | 输出特征图数量 |
| `kernel_size` | 卷积核大小 |
| `padding` | 给图片边缘补 0，控制输出尺寸 |
| `ReLU` | 增加非线性表达能力 |
| `MaxPool2d` | 缩小图片尺寸，保留重要特征 |
| `flatten` | 把多维特征图拉平成一维向量 |
| `Linear` | 根据提取到的特征进行分类 |

MNIST CNN 分类任务中，模型输出 10 个数字，对应类别：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

这 10 个输出不是概率，而是每个类别的原始分数。

训练时使用：

```python
loss_fn = nn.CrossEntropyLoss()
```

预测时使用：

```python
_, predicted = torch.max(outputs, dim=1)
```

含义是：

> 选择分数最高的类别作为预测结果。

---

## 27. 过拟合与正则化总结

在 `09_overfitting_regularization.ipynb` 中，我学习了过拟合。

过拟合是指：

> 模型在训练集上表现很好，但在测试集或新数据上表现不好。

简单说，就是模型把训练数据“记住了”，但没有真正学会一般规律。

常见现象：

| 情况 | 说明 |
| --- | --- |
| 训练 loss 很低，测试 loss 很高 | 可能过拟合 |
| 训练准确率很高，测试准确率明显低 | 可能过拟合 |
| 模型太复杂、数据太少 | 更容易过拟合 |

缓解过拟合的方法包括：

### 27.1 Dropout

Dropout 会在训练过程中随机关闭一部分神经元。

这样可以避免模型过度依赖某几个神经元，提高泛化能力。

示例：

```python
nn.Dropout(p=0.5)
```

表示训练时随机让 50% 的神经元暂时失效。

### 27.2 weight_decay

`weight_decay` 是一种权重衰减方法。

它会限制模型参数不要变得过大，从而降低过拟合风险。

示例：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

### 27.3 观察训练集和测试集

判断模型好不好，不能只看训练集。

应该同时看：

```text
训练集表现
测试集表现
```

如果训练集越来越好，但测试集不提升甚至下降，就要警惕过拟合。

---

## 28. 模型保存与加载总结

在 `10_model_save_load.ipynb` 中，我学习了如何保存和加载模型。

训练好的模型如果不保存，程序关闭后参数就丢失了。

所以实际项目中通常需要保存：

```text
模型参数
训练 checkpoint
标准化器 scaler
```

### 28.1 `.pth` 文件

`.pth` 通常是 PyTorch 保存模型参数的文件。

常见保存方式：

```python
torch.save(model.state_dict(), "model.pth")
```

加载方式：

```python
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

`state_dict` 保存的是模型每一层的参数，例如：

```text
weight
bias
```

### 28.2 checkpoint

checkpoint 通常保存更多信息，例如：

```text
模型参数
优化器状态
训练轮数
损失值
```

它适合中断后继续训练。

### 28.3 `.pkl` 文件

`.pkl` 是 Python 对象保存文件。

在本阶段中，`digits_scaler.pkl` 用来保存标准化器。

保存 scaler 很重要，因为预测新数据时必须使用训练时同一个标准化规则。

如果训练时用了一个 scaler，预测时用了另一个 scaler，模型输入分布就会变，预测结果可能不准。

### 28.4 saved_models 目录

本阶段保存的模型文件包括：

```text
saved_models/digits_mlp_state_dict.pth
saved_models/digits_mlp_checkpoint.pth
saved_models/digits_scaler.pkl
```

这些文件不是普通文本文件，不能直接像 `.py`、`.md`、`.csv` 那样打开阅读。

它们需要用 Python 加载。

---

## 29. 本阶段完整能力总结

经过第三阶段学习，我现在已经能够完成一个比较完整的 PyTorch 小项目流程：

```text
1. 准备数据
2. 转换成 Tensor
3. 划分训练集和测试集
4. 标准化数据
5. 定义模型结构
6. 选择损失函数
7. 选择优化器
8. 编写训练循环
9. 观察 loss 是否下降
10. 在测试集上评估模型
11. 保存模型参数
12. 加载模型并复用
```

这说明我已经从“只会调用现成模型”，进入到“能自己搭建和训练神经网络”的阶段。

---

## 30. 本阶段遗留问题

当前阶段已经掌握了深度学习的基础流程、MLP、CNN、正则化和模型保存。

后面还需要继续学习：

1. RNN / LSTM
2. Attention 注意力机制
3. Transformer
4. Embedding
5. NLP 文本处理
6. 大模型基础
7. RAG
8. Agent
9. 模型部署与推理服务

---

## 31. 下一阶段学习方向

下一阶段进入：

```text
第四阶段：NLP 与大模型基础
```

学习顺序：

| 顺序 | 内容          | 目标                |
| -- | ----------- | ----------------- |
| 1  | 文本如何变成数字    | 理解 Token、词表、编码    |
| 2  | Embedding   | 理解词向量             |
| 3  | 文本分类        | 用 PyTorch 做简单文本分类 |
| 4  | Attention   | 理解注意力机制           |
| 5  | Transformer | 理解大模型核心结构         |
| 6  | GPT / BERT  | 理解生成式模型和理解式模型区别   |
| 7  | Prompt      | 学会和大模型交互          |
| 8  | RAG         | 理解文档问答系统原理        |

---

## 32. 阶段总结

第三阶段让我从传统机器学习进入了深度学习。

传统机器学习中，我主要使用 scikit-learn 调用现成模型，例如逻辑回归、决策树、随机森林。

而在 PyTorch 中，我开始自己定义模型结构、损失函数和训练循环。

我现在已经理解：

> 深度学习不是简单调用一个模型，而是通过 Tensor、神经网络、损失函数、自动求导和优化器共同完成训练过程。

本阶段最重要的一句话是：

```text
模型预测 → 计算损失 → 反向传播 → 更新参数 → 重复训练
```

现在我还进一步理解了：

```text
模型训练 → 测试评估 → 判断过拟合 → 正则化优化 → 保存模型 → 加载复用
```

只要理解这两条主线，就抓住了 PyTorch 深度学习入门阶段的核心。
