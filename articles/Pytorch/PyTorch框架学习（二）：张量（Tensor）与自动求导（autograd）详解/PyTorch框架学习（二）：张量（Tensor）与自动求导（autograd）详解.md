---
title: PyTorch框架学习（二）：张量（Tensor）与自动求导（autograd）详解
date: 2026-02-05
summary: 本期内容，我们深入拆解了张量（Tensor）与自动求导（autograd）的核心原理与实操方法，明确了张量作为底层数据结构的核心作用，掌握了自动求导的“标记-正向传播-反向传播-梯度清零”全流程，也解决了初学者最易踩的高频误区。这些内容，是后续学习数据处理、模型构建的基础——只有熟练掌握张量的操作和自动求导的逻辑，才能真正理解模型训练的底层机制，避免“只会调API，不懂原理”的问题。
---

### 导言

在第1期内容中，我们梳理了PyTorch的四大核心模块，明确了**张量（Tensor）是所有模块的底层依赖**——数据模块输出张量、模型对张量做运算、损失函数与优化器基于张量的梯度工作。对于深度学习初学者而言，张量就像是“深度学习的积木”，自动求导则是“让积木能够自我调整的魔法”，两者共同构成了PyTorch框架的核心底层逻辑。

本期作为系列第2期，将聚焦张量（Tensor）与自动求导（autograd）两大核心，从“是什么、怎么用、注意什么”三个维度，深入拆解其原理与实操方法，搭配可直接复制运行的代码案例，解决初学者最易踩的“张量不会用”“梯度报错”等问题，夯实PyTorch学习的底层基础，为后续数据处理、模型构建等模块的深入学习铺路。

### 一、Tensor张量：PyTorch的核心数据结构

简单来说，**张量（Tensor）就是PyTorch中用于存储数据的多维数组**，类比于NumPy中的ndarray，但它最大的优势的是——支持GPU加速计算，且能与自动求导机制深度结合，承载模型的输入、参数、中间结果和损失值。

### 1.1 张量的核心特性（初学者必掌握）

张量的三大核心特性，直接决定了它在PyTorch中的使用逻辑，也是后续避免报错的关键：

#### （1）数据类型（dtype）

张量的数据类型与Python、NumPy的数据类型对应，常用的有以下几种，需根据任务合理选择（避免类型不匹配报错）：

- 浮点型（最常用）：torch.float32（默认）、torch.float64（精度更高，适合复杂计算）；

- 整型：torch.int32、torch.int64（常用于索引、标签存储）；

- 布尔型：torch.bool（用于逻辑判断，如筛选数据）。

注意：模型训练中，输入张量与模型参数的 dtype 必须一致，否则会报“类型不匹配”错误。

#### （2）设备类型（device）

这是张量与NumPy数组最核心的区别——张量可以放在CPU或GPU上运行，实现加速计算：

- CPU设备：torch.device("cpu")（默认，适合小规模数据、调试代码）；

- GPU设备：torch.device("cuda")（需电脑有NVIDIA显卡且安装CUDA，适合大规模数据、模型训练）。

注意：同一计算中，所有张量必须放在**同一设备**上（如不能用CPU张量和GPU张量做运算），否则会报错。

#### （3）形状操作（shape）

张量的形状（shape）描述了张量的维度和各维度的元素个数，比如：

- 0维张量（标量）：shape=()，用于存储单个数值（如损失值loss）；

- 1维张量（向量）：shape=(n,)，用于存储一维数据（如单个样本的特征）；

- 2维张量（矩阵）：shape=(m,n)，用于存储二维数据（如多个样本的特征，shape=(batch_size, feature_num)）；

- 3维及以上张量：常用于存储图片（shape=(batch_size, channel, height, width)）、文本等复杂数据。

形状的灵活调整，是后续数据适配模型输入的关键。

### 1.2 张量常用API实战（可直接复制运行）

掌握以下常用API，就能应对80%的基础场景，建议边看边运行代码，加深记忆（代码中包含详细注释）：

```python
import torch

# 1. 张量的创建（最常用3种方式）
# 方式1：从Python列表/元组创建
tensor1 = torch.tensor([1, 2, 3], dtype=torch.float32, device="cpu")
print("从列表创建：", tensor1, "shape:", tensor1.shape, "dtype:", tensor1.dtype)

# 方式2：创建全0/全1张量（常用作初始化）
tensor2 = torch.zeros((3, 4))  # 全0张量，shape=(3,4)
tensor3 = torch.ones((2, 3), dtype=torch.int64)  # 全1张量，shape=(2,3)
print("全0张量：", tensor2, "\n全1张量：", tensor3)

# 方式3：从NumPy数组创建（衔接NumPy数据）
import numpy as np
np_arr = np.array([[1, 2], [3, 4]])
tensor4 = torch.from_numpy(np_arr)  # 从NumPy转换，共享内存（修改一方，另一方会变）
print("从NumPy创建：", tensor4)

# 2. 张量的索引与切片（和Python、NumPy用法一致）
tensor5 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("原始张量：", tensor5)
print("取第2行（索引从0开始）：", tensor5[1])
print("取第1列第2个元素：", tensor5[1, 0])
print("取前2行、后2列：", tensor5[:2, 1:])

# 3. 张量的形状调整（核心API：view、reshape、squeeze、unsqueeze）
tensor6 = torch.tensor([[1, 2], [3, 4], [5, 6]])  # shape=(3,2)
# view：调整形状（要求元素总数不变，推荐使用）
tensor7 = tensor6.view(2, 3)  # 调整为shape=(2,3)
# unsqueeze：增加维度（常用在模型输入，补充batch维度）
tensor8 = tensor6.unsqueeze(0)  # shape从(3,2)变为(1,3,2)
# squeeze：删除维度（删除维度为1的维度）
tensor9 = tensor8.squeeze(0)  # 恢复为shape=(3,2)
print("形状调整后：", tensor7.shape, tensor8.shape, tensor9.shape)

# 4. 张量的拼接与拆分
tensor10 = torch.tensor([[1, 2], [3, 4]])
tensor11 = torch.tensor([[5, 6], [7, 8]])
# 拼接（dim=0：按行拼接，dim=1：按列拼接）
tensor12 = torch.cat([tensor10, tensor11], dim=0)  # shape=(4,2)
# 拆分（按dim=0拆分，分成2个张量）
tensor13, tensor14 = torch.split(tensor12, 2, dim=0)
print("拼接后：", tensor12, "\n拆分后：", tensor13, tensor14)

# 5. 张量的设备转换（CPU ↔ GPU）
if torch.cuda.is_available():  # 判断GPU是否可用
    tensor15 = tensor10.to("cuda")  # CPU张量转GPU
    tensor16 = tensor15.to("cpu")  # GPU张量转CPU
    print("GPU张量：", tensor15.device)
else:
    print("当前设备不支持GPU，无法转换")

# 6. 张量与NumPy的转换
tensor17 = torch.tensor([1, 2, 3])
np_arr2 = tensor17.numpy()  # 张量转NumPy
tensor18 = torch.from_numpy(np_arr2)  # NumPy转张量
print("张量转NumPy：", np_arr2, "\nNumPy转张量：", tensor18)

```

### 1.3 张量使用注意事项（避坑重点）

- 避免“原地操作”（in-place）：如 tensor.add_(1) 是原地修改张量，会破坏计算图，导致自动求导报错；建议使用 tensor = tensor.add(1) 替代。

- 设备一致性：同一计算中，所有张量（输入、模型参数、标签）必须在同一设备（CPU/GPU），否则报错。

- 数据类型一致性：模型参数默认是float32，输入张量需与之匹配，避免int型张量输入模型。

- 标量取值：0维张量（标量）需用 .item() 取值（如 loss.item()），直接打印会包含张量相关信息，不利于后续计算。

### 二、自动求导（autograd）：模型训练的“核心魔法”

在第1期的训练流程中，我们提到“通过 loss.backward() 计算梯度，再用优化器更新参数”——这背后的核心就是PyTorch的自动求导（autograd）机制。它能自动构建计算图，追踪张量的所有运算，然后反向传播计算出每个可求导张量的梯度（gradient），无需手动推导复杂的求导公式。

### 2.1 自动求导的核心原理：计算图

自动求导的本质是“计算图的构建与反向传播”，我们用一个简单的例子理解：

假设我们有运算流程： $x \rightarrow y = x^2 \rightarrow z = \frac{1}{n}\sum y$ （x是输入张量，z是损失值）

- 正向传播：构建计算图，记录张量的运算关系（x → y → z）；

- 反向传播：从z（损失值）出发，自动计算z对每个可求导张量（x）的梯度（ $\frac{\partial z}{\partial x}$ ），并将梯度存储在张量的 .grad 属性中。

关键特点：PyTorch的计算图是**动态图**——每次正向传播都会重新构建计算图，灵活性极高，适合调试代码和动态调整运算逻辑（这也是PyTorch比TensorFlow 1.x更受初学者欢迎的原因）。

### 2.2 自动求导的核心使用方法

自动求导的使用核心是“标记可求导张量”和“触发反向传播”，结合代码案例详解（重点看注释）：

```python
import torch

# 1. 标记可求导张量（requires_grad=True）
# 方法1：创建张量时指定requires_grad=True
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # 标记x为可求导张量
print("x:", x, "requires_grad:", x.requires_grad)

# 方法2：对已创建的张量，通过.requires_grad_()标记（原地操作，仅此处推荐）
y = torch.tensor([4.0, 5.0, 6.0])
y.requires_grad_()  # 标记y为可求导张量
print("y:", y, "requires_grad:", y.requires_grad)

# 2. 正向传播：构建计算图
z = x ** 2 + y  # 运算1：x平方 + y
loss = z.mean()  # 运算2：求平均值（模拟损失函数）
print("z:", z, "\nloss:", loss)

# 3. 反向传播：计算梯度（loss.backward()触发）
loss.backward()  # 自动计算loss对所有可求导张量（x、y）的梯度

# 4. 查看梯度（梯度存储在张量的.grad属性中）
print("loss对x的梯度（dz/dx）：", x.grad)  # 梯度值 = 2x/3（因loss是平均值，n=3）
print("loss对y的梯度（dz/dy）：", y.grad)  # 梯度值 = 1/3

# 5. 禁止求导（上下文管理器，用于验证/推理阶段）
with torch.no_grad():  # 该上下文内，所有运算不构建计算图，不计算梯度
    z_no_grad = x ** 2 + y
    print("禁止求导后的z：", z_no_grad)
    print("z_no_grad.requires_grad：", z_no_grad.requires_grad)  # 输出False

# 6. 梯度清零（核心！避免梯度累积）
# 多次反向传播前，必须清零梯度，否则梯度会累加（导致参数更新错误）
x.grad.zero_()  # 清零x的梯度
y.grad.zero_()  # 清零y的梯度
print("清零后的x梯度：", x.grad)  # 输出None（或全0）

# 补充：禁止某个张量参与求导（detach()）
x_detach = x.detach()  # 生成x的副本，不参与求导，不影响原计算图
z_detach = x_detach ** 2 + y
z_detach.mean().backward()
print("x_detach.requires_grad：", x_detach.requires_grad)  # False
print("原x的梯度：", x.grad)  # 因x_detach不参与求导，原x梯度仍为0

```

### 2.3 自动求导的注意事项（训练报错高频点）

这部分是初学者最易踩坑的地方，一定要重点记住，避免训练时出现梯度相关报错：

#### （1）梯度清零是必须操作

PyTorch中，张量的梯度会**自动累积**（即每次调用 backward()，梯度都会叠加到 .grad 中）。而模型训练中，每个批次的梯度都是独立的，因此每次反向传播前，必须用 optimizer.zero_grad()（或张量.grad.zero_()）清零梯度。

错误示例：未清零梯度，导致梯度累积，参数更新错误；正确示例：见上述代码中的梯度清零操作，或第1期训练循环中的 optimizer.zero_grad()。

#### （2）只有标量才能调用 backward()

backward() 只能用于0维张量（标量，如loss），如果是高维张量（如z是1维/2维），直接调用 backward() 会报错，需指定 grad_tensors 参数（用于加权求和，将高维转为标量）。

示例（高维张量反向传播）：

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
z = x ** 2  # z是1维张量（shape=(2,)）
# z.backward()  # 直接调用会报错
z.backward(torch.tensor([1.0, 1.0]))  # 指定grad_tensors，将z转为标量（z1 + z2）
print("x的梯度：", x.grad)  # 输出[2.0, 4.0]

```

#### （3）避免原地操作破坏计算图

如前所述，原地操作（如 x.add_(1)、x[0] = 0）会直接修改张量的值，破坏计算图的完整性，导致 backward() 报错。建议使用非原地操作替代。

#### （4）可求导张量的运算限制

只有 requires_grad=True 的张量，才会被追踪运算、计算梯度；如果张量的 requires_grad=False（默认），则其运算不会被追踪，也不会产生梯度。

注意：模型参数（如 model.parameters()）默认 requires_grad=True，无需手动标记；而输入数据、标签的 requires_grad 需设为 False（默认就是False），避免不必要的梯度计算。

### 三、实操案例：结合张量与自动求导，模拟简单模型训练

为了让大家更好地衔接第1期的训练流程，我们用张量和自动求导，模拟一个最简单的线性回归训练过程（无复杂模型，聚焦底层逻辑），完整代码如下：

```python
import torch

# 1. 准备数据（张量形式，模拟线性回归的输入x和标签y）
# 假设真实模型：y = 2x + 1 + 噪声
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)  # 输入，无需求导
y_true = torch.tensor([[3.1], [5.2], [7.0], [9.1]], requires_grad=False)  # 真实标签，无需求导

# 2. 定义模型参数（可求导张量，模拟线性层的权重和偏置）
w = torch.tensor([[0.5]], requires_grad=True)  # 权重，初始值随机
b = torch.tensor([[0.1]], requires_grad=True)  # 偏置，初始值随机

# 3. 定义损失函数（均方误差，模拟回归任务的损失）
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

# 4. 模拟训练循环（5轮，核心：前向传播→计算损失→反向传播→参数更新）
learning_rate = 0.01  # 学习率，控制参数更新幅度
for epoch in range(5):
    # 步骤1：前向传播（计算预测值y_pred）
    y_pred = x @ w + b  # 线性运算：y_pred = w*x + b（矩阵乘法@）
    
    # 步骤2：计算损失
    loss = mse_loss(y_pred, y_true)
    
    # 步骤3：反向传播（计算梯度）
    loss.backward()  # 自动计算loss对w、b的梯度，存储在w.grad、b.grad中
    
    # 步骤4：参数更新（手动模拟优化器，后续会讲torch.optim）
    # 注意：参数更新是原地操作，但此处用torch.no_grad()包裹，避免破坏计算图
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # 步骤5：梯度清零（关键！避免梯度累积）
    w.grad.zero_()
    b.grad.zero_()
    
    # 打印每轮训练结果
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

# 训练结束：查看最终参数（接近真实值w=2，b=1）
print("\n训练完成，最终参数：w =", w.item(), "b =", b.item())

```

运行代码后会发现，随着训练轮次增加，损失值逐渐减小，w和b逐渐接近真实值（w=2、b=1）——这就是模型训练的底层逻辑，而张量和自动求导，正是支撑这一逻辑的核心。

### 四、初学者常见误区总结

结合本期内容，梳理初学者最易踩的5个误区，帮大家快速避坑：

1. 误区1：忘记梯度清零，导致梯度累积，参数更新错误；→ 解决方案：每次 backward() 后，用 optimizer.zero_grad() 或张量.grad.zero_() 清零。

2. 误区2：用高维张量直接调用 backward()，导致报错；→ 解决方案：要么将高维张量转为标量，要么指定 grad_tensors 参数。

3. 误区3：CPU张量与GPU张量混合运算，导致报错；→ 解决方案：用 .to() 方法，将所有张量转换到同一设备。

4. 误区4：原地操作（如 x.add_(1)）破坏计算图，导致求导报错；→ 解决方案：使用非原地操作（如 x = x.add(1)）。

5. 误区5：标量取值不用 .item()，直接用于后续计算；→ 解决方案：标量张量（如loss）用 .item() 取值，得到Python数值。

### 结尾与预告

本期内容，我们深入拆解了张量（Tensor）与自动求导（autograd）的核心原理与实操方法，明确了张量作为底层数据结构的核心作用，掌握了自动求导的“标记-正向传播-反向传播-梯度清零”全流程，也解决了初学者最易踩的高频误区。这些内容，是后续学习数据处理、模型构建的基础——只有熟练掌握张量的操作和自动求导的逻辑，才能真正理解模型训练的底层机制，避免“只会调API，不懂原理”的问题。

下一期（第3期），我们将聚焦**数据处理模块（torch.utils.data）**，手把手教大家自定义Dataset、使用DataLoader加载数据，以及用torchvision.transforms做数据预处理，解决“数据加载不了”“预处理不会做”的实战难题，为后续模型训练准备好“原材料”。

如果大家在本期内容的实操中遇到问题，或者对某个API、知识点有疑问，欢迎在评论区留言，我们下期再见！
