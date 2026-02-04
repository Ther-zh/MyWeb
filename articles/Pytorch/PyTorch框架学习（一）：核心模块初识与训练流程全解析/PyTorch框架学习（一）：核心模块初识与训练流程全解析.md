---
title: PyTorch框架学习（一）：核心模块初识与训练流程全解析
date: 2026-02-05
summary: 在PyTorch中，各个模块的设计遵循“分工明确、协同工作”的原则，核心目标是支持从数据加载到模型训练、推理的全流程。理解模块间的依赖关系和训练流程的逻辑，能帮你更清晰地掌握PyTorch的使用框架。
---

在深度学习的技术生态中，PyTorch 凭借简洁的语法、灵活的动态计算图以及丰富的生态支持，成为了科研与工业界最受欢迎的框架之一。对于初学者而言，PyTorch 的学习并非从零散的 API 开始，而是要先理清其核心设计逻辑，掌握各模块的职责与协作关系 —— 这是搭建深度学习模型、完成训练任务的基础。本系列将从 PyTorch 的基础核心入手，由浅入深拆解框架使用逻辑，本篇作为系列第一篇，将聚焦 PyTorch 的核心模块体系，梳理各模块的依赖关系，并完整讲解模型训练的全流程，帮助大家建立对 PyTorch 的整体认知，为后续的深度学习实践打下框架性基础。

## 一、PyTorch核心模块的依赖关系

PyTorch的核心模块主要围绕“数据→模型→损失→优化”这一主线设计，各模块职责和依赖关系如下：

#### 1. 数据处理模块：`torch.utils.data`

**核心组件**：`Dataset`（数据集抽象类）、`DataLoader`（数据加载器）。  
**作用**：负责数据的读取、预处理（如归一化、转换）、批量加载、打乱顺序等。  
**依赖**：无强依赖其他模块，但需要用户自定义数据格式（如将图片/标签封装成`Dataset`），最终为模型提供输入数据（`input`）和标签（`label`）。  

#### 2. 模型构建模块：`torch.nn`

**核心组件**：`Module`（所有模型的基类）、`Linear`/`Conv2d`（层）、`ReLU`/`Softmax`（激活函数）、`CrossEntropyLoss`（损失函数）等。  
**作用**：定义模型的网络结构（层的堆叠、计算逻辑），并提供损失函数（用于衡量预测误差）。  
**依赖**：  
- 模型的构建依赖`nn.Module`（所有自定义模型必须继承它）；  
- 损失函数（如`nn.CrossEntropyLoss`）依赖模型的输出（`output`）和真实标签（`label`），用于计算损失值（`loss`）。  

#### 3. 优化器模块：`torch.optim`

**核心组件**：`SGD`、`Adam`等优化器。  
**作用**：根据损失的梯度（`gradient`）更新模型参数（`parameters`），最小化损失函数。  
**依赖**：  
- 必须依赖模型的参数（`model.parameters()`）—— 优化器需要知道“要更新哪些参数”；  
- 依赖损失的梯度（通过`loss.backward()`计算得到）—— 优化器需要根据梯度方向调整参数。  

#### 4. 张量与计算图：`torch.Tensor`

**核心组件**：`Tensor`（张量，PyTorch的基本数据结构）、自动求导机制（`autograd`）。  
**作用**：所有数据（输入、模型参数、中间结果、损失）都以张量形式存在；`autograd`自动构建计算图，支持梯度反向传播。  
**依赖**：是所有模块的底层依赖——数据模块输出张量，模型对张量做运算，损失函数和优化器基于张量的梯度工作。  


#### 模块依赖关系总结：  
```  
数据模块（Dataset/DataLoader）→ 输出输入张量（input）和标签（label）  
↓  
模型模块（nn.Module）→ 接收input，输出预测值（output）  
↓  
损失函数（nn.xxxLoss）→ 接收output和label，输出损失值（loss）  
↓  
自动求导（autograd）→ 基于loss计算所有参数的梯度（通过loss.backward()）  
↓  
优化器（optim.xxx）→ 接收参数梯度，更新模型参数（通过optimizer.step()）  
```

## 二、模型训练的完整流程及逻辑

训练流程的核心是“迭代优化”：通过不断输入数据、计算误差、调整参数，让模型逐渐学会拟合数据。具体步骤和逻辑如下：  


#### 阶段1：准备工作（只执行1次）  
在开始训练循环前，需要完成以下初始化：  

1. **准备数据**  
   - 定义`Dataset`：将原始数据（如图片、标签）封装成PyTorch可识别的格式（需实现`__getitem__`和`__len__`方法）；  
   - 定义`DataLoader`：对`Dataset`进行批量处理（`batch_size`）、打乱（`shuffle=True`）、多进程加载（`num_workers`）等，最终得到可迭代的数据批次（`batch`）。  

   ```python  
   from torch.utils.data import Dataset, DataLoader  
   
   class MyDataset(Dataset):  
       def __init__(self, data, labels):  
           self.data = data  
           self.labels = labels  
       def __getitem__(self, idx):  
           return self.data[idx], self.labels[idx]  
       def __len__(self):  
           return len(self.data)  
   
   dataset = MyDataset(train_data, train_labels)  
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  
   ```


2. **定义模型**  
   - 继承`nn.Module`自定义模型，在`__init__`中定义层（如`Conv2d`、`Linear`），在`forward`中定义计算逻辑（输入→输出的映射）。  

   ```python  
   import torch.nn as nn  
   
   class MyModel(nn.Module):  
       def __init__(self):  
           super().__init__()  
           self.fc1 = nn.Linear(784, 256)  # 假设输入是28x28的MNIST图片（展平后784维）  
           self.fc2 = nn.Linear(256, 10)   # 输出10类（0-9）  
       def forward(self, x):  
           x = x.view(-1, 784)  # 展平（batch_size, 784）  
           x = nn.functional.relu(self.fc1(x))  
           x = self.fc2(x)  
           return x  
   
   model = MyModel()  # 实例化模型  
   ```


3. **定义损失函数和优化器**  
   - 损失函数：衡量预测值（`output`）与真实标签（`label`）的差距（如分类用`CrossEntropyLoss`）；  
   - 优化器：接收模型参数，定义更新策略（如`Adam`、`SGD`）。  

   ```python  
   criterion = nn.CrossEntropyLoss()  # 损失函数  
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器（依赖模型参数）  
   ```


#### 阶段2：训练循环（迭代执行，直到收敛）  
每次循环处理一个批次的数据，核心是“前向传播→计算损失→反向传播→参数更新”的闭环。  

1. **设置模型为训练模式**  
   - 调用`model.train()`：启用 dropout、batch normalization 等训练特有的层（这些层在训练和推理时行为不同）。  

   ```python  
   model.train()  # 训练模式  
   ```


2. **迭代数据批次**  
   对`dataloader`中的每个批次（`inputs, labels`）执行以下步骤：  

   ```python  
   for epoch in range(10):  # 训练10轮（所有数据过一遍为1轮）  
       running_loss = 0.0  
       for inputs, labels in dataloader:  # 遍历每个批次  
           # 步骤1：清空梯度（关键！避免梯度累积）  
           optimizer.zero_grad()  
   
           # 步骤2：前向传播（ Forward ）  
           outputs = model(inputs)  # 模型预测：inputs → outputs  
   
           # 步骤3：计算损失（ Loss ）  
           loss = criterion(outputs, labels)  # 对比outputs和labels，得到损失  
   
           # 步骤4：反向传播（ Backward ）  
           loss.backward()  # 自动计算所有参数的梯度（基于计算图）  
   
           # 步骤5：参数更新（ Optimize ）  
           optimizer.step()  # 优化器根据梯度更新模型参数  
   
           # 统计损失（可选）  
           running_loss += loss.item()  
       print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")  
   ```


#### 阶段3：验证/测试（可选，通常每轮训练后执行）  
验证模型在未见过的数据上的性能，判断是否过拟合。  

1. **设置模型为评估模式**  
   - 调用`model.eval()`：关闭 dropout、固定 batch normalization 的统计量（用训练时的均值/方差）。  

   ```python  
   model.eval()  # 评估模式  
   ```


2. **关闭梯度计算（节省资源）**  
   - 用`torch.no_grad()`上下文管理器：验证时不需要计算梯度（无需更新参数），减少内存占用。  

   ```python  
   correct = 0  
   total = 0  
   with torch.no_grad():  # 关闭梯度计算  
       for inputs, labels in test_dataloader:  
           outputs = model(inputs)  
           _, predicted = torch.max(outputs.data, 1)  # 取预测概率最大的类别  
           total += labels.size(0)  
           correct += (predicted == labels).sum().item()  
   print(f"测试准确率：{100 * correct / total}%")  
   ```

## 三、总结

本篇作为 PyTorch 框架学习的开篇，核心围绕**模块认知**与**流程梳理**两大核心展开，为大家搭建了 PyTorch 的基础使用框架。我们首先明确了 PyTorch 围绕 “数据→模型→损失→优化” 这一深度学习核心主线设计的四大核心模块，解析了`torch.utils.data`、`torch.nn`、`torch.optim`与`torch.Tensor`的核心职责、关键组件，以及各模块间层层递进的依赖关系，理解了张量作为底层数据结构，是所有模块协作的基础。同时，我们还拆解了 PyTorch 模型训练的完整流程，从仅需执行一次的准备工作（数据封装、模型定义、损失函数与优化器初始化），到迭代执行的训练循环（前向传播→计算损失→反向传播→参数更新），再到可选的验证测试环节，掌握了`model.train()`/`model.eval()`模式切换、梯度清空、关闭梯度计算等关键操作的意义与用法。

这些内容是 PyTorch 入门的核心基础，理清模块间的协作逻辑，掌握训练流程的核心闭环，能够让我们跳出零散 API 的误区，从整体上理解 PyTorch 构建和训练模型的底层逻辑，避免后续实践中出现 “知其然不知其所以然” 的问题。后续本系列将基于本次的基础认知，对各核心模块进行深度拆解，结合实战案例讲解 API 的具体使用、常见问题解决与优化技巧，让大家从 “认识框架” 逐步走向 “熟练使用框架”。