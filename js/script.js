// 博客文章数据
const columns = [
    "Pytorch"
];

const articles = [
    {
        "id": "pytorch_20260205014107",
        "title": "PyTorch框架学习（一）：核心模块初识与训练流程全解析",
        "date": "2026-02-05",
        "summary": "在PyTorch中，各个模块的设计遵循“分工明确、协同工作”的原则，核心目标是支持从数据加载到模型训练、推理的全流程。理解模块间的依赖关系和训练流程的逻辑，能帮你更清晰地掌握PyTorch的使用框架。",
        "content": "在深度学习的技术生态中，PyTorch 凭借简洁的语法、灵活的动态计算图以及丰富的生态支持，成为了科研与工业界最受欢迎的框架之一。对于初学者而言，PyTorch 的学习并非从零散的 API 开始，而是要先理清其核心设计逻辑，掌握各模块的职责与协作关系 —— 这是搭建深度学习模型、完成训练任务的基础。本系列将从 PyTorch 的基础核心入手，由浅入深拆解框架使用逻辑，本篇作为系列第一篇，将聚焦 PyTorch 的核心模块体系，梳理各模块的依赖关系，并完整讲解模型训练的全流程，帮助大家建立对 PyTorch 的整体认知，为后续的深度学习实践打下框架性基础。\n\n## 一、PyTorch核心模块的依赖关系\n\nPyTorch的核心模块主要围绕“数据→模型→损失→优化”这一主线设计，各模块职责和依赖关系如下：\n\n#### 1. 数据处理模块：`torch.utils.data`\n\n**核心组件**：`Dataset`（数据集抽象类）、`DataLoader`（数据加载器）。  \n**作用**：负责数据的读取、预处理（如归一化、转换）、批量加载、打乱顺序等。  \n**依赖**：无强依赖其他模块，但需要用户自定义数据格式（如将图片/标签封装成`Dataset`），最终为模型提供输入数据（`input`）和标签（`label`）。  \n\n#### 2. 模型构建模块：`torch.nn`\n\n**核心组件**：`Module`（所有模型的基类）、`Linear`/`Conv2d`（层）、`ReLU`/`Softmax`（激活函数）、`CrossEntropyLoss`（损失函数）等。  \n**作用**：定义模型的网络结构（层的堆叠、计算逻辑），并提供损失函数（用于衡量预测误差）。  \n**依赖**：  \n- 模型的构建依赖`nn.Module`（所有自定义模型必须继承它）；  \n- 损失函数（如`nn.CrossEntropyLoss`）依赖模型的输出（`output`）和真实标签（`label`），用于计算损失值（`loss`）。  \n\n#### 3. 优化器模块：`torch.optim`\n\n**核心组件**：`SGD`、`Adam`等优化器。  \n**作用**：根据损失的梯度（`gradient`）更新模型参数（`parameters`），最小化损失函数。  \n**依赖**：  \n- 必须依赖模型的参数（`model.parameters()`）—— 优化器需要知道“要更新哪些参数”；  \n- 依赖损失的梯度（通过`loss.backward()`计算得到）—— 优化器需要根据梯度方向调整参数。  \n\n#### 4. 张量与计算图：`torch.Tensor`\n\n**核心组件**：`Tensor`（张量，PyTorch的基本数据结构）、自动求导机制（`autograd`）。  \n**作用**：所有数据（输入、模型参数、中间结果、损失）都以张量形式存在；`autograd`自动构建计算图，支持梯度反向传播。  \n**依赖**：是所有模块的底层依赖——数据模块输出张量，模型对张量做运算，损失函数和优化器基于张量的梯度工作。  \n\n\n#### 模块依赖关系总结：  \n```  \n数据模块（Dataset/DataLoader）→ 输出输入张量（input）和标签（label）  \n↓  \n模型模块（nn.Module）→ 接收input，输出预测值（output）  \n↓  \n损失函数（nn.xxxLoss）→ 接收output和label，输出损失值（loss）  \n↓  \n自动求导（autograd）→ 基于loss计算所有参数的梯度（通过loss.backward()）  \n↓  \n优化器（optim.xxx）→ 接收参数梯度，更新模型参数（通过optimizer.step()）  \n```\n\n## 二、模型训练的完整流程及逻辑\n\n训练流程的核心是“迭代优化”：通过不断输入数据、计算误差、调整参数，让模型逐渐学会拟合数据。具体步骤和逻辑如下：  \n\n\n#### 阶段1：准备工作（只执行1次）  \n在开始训练循环前，需要完成以下初始化：  \n\n1. **准备数据**  \n   - 定义`Dataset`：将原始数据（如图片、标签）封装成PyTorch可识别的格式（需实现`__getitem__`和`__len__`方法）；  \n   - 定义`DataLoader`：对`Dataset`进行批量处理（`batch_size`）、打乱（`shuffle=True`）、多进程加载（`num_workers`）等，最终得到可迭代的数据批次（`batch`）。  \n\n   ```python  \n   from torch.utils.data import Dataset, DataLoader  \n   \n   class MyDataset(Dataset):  \n       def __init__(self, data, labels):  \n           self.data = data  \n           self.labels = labels  \n       def __getitem__(self, idx):  \n           return self.data[idx], self.labels[idx]  \n       def __len__(self):  \n           return len(self.data)  \n   \n   dataset = MyDataset(train_data, train_labels)  \n   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  \n   ```\n\n\n2. **定义模型**  \n   - 继承`nn.Module`自定义模型，在`__init__`中定义层（如`Conv2d`、`Linear`），在`forward`中定义计算逻辑（输入→输出的映射）。  \n\n   ```python  \n   import torch.nn as nn  \n   \n   class MyModel(nn.Module):  \n       def __init__(self):  \n           super().__init__()  \n           self.fc1 = nn.Linear(784, 256)  # 假设输入是28x28的MNIST图片（展平后784维）  \n           self.fc2 = nn.Linear(256, 10)   # 输出10类（0-9）  \n       def forward(self, x):  \n           x = x.view(-1, 784)  # 展平（batch_size, 784）  \n           x = nn.functional.relu(self.fc1(x))  \n           x = self.fc2(x)  \n           return x  \n   \n   model = MyModel()  # 实例化模型  \n   ```\n\n\n3. **定义损失函数和优化器**  \n   - 损失函数：衡量预测值（`output`）与真实标签（`label`）的差距（如分类用`CrossEntropyLoss`）；  \n   - 优化器：接收模型参数，定义更新策略（如`Adam`、`SGD`）。  \n\n   ```python  \n   criterion = nn.CrossEntropyLoss()  # 损失函数  \n   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器（依赖模型参数）  \n   ```\n\n\n#### 阶段2：训练循环（迭代执行，直到收敛）  \n每次循环处理一个批次的数据，核心是“前向传播→计算损失→反向传播→参数更新”的闭环。  \n\n1. **设置模型为训练模式**  \n   - 调用`model.train()`：启用 dropout、batch normalization 等训练特有的层（这些层在训练和推理时行为不同）。  \n\n   ```python  \n   model.train()  # 训练模式  \n   ```\n\n\n2. **迭代数据批次**  \n   对`dataloader`中的每个批次（`inputs, labels`）执行以下步骤：  \n\n   ```python  \n   for epoch in range(10):  # 训练10轮（所有数据过一遍为1轮）  \n       running_loss = 0.0  \n       for inputs, labels in dataloader:  # 遍历每个批次  \n           # 步骤1：清空梯度（关键！避免梯度累积）  \n           optimizer.zero_grad()  \n   \n           # 步骤2：前向传播（ Forward ）  \n           outputs = model(inputs)  # 模型预测：inputs → outputs  \n   \n           # 步骤3：计算损失（ Loss ）  \n           loss = criterion(outputs, labels)  # 对比outputs和labels，得到损失  \n   \n           # 步骤4：反向传播（ Backward ）  \n           loss.backward()  # 自动计算所有参数的梯度（基于计算图）  \n   \n           # 步骤5：参数更新（ Optimize ）  \n           optimizer.step()  # 优化器根据梯度更新模型参数  \n   \n           # 统计损失（可选）  \n           running_loss += loss.item()  \n       print(f\"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}\")  \n   ```\n\n\n#### 阶段3：验证/测试（可选，通常每轮训练后执行）  \n验证模型在未见过的数据上的性能，判断是否过拟合。  \n\n1. **设置模型为评估模式**  \n   - 调用`model.eval()`：关闭 dropout、固定 batch normalization 的统计量（用训练时的均值/方差）。  \n\n   ```python  \n   model.eval()  # 评估模式  \n   ```\n\n\n2. **关闭梯度计算（节省资源）**  \n   - 用`torch.no_grad()`上下文管理器：验证时不需要计算梯度（无需更新参数），减少内存占用。  \n\n   ```python  \n   correct = 0  \n   total = 0  \n   with torch.no_grad():  # 关闭梯度计算  \n       for inputs, labels in test_dataloader:  \n           outputs = model(inputs)  \n           _, predicted = torch.max(outputs.data, 1)  # 取预测概率最大的类别  \n           total += labels.size(0)  \n           correct += (predicted == labels).sum().item()  \n   print(f\"测试准确率：{100 * correct / total}%\")  \n   ```\n\n## 三、总结\n\n本篇作为 PyTorch 框架学习的开篇，核心围绕**模块认知**与**流程梳理**两大核心展开，为大家搭建了 PyTorch 的基础使用框架。我们首先明确了 PyTorch 围绕 “数据→模型→损失→优化” 这一深度学习核心主线设计的四大核心模块，解析了`torch.utils.data`、`torch.nn`、`torch.optim`与`torch.Tensor`的核心职责、关键组件，以及各模块间层层递进的依赖关系，理解了张量作为底层数据结构，是所有模块协作的基础。同时，我们还拆解了 PyTorch 模型训练的完整流程，从仅需执行一次的准备工作（数据封装、模型定义、损失函数与优化器初始化），到迭代执行的训练循环（前向传播→计算损失→反向传播→参数更新），再到可选的验证测试环节，掌握了`model.train()`/`model.eval()`模式切换、梯度清空、关闭梯度计算等关键操作的意义与用法。\n\n这些内容是 PyTorch 入门的核心基础，理清模块间的协作逻辑，掌握训练流程的核心闭环，能够让我们跳出零散 API 的误区，从整体上理解 PyTorch 构建和训练模型的底层逻辑，避免后续实践中出现 “知其然不知其所以然” 的问题。后续本系列将基于本次的基础认知，对各核心模块进行深度拆解，结合实战案例讲解 API 的具体使用、常见问题解决与优化技巧，让大家从 “认识框架” 逐步走向 “熟练使用框架”。",
        "column": "Pytorch",
        "path": "Pytorch/PyTorch框架学习（一）：核心模块初识与训练流程全解析"
    },
    {
        "id": "pytorchtensorautograd_20260205014107",
        "title": "PyTorch框架学习（二）：张量（Tensor）与自动求导（autograd）详解",
        "date": "2026-02-05",
        "summary": "本期内容，我们深入拆解了张量（Tensor）与自动求导（autograd）的核心原理与实操方法，明确了张量作为底层数据结构的核心作用，掌握了自动求导的“标记-正向传播-反向传播-梯度清零”全流程，也解决了初学者最易踩的高频误区。这些内容，是后续学习数据处理、模型构建的基础——只有熟练掌握张量的操作和自动求导的逻辑，才能真正理解模型训练的底层机制，避免“只会调API，不懂原理”的问题。",
        "content": "### 导言\n\n在第1期内容中，我们梳理了PyTorch的四大核心模块，明确了**张量（Tensor）是所有模块的底层依赖**——数据模块输出张量、模型对张量做运算、损失函数与优化器基于张量的梯度工作。对于深度学习初学者而言，张量就像是“深度学习的积木”，自动求导则是“让积木能够自我调整的魔法”，两者共同构成了PyTorch框架的核心底层逻辑。\n\n本期作为系列第2期，将聚焦张量（Tensor）与自动求导（autograd）两大核心，从“是什么、怎么用、注意什么”三个维度，深入拆解其原理与实操方法，搭配可直接复制运行的代码案例，解决初学者最易踩的“张量不会用”“梯度报错”等问题，夯实PyTorch学习的底层基础，为后续数据处理、模型构建等模块的深入学习铺路。\n\n### 一、Tensor张量：PyTorch的核心数据结构\n\n简单来说，**张量（Tensor）就是PyTorch中用于存储数据的多维数组**，类比于NumPy中的ndarray，但它最大的优势的是——支持GPU加速计算，且能与自动求导机制深度结合，承载模型的输入、参数、中间结果和损失值。\n\n### 1.1 张量的核心特性（初学者必掌握）\n\n张量的三大核心特性，直接决定了它在PyTorch中的使用逻辑，也是后续避免报错的关键：\n\n#### （1）数据类型（dtype）\n\n张量的数据类型与Python、NumPy的数据类型对应，常用的有以下几种，需根据任务合理选择（避免类型不匹配报错）：\n\n- 浮点型（最常用）：torch.float32（默认）、torch.float64（精度更高，适合复杂计算）；\n\n- 整型：torch.int32、torch.int64（常用于索引、标签存储）；\n\n- 布尔型：torch.bool（用于逻辑判断，如筛选数据）。\n\n注意：模型训练中，输入张量与模型参数的 dtype 必须一致，否则会报“类型不匹配”错误。\n\n#### （2）设备类型（device）\n\n这是张量与NumPy数组最核心的区别——张量可以放在CPU或GPU上运行，实现加速计算：\n\n- CPU设备：torch.device(\"cpu\")（默认，适合小规模数据、调试代码）；\n\n- GPU设备：torch.device(\"cuda\")（需电脑有NVIDIA显卡且安装CUDA，适合大规模数据、模型训练）。\n\n注意：同一计算中，所有张量必须放在**同一设备**上（如不能用CPU张量和GPU张量做运算），否则会报错。\n\n#### （3）形状操作（shape）\n\n张量的形状（shape）描述了张量的维度和各维度的元素个数，比如：\n\n- 0维张量（标量）：shape=()，用于存储单个数值（如损失值loss）；\n\n- 1维张量（向量）：shape=(n,)，用于存储一维数据（如单个样本的特征）；\n\n- 2维张量（矩阵）：shape=(m,n)，用于存储二维数据（如多个样本的特征，shape=(batch_size, feature_num)）；\n\n- 3维及以上张量：常用于存储图片（shape=(batch_size, channel, height, width)）、文本等复杂数据。\n\n形状的灵活调整，是后续数据适配模型输入的关键。\n\n### 1.2 张量常用API实战（可直接复制运行）\n\n掌握以下常用API，就能应对80%的基础场景，建议边看边运行代码，加深记忆（代码中包含详细注释）：\n\n```python\nimport torch\n\n# 1. 张量的创建（最常用3种方式）\n# 方式1：从Python列表/元组创建\ntensor1 = torch.tensor([1, 2, 3], dtype=torch.float32, device=\"cpu\")\nprint(\"从列表创建：\", tensor1, \"shape:\", tensor1.shape, \"dtype:\", tensor1.dtype)\n\n# 方式2：创建全0/全1张量（常用作初始化）\ntensor2 = torch.zeros((3, 4))  # 全0张量，shape=(3,4)\ntensor3 = torch.ones((2, 3), dtype=torch.int64)  # 全1张量，shape=(2,3)\nprint(\"全0张量：\", tensor2, \"\n全1张量：\", tensor3)\n\n# 方式3：从NumPy数组创建（衔接NumPy数据）\nimport numpy as np\nnp_arr = np.array([[1, 2], [3, 4]])\ntensor4 = torch.from_numpy(np_arr)  # 从NumPy转换，共享内存（修改一方，另一方会变）\nprint(\"从NumPy创建：\", tensor4)\n\n# 2. 张量的索引与切片（和Python、NumPy用法一致）\ntensor5 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\nprint(\"原始张量：\", tensor5)\nprint(\"取第2行（索引从0开始）：\", tensor5[1])\nprint(\"取第1列第2个元素：\", tensor5[1, 0])\nprint(\"取前2行、后2列：\", tensor5[:2, 1:])\n\n# 3. 张量的形状调整（核心API：view、reshape、squeeze、unsqueeze）\ntensor6 = torch.tensor([[1, 2], [3, 4], [5, 6]])  # shape=(3,2)\n# view：调整形状（要求元素总数不变，推荐使用）\ntensor7 = tensor6.view(2, 3)  # 调整为shape=(2,3)\n# unsqueeze：增加维度（常用在模型输入，补充batch维度）\ntensor8 = tensor6.unsqueeze(0)  # shape从(3,2)变为(1,3,2)\n# squeeze：删除维度（删除维度为1的维度）\ntensor9 = tensor8.squeeze(0)  # 恢复为shape=(3,2)\nprint(\"形状调整后：\", tensor7.shape, tensor8.shape, tensor9.shape)\n\n# 4. 张量的拼接与拆分\ntensor10 = torch.tensor([[1, 2], [3, 4]])\ntensor11 = torch.tensor([[5, 6], [7, 8]])\n# 拼接（dim=0：按行拼接，dim=1：按列拼接）\ntensor12 = torch.cat([tensor10, tensor11], dim=0)  # shape=(4,2)\n# 拆分（按dim=0拆分，分成2个张量）\ntensor13, tensor14 = torch.split(tensor12, 2, dim=0)\nprint(\"拼接后：\", tensor12, \"\n拆分后：\", tensor13, tensor14)\n\n# 5. 张量的设备转换（CPU ↔ GPU）\nif torch.cuda.is_available():  # 判断GPU是否可用\n    tensor15 = tensor10.to(\"cuda\")  # CPU张量转GPU\n    tensor16 = tensor15.to(\"cpu\")  # GPU张量转CPU\n    print(\"GPU张量：\", tensor15.device)\nelse:\n    print(\"当前设备不支持GPU，无法转换\")\n\n# 6. 张量与NumPy的转换\ntensor17 = torch.tensor([1, 2, 3])\nnp_arr2 = tensor17.numpy()  # 张量转NumPy\ntensor18 = torch.from_numpy(np_arr2)  # NumPy转张量\nprint(\"张量转NumPy：\", np_arr2, \"\nNumPy转张量：\", tensor18)\n\n```\n\n### 1.3 张量使用注意事项（避坑重点）\n\n- 避免“原地操作”（in-place）：如 tensor.add_(1) 是原地修改张量，会破坏计算图，导致自动求导报错；建议使用 tensor = tensor.add(1) 替代。\n\n- 设备一致性：同一计算中，所有张量（输入、模型参数、标签）必须在同一设备（CPU/GPU），否则报错。\n\n- 数据类型一致性：模型参数默认是float32，输入张量需与之匹配，避免int型张量输入模型。\n\n- 标量取值：0维张量（标量）需用 .item() 取值（如 loss.item()），直接打印会包含张量相关信息，不利于后续计算。\n\n### 二、自动求导（autograd）：模型训练的“核心魔法”\n\n在第1期的训练流程中，我们提到“通过 loss.backward() 计算梯度，再用优化器更新参数”——这背后的核心就是PyTorch的自动求导（autograd）机制。它能自动构建计算图，追踪张量的所有运算，然后反向传播计算出每个可求导张量的梯度（gradient），无需手动推导复杂的求导公式。\n\n### 2.1 自动求导的核心原理：计算图\n\n自动求导的本质是“计算图的构建与反向传播”，我们用一个简单的例子理解：\n\n假设我们有运算流程： $x \rightarrow y = x^2 \rightarrow z = \frac{1}{n}\sum y$ （x是输入张量，z是损失值）\n\n- 正向传播：构建计算图，记录张量的运算关系（x → y → z）；\n\n- 反向传播：从z（损失值）出发，自动计算z对每个可求导张量（x）的梯度（ $\frac{\partial z}{\partial x}$ ），并将梯度存储在张量的 .grad 属性中。\n\n关键特点：PyTorch的计算图是**动态图**——每次正向传播都会重新构建计算图，灵活性极高，适合调试代码和动态调整运算逻辑（这也是PyTorch比TensorFlow 1.x更受初学者欢迎的原因）。\n\n### 2.2 自动求导的核心使用方法\n\n自动求导的使用核心是“标记可求导张量”和“触发反向传播”，结合代码案例详解（重点看注释）：\n\n```python\nimport torch\n\n# 1. 标记可求导张量（requires_grad=True）\n# 方法1：创建张量时指定requires_grad=True\nx = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # 标记x为可求导张量\nprint(\"x:\", x, \"requires_grad:\", x.requires_grad)\n\n# 方法2：对已创建的张量，通过.requires_grad_()标记（原地操作，仅此处推荐）\ny = torch.tensor([4.0, 5.0, 6.0])\ny.requires_grad_()  # 标记y为可求导张量\nprint(\"y:\", y, \"requires_grad:\", y.requires_grad)\n\n# 2. 正向传播：构建计算图\nz = x ** 2 + y  # 运算1：x平方 + y\nloss = z.mean()  # 运算2：求平均值（模拟损失函数）\nprint(\"z:\", z, \"\nloss:\", loss)\n\n# 3. 反向传播：计算梯度（loss.backward()触发）\nloss.backward()  # 自动计算loss对所有可求导张量（x、y）的梯度\n\n# 4. 查看梯度（梯度存储在张量的.grad属性中）\nprint(\"loss对x的梯度（dz/dx）：\", x.grad)  # 梯度值 = 2x/3（因loss是平均值，n=3）\nprint(\"loss对y的梯度（dz/dy）：\", y.grad)  # 梯度值 = 1/3\n\n# 5. 禁止求导（上下文管理器，用于验证/推理阶段）\nwith torch.no_grad():  # 该上下文内，所有运算不构建计算图，不计算梯度\n    z_no_grad = x ** 2 + y\n    print(\"禁止求导后的z：\", z_no_grad)\n    print(\"z_no_grad.requires_grad：\", z_no_grad.requires_grad)  # 输出False\n\n# 6. 梯度清零（核心！避免梯度累积）\n# 多次反向传播前，必须清零梯度，否则梯度会累加（导致参数更新错误）\nx.grad.zero_()  # 清零x的梯度\ny.grad.zero_()  # 清零y的梯度\nprint(\"清零后的x梯度：\", x.grad)  # 输出None（或全0）\n\n# 补充：禁止某个张量参与求导（detach()）\nx_detach = x.detach()  # 生成x的副本，不参与求导，不影响原计算图\nz_detach = x_detach ** 2 + y\nz_detach.mean().backward()\nprint(\"x_detach.requires_grad：\", x_detach.requires_grad)  # False\nprint(\"原x的梯度：\", x.grad)  # 因x_detach不参与求导，原x梯度仍为0\n\n```\n\n### 2.3 自动求导的注意事项（训练报错高频点）\n\n这部分是初学者最易踩坑的地方，一定要重点记住，避免训练时出现梯度相关报错：\n\n#### （1）梯度清零是必须操作\n\nPyTorch中，张量的梯度会**自动累积**（即每次调用 backward()，梯度都会叠加到 .grad 中）。而模型训练中，每个批次的梯度都是独立的，因此每次反向传播前，必须用 optimizer.zero_grad()（或张量.grad.zero_()）清零梯度。\n\n错误示例：未清零梯度，导致梯度累积，参数更新错误；正确示例：见上述代码中的梯度清零操作，或第1期训练循环中的 optimizer.zero_grad()。\n\n#### （2）只有标量才能调用 backward()\n\nbackward() 只能用于0维张量（标量，如loss），如果是高维张量（如z是1维/2维），直接调用 backward() 会报错，需指定 grad_tensors 参数（用于加权求和，将高维转为标量）。\n\n示例（高维张量反向传播）：\n\n```python\nx = torch.tensor([1.0, 2.0], requires_grad=True)\nz = x ** 2  # z是1维张量（shape=(2,)）\n# z.backward()  # 直接调用会报错\nz.backward(torch.tensor([1.0, 1.0]))  # 指定grad_tensors，将z转为标量（z1 + z2）\nprint(\"x的梯度：\", x.grad)  # 输出[2.0, 4.0]\n\n```\n\n#### （3）避免原地操作破坏计算图\n\n如前所述，原地操作（如 x.add_(1)、x[0] = 0）会直接修改张量的值，破坏计算图的完整性，导致 backward() 报错。建议使用非原地操作替代。\n\n#### （4）可求导张量的运算限制\n\n只有 requires_grad=True 的张量，才会被追踪运算、计算梯度；如果张量的 requires_grad=False（默认），则其运算不会被追踪，也不会产生梯度。\n\n注意：模型参数（如 model.parameters()）默认 requires_grad=True，无需手动标记；而输入数据、标签的 requires_grad 需设为 False（默认就是False），避免不必要的梯度计算。\n\n### 三、实操案例：结合张量与自动求导，模拟简单模型训练\n\n为了让大家更好地衔接第1期的训练流程，我们用张量和自动求导，模拟一个最简单的线性回归训练过程（无复杂模型，聚焦底层逻辑），完整代码如下：\n\n```python\nimport torch\n\n# 1. 准备数据（张量形式，模拟线性回归的输入x和标签y）\n# 假设真实模型：y = 2x + 1 + 噪声\nx = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)  # 输入，无需求导\ny_true = torch.tensor([[3.1], [5.2], [7.0], [9.1]], requires_grad=False)  # 真实标签，无需求导\n\n# 2. 定义模型参数（可求导张量，模拟线性层的权重和偏置）\nw = torch.tensor([[0.5]], requires_grad=True)  # 权重，初始值随机\nb = torch.tensor([[0.1]], requires_grad=True)  # 偏置，初始值随机\n\n# 3. 定义损失函数（均方误差，模拟回归任务的损失）\ndef mse_loss(y_pred, y_true):\n    return ((y_pred - y_true) ** 2).mean()\n\n# 4. 模拟训练循环（5轮，核心：前向传播→计算损失→反向传播→参数更新）\nlearning_rate = 0.01  # 学习率，控制参数更新幅度\nfor epoch in range(5):\n    # 步骤1：前向传播（计算预测值y_pred）\n    y_pred = x @ w + b  # 线性运算：y_pred = w*x + b（矩阵乘法@）\n    \n    # 步骤2：计算损失\n    loss = mse_loss(y_pred, y_true)\n    \n    # 步骤3：反向传播（计算梯度）\n    loss.backward()  # 自动计算loss对w、b的梯度，存储在w.grad、b.grad中\n    \n    # 步骤4：参数更新（手动模拟优化器，后续会讲torch.optim）\n    # 注意：参数更新是原地操作，但此处用torch.no_grad()包裹，避免破坏计算图\n    with torch.no_grad():\n        w -= learning_rate * w.grad\n        b -= learning_rate * b.grad\n    \n    # 步骤5：梯度清零（关键！避免梯度累积）\n    w.grad.zero_()\n    b.grad.zero_()\n    \n    # 打印每轮训练结果\n    print(f\"Epoch {epoch+1}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}\")\n\n# 训练结束：查看最终参数（接近真实值w=2，b=1）\nprint(\"\n训练完成，最终参数：w =\", w.item(), \"b =\", b.item())\n\n```\n\n运行代码后会发现，随着训练轮次增加，损失值逐渐减小，w和b逐渐接近真实值（w=2、b=1）——这就是模型训练的底层逻辑，而张量和自动求导，正是支撑这一逻辑的核心。\n\n### 四、初学者常见误区总结\n\n结合本期内容，梳理初学者最易踩的5个误区，帮大家快速避坑：\n\n1. 误区1：忘记梯度清零，导致梯度累积，参数更新错误；→ 解决方案：每次 backward() 后，用 optimizer.zero_grad() 或张量.grad.zero_() 清零。\n\n2. 误区2：用高维张量直接调用 backward()，导致报错；→ 解决方案：要么将高维张量转为标量，要么指定 grad_tensors 参数。\n\n3. 误区3：CPU张量与GPU张量混合运算，导致报错；→ 解决方案：用 .to() 方法，将所有张量转换到同一设备。\n\n4. 误区4：原地操作（如 x.add_(1)）破坏计算图，导致求导报错；→ 解决方案：使用非原地操作（如 x = x.add(1)）。\n\n5. 误区5：标量取值不用 .item()，直接用于后续计算；→ 解决方案：标量张量（如loss）用 .item() 取值，得到Python数值。\n\n### 结尾与预告\n\n本期内容，我们深入拆解了张量（Tensor）与自动求导（autograd）的核心原理与实操方法，明确了张量作为底层数据结构的核心作用，掌握了自动求导的“标记-正向传播-反向传播-梯度清零”全流程，也解决了初学者最易踩的高频误区。这些内容，是后续学习数据处理、模型构建的基础——只有熟练掌握张量的操作和自动求导的逻辑，才能真正理解模型训练的底层机制，避免“只会调API，不懂原理”的问题。\n\n下一期（第3期），我们将聚焦**数据处理模块（torch.utils.data）**，手把手教大家自定义Dataset、使用DataLoader加载数据，以及用torchvision.transforms做数据预处理，解决“数据加载不了”“预处理不会做”的实战难题，为后续模型训练准备好“原材料”。\n\n如果大家在本期内容的实操中遇到问题，或者对某个API、知识点有疑问，欢迎在评论区留言，我们下期再见！",
        "column": "Pytorch",
        "path": "Pytorch/PyTorch框架学习（二）：张量（Tensor）与自动求导（autograd）详解"
    },
    {
        "id": "kl_20260205014107",
        "title": "信息量、信息熵、KL散度交叉熵损失",
        "date": "2026-01-30",
        "summary": "信息量、信息熵、KL散度交叉熵损失",
        "content": "## 一、信息量\n\n### **定义**\n\n通过==**概率**==来定义信息量：即一个信息的信息量有多大取决于它**能为我想知道的东西缩小多少概率**，自变量是概率（一个事情发生概率低，那么如果它实现了带来的信息量就多）接下来我们要考虑用数学表达：\n\n![img](articles/信息论/3423c2ad4f6886ee9dfbcc9db05d7444e6ee99c5.png@500w_!web-note.webp)\n\n![img](articles/信息论/fbbce2f3807680a22629222fc094b4bd33697e45.jpg@690w_!web-note.webp)\n\n由于概率运算为相乘，就有以下关系，而要刻画这种关系的函数就是log\n\n![img](articles/信息论/b6f7494c22c37275b571804d606331edb19ed645.jpg@690w_!web-note.webp)\n\n![img](articles/信息论/107a8c5d3d44a86ae6d84207353df5e789400cf9.jpg@690w_!web-note.webp)\n\n接下来就是确定系数，概率越低信息越大，要乘个负号，底数可以任取，不过在计算机中用二进制比较多就取2为底数，相当于用二进制表达概率，给他个单位即比特\n$$\nf(x)=-\log_2(x)\n$$\n![img](articles/信息论/9e17c06790ceccc800585217f6d7e8854cf1210d.jpg@690w_!web-note.webp)\n\n## **二、信息熵**\n\n而对于一个系统想要刻画它的信息量是不是把它简单相加呢？并非，如以下例子，右边系统法国队赢得概率很大理论上来说右边更稳定\n\n![img](articles/信息论/2f496103b27025399ad704030cd767c8f7f66fdf.jpg@690w_!web-note.webp)\n\n先前我们计算信息量是**以这个事情发生为前提**的，所以对于一个系统的信息量（信息熵），要给**每个信息量 乘上它的概率**加权求和得到系统信息量（信息熵）\n\n![img](articles/信息论/3e3d3668b1e1718f6b8f42597b6d50bf9a1a33dc.jpg@690w_!web-note.webp)\n$$\nf(p_1,p_2...p_k)=\Sigma_1^k p_i*(-\log_2(p_i))\n$$\n![img](articles/信息论/6f865e629e845dbcbc6e095cf402dd053a20f51e.jpg@690w_!web-note.webp)\n\n## **三、相对熵——KL散度**\n\n而对于两个系统，我们希望比较它们的异同，我们使用熵进行刻画\n\n![img](articles/信息论/cd1ee5f4503d9643bc03fdb93866f50530bb2737.jpg@690w_!web-note.webp)\n\n![img](articles/信息论/559236cf30ac848b0b016366fc7333230a594f88.jpg@690w_!web-note.webp)\n\n公式中，$D_{KL}(P||Q)$的`||`符号前面是基准，用基准的概率乘各项相差的信息量化简后得到：\n\n**交叉熵(基准概率x另一个的信息量）- 基准熵**的形式\n\n![img](articles/信息论/43715b7ac9a569d9e400b69c2edbb4886ed540b0.png@690w_!web-note.webp)\n\n吉布斯不等式：交叉熵>基准熵——》KL散度始终为正数\n\n$p_i$只关乎基准本身，而KL散度取决于交叉熵大小——》使用交叉熵计算损失\n\n**把交叉熵应用到神经网络中**\n\n训练时标签是已知的，只要以真实标签为基准，把得到的分类概率与真实结果算交叉熵损失即可\n\n![img](articles/信息论/7f903002ee859ad7144692950dd5ad62241b163b.jpg@690w_!web-note.webp)\n\n如：\n\n|          | A    | B    | C    |\n| :------: | ---- | ---- | ---- |\n| 真实标签 | 1    | 0    | 0    |\n| 神经网络 | 0.9  | 0.04 | 0.06 |\n\n$$\nD_{KL}(P||Q)=-1*\log_20.9+0+0\n$$\n\n（二分类时只要手动分开“是”和“不是”两类即可）\n\n## 四、最大似然估计\n\n利用数据的概率估计最有可能的概率分布（让**已有数据出现的可能性最大**）\n\n数据中进行n次实验，某事件发生了n次，设其概率为$p_i$，则样本数据出现的概率为$p=p_i^k*(1-p_i)^{(n-k)}$\n\n![image-20250726190358431](articles/信息论/image-20250726190358431.png)\n\n在最外层套个log让相乘变成相加方便求导：$f(p)=k\log_2p_i+(n-k)\log_2(1-p_i)$\n\n发生的次数可替换为频率$q_i$（两边除以总次数）：$f(p)=q_i\log_2p_i+(1-q_i)\log_2(1-p_i)$\n\n使其最大即使其相反数最小——》使交叉熵损失最小\n\n同样的，对于多分类问题可以进行拓展：\n\n![image-20250726192237390](articles/信息论/image-20250726192237390.png)",
        "column": null,
        "path": "信息论"
    }
];

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    console.log('===== 页面加载完成，开始初始化 =====');
    
    // 检查当前页面
    const currentPath = window.location.pathname;
    const currentFileName = currentPath.split('/').pop();
    
    console.log('当前页面路径:', currentPath);
    console.log('当前页面文件名:', currentFileName);
    
    // 检查articles数组是否存在
    console.log('articles变量是否存在:', typeof articles !== 'undefined');
    if (typeof articles !== 'undefined') {
        console.log('articles数组长度:', articles.length);
    }
    
    // 检查columns数组是否存在
    console.log('columns变量是否存在:', typeof columns !== 'undefined');
    if (typeof columns !== 'undefined') {
        console.log('columns数组长度:', columns.length);
    }
    
    // 如果是博客页面，加载文章列表
    if (currentFileName === 'blog.html' || currentFileName === 'blog') {
        console.log('===== 检测到博客页面，开始加载文章列表 =====');
        loadArticles();
        
        // 初始化专栏筛选
        initColumnFilter();
        
        // 为搜索按钮添加事件监听器
        const searchButton = document.getElementById('search-button');
        console.log('搜索按钮元素:', searchButton);
        if (searchButton) {
            searchButton.addEventListener('click', searchArticles);
            console.log('已为搜索按钮添加事件监听器');
        }
        
        // 为搜索输入框添加回车事件监听器
        const searchInput = document.getElementById('search-input');
        console.log('搜索输入框元素:', searchInput);
        if (searchInput) {
            searchInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    searchArticles();
                }
            });
            console.log('已为搜索输入框添加回车事件监听器');
        }
        
        // 为重置按钮添加事件监听器
        const resetButton = document.getElementById('reset-button');
        console.log('重置按钮元素:', resetButton);
        if (resetButton) {
            resetButton.addEventListener('click', resetSearch);
            console.log('已为重置按钮添加事件监听器');
        }
        
        // 为排序选择框添加事件监听器
        const sortSelect = document.getElementById('sort-select');
        console.log('排序选择框元素:', sortSelect);
        if (sortSelect) {
            sortSelect.addEventListener('change', sortArticles);
            console.log('已为排序选择框添加事件监听器');
        }
        
        // 为专栏筛选添加事件监听器
        const columnFilter = document.getElementById('column-filter');
        console.log('专栏筛选元素:', columnFilter);
        if (columnFilter) {
            columnFilter.addEventListener('change', filterArticlesByColumn);
            console.log('已为专栏筛选添加事件监听器');
        }
    }
    
    // 如果是文章详情页面，加载文章内容
    if (currentFileName === 'article.html') {
        console.log('===== 检测到文章详情页面，开始加载文章内容 =====');
        loadArticleDetail();
    }
    
    console.log('===== 页面初始化完成 =====');
});

// 初始化专栏筛选
function initColumnFilter() {
    console.log('===== 开始初始化专栏筛选 =====');
    
    // 检查columns数组是否存在
    if (typeof columns === 'undefined') {
        console.error('columns数组未定义');
        return;
    }
    
    // 获取专栏筛选元素
    const columnFilter = document.getElementById('column-filter');
    console.log('专栏筛选元素:', columnFilter);
    if (!columnFilter) {
        console.error('未找到 column-filter 元素');
        return;
    }
    
    // 清空现有选项（保留"全部专栏"）
    while (columnFilter.options.length > 1) {
        columnFilter.remove(1);
    }
    
    // 添加专栏选项
    console.log('开始添加专栏选项');
    columns.forEach((column, index) => {
        console.log(`添加专栏 ${index + 1}:`, column);
        const option = document.createElement('option');
        option.value = column;
        option.textContent = column;
        columnFilter.appendChild(option);
    });
    
    console.log('专栏筛选初始化完成');
}

// 按专栏筛选文章
function filterArticlesByColumn() {
    console.log('===== 开始按专栏筛选文章 =====');
    
    // 获取选择的专栏
    const columnFilter = document.getElementById('column-filter');
    if (!columnFilter) {
        console.error('未找到 column-filter 元素');
        return;
    }
    
    const selectedColumn = columnFilter.value;
    console.log('选择的专栏:', selectedColumn);
    
    // 筛选文章
    let filteredArticles;
    if (selectedColumn === 'all') {
        // 显示所有文章
        filteredArticles = articles;
    } else {
        // 显示指定专栏的文章
        filteredArticles = articles.filter(article => article.column === selectedColumn);
    }
    
    console.log('筛选后的文章数量:', filteredArticles.length);
    console.log('筛选后的文章数据:', filteredArticles);
    
    // 加载筛选后的文章
    loadArticles(filteredArticles);
    
    // 更新搜索结果提示
    const searchResult = document.getElementById('search-result');
    if (searchResult) {
        if (selectedColumn === 'all') {
            searchResult.textContent = '';
            searchResult.classList.remove('show');
        } else {
            searchResult.textContent = `找到 ${filteredArticles.length} 篇来自专栏 "${selectedColumn}" 的文章`;
            searchResult.classList.add('show');
        }
    }
    
    console.log('按专栏筛选文章完成');
}

// 加载文章列表
function loadArticles(filteredArticles = null) {
    console.log('===== 开始加载文章列表 =====');
    
    // 检查articles数组是否存在
    console.log('articles变量是否存在:', typeof articles !== 'undefined');
    if (typeof articles === 'undefined') {
        console.error('articles数组未定义');
        return;
    }
    
    // 检查articles数组内容
    console.log('原始articles数组长度:', articles.length);
    console.log('原始articles数组内容:', articles);
    
    // 获取文章容器
    const articlesContainer = document.getElementById('articles-container');
    console.log('articles-container元素:', articlesContainer);
    if (!articlesContainer) {
        console.error('未找到 articles-container 元素');
        return;
    }
    
    // 使用过滤后的文章或原始文章
    const displayArticles = filteredArticles || articles;
    
    console.log('要显示的文章数量:', displayArticles.length);
    console.log('要显示的文章数据:', displayArticles);
    
    // 清空容器
    console.log('清空文章容器');
    articlesContainer.innerHTML = '';
    
    // 添加加载动画
    console.log('添加加载动画');
    articlesContainer.innerHTML = '<div class="loading"></div>';
    
    // 模拟异步加载
    console.log('开始模拟异步加载');
    setTimeout(() => {
        console.log('异步加载完成，开始显示文章');
        // 清空加载动画
        articlesContainer.innerHTML = '';
        
        // 检查文章数组
        if (displayArticles.length === 0) {
            console.log('文章数量为0，显示"暂无文章"');
            articlesContainer.innerHTML = '<p>暂无文章</p>';
            return;
        }
        
        // 遍历文章数据，生成文章卡片
        console.log('开始遍历文章数据，生成文章卡片');
        displayArticles.forEach((article, index) => {
            console.log(`加载文章 ${index + 1}:`, article.title);
            console.log(`文章ID:`, article.id);
            console.log(`文章摘要:`, article.summary);
            console.log(`文章日期:`, article.date);
            console.log(`文章专栏:`, article.column);
            
            const articleCard = document.createElement('a');
            articleCard.href = `article.html?id=${article.id}`;
            articleCard.className = 'article-card';
            
            // 生成文章卡片HTML，包含专栏信息
            let cardHtml = `
                <h3>${article.title}</h3>
                <p>${article.summary}</p>
                <div class="article-meta">
                    <div class="article-date">${article.date}</div>
            `;
            
            // 如果文章属于某个专栏，添加专栏标签
            if (article.column) {
                cardHtml += `
                    <div class="article-column">专栏：${article.column}</div>
                `;
            }
            
            cardHtml += `
                </div>
            `;
            
            articleCard.innerHTML = cardHtml;
            
            console.log('创建文章卡片:', articleCard);
            articlesContainer.appendChild(articleCard);
            console.log('文章卡片添加到容器');
        });
        
        console.log('文章加载完成');
    }, 500);
}

// 搜索文章
function searchArticles() {
    const searchInput = document.getElementById('search-input');
    const searchResult = document.getElementById('search-result');
    
    if (!searchInput || !searchResult) {
        console.error('未找到搜索相关元素');
        return;
    }
    
    const searchTerm = searchInput.value.trim().toLowerCase();
    
    if (!searchTerm) {
        // 搜索词为空，显示所有文章
        searchResult.textContent = '';
        searchResult.classList.remove('show');
        loadArticles();
        return;
    }
    
    // 过滤文章
    const filteredArticles = articles.filter(article => {
        return (
            article.title.toLowerCase().includes(searchTerm) ||
            article.summary.toLowerCase().includes(searchTerm) ||
            article.content.toLowerCase().includes(searchTerm)
        );
    });
    
    // 显示搜索结果
    searchResult.textContent = `找到 ${filteredArticles.length} 篇相关文章`;
    searchResult.classList.add('show');
    
    // 加载过滤后的文章
    loadArticles(filteredArticles);
}

// 重置搜索
function resetSearch() {
    const searchInput = document.getElementById('search-input');
    const searchResult = document.getElementById('search-result');
    
    if (!searchInput || !searchResult) {
        console.error('未找到搜索相关元素');
        return;
    }
    
    // 清空搜索输入
    searchInput.value = '';
    
    // 隐藏搜索结果
    searchResult.textContent = '';
    searchResult.classList.remove('show');
    
    // 加载所有文章
    loadArticles();
}

// 排序文章
function sortArticles() {
    const sortSelect = document.getElementById('sort-select');
    if (!sortSelect) {
        console.error('未找到 sort-select 元素');
        return;
    }
    
    const sortValue = sortSelect.value;
    
    // 创建文章副本
    const sortedArticles = [...articles];
    
    // 根据排序值排序
    switch (sortValue) {
        case 'date-desc':
            // 按日期降序
            sortedArticles.sort((a, b) => new Date(b.date) - new Date(a.date));
            break;
        case 'date-asc':
            // 按日期升序
            sortedArticles.sort((a, b) => new Date(a.date) - new Date(b.date));
            break;
        case 'title-asc':
            // 按标题升序
            sortedArticles.sort((a, b) => a.title.localeCompare(b.title));
            break;
        case 'title-desc':
            // 按标题降序
            sortedArticles.sort((a, b) => b.title.localeCompare(a.title));
            break;
    }
    
    // 加载排序后的文章
    loadArticles(sortedArticles);
}

// 加载文章详情
function loadArticleDetail() {
    const articleDetailContainer = document.getElementById('article-detail-container');
    if (!articleDetailContainer) {
        console.error('未找到 article-detail-container 元素');
        return;
    }
    
    // 清空容器
    articleDetailContainer.innerHTML = '';
    
    // 添加加载动画
    articleDetailContainer.innerHTML = '<div class="loading"></div>';
    
    // 获取文章ID
    const urlParams = new URLSearchParams(window.location.search);
    const articleId = urlParams.get('id');
    
    console.log('文章ID:', articleId);
    
    // 模拟异步加载
    setTimeout(() => {
        // 查找文章
        const article = articles.find(a => a.id === articleId);
        
        console.log('找到文章:', article);
        
        if (article) {
            // 清空加载动画
            articleDetailContainer.innerHTML = '';
            
            // 生成文章内容
            const articleElement = document.createElement('div');
            articleElement.className = 'article-detail';
            
            // 解析Markdown内容
            const parsedContent = parseMarkdown(article.content);
            
            // 生成文章元信息，包含专栏信息
            let metaHtml = `
                <div class="article-meta">
                    <span>发布日期：${article.date}</span>
            `;
            
            // 如果文章属于某个专栏，添加专栏信息
            if (article.column) {
                metaHtml += `
                    <span>专栏：${article.column}</span>
                `;
            }
            
            metaHtml += `
                </div>
            `;
            
            articleElement.innerHTML = `
                <h1>${article.title}</h1>
                ${metaHtml}
                <div class="article-content">
                    ${parsedContent}
                </div>
            `;
            
            articleDetailContainer.appendChild(articleElement);
        } else {
            // 文章不存在
            articleDetailContainer.innerHTML = '<h2>文章不存在</h2><p>抱歉，您访问的文章不存在。</p>';
        }
    }, 500);
}

// 增强的Markdown解析函数（使用正则表达式直接处理）
function parseMarkdown(markdown) {
    console.log('解析Markdown内容:', markdown.substring(0, 100) + '...');
    
    // 1. 先处理代码块，避免被其他规则干扰
    markdown = markdown.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');
    
    // 2. 处理LaTeX公式，避免被其他规则干扰
    // 先保存行间公式
    const blockFormulas = [];
    let blockFormulaIndex = 0;
    markdown = markdown.replace(/\$\$([\s\S]*?)\$\$/g, (_, content) => {
        const placeholder = `__BLOCK_FORMULA_${blockFormulaIndex}__`;
        blockFormulas[blockFormulaIndex] = content.trim();
        blockFormulaIndex++;
        return placeholder;
    });
    // 保存行内公式
    const inlineFormulas = [];
    let inlineFormulaIndex = 0;
    markdown = markdown.replace(/(?<!\\)\$(?!\$)([^\$]+?)(?<!\\)\$(?!\$)/g, (_, content) => {
        const placeholder = `__INLINE_FORMULA_${inlineFormulaIndex}__`;
        inlineFormulas[inlineFormulaIndex] = content.trim();
        inlineFormulaIndex++;
        return placeholder;
    });
    
    // 3. 处理图片语法
    // 直接替换图片语法为img标签
    markdown = markdown.replace(/!\[([^\]]*)\]\(([^\)]+)\)/g, function(match, alt, src) {
        // 确保图片路径正确
        // 检查src是否已经包含articles/前缀
        let imgSrc;
        if (src.startsWith('articles/')) {
            // 已经包含articles/前缀，直接使用
            imgSrc = src;
        } else {
            // 不包含articles/前缀，添加前缀
            imgSrc = `articles/${src}`;
        }
        return `<img src="${imgSrc}" alt="${alt}" style="max-width: 100%; height: auto; margin: 1em 0;">`;
    });
    
    // 4. 处理表格语法
    // 匹配完整的表格结构
    const tableRegex = /(?:^\|.*\|$\s*)+/gm;
    markdown = markdown.replace(tableRegex, (tableContent) => {
        const rows = tableContent.trim().split('\n').filter(row => row.trim());
        if (rows.length < 2) return tableContent;
        
        let tableHtml = '<table style="border-collapse: collapse; width: 100%; margin: 1em 0;">';
        
        // 处理表头
        const headerRow = rows[0];
        const headerCells = headerRow.split('|').filter(cell => cell !== '').map(cell => cell.trim());
        tableHtml += '<tr style="border: 1px solid #ddd;">';
        headerCells.forEach(cell => {
            tableHtml += `<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">${cell}</th>`;
        });
        tableHtml += '</tr>';
        
        // 跳过可能的分隔行
        let startRow = 1;
        if (rows[1] && rows[1].match(/^\|\s*[:\-]+\s*\|/)) {
            startRow = 2;
        }
        
        // 处理表格内容行
        for (let i = startRow; i < rows.length; i++) {
            const row = rows[i];
            const cells = row.split('|').filter(cell => cell !== '').map(cell => cell.trim());
            tableHtml += '<tr style="border: 1px solid #ddd;">';
            cells.forEach(cell => {
                tableHtml += `<td style="border: 1px solid #ddd; padding: 8px;">${cell}</td>`;
            });
            tableHtml += '</tr>';
        }
        
        tableHtml += '</table>';
        return tableHtml;
    });
    
    // 5. 处理标题
    markdown = markdown.replace(/^(#{1,6})\s+(.*?)$/gim, function(match, hashes, content) {
        const level = hashes.length;
        return `<h${level}>${content}</h${level}>`;
    });
    
    // 6. 处理加粗
    markdown = markdown.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // 7. 处理斜体
    markdown = markdown.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // 8. 处理高亮
    markdown = markdown.replace(/==(.*?)==/g, '<mark>$1</mark>');
    
    // 9. 处理行内代码
    markdown = markdown.replace(/`(.*?)`/g, '<code>$1</code>');
    
    // 10. 处理列表
    // 处理无序列表
    markdown = markdown.replace(/^-\s+(.*?)$/gim, '<li>$1</li>');
    // 处理有序列表
    markdown = markdown.replace(/^\d+\.\s+(.*?)$/gim, '<li>$1</li>');
    // 包裹列表
    markdown = markdown.replace(/(<li>.*?<\/li>)/gs, '<ul>$1</ul>');
    
    // 11. 处理段落
    markdown = markdown.replace(/^(?!<h[1-6]>)(?!<ul>)(?!<li>)(?!<pre>)(?!<code>)(?!<table>)(?!<tr>)(?!<td>)(?!<th>)(?!<img>)(.*?)$/gim, function(match) {
        if (match.trim()) {
            return '<p>' + match + '</p>';
        }
        return match;
    });
    
    // 12. 恢复并渲染LaTeX公式
    // 渲染行间公式
    markdown = markdown.replace(/__BLOCK_FORMULA_([0-9]+)__/g, (_, index) => {
        try {
            const formula = blockFormulas[index];
            if (formula) {
                return katex.renderToString(formula, {
                    displayMode: true,
                    throwOnError: false,
                    strict: 'ignore'
                });
            } else {
                return '$$' + index + '$$';
            }
        } catch (e) {
            return '$$' + index + '$$';
        }
    });
    // 渲染行内公式
    markdown = markdown.replace(/__INLINE_FORMULA_([0-9]+)__/g, (_, index) => {
        try {
            const formula = inlineFormulas[index];
            if (formula) {
                return katex.renderToString(formula, {
                    displayMode: false,
                    throwOnError: false,
                    strict: 'ignore'
                });
            } else {
                return '$' + index + '$';
            }
        } catch (e) {
            return '$' + index + '$';
        }
    });
    
    // 13. 清理多余的标签和空格
    markdown = markdown.replace(/<\/p>\s*<p>/g, '</p><p>');
    markdown = markdown.replace(/\s+/g, ' ');
    markdown = markdown.replace(/\s*<\/p>/g, '</p>');
    markdown = markdown.replace(/<p>\s*/g, '<p>');
    
    console.log('解析结果:', markdown.substring(0, 100) + '...');
    return markdown;
}
