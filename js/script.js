// 博客文章数据
const articles = [
    {
        "id": "pytorch_20260130133638",
        "title": "PyTorch核心模块的依赖关系",
        "date": "2026-01-30",
        "summary": "在PyTorch中，各个模块的设计遵循“分工明确、协同工作”的原则，核心目标是支持从数据加载到模型训练、推理的全流程。理解模块间的依赖关系和训练流程的逻辑，能帮你更清晰地掌握PyTorch的使用框架。",
        "content": "### 一、PyTorch核心模块的依赖关系\n\nPyTorch的核心模块主要围绕“数据→模型→损失→优化”这一主线设计，各模块职责和依赖关系如下：\n\n\n#### 1. 数据处理模块：`torch.utils.data`  \n**核心组件**：`Dataset`（数据集抽象类）、`DataLoader`（数据加载器）。  \n**作用**：负责数据的读取、预处理（如归一化、转换）、批量加载、打乱顺序等。  \n**依赖**：无强依赖其他模块，但需要用户自定义数据格式（如将图片/标签封装成`Dataset`），最终为模型提供输入数据（`input`）和标签（`label`）。  \n\n\n#### 2. 模型构建模块：`torch.nn`  \n**核心组件**：`Module`（所有模型的基类）、`Linear`/`Conv2d`（层）、`ReLU`/`Softmax`（激活函数）、`CrossEntropyLoss`（损失函数）等。  \n**作用**：定义模型的网络结构（层的堆叠、计算逻辑），并提供损失函数（用于衡量预测误差）。  \n**依赖**：  \n- 模型的构建依赖`nn.Module`（所有自定义模型必须继承它）；  \n- 损失函数（如`nn.CrossEntropyLoss`）依赖模型的输出（`output`）和真实标签（`label`），用于计算损失值（`loss`）。  \n\n\n#### 3. 优化器模块：`torch.optim`  \n**核心组件**：`SGD`、`Adam`等优化器。  \n**作用**：根据损失的梯度（`gradient`）更新模型参数（`parameters`），最小化损失函数。  \n**依赖**：  \n- 必须依赖模型的参数（`model.parameters()`）—— 优化器需要知道“要更新哪些参数”；  \n- 依赖损失的梯度（通过`loss.backward()`计算得到）—— 优化器需要根据梯度方向调整参数。  \n\n\n#### 4. 张量与计算图：`torch.Tensor`  \n**核心组件**：`Tensor`（张量，PyTorch的基本数据结构）、自动求导机制（`autograd`）。  \n**作用**：所有数据（输入、模型参数、中间结果、损失）都以张量形式存在；`autograd`自动构建计算图，支持梯度反向传播。  \n**依赖**：是所有模块的底层依赖——数据模块输出张量，模型对张量做运算，损失函数和优化器基于张量的梯度工作。  \n\n\n#### 模块依赖关系总结：  \n```  \n数据模块（Dataset/DataLoader）→ 输出输入张量（input）和标签（label）  \n↓  \n模型模块（nn.Module）→ 接收input，输出预测值（output）  \n↓  \n损失函数（nn.xxxLoss）→ 接收output和label，输出损失值（loss）  \n↓  \n自动求导（autograd）→ 基于loss计算所有参数的梯度（通过loss.backward()）  \n↓  \n优化器（optim.xxx）→ 接收参数梯度，更新模型参数（通过optimizer.step()）  \n```\n\n\n### 二、模型训练的完整流程及逻辑  \n训练流程的核心是“迭代优化”：通过不断输入数据、计算误差、调整参数，让模型逐渐学会拟合数据。具体步骤和逻辑如下：  \n\n\n#### 阶段1：准备工作（只执行1次）  \n在开始训练循环前，需要完成以下初始化：  \n\n1. **准备数据**  \n   - 定义`Dataset`：将原始数据（如图片、标签）封装成PyTorch可识别的格式（需实现`__getitem__`和`__len__`方法）；  \n   - 定义`DataLoader`：对`Dataset`进行批量处理（`batch_size`）、打乱（`shuffle=True`）、多进程加载（`num_workers`）等，最终得到可迭代的数据批次（`batch`）。  \n\n   ```python  \n   from torch.utils.data import Dataset, DataLoader  \n   \n   class MyDataset(Dataset):  \n       def __init__(self, data, labels):  \n           self.data = data  \n           self.labels = labels  \n       def __getitem__(self, idx):  \n           return self.data[idx], self.labels[idx]  \n       def __len__(self):  \n           return len(self.data)  \n   \n   dataset = MyDataset(train_data, train_labels)  \n   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  \n   ```\n\n\n2. **定义模型**  \n   - 继承`nn.Module`自定义模型，在`__init__`中定义层（如`Conv2d`、`Linear`），在`forward`中定义计算逻辑（输入→输出的映射）。  \n\n   ```python  \n   import torch.nn as nn  \n   \n   class MyModel(nn.Module):  \n       def __init__(self):  \n           super().__init__()  \n           self.fc1 = nn.Linear(784, 256)  # 假设输入是28x28的MNIST图片（展平后784维）  \n           self.fc2 = nn.Linear(256, 10)   # 输出10类（0-9）  \n       def forward(self, x):  \n           x = x.view(-1, 784)  # 展平（batch_size, 784）  \n           x = nn.functional.relu(self.fc1(x))  \n           x = self.fc2(x)  \n           return x  \n   \n   model = MyModel()  # 实例化模型  \n   ```\n\n\n3. **定义损失函数和优化器**  \n   - 损失函数：衡量预测值（`output`）与真实标签（`label`）的差距（如分类用`CrossEntropyLoss`）；  \n   - 优化器：接收模型参数，定义更新策略（如`Adam`、`SGD`）。  \n\n   ```python  \n   criterion = nn.CrossEntropyLoss()  # 损失函数  \n   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器（依赖模型参数）  \n   ```\n\n\n#### 阶段2：训练循环（迭代执行，直到收敛）  \n每次循环处理一个批次的数据，核心是“前向传播→计算损失→反向传播→参数更新”的闭环。  \n\n1. **设置模型为训练模式**  \n   - 调用`model.train()`：启用 dropout、batch normalization 等训练特有的层（这些层在训练和推理时行为不同）。  \n\n   ```python  \n   model.train()  # 训练模式  \n   ```\n\n\n2. **迭代数据批次**  \n   对`dataloader`中的每个批次（`inputs, labels`）执行以下步骤：  \n\n   ```python  \n   for epoch in range(10):  # 训练10轮（所有数据过一遍为1轮）  \n       running_loss = 0.0  \n       for inputs, labels in dataloader:  # 遍历每个批次  \n           # 步骤1：清空梯度（关键！避免梯度累积）  \n           optimizer.zero_grad()  \n   \n           # 步骤2：前向传播（ Forward ）  \n           outputs = model(inputs)  # 模型预测：inputs → outputs  \n   \n           # 步骤3：计算损失（ Loss ）  \n           loss = criterion(outputs, labels)  # 对比outputs和labels，得到损失  \n   \n           # 步骤4：反向传播（ Backward ）  \n           loss.backward()  # 自动计算所有参数的梯度（基于计算图）  \n   \n           # 步骤5：参数更新（ Optimize ）  \n           optimizer.step()  # 优化器根据梯度更新模型参数  \n   \n           # 统计损失（可选）  \n           running_loss += loss.item()  \n       print(f\"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}\")  \n   ```\n\n\n#### 阶段3：验证/测试（可选，通常每轮训练后执行）  \n验证模型在未见过的数据上的性能，判断是否过拟合。  \n\n1. **设置模型为评估模式**  \n   - 调用`model.eval()`：关闭 dropout、固定 batch normalization 的统计量（用训练时的均值/方差）。  \n\n   ```python  \n   model.eval()  # 评估模式  \n   ```\n\n\n2. **关闭梯度计算（节省资源）**  \n   - 用`torch.no_grad()`上下文管理器：验证时不需要计算梯度（无需更新参数），减少内存占用。  \n\n   ```python  \n   correct = 0  \n   total = 0  \n   with torch.no_grad():  # 关闭梯度计算  \n       for inputs, labels in test_dataloader:  \n           outputs = model(inputs)  \n           _, predicted = torch.max(outputs.data, 1)  # 取预测概率最大的类别  \n           total += labels.size(0)  \n           correct += (predicted == labels).sum().item()  \n   print(f\"测试准确率：{100 * correct / total}%\")  \n   ```\n\n\n### 三、关键逻辑总结  \n1. **依赖逻辑**：数据是输入，模型是核心计算单元，损失是误差指标，优化器是参数调整工具，四者环环相扣；  \n2. **训练闭环**：必须严格遵循“清空梯度→前向传播→计算损失→反向传播→更新参数”的顺序（比如梯度不清空会累积，导致更新混乱；没有损失无法反向传播）；  \n3. **模式切换**：训练用`model.train()`，验证用`model.eval()`+`torch.no_grad()`，确保层的行为符合场景需求。  \n\n理解这些逻辑后，无论模型简单（如线性回归）还是复杂（如Transformer），训练流程的核心框架都是一致的，只是细节（如模型结构、数据预处理）有所差异。"
    },
    {
        "id": "article_20260130133638",
        "title": "示例文章",
        "date": "2026-01-30",
        "summary": "这是一篇示例文章，用于测试文章更新脚本。",
        "content": "# 示例文章\n\n## 什么是示例文章？\n\n示例文章是用于测试和演示的文章，通常包含一些基本的内容结构和格式。\n\n## 示例内容\n\n### 列表示例\n\n- 项目1\n- 项目2\n- 项目3\n\n### 代码示例\n\n```python\nprint(\"Hello, World!\")\n```\n\n### 引用示例\n\n> 这是一段引用文字，用于展示引用格式。\n\n## 总结\n\n这篇示例文章展示了Markdown的基本语法和格式，包括标题、列表、代码块和引用等。"
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
    
    // 如果是博客页面，加载文章列表
    if (currentFileName === 'blog.html') {
        console.log('===== 检测到博客页面，开始加载文章列表 =====');
        loadArticles();
        
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
    }
    
    // 如果是文章详情页面，加载文章内容
    if (currentFileName === 'article.html') {
        console.log('===== 检测到文章详情页面，开始加载文章内容 =====');
        loadArticleDetail();
    }
    
    console.log('===== 页面初始化完成 =====');
});

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
            
            const articleCard = document.createElement('a');
            articleCard.href = `article.html?id=${article.id}`;
            articleCard.className = 'article-card';
            
            articleCard.innerHTML = `
                <h3>${article.title}</h3>
                <p>${article.summary}</p>
                <div class="article-date">${article.date}</div>
            `;
            
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
            
            articleElement.innerHTML = `
                <h1>${article.title}</h1>
                <div class="article-meta">
                    <span>发布日期：${article.date}</span>
                </div>
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

// 简单的Markdown解析函数
function parseMarkdown(markdown) {
    console.log('解析Markdown内容:', markdown.substring(0, 100) + '...');
    
    // 标题解析
    markdown = markdown.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    markdown = markdown.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    markdown = markdown.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
    // 段落解析
    markdown = markdown.replace(/^(?!<h[1-6]>)(.*$)/gim, function(match) {
        if (match.trim()) {
            return '<p>' + match + '</p>';
        }
        return match;
    });
    
    // 列表解析
    markdown = markdown.replace(/^- (.*$)/gim, '<li>$1</li>');
    markdown = markdown.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    
    // 代码块解析
    markdown = markdown.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');
    
    // 行内代码解析
    markdown = markdown.replace(/`(.*?)`/g, '<code>$1</code>');
    
    console.log('解析结果:', markdown.substring(0, 100) + '...');
    
    return markdown;
}
