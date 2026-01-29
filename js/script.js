// 博客文章数据
const articles = [
    {
        id: 'article1',
        title: '深度学习入门指南',
        date: '2026-01-01',
        summary: '本文介绍深度学习的基本概念、常用框架和实践方法，帮助初学者快速入门。',
        content: `# 深度学习入门指南

## 什么是深度学习？
深度学习是机器学习的一个分支，通过模拟人脑的神经网络结构，实现对复杂数据的自动特征提取和模式识别。

## 常用深度学习框架
- **TensorFlow**：Google开发的开源框架，功能强大，生态丰富
- **PyTorch**：Facebook开发的框架，动态计算图，易于调试
- **Keras**：高级神经网络API，可基于TensorFlow或PyTorch后端

## 深度学习应用场景
- 计算机视觉：图像分类、目标检测、人脸识别
- 自然语言处理：机器翻译、情感分析、文本生成
- 语音识别：语音转文本、声纹识别
- 推荐系统：个性化推荐、内容过滤

## 学习资源推荐
- 书籍：《深度学习》（花书）、《动手学深度学习》
- 在线课程：Coursera深度学习专项课程、Fast.ai
- 实践项目：Kaggle竞赛、GitHub开源项目

## 总结
深度学习是一门需要理论与实践相结合的学科，通过不断学习和动手实践，才能真正掌握其精髓。`
    },
    {
        id: 'article2',
        title: 'Python数据分析入门',
        date: '2025-12-15',
        summary: '本文介绍Python数据分析的常用库和工具，包括NumPy、Pandas、Matplotlib等，帮助读者快速上手数据分析。',
        content: `# Python数据分析入门

## 为什么选择Python进行数据分析？
Python具有丰富的数据分析库生态，语法简洁易懂，社区活跃，是数据分析的理想选择。

## 常用数据分析库
- **NumPy**：数值计算库，提供高效的数组操作
- **Pandas**：数据处理库，提供DataFrame数据结构，方便数据清洗和分析
- **Matplotlib**：数据可视化库，支持多种图表类型
- **Seaborn**：基于Matplotlib的高级可视化库，提供更美观的图表样式
- **Scikit-learn**：机器学习库，提供常用的机器学习算法

## 数据分析流程
1. **数据获取**：从文件、数据库或API获取数据
2. **数据清洗**：处理缺失值、异常值，数据类型转换
3. **数据探索**：描述性统计，数据可视化
4. **特征工程**：特征选择、特征提取、特征转换
5. **模型构建**：选择合适的模型，训练模型
6. **模型评估**：评估模型性能，调参优化
7. **结果可视化**：展示分析结果，生成报告

## 实践案例
### 示例：分析学生成绩数据

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv('student_scores.csv')

# 查看数据基本信息
print(df.info())
print(df.describe())

# 数据可视化
sns.scatterplot(x='hours', y='score', data=df)
plt.title('学习时间与成绩关系')
plt.show()

# 简单线性回归
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[['hours']]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f'模型准确率：{model.score(X_test, y_test)}')
```

## 总结
Python数据分析是一项实用技能，通过掌握相关库和工具，可以快速从数据中提取有价值的信息，为决策提供支持。`
    },
    {
        id: 'article3',
        title: 'Git版本控制最佳实践',
        date: '2025-11-20',
        summary: '本文介绍Git版本控制的最佳实践，包括分支管理、提交规范、代码审查等，帮助团队提高协作效率。',
        content: `# Git版本控制最佳实践

## 什么是Git？
Git是一个分布式版本控制系统，用于跟踪代码的变更，支持多人协作开发。

## Git工作流程
### 1. 分支管理
- **main/master分支**：存放稳定的生产代码
- **develop分支**：集成开发分支，包含最新的开发代码
- **feature分支**：用于开发新功能
- **bugfix分支**：用于修复bug
- **release分支**：用于发布版本

### 2. 提交规范
提交信息应清晰明了，遵循以下格式：

```
<类型>(<范围>): <描述>

<详细描述>

<footer>
```

类型包括：
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码样式调整
- refactor: 代码重构
- test: 测试相关
- chore: 构建或依赖更新

### 3. 代码审查
- 使用Pull Request进行代码审查
- 至少有一人审核通过后才能合并
- 审查重点：代码质量、功能正确性、性能影响

## Git常用命令

### 基本操作
- `git init`：初始化仓库
- `git clone <url>`：克隆仓库
- `git add <file>`：添加文件到暂存区
- `git commit -m "message"`：提交更改
- `git push`：推送更改到远程仓库
- `git pull`：从远程仓库拉取更改

### 分支操作
- `git branch`：查看分支
- `git branch <name>`：创建分支
- `git checkout <name>`：切换分支
- `git merge <name>`：合并分支
- `git branch -d <name>`：删除分支

### 撤销操作
- `git reset HEAD <file>`：从暂存区移除文件
- `git checkout -- <file>`：撤销工作区更改
- `git revert <commit>`：撤销提交

## 最佳实践建议
1. **频繁提交**：每次提交应包含一个逻辑完整的更改
2. **合理使用分支**：不同功能使用不同分支，避免直接在main分支开发
3. **定期合并**：将develop分支的更改定期合并到main分支
4. **编写清晰的提交信息**：便于后续查找和理解更改
5. **使用.gitignore文件**：忽略不需要版本控制的文件
6. **定期备份**：确保远程仓库的安全

## 总结
Git版本控制是现代软件开发中不可或缺的工具，遵循最佳实践可以提高团队协作效率，减少代码冲突，保证代码质量。`
    }
];

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 检查当前页面
    const currentPath = window.location.pathname;
    
    // 如果是博客页面，加载文章列表
    if (currentPath.includes('blog.html')) {
        loadArticles();
    }
    
    // 如果是文章详情页面，加载文章内容
    if (currentPath.includes('article.html')) {
        loadArticleDetail();
    }
});

// 加载文章列表
function loadArticles() {
    const articlesContainer = document.getElementById('articles-container');
    if (!articlesContainer) return;
    
    // 清空容器
    articlesContainer.innerHTML = '';
    
    // 添加加载动画
    articlesContainer.innerHTML = '<div class="loading"></div>';
    
    // 模拟异步加载
    setTimeout(() => {
        // 清空加载动画
        articlesContainer.innerHTML = '';
        
        // 遍历文章数据，生成文章卡片
        articles.forEach(article => {
            const articleCard = document.createElement('a');
            articleCard.href = `article.html?id=${article.id}`;
            articleCard.className = 'article-card';
            
            articleCard.innerHTML = `
                <h3>${article.title}</h3>
                <p>${article.summary}</p>
                <div class="article-date">${article.date}</div>
            `;
            
            articlesContainer.appendChild(articleCard);
        });
    }, 500);
}

// 加载文章详情
function loadArticleDetail() {
    const articleDetailContainer = document.getElementById('article-detail-container');
    if (!articleDetailContainer) return;
    
    // 清空容器
    articleDetailContainer.innerHTML = '';
    
    // 添加加载动画
    articleDetailContainer.innerHTML = '<div class="loading"></div>';
    
    // 获取文章ID
    const urlParams = new URLSearchParams(window.location.search);
    const articleId = urlParams.get('id');
    
    // 模拟异步加载
    setTimeout(() => {
        // 查找文章
        const article = articles.find(a => a.id === articleId);
        
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
    
    return markdown;
}
