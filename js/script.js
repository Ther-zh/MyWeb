// 博客文章数据
const articles = [
    {
        "id": "article_20260130032908",
        "title": "示例文章",
        "date": "2026-01-30",
        "summary": "这是一篇示例文章，用于测试文章更新脚本。",
        "content": "# 示例文章

## 什么是示例文章？

示例文章是用于测试和演示的文章，通常包含一些基本的内容结构和格式。

## 示例内容

### 列表示例

- 项目1
- 项目2
- 项目3

### 代码示例

```python
print(\"Hello, World!\")
```

### 引用示例

> 这是一段引用文字，用于展示引用格式。

## 总结

这篇示例文章展示了Markdown的基本语法和格式，包括标题、列表、代码块和引用等。"
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
