// 博客文章数据
const articles = [
    {
        "id": "article_20260130034212",
        "title": "示例文章",
        "date": "2026-01-30",
        "summary": "这是一篇示例文章，用于测试文章更新脚本。",
        "content": "# 示例文章\n\n## 什么是示例文章？\n\n示例文章是用于测试和演示的文章，通常包含一些基本的内容结构和格式。\n\n## 示例内容\n\n### 列表示例\n\n- 项目1\n- 项目2\n- 项目3\n\n### 代码示例\n\n```python\nprint(\"Hello, World!\")\n```\n\n### 引用示例\n\n> 这是一段引用文字，用于展示引用格式。\n\n## 总结\n\n这篇示例文章展示了Markdown的基本语法和格式，包括标题、列表、代码块和引用等。"
    }
];

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 检查当前页面
    const currentPath = window.location.pathname;
    const currentFileName = currentPath.split('/').pop();
    
    console.log('当前页面:', currentFileName);
    
    // 如果是博客页面，加载文章列表
    if (currentFileName === 'blog.html') {
        console.log('加载文章列表...');
        loadArticles();
        
        // 为搜索按钮添加事件监听器
        const searchButton = document.getElementById('search-button');
        if (searchButton) {
            searchButton.addEventListener('click', searchArticles);
        }
        
        // 为搜索输入框添加回车事件监听器
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    searchArticles();
                }
            });
        }
        
        // 为重置按钮添加事件监听器
        const resetButton = document.getElementById('reset-button');
        if (resetButton) {
            resetButton.addEventListener('click', resetSearch);
        }
        
        // 为排序选择框添加事件监听器
        const sortSelect = document.getElementById('sort-select');
        if (sortSelect) {
            sortSelect.addEventListener('change', sortArticles);
        }
    }
    
    // 如果是文章详情页面，加载文章内容
    if (currentFileName === 'article.html') {
        console.log('加载文章详情...');
        loadArticleDetail();
    }
});

// 加载文章列表
function loadArticles(filteredArticles = null) {
    const articlesContainer = document.getElementById('articles-container');
    if (!articlesContainer) {
        console.error('未找到 articles-container 元素');
        return;
    }
    
    // 使用过滤后的文章或原始文章
    const displayArticles = filteredArticles || articles;
    
    console.log('文章数量:', displayArticles.length);
    console.log('文章数据:', displayArticles);
    
    // 清空容器
    articlesContainer.innerHTML = '';
    
    // 添加加载动画
    articlesContainer.innerHTML = '<div class="loading"></div>';
    
    // 模拟异步加载
    setTimeout(() => {
        // 清空加载动画
        articlesContainer.innerHTML = '';
        
        // 检查文章数组
        if (displayArticles.length === 0) {
            articlesContainer.innerHTML = '<p>暂无文章</p>';
            return;
        }
        
        // 遍历文章数据，生成文章卡片
        displayArticles.forEach((article, index) => {
            console.log('加载文章:', index, article.title);
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
