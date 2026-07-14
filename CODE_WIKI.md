# Zhihang Zheng个人博客 - Code Wiki 文档

> 项目完整技术文档，包含架构、模块、API、配置和开发指南

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
- [3. 核心模块说明](#3-核心模块说明)
- [4. 文件结构](#4-文件结构)
- [5. 主要类与函数](#5-主要类与函数)
- [6. 依赖关系](#6-依赖关系)
- [7. 配置与部署](#7-配置与部署)
- [8. 开发指南](#8-开发指南)

---

## 1. 项目概述

### 1.1 项目简介

**Zhihang Zheng个人博客**是一个轻量级、无框架的个人博客网站，采用纯前端技术实现，支持 Markdown 文章、LaTeX 公式渲染、专栏分类、搜索和排序等功能。

### 1.2 主要功能

- **个人主页**：展示个人信息、技术栈、项目实践、荣誉奖项和未来展望
- **博客列表**：文章卡片展示、支持搜索、专栏筛选和日期/标题排序
- **文章详情**：支持 Markdown 解析、LaTeX 公式渲染、代码高亮和表格展示
- **文章管理**：通过 Python 脚本自动同步 Markdown 文章到前端数据
- **响应式设计**：完美适配桌面端和移动端

### 1.3 技术特点

- **无框架依赖**：纯 HTML5 + CSS3 + 原生 JavaScript 实现
- **动态内容**：前端路由和数据动态渲染
- **轻量级 Markdown 解析**：自研解析器，支持扩展
- **LaTeX 公式支持**：集成 KaTeX 渲染复杂数学公式
- **易于部署**：静态站点，支持 GitHub Pages、Netlify 等

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                      用户浏览器                          │
├─────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────┐  │
│  │  index.html (个人主页)   blog.html (博客列表)     │  │
│  │  article.html (文章详情)                          │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │           css/style.css (样式系统)                 │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  js/script.js                                     │  │
│  │  ├─ columns/articles 数组 (数据层)                │  │
│  │  ├─ 页面路由与初始化                              │  │
│  │  ├─ 博客列表/搜索/筛选/排序                       │  │
│  │  ├─ Markdown 解析器 + KaTeX 集成                  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                              ↑
                              │ (CDN 引用)
                              │
┌─────────────────────────────────────────────────────────┐
│        KaTeX (公式渲染)   marked (备用解析)            │
└─────────────────────────────────────────────────────────┘
                              ↑
                              │ (离线同步)
                              │
┌─────────────────────────────────────────────────────────┐
│      update_articles.py (Python 脚本)                  │
│      articles/ (Markdown 文章源)                       │
└─────────────────────────────────────────────────────────┘
```

### 2.2 层次架构

| 层级 | 职责 | 文件 |
|------|------|------|
| **表现层** | 页面展示与用户交互 | index.html, blog.html, article.html |
| **样式层** | 视觉呈现与响应式 | css/style.css |
| **逻辑层** | 业务逻辑与数据处理 | js/script.js |
| **数据层** | 文章与专栏数据 | articles/ (源数据), js/script.js (渲染数据) |
| **工具层** | 文章同步与管理 | update_articles.py |

---

## 3. 核心模块说明

### 3.1 个人主页模块 (index.html)

#### 功能概述
展示个人信息、技术背景、项目成果等静态内容。

#### 页面结构
```
┌─────────────────────────────┐
│        导航栏               │
├─────────────────────────────┤
│       英雄区域              │
│   (Hi，我是Zhihang Zheng + 简介)     │
├─────────────────────────────┤
│      关于我区域             │
├─────────────────────────────┤
│      技术栈网格             │
│ (核心语言/工具/能力/方向)   │
├─────────────────────────────┤
│      近期实践列表           │
├─────────────────────────────┤
│      竞赛荣誉列表           │
├─────────────────────────────┤
│      未来展望               │
├─────────────────────────────┤
│      联系区域               │
├─────────────────────────────┤
│         页脚                │
└─────────────────────────────┘
```

### 3.2 博客列表模块 (blog.html)

#### 功能概述
展示所有文章卡片，支持搜索、专栏筛选和排序功能。

#### 主要特性
- **搜索功能**：支持按文章标题、摘要和内容搜索
- **专栏筛选**：按文章所属专栏分类筛选
- **排序选项**：
  - 日期（最新优先）
  - 日期（最早优先）
  - 标题（A-Z）
  - 标题（Z-A）
- **文章卡片**：包含标题、摘要、日期、专栏标签

### 3.3 文章详情模块 (article.html)

#### 功能概述
渲染单篇文章的完整内容，支持 Markdown 和 LaTeX 公式。

#### 支持的 Markdown 特性
- 标题 (h1-h6)
- 加粗、斜体、高亮
- 有序/无序列表
- 行内代码和代码块
- 表格
- 图片
- LaTeX 公式（行内 `$公式$` 和块级 `$$公式$$`）

### 3.4 文章管理模块 (update_articles.py)

#### 功能概述
自动扫描 `articles/` 目录下的 Markdown 文件，解析并更新前端数据。

#### 支持的文章结构
```
articles/
├── 专栏1/
│   ├── 文章1/
│   │   ├── 文章1.md (或 index.md)
│   │   └── 图片资源...
│   └── 文章2/
│       └── ...
└── 无专栏文章/
    └── 文章.md
```

#### Markdown 格式要求
```markdown
---
title: 文章标题
date: 2026-01-30
summary: 文章摘要
---

正文内容...
```

---

## 4. 文件结构

### 4.1 完整目录树

```
MyWeb/
├── index.html                  # 个人主页
├── blog.html                   # 博客列表页
├── article.html                # 文章详情页
├── css/
│   └── style.css               # 全局样式
├── js/
│   └── script.js               # 核心 JavaScript 逻辑 + 文章数据
├── articles/                   # 文章源文件目录
│   ├── Pytorch/
│   │   ├── PyTorch框架学习（一）/
│   │   │   └── ...
│   │   └── PyTorch框架学习（二）/
│   │       └── ...
│   └── 信息论/
│       └── 信息论.md
├── update_articles.py          # 文章更新脚本
├── README.md                   # 项目说明
├── PROJECT_DOCUMENTATION.md    # 技术文档
├── ARTICLES_GUIDE.md           # 文章编写指南
└── CODE_WIKI.md                # 本文档
```

### 4.2 关键文件说明

| 文件 | 说明 |
|------|------|
| `index.html` | 个人主页，纯静态内容 |
| `blog.html` | 博客列表，包含搜索、筛选、排序 UI |
| `article.html` | 文章详情页，引入 KaTeX 和 marked CDN |
| `css/style.css` | 全局样式，包含响应式设计 |
| `js/script.js` | 核心逻辑 + 数据数组 |
| `update_articles.py` | Python 脚本，用于同步文章 |

---

## 5. 主要类与函数

### 5.1 js/script.js - 核心函数

#### 数据定义
```javascript
// 专栏数组
const columns = ["Pytorch"];

// 文章数组
const articles = [
  {
    id: "...",
    title: "...",
    date: "YYYY-MM-DD",
    summary: "...",
    content: "...",
    column: "...",  // 或 null
    path: "..."
  },
  // ...
];
```

#### 页面初始化与路由

| 函数 | 说明 |
|------|------|
| `DOMContentLoaded` 事件监听器 | 页面加载完成后执行，判断当前页面并初始化相应功能 |

#### 博客列表相关

| 函数 | 说明 |
|------|------|
| `loadArticles(filteredArticles)` | 渲染文章卡片到页面，支持传入过滤/排序后的数组 |
| `initColumnFilter()` | 初始化专栏筛选下拉框，填充 `columns` 数据 |
| `filterArticlesByColumn()` | 根据选择的专栏筛选文章 |
| `searchArticles()` | 按关键词搜索文章（标题/摘要/内容） |
| `resetSearch()` | 重置搜索，显示所有文章 |
| `sortArticles()` | 根据选择的排序方式对文章排序 |

#### 文章详情相关

| 函数 | 说明 |
|------|------|
| `loadArticleDetail()` | 从 URL 参数获取文章 ID，查找并渲染文章内容 |
| `parseMarkdown(markdown)` | 自研 Markdown 解析器，支持 LaTeX 公式 |

#### Markdown 解析器核心流程
```
parseMarkdown(markdown):
  1. 提取并保存 LaTeX 公式（块级和行内）
  2. 提取并保存代码块
  3. 处理标题语法
  4. 处理图片语法
  5. 处理表格语法
  6. 处理加粗/斜体/高亮/行内代码
  7. 处理列表
  8. 处理段落
  9. 恢复代码块
  10. 使用 KaTeX 渲染并恢复 LaTeX 公式
```

### 5.2 update_articles.py - Python 脚本函数

| 函数 | 说明 |
|------|------|
| `parse_front_matter(content)` | 解析 Markdown 文件的前言部分（YAML 风格） |
| `extract_content(content)` | 提取 Markdown 正文（去除前言） |
| `generate_article_id(title)` | 根据标题生成唯一文章 ID |
| `read_articles()` | 递归读取 articles/ 目录，解析所有文章 |
| `update_script(articles, columns)` | 更新 js/script.js 中的数据数组 |
| `git_push()` | 执行 Git 提交推送操作（可选） |
| `main()` | 主函数，流程控制 |

---

## 6. 依赖关系

### 6.1 外部依赖（CDN）

| 库 | 用途 | 引入位置 |
|----|------|----------|
| **KaTeX** | LaTeX 数学公式渲染 | article.html |
| **marked.js** | 备用 Markdown 解析器（当前未使用） | article.html |

### 6.2 内部模块依赖

```
article.html
    ├── depends on → css/style.css
    ├── depends on → js/script.js
    ├── depends on → KaTeX (CDN)
    └── depends on → marked (CDN)

blog.html
    ├── depends on → css/style.css
    └── depends on → js/script.js

index.html
    ├── depends on → css/style.css
    └── depends on → js/script.js

js/script.js
    ├── contains → columns / articles 数据（由 update_articles.py 生成）
    └── 无其他内部依赖

update_articles.py
    ├── reads from → articles/ 目录
    └── writes to → js/script.js
```

---

## 7. 配置与部署

### 7.1 本地开发

#### 方式一：直接打开文件
```bash
# 直接在浏览器中打开 index.html
```

#### 方式二：使用本地服务器（推荐）
```bash
# 使用 Python 3
python -m http.server

# 或使用 Python 2
python -m SimpleHTTPServer

# 然后访问 http://localhost:8000
```

### 7.2 部署到 GitHub Pages

1. 将项目推送到 GitHub 仓库
2. 在仓库设置中开启 GitHub Pages
3. 选择源分支（如 `main`）和根目录
4. 等待部署，访问 `https://<username>.github.io/<repo>`

### 7.3 部署到 Netlify

1. 将代码推送到 GitHub
2. 在 Netlify 中导入仓库
3. 配置：
   - Build command: 留空
   - Publish directory: 根目录
4. 部署完成

---

## 8. 开发指南

### 8.1 添加新文章

1. 在 `articles/` 目录下创建文章文件夹（专栏文章放在对应专栏文件夹下）
2. 创建 Markdown 文件（包含前言）
3. 运行更新脚本：
   ```bash
   cd MyWeb
   python update_articles.py
   ```
4. 在本地预览或部署

### 8.2 修改样式

直接编辑 `css/style.css`：
- 使用渐变色彩 `#1a1a2e` 到 `#16213e` 作为主题
- 蓝色系 `#4facfe` 作为主色调
- `@media (max-width: 768px)` 用于响应式

### 8.3 扩展功能

#### 示例：添加新的排序方式
1. 在 `blog.html` 的 `sort-select` 中添加新 option
2. 在 `js/script.js` 的 `sortArticles()` 函数中添加 case 分支

#### 示例：增强 Markdown 解析
在 `parseMarkdown()` 函数中添加新的正则表达式替换规则。

---

## 附录

### A. 现有文章一览

| 标题 | 日期 | 专栏 |
|------|------|------|
| PyTorch框架学习（一）：核心模块初识与训练流程全解析 | 2026-02-05 | Pytorch |
| PyTorch框架学习（二）：张量（Tensor）与自动求导（autograd）详解 | 2026-02-05 | Pytorch |
| 信息量、信息熵、KL散度交叉熵损失 | 2026-01-30 | - |

### B. 配色规范

| 用途 | 颜色值 |
|------|--------|
| 深色背景 | `#1a1a2e` |
| 次要深色 | `#16213e` |
| 主色调 | `#4facfe` |
| 辅助色 | `#00f2fe` |
| 文本色 | `#333` |
| 浅色背景 | `#f5f5f5` |

---

**文档版本**: 1.0  
**最后更新**: 2026-05-11
