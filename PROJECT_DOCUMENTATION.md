# MyWeb 项目文档

> 本文档由项目扫描自动生成，描述「知帆个人博客网站」的架构、文件与使用方式。

---

## 一、项目概述

**项目名称**：知帆个人博客网站（MyWeb）  
**类型**：静态个人博客 / 作品展示站  
**技术选型**：纯 HTML + CSS + JavaScript，无前端框架；文章通过 Markdown 管理，由 Python 脚本同步到前端数据。

**主要能力**：
- 个人主页：关于我、技术栈、近期实践、竞赛荣誉、未来展望、联系
- 博客列表：文章展示、搜索、专栏筛选、排序（日期/标题）
- 文章详情：Markdown 渲染、LaTeX 公式（KaTeX）、代码高亮、表格与图片
- 文章管理：在 `articles/` 下添加 MD 文件，运行 `update_articles.py` 同步到 `js/script.js`

---

## 二、技术栈

| 类别     | 技术 / 说明 |
|----------|--------------|
| 前端     | HTML5、CSS3、原生 JavaScript |
| 文章格式 | Markdown（含 Front Matter） |
| 公式渲染 | KaTeX（CDN，行内 `$...$`、块级 `$$...$$`） |
| Markdown 解析 | 自研 `parseMarkdown()`（script.js），支持标题、列表、代码块、表格、图片、加粗、高亮、LaTeX |
| 构建/工具 | 无；文章同步依赖 Python 3 脚本 `update_articles.py` |
| 部署     | 静态站点，可部署至 GitHub Pages、Netlify 等 |

---

## 三、目录结构（实际扫描）

```
MyWeb/
├── index.html          # 个人主页
├── blog.html           # 博客列表页（含搜索、专栏、排序）
├── article.html        # 文章详情页（根据 URL ?id= 渲染）
├── css/
│   └── style.css       # 全局样式（导航、英雄区、卡片、博客、文章、响应式、加载动画）
├── js/
│   └── script.js       # 文章数据(articles/columns)、列表/详情/搜索/筛选/排序、Markdown+LaTeX 解析
├── articles/           # 文章与专栏根目录
│   ├── Pytorch/        # 专栏：Pytorch
│   │   ├── PyTorch框架学习（一）：.../  # 文章目录（内含同名 .md）
│   │   └── PyTorch框架学习（二）：.../
│   └── 信息论/
│       └── 信息论.md   # 单文件文章（无子目录时无专栏）
├── images/             # 图片资源目录（可选）
├── update_articles.py  # 从 articles/ 扫描 MD，更新 script.js 中的 articles/columns
├── README.md           # 项目说明与使用指南
├── ARTICLES_GUIDE.md   # 文章编写与更新指南
└── PROJECT_DOCUMENTATION.md  # 本项目的技术文档（本文档）
```

---

## 四、核心文件说明

### 4.1 页面文件

- **index.html**  
  个人主页。包含：导航、英雄区（Hi 我是知帆）、关于我、技术栈（核心语言/开发工具/技术能力/研究方向）、近期实践、竞赛荣誉、未来展望、联系、页脚。  
  引入：`css/style.css`、`js/script.js`。

- **blog.html**  
  博客列表页。包含：导航、标题与说明、搜索框（输入+搜索+重置）、专栏下拉筛选、排序（日期升降序、标题 A-Z/Z-A）、文章卡片容器 `#articles-container`。  
  列表与筛选逻辑由 `script.js` 在 `DOMContentLoaded` 中根据路径识别 `blog.html` 后执行。

- **article.html**  
  文章详情页。包含：导航、`#article-detail-container`。  
  通过 `article.html?id=<文章id>` 定位文章，由 `script.js` 的 `loadArticleDetail()` 渲染；引入 KaTeX CSS/JS、marked（若使用）及 `script.js`。  
  当前实现以自研 `parseMarkdown()` 为主，文章内容支持 LaTeX（KaTeX 渲染）。

### 4.2 样式：css/style.css

- **全局**：重置、`body` 字体与背景。
- **布局**：`.container` 最大宽度 1000px，居中。
- **导航**：深色背景、sticky、Logo「知帆」、首页/博客链接。
- **首页区块**：hero 渐变、about/skills/projects/honors/future/contact 卡片式、section 标题带下划线。
- **博客页**：搜索栏、专栏筛选、排序下拉、`.articles-grid` 网格、`.article-card` 卡片、`.article-meta`（日期、专栏）。
- **文章详情**：`.article-detail`、标题与元信息、`.article-content` 段落/标题/列表/代码/表格样式。
- **通用**：footer、`.loading` 旋转动画、`@media (max-width: 768px)` 响应式（导航、hero 字号、单列网格）。

### 4.3 脚本：js/script.js

- **数据**  
  - `columns`：专栏名称数组（如 `["Pytorch"]`）。  
  - `articles`：文章数组，每项包含 `id`、`title`、`date`、`summary`、`content`、`column`（可选）、`path`（相对 `articles/` 的路径，用于图片等）。

- **路由与初始化**  
  - 根据 `window.location.pathname` 判断当前页为 `blog.html` 或 `article.html`，分别执行博客列表初始化或文章详情加载。

- **博客页**  
  - `loadArticles(filteredArticles?)`：向 `#articles-container` 渲染文章卡片；支持传入已筛选/排序的数组。  
  - `initColumnFilter()`：用 `columns` 填充 `#column-filter` 选项。  
  - `filterArticlesByColumn()`：按专栏筛选后调用 `loadArticles`。  
  - `searchArticles()`：按标题、摘要、正文关键词过滤后显示并更新 `#search-result`。  
  - `resetSearch()`：清空搜索并重新 `loadArticles()`。  
  - `sortArticles()`：按 `#sort-select` 对 `articles` 排序（日期升/降、标题 A-Z/Z-A）后 `loadArticles`。

- **文章详情页**  
  - `loadArticleDetail()`：从 `URLSearchParams` 取 `id`，在 `articles` 中查找，将 `content` 经 `parseMarkdown()` 转 HTML 后写入 `#article-detail-container`。

- **Markdown 与公式**  
  - `parseMarkdown(markdown)`：  
    - 先保护并替换块级 `$$...$$`、行内 `$...$` 为占位符；  
    - 再处理代码块、标题、图片、表格、加粗、斜体、高亮、行内代码、列表、段落；  
    - 最后将占位符用 KaTeX 渲染回 HTML。  
  - 图片路径：支持 `articles/` 前缀或相对路径，会补全为站内路径。

### 4.4 文章更新：update_articles.py

- **作用**：递归扫描 `articles/` 下所有 `.md` 文件，解析 Front Matter（`title`、`date`、`summary`），生成 `id`、`path`、`column`，将正文转成单行字符串（换行 `\n`）并处理图片路径，然后写回 `js/script.js` 中的 `const articles = [...]` 和 `const columns = [...]`。
- **文章结构约定**：  
  - 专栏：`articles/专栏名/文章目录/`，目录内放一个 `.md`（可为 `index.md` 或与文章同名的 `.md`）。  
  - 无专栏：`articles/文章名/xxx.md` 或 `articles/xxx.md`（单文件）。  
- **可选**：脚本末尾询问是否执行 `git add/commit/push`，便于发布流程自动化。

---

## 五、文章系统

### 5.1 数据流

1. 作者在 `articles/` 下按上述结构添加/修改 `.md`。  
2. 运行 `python update_articles.py`（或 `python3 update_articles.py`）。  
3. 脚本重写 `js/script.js` 中的 `articles` 与 `columns`。  
4. 打开 `blog.html` / `article.html?id=xxx` 即可看到最新内容。

### 5.2 Markdown 约定

- **Front Matter**（必选）：  
  ```yaml
  ---
  title: 文章标题
  date: 2026-01-30
  summary: 文章摘要
  ---
  ```
- **正文**：标准 Markdown；支持 LaTeX（`$...$`、`$$...$$`）、表格、图片（相对路径会按 `path` 解析）。  
- 更多示例与故障排除见 **ARTICLES_GUIDE.md**。

### 5.3 当前内容概览（扫描时）

- **专栏**：Pytorch  
- **文章**：  
  - PyTorch 框架学习（一）：核心模块初识与训练流程全解析  
  - PyTorch 框架学习（二）：张量（Tensor）与自动求导（autograd）详解  
  - 信息量、信息熵、KL 散度交叉熵损失（信息论）

---

## 六、外部依赖（CDN）

- **article.html** 中引入：  
  - KaTeX：`katex.min.css`、`katex.min.js`（公式渲染）  
  - marked：`marked.min.js`（若后续改用库解析 MD，当前以自研解析为主）

其余页面无外部脚本依赖，仅依赖本地 `css/style.css` 与 `js/script.js`。

---

## 七、部署说明

- **本地**：直接打开 `index.html` 或通过任意静态服务器（如 `python -m http.server`）访问根目录。  
- **GitHub Pages**：仓库开启 Pages，源选择对应分支（如 `main`），发布目录为仓库根目录。  
- **Netlify**：从 Git 拉取，构建命令留空，发布目录为根目录即可。

注意：若使用多级路径（如 `username.github.io/MyWeb`），需确保资源与链接使用相对路径（当前已为相对路径，一般无需改动）。

---

## 八、开发与维护

- **修改样式**：仅改 `css/style.css`。  
- **修改逻辑**：仅改 `js/script.js`（数据由脚本更新，勿手改 `articles`/`columns` 长期内容）。  
- **新增页面**：可复制现有 HTML，保留导航与 footer，按需挂载 `script.js` 中的逻辑或新写函数。  
- **新增功能**：在 `script.js` 中扩展（如新筛选条件、新排序方式），并在对应 HTML 增加 DOM 与事件绑定。  
- **添加/修改文章**：只改 `articles/` 下 MD，再运行 `update_articles.py`。

---

## 九、文档与规范

- **README.md**：面向访客与克隆者，介绍项目、目录、本地运行、添加文章、部署（GitHub Pages / Netlify）、技术特点与未来扩展。  
- **ARTICLES_GUIDE.md**：面向写作者，说明如何创建 MD、Front Matter、正文格式、运行脚本、示例与故障排除。  
- **PROJECT_DOCUMENTATION.md**（本文档）：面向开发者与维护者，描述项目架构、文件职责、数据流与部署方式。

---

*文档生成时间：基于 2026-02 项目扫描。*
