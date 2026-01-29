# 知帆个人博客网站

## 项目介绍

这是一个个人博客网站，包含个人介绍页面和博客文章页面。网站采用纯HTML、CSS和JavaScript实现，不依赖任何框架，支持通过添加MD文件来上传文章。

## 目录结构

```
├── index.html          # 个人主页
├── blog.html           # 博客页面
├── article.html        # 文章详情页面
├── css/
│   └── style.css       # 样式文件
├── js/
│   └── script.js       # JavaScript文件
├── images/             # 图片目录
├── articles/           # 文章MD文件目录
└── README.md           # 项目说明
```

## 如何使用

### 本地运行

1. 克隆或下载项目到本地
2. 直接打开 `index.html` 文件即可在浏览器中查看

### 添加文章

1. 在 `js/script.js` 文件中，找到 `articles` 数组
2. 添加新的文章对象，包含以下字段：
   - id: 文章唯一标识
   - title: 文章标题
   - date: 发布日期
   - summary: 文章摘要
   - content: 文章内容（支持Markdown格式）

### 部署到GitHub Pages

1. 在GitHub上创建一个新的仓库
2. 将项目文件上传到仓库
3. 在仓库设置中，开启GitHub Pages功能，选择 `main` 分支作为源
4. 等待几分钟后，网站将通过 `https://<username>.github.io/<repository>` 访问

### 部署到Netlify

1. 在GitHub上创建一个新的仓库
2. 将项目文件上传到仓库
3. 登录Netlify，选择 "New site from Git"
4. 选择GitHub作为Git provider，找到并选择你的仓库
5. 配置构建设置（不需要构建命令，发布目录为仓库根目录）
6. 点击 "Deploy site" 按钮，等待部署完成
7. 部署完成后，Netlify将提供一个随机域名，你可以自定义域名

## 技术特点

- **响应式设计**：适配不同屏幕尺寸
- **现代化UI**：采用渐变色彩和卡片式设计
- **Markdown支持**：文章内容支持Markdown格式
- **动态加载**：文章列表和详情通过JavaScript动态加载
- **无依赖**：纯HTML、CSS和JavaScript实现，不依赖任何框架

## 未来扩展

- 添加文章评论功能
- 实现文章分类和标签
- 添加搜索功能
- 实现深色模式
- 优化Markdown解析，支持更多语法
