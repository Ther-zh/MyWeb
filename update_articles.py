#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文章更新脚本
用于自动读取articles文件夹下的md文件，并更新js/script.js文件中的articles数组
支持专栏功能和新的文件结构
"""

import os
import re
import json
import subprocess
from datetime import datetime

# 配置
ARTICLES_DIR = 'articles'
SCRIPT_FILE = 'js/script.js'


def parse_front_matter(content):
    """
    解析Markdown文件的前言部分
    前言格式：
    ---
    title: 文章标题
    date: 2026-01-01
    summary: 文章摘要
    ---
    """
    front_matter_pattern = re.compile(r'^---\n(.*?)\n---\n', re.DOTALL)
    match = front_matter_pattern.match(content)
    if not match:
        return {}
    
    front_matter = {}
    for line in match.group(1).strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            front_matter[key.strip()] = value.strip()
    
    return front_matter


def extract_content(content):
    """
    提取Markdown文件的内容部分（去除前言）
    """
    front_matter_pattern = re.compile(r'^---\n(.*?)\n---\n', re.DOTALL)
    match = front_matter_pattern.match(content)
    if match:
        return content[match.end():].strip()
    return content.strip()


def generate_article_id(title):
    """
    根据文章标题生成唯一ID
    """
    # 移除特殊字符，替换空格为下划线，转为小写
    id_str = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    id_str = id_str.replace(' ', '_').lower()
    # 添加时间戳确保唯一性
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    return f"{id_str}_{timestamp}" if id_str else f"article_{timestamp}"


def read_articles():
    """
    读取articles文件夹下的所有md文件，并解析为文章对象
    支持新的文件结构：
    articles/
    ├── 专栏1/
    │   ├── 文章1/
    │   │   ├── index.md
    │   │   └── 图片1.jpg
    │   └── 文章2/
    │       ├── index.md
    │       └── 图片2.jpg
    └── 文章3/
        ├── index.md
        └── 图片3.jpg
    """
    articles = []
    columns = []
    
    if not os.path.exists(ARTICLES_DIR):
        os.makedirs(ARTICLES_DIR)
        return articles, columns
    
    # 递归遍历articles文件夹
    for root, dirs, files in os.walk(ARTICLES_DIR):
        # 检查是否是文章文件夹（包含.md文件）
        md_files = [f for f in files if f.endswith('.md')]
        if md_files:
            # 选择第一个md文件作为文章内容
            filepath = os.path.join(root, md_files[0])
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析前言
            front_matter = parse_front_matter(content)
            
            # 提取信息
            title = front_matter.get('title', os.path.basename(root))
            date = front_matter.get('date', datetime.now().strftime('%Y-%m-%d'))
            summary = front_matter.get('summary', '')
            article_content = extract_content(content)
            article_id = generate_article_id(title)
            
            # 计算相对路径，用于图片引用
            relative_path = os.path.relpath(root, ARTICLES_DIR).replace('\\', '/')
            
            # 提取专栏信息
            # 基于路径层级结构判断：
            # - 如果路径是 articles/专栏名/文章名/，那么专栏名就是专栏
            # - 如果路径是 articles/文章名/，那么没有专栏
            relative_path_parts = relative_path.split('/')
            column = None
            if len(relative_path_parts) >= 2:
                # 检查最后一个部分是否包含.md文件
                last_part = relative_path_parts[-1]
                has_md_file = any(f.endswith('.md') for f in files)
                
                if has_md_file:
                    # 检查是否有子文件夹
                    has_subfolders = any(os.path.isdir(os.path.join(root, d)) for d in dirs)
                    
                    if not has_subfolders:
                        # 如果当前文件夹包含md文件且没有子文件夹，那么它是一个文章文件夹
                        # 检查父目录是否是专栏目录
                        if len(relative_path_parts) == 2:
                            # 路径格式：专栏名/文章名
                            column = relative_path_parts[0]
            
            # 创建文章对象
            article = {
                'id': article_id,
                'title': title,
                'date': date,
                'summary': summary,
                'content': article_content,
                'column': column,
                'path': relative_path
            }
            
            articles.append(article)
            
            # 添加专栏到columns列表
            if column and column not in columns:
                columns.append(column)
    
    # 按日期降序排序
    articles.sort(key=lambda x: x['date'], reverse=True)
    
    return articles, columns


def update_script(articles, columns):
    """
    更新js/script.js文件中的articles数组和columns数组
    """
    # 读取现有脚本文件
    with open(SCRIPT_FILE, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # 处理文章内容中的换行符和路径
    for article in articles:
        if 'content' in article:
            # 将内容转换为单行字符串，使用\n表示换行
            article['content'] = article['content'].replace('\n', '\\n')
            
            # 处理图片路径，确保图片引用正确
            # 查找所有图片引用
            img_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)', re.DOTALL)
            def replace_img_path(match):
                alt = match.group(1)
                src = match.group(2)
                # 如果是相对路径，添加文章路径前缀
                if not src.startswith('http') and not src.startswith('/'):
                    new_src = f"articles/{article['path']}/{src}"
                    return f"![{alt}]({new_src})"
                return match.group(0)
            article['content'] = img_pattern.sub(replace_img_path, article['content'])
    
    # 生成新的articles数组
    articles_json = json.dumps(articles, ensure_ascii=False, indent=4)
    articles_array = f'const articles = {articles_json};'
    
    # 生成新的columns数组
    columns_json = json.dumps(columns, ensure_ascii=False, indent=4)
    columns_array = f'const columns = {columns_json};'
    
    # 替换现有articles数组
    pattern = re.compile(r'const articles = \[.*?\];', re.DOTALL)
    new_script_content = pattern.sub(articles_array, script_content)
    
    # 检查是否已有columns数组，如果没有则添加
    if 'const columns = [' not in new_script_content:
        # 在articles数组后添加columns数组
        new_script_content = new_script_content.replace('const articles = [', 'const columns = [];\n\nconst articles = [')
    
    # 替换现有columns数组
    pattern = re.compile(r'const columns = \[.*?\];', re.DOTALL)
    new_script_content = pattern.sub(columns_array, new_script_content)
    
    # 写入更新后的脚本文件
    with open(SCRIPT_FILE, 'w', encoding='utf-8') as f:
        f.write(new_script_content)


def git_push():
    """
    执行Git推送操作
    """
    try:
        print("\n开始执行Git推送操作...")
        
        # 检查Git状态
        print("检查Git状态...")
        subprocess.run(['git', 'status'], check=True, shell=True)
        
        # 添加所有修改的文件
        print("添加修改的文件...")
        subprocess.run(['git', 'add', '.'], check=True, shell=True)
        
        # 提交修改
        commit_message = f"更新文章 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"提交修改: {commit_message}")
        subprocess.run(['git', 'commit', '-m', commit_message], check=True, shell=True)
        
        # 推送到远程仓库
        print("推送到远程仓库...")
        subprocess.run(['git', 'push'], check=True, shell=True)
        
        print("Git推送操作完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git操作失败: {e}")
        print("请检查Git环境和仓库配置")
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        return False


def main():
    """
    主函数
    """
    print("开始更新文章...")
    
    # 读取文章和专栏
    articles, columns = read_articles()
    print(f"找到 {len(articles)} 篇文章")
    print(f"找到 {len(columns)} 个专栏")
    
    # 更新脚本
    update_script(articles, columns)
    print("文章更新完成！")
    
    # 询问是否执行Git推送
    push_confirm = input("是否执行Git推送操作？(y/n): ")
    if push_confirm.lower() == 'y':
        git_push()
    else:
        print("跳过Git推送操作")


if __name__ == '__main__':
    main()
