#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文章更新脚本
用于自动读取articles文件夹下的md文件，并更新js/script.js文件中的articles数组
"""

import os
import re
import json
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
    """
    articles = []
    
    if not os.path.exists(ARTICLES_DIR):
        os.makedirs(ARTICLES_DIR)
        return articles
    
    for filename in os.listdir(ARTICLES_DIR):
        if filename.endswith('.md'):
            filepath = os.path.join(ARTICLES_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析前言
            front_matter = parse_front_matter(content)
            
            # 提取信息
            title = front_matter.get('title', filename[:-3])
            date = front_matter.get('date', datetime.now().strftime('%Y-%m-%d'))
            summary = front_matter.get('summary', '')
            article_content = extract_content(content)
            article_id = generate_article_id(title)
            
            # 创建文章对象
            article = {
                'id': article_id,
                'title': title,
                'date': date,
                'summary': summary,
                'content': article_content
            }
            
            articles.append(article)
    
    # 按日期降序排序
    articles.sort(key=lambda x: x['date'], reverse=True)
    
    return articles


def update_script(articles):
    """
    更新js/script.js文件中的articles数组
    """
    # 读取现有脚本文件
    with open(SCRIPT_FILE, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # 生成新的articles数组
    articles_json = json.dumps(articles, ensure_ascii=False, indent=4)
    articles_array = f'const articles = {articles_json};'
    
    # 替换现有articles数组
    pattern = re.compile(r'const articles = \[.*?\];', re.DOTALL)
    new_script_content = pattern.sub(articles_array, script_content)
    
    # 写入更新后的脚本文件
    with open(SCRIPT_FILE, 'w', encoding='utf-8') as f:
        f.write(new_script_content)


def main():
    """
    主函数
    """
    print("开始更新文章...")
    
    # 读取文章
    articles = read_articles()
    print(f"找到 {len(articles)} 篇文章")
    
    # 更新脚本
    update_script(articles)
    print("文章更新完成！")


if __name__ == '__main__':
    main()
