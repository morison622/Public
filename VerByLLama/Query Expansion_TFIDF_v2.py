#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 09:19:02 2025

@author: morison
"""

# pip install jieba scikit-learn nltk

import re
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
import nltk

def download_nltk_dependencies():
    """下載必要的 NLTK 資源（第一次執行需要）"""
    try:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception as e:
        print(f"NLTK 資源下載失敗：{e}")

def auto_query_expansion_with_tfidf(text, top_keywords=15, expand_synonyms=True):
    """
    使用 TF-IDF 和 jieba 從文字中生成中、英文關鍵字並進行查詢擴展。
    
    Args:
        text (str): 輸入文字
        top_keywords (int): 提取的關鍵詞數量，預設 15
        expand_synonyms (bool): 是否進行英文同義詞擴展，預設 True
    
    Returns:
        list: 擴展後的查詢列表
    """
    
    # 步驟一：初始化和前處理
    print("--- 步驟一：初始化和文本前處理 ---")
    
    # 確保 NLTK 資源已下載
    if expand_synonyms:
        download_nltk_dependencies()
    
    # 停用詞表
    stopwords = {"的", "是", "在", "和", "有", "也", "就", "對", "我們", "它", "從", "與", "或", "但", "這", "那"}
    
    expanded_queries = [text.strip()]  # 包含原始文字
    
    # 步驟二：使用 jieba 提取中文關鍵詞
    print("\n--- 步驟二：使用 jieba 提取中文關鍵詞 ---")
    try:
        jieba_keywords = jieba.analyse.extract_tags(text, topK=30, withWeight=False)
        jieba_keywords = [w for w in jieba_keywords if w not in stopwords and len(w) > 1]
        print(f"jieba 提取到 {len(jieba_keywords)} 個關鍵詞")
    except Exception as e:
        print(f"jieba 關鍵詞提取失敗：{e}")
        jieba_keywords = []

    # 步驟三：TF-IDF 分析
    print("\n--- 步驟三：TF-IDF 分析 ---")
    try:
        words = [w for w in jieba.lcut(text) if w not in stopwords and len(w) > 1]
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1,2))
        X = vectorizer.fit_transform([" ".join(words)])
        tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))
        print(f"TF-IDF 分析完成，共 {len(tfidf_scores)} 個詞項")
    except Exception as e:
        print(f"TF-IDF 分析失敗：{e}")
        tfidf_scores = {}

    # 步驟四：英文縮寫識別與加權
    print("\n--- 步驟四：英文縮寫識別與加權 ---")
    english_terms = re.findall(r"\b[A-Z]{2,}\b", text)
    for term in english_terms:
        tfidf_scores[term] = tfidf_scores.get(term, 1.0) * 3  # 英文縮寫加權
    print(f"識別到 {len(english_terms)} 個英文縮寫：{english_terms}")

    # 步驟五：合併所有候選詞並排序
    print("\n--- 步驟五：合併關鍵詞並排序 ---")
    all_candidates = set(jieba_keywords) | set(english_terms)
    final_scores = {k: tfidf_scores.get(k, 1.0) for k in all_candidates if k not in stopwords}
    
    # 排序取前 N 個
    final_keywords = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_keywords]
    final_keywords = [k for k, _ in final_keywords]
    
    print(f"最終關鍵詞 ({len(final_keywords)} 個):")
    for i, kw in enumerate(final_keywords, 1):
        print(f"  {i}. {kw}")
        expanded_queries.append(kw)

    # 步驟六：英文同義詞擴展（可選）
    if expand_synonyms:
        print("\n--- 步驟六：英文同義詞擴展 ---")
        try:
            expanded_count = 0
            for kw in final_keywords:
                if kw.isascii() and len(kw) > 2:  # 只處理英文詞且長度大於2
                    synonyms = set()
                    for syn in wn.synsets(kw.lower()):
                        for lemma in syn.lemmas():
                            synonym = lemma.name().replace("_", " ")
                            if synonym != kw.lower() and len(synonym) > 2:
                                synonyms.add(synonym)
                    
                    if synonyms:
                        synonym_list = list(synonyms)[:3]  # 每個詞最多3個同義詞
                        print(f"  {kw}: {synonym_list}")
                        expanded_queries.extend(synonym_list)
                        expanded_count += len(synonym_list)
            
            print(f"共擴展了 {expanded_count} 個英文同義詞")
        
        except Exception as e:
            print(f"同義詞擴展失敗：{e}")

    return expanded_queries


# 測試函式
if __name__ == "__main__":
    text = """
    人工智能（AI）正在迅速改變我們的生活和工作方式。從智能手機中的語音助手到自動駕駛汽車，AI 的應用無處不在。AI 的核心目標是使機器能夠像人類一樣思考、學習和解決問題。
    機器學習（Machine learning）是AI的一個重要分支，它使計算機系統能夠從數據中學習而無需明確編程。深度學習（Deep learning）是機器學習的子領域，它利用人工神經網絡，特別是多層次網絡，來處理複雜模式識別任務，例如圖像識別和自然語言處理（NLP）。
    自然語言處理是AI的另一個關鍵領域，它專注於讓計算機理解、解釋和生成人類語言。語音識別（Voice recognition）、機器翻譯和情感分析都是NLP的應用。
    """

    expanded_queries = auto_query_expansion_with_tfidf(text, top_keywords=15, expand_synonyms=True)
    
    print("\n" + "="*50)
    print("--- 最終結果 ---")
    print("原始文字:")
    print(text.strip())
    print(f"\n擴展後的查詢列表 (共 {len(expanded_queries)} 項):")
    for i, query in enumerate(expanded_queries, 1):
        print(f"{i:2d}. {query}")