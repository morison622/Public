# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:37:42 2025

@author: morison.su
"""

import requests
import json
import re
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
import nltk


GEMINI_API_KEY = ""

def auto_query_expansion_with_gemini(text, api_key):
    """
    使用 Google Gemini API，從文字中同時生成中、英文關鍵字。
    """
    print("--- 啟動高成本模式：使用 Gemini API 進行查詢擴展 ---")
    
    model_name = "gemini-2.5-flash-preview-05-20"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    prompt = f"請從以下文字中，提取最重要的中文核心關鍵詞和英文核心關鍵詞。請將結果以 JSON 格式回傳，格式為: {{'chinese_keywords': [], 'english_keywords': []}}。文字內容：\n\n{text}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        
        generated_json_str = response.json()['candidates'][0]['content']['parts'][0]['text']
        print("Gemini 模型生成成功。")
        
        keywords_data = json.loads(generated_json_str)
        chinese_keywords = keywords_data.get('chinese_keywords', [])
        english_keywords = keywords_data.get('english_keywords', [])
        
        return list(set(chinese_keywords + english_keywords))
    
    except requests.exceptions.RequestException as e:
        print(f"錯誤：Gemini API 服務調用失敗。詳情：{e}")
        return []
    
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"錯誤：API 回應格式不正確。詳情：{e}")
        return []

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
        top_keywords (int): 提取的關鍵詞數量
        expand_synonyms (bool): 是否進行英文同義詞擴展
    
    Returns:
        tuple: (list: 擴展後的查詢, float: 最高 TF-IDF 分數)
    """
    
    print("--- 步驟一：啟動低成本模式：TF-IDF 分析 ---")
    
    if expand_synonyms:
        download_nltk_dependencies()
    
    stopwords = {"的", "是", "在", "和", "有", "也", "就", "對", "我們", "它", "從", "與", "或", "但", "這", "那"}
    expanded_queries = [text.strip()]
    
    try:
        words = [w for w in jieba.lcut(text) if w not in stopwords and len(w) > 1]
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1,2))
        X = vectorizer.fit_transform([" ".join(words)])
        tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))
    except Exception as e:
        print(f"TF-IDF 分析失敗：{e}")
        return expanded_queries, 0.0

    english_terms = re.findall(r"\b[A-Z]{2,}\b", text)
    for term in english_terms:
        tfidf_scores[term] = tfidf_scores.get(term, 1.0) * 3

    all_candidates = set(jieba.analyse.extract_tags(text, topK=30, withWeight=False)) | set(english_terms)
    final_scores = {k: tfidf_scores.get(k, 0) for k in all_candidates if k not in stopwords and len(k) > 1}
    
    if not final_scores:
        return expanded_queries, 0.0

    max_score = max(final_scores.values())

    final_keywords = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_keywords]
    final_keywords = [k for k, _ in final_keywords]

    for kw in final_keywords:
        if kw not in expanded_queries:
            expanded_queries.append(kw)

    if expand_synonyms:
        for kw in final_keywords:
            if kw.isascii() and len(kw) > 2:
                synonyms = set()
                for syn in wn.synsets(kw.lower()):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace("_", " ")
                        if synonym != kw.lower() and len(synonym) > 2:
                            synonyms.add(synonym)
                expanded_queries.extend([s for s in synonyms][:3])

    return list(set(expanded_queries)), max_score


def run_cost_effective_retrieval(text, tfidf_confidence_threshold=0.5):
    """
    根據 TF-IDF 分數，動態選擇使用低成本或高成本方法進行查詢擴展。
    
    Args:
        text (str): 待處理的文字
        tfidf_confidence_threshold (float): TF-IDF 信心分數閾值
        
    Returns:
        list: 最終擴展後的查詢列表
    """
    print("\n" + "="*50)
    print("--- 啟動成本效益驅動的檢索模型 ---")
    
    # 步驟一：優先使用低成本的 TF-IDF 方法
    tfidf_queries, max_score = auto_query_expansion_with_tfidf(text)
    
    print(f"\n最高 TF-IDF 關鍵詞分數: {max_score:.4f}")
    
    # 步驟二：檢查信心分數，決定是否切換到高成本模式
    if max_score < tfidf_confidence_threshold:
        print(f"最高分數 {max_score:.4f} 低於閾值 {tfidf_confidence_threshold}。切換到高成本模式...")
        
        # 啟動 Gemini API
        gemini_queries = auto_query_expansion_with_gemini(text, GEMINI_API_KEY)
        
        # 合併並去重
        final_queries = list(set(tfidf_queries + gemini_queries))
        
        print("\n--- 檢索模式：高成本 (Gemini API) ---")
    else:
        print(f"最高分數 {max_score:.4f} 高於閾值 {tfidf_confidence_threshold}。使用低成本模式...")
        final_queries = tfidf_queries
        print("\n--- 檢索模式：低成本 (TF-IDF) ---")
    
    return final_queries


# 測試主程式
if __name__ == "__main__":
    test_text_high_confidence = """
    自然語言處理（NLP）是 AI 的一個關鍵分支。它致力於讓電腦理解、解釋和生成人類語言，應用於語音識別、機器翻譯和情感分析等領域。
    """
    
    test_text_low_confidence = """
    在未來，我們將看到許多新技術。這些技術會影響我們的生活。
    """

    print("\n" + "="*50)
    print("測試範例一：高信心度文字 (預期使用 TF-IDF)")
    expanded_queries_1 = run_cost_effective_retrieval(test_text_high_confidence)
    print(f"\n最終擴展查詢列表 ({len(expanded_queries_1)} 項):")
    for i, q in enumerate(expanded_queries_1, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "="*50)
    print("\n" + "="*50)
    print("測試範例二：低信心度文字 (預期切換到 Gemini API)")
    expanded_queries_2 = run_cost_effective_retrieval(test_text_low_confidence)
    print(f"\n最終擴展查詢列表 ({len(expanded_queries_2)} 項):")
    for i, q in enumerate(expanded_queries_2, 1):
        print(f"  {i}. {q}")
        
