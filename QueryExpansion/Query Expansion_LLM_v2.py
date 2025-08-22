#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 09:24:33 2025

@author: morison
"""

import ollama

def auto_query_expansion_with_ollama(text):
    """
    使用 Ollama 服務上的模型，從文字中同時生成中、英文關鍵字。
    """
    # 步驟一：調用 Ollama 模型
    print("--- 步驟一：嘗試調用 Ollama 服務上的模型 ---")
    try:
        # 改進提示詞，要求更結構化的輸出
        prompt = f"""從這篇文章中，分別提取中文核心關鍵詞和英文核心關鍵詞。
請按照以下格式輸出，每個關鍵詞用逗號分隔：

中文關鍵詞：關鍵詞1, 關鍵詞2, 關鍵詞3...
英文關鍵詞：keyword1, keyword2, keyword3...

文章內容：
{text}"""
        
        # 這裡使用 'llama3' 模型，你需要確保它已經在 Ollama 服務上運行
        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        
        generated_keywords_str = response['message']['content']
        print("模型生成成功。")
        print(f"模型回應：\n{generated_keywords_str}")
    
    except Exception as e:
        print(f"錯誤：無法連線到 Ollama 服務或調用模型失敗。")
        print(f"錯誤詳情：{e}")
        # 如果失敗，返回原始文字
        return [text.strip()]

    # 步驟二：解析生成的關鍵字
    print("\n--- 步驟二：解析出的擴展查詢 ---")
    
    # 使用集合來避免重複，但保持順序
    expanded_queries = [text.strip()]  # 原始文字
    seen_queries = {text.strip()}  # 用於檢查重複
    
    # 解析模型回應
    lines = generated_keywords_str.split('\n')
    chinese_keywords = []
    english_keywords = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 解析中文關鍵詞
        if '中文關鍵詞' in line or '中文关键词' in line:
            # 提取冒號後的內容
            if '：' in line:
                keywords_part = line.split('：', 1)[1].strip()
            elif ':' in line:
                keywords_part = line.split(':', 1)[1].strip()
            else:
                continue
                
            # 分割關鍵詞
            keywords = [kw.strip() for kw in keywords_part.split(',') if kw.strip()]
            chinese_keywords.extend(keywords)
            
        # 解析英文關鍵詞
        elif '英文關鍵詞' in line or '英文关键词' in line or 'english' in line.lower() or 'keyword' in line.lower():
            # 提取冒號後的內容
            if '：' in line:
                keywords_part = line.split('：', 1)[1].strip()
            elif ':' in line:
                keywords_part = line.split(':', 1)[1].strip()
            else:
                continue
                
            # 分割關鍵詞
            keywords = [kw.strip() for kw in keywords_part.split(',') if kw.strip()]
            english_keywords.extend(keywords)
        
        # 如果沒有明確標示，嘗試直接解析逗號分隔的關鍵詞
        elif ',' in line and not any(x in line for x in ['：', ':', '關鍵詞', '关键词']):
            keywords = [kw.strip() for kw in line.split(',') if kw.strip()]
            # 判斷是中文還是英文（簡單啟發式）
            for kw in keywords:
                if any('\u4e00' <= char <= '\u9fff' for char in kw):  # 包含中文字符
                    chinese_keywords.append(kw)
                elif kw.isascii() and len(kw) > 1:  # 英文字符
                    english_keywords.append(kw)

    # 清理和去重關鍵詞
    def clean_and_dedupe_keywords(keywords_list):
        """清理關鍵詞並去除重複"""
        cleaned = []
        seen = set()
        for kw in keywords_list:
            # 移除多餘的標點符號和空白
            kw = kw.strip('.,，。、；;').strip()
            if kw and len(kw) > 1 and kw.lower() not in seen:
                cleaned.append(kw)
                seen.add(kw.lower())
        return cleaned

    chinese_keywords = clean_and_dedupe_keywords(chinese_keywords)
    english_keywords = clean_and_dedupe_keywords(english_keywords)
    
    # 添加中文關鍵詞到擴展查詢
    print("中文關鍵詞：")
    for kw in chinese_keywords:
        if kw not in seen_queries:
            expanded_queries.append(kw)
            seen_queries.add(kw)
            print(f"- {kw}")
    
    # 添加英文關鍵詞到擴展查詢
    print("\n英文關鍵詞：")
    for kw in english_keywords:
        if kw not in seen_queries:
            expanded_queries.append(kw)
            seen_queries.add(kw)
            print(f"- {kw}")
    
    # 如果上述解析失敗，使用備用解析方法
    if len(expanded_queries) == 1:  # 只有原始文字
        print("\n--- 使用備用解析方法 ---")
        # 簡單地用逗號分割所有內容
        all_keywords = []
        for line in generated_keywords_str.split('\n'):
            if line.strip() and '：' not in line and ':' not in line:
                keywords = [kw.strip() for kw in line.split(',') if kw.strip()]
                all_keywords.extend(keywords)
        
        # 清理並添加到結果
        all_keywords = clean_and_dedupe_keywords(all_keywords)
        for kw in all_keywords:
            if kw not in seen_queries and len(kw) > 1:
                expanded_queries.append(kw)
                seen_queries.add(kw)
                print(f"- {kw}")

    return expanded_queries

# 測試函式
if __name__ == "__main__":
    text = """
    人工智能（AI）正在迅速改變我們的生活和工作方式。從智能手機中的語音助手到自動駕駛汽車，AI 的應用無處不在。AI 的核心目標是使機器能夠像人類一樣思考、學習和解決問題。
    機器學習（Machine learning）是AI的一個重要分支，它使計算機系統能夠從數據中學習而無需明確編程。深度學習（Deep learning）是機器學習的子領域，它利用人工神經網絡，特別是多層次網絡，來處理複雜模式識別任務，例如圖像識別和自然語言處理（NLP）。
    自然語言處理是AI的另一個關鍵領域，它專注於讓計算機理解、解釋和生成人類語言。語音識別（Voice recognition）、機器翻譯和情感分析都是NLP的應用。
    """

    expanded_queries = auto_query_expansion_with_ollama(text)
    
    print("\n" + "="*50)
    print("--- 最終結果 ---")
    print("原始文字:")
    print(text.strip())
    print(f"\n擴展後的查詢列表 (共 {len(expanded_queries)} 項):")
    for i, query in enumerate(expanded_queries, 1):
        print(f"{i:2d}. {query}")