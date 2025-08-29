#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:14:52 2025

@author: morison
"""

# -*- coding: utf-8 -*-
"""
完全本地化的動態文本切塊系統
- 使用本地 Ollama LLM 
- 可選的本地嵌入模型或簡單相似度算法
- 無需網路連接

Modified from original DynamicChunking_HC_v2.py
Author: morison.su (Modified for local execution)
"""

import numpy as np
import ollama
import os
import json
from pathlib import Path
import re
from collections import Counter
import math

# --- 本地化配置 ---
OLLAMA_MODEL = "llama3.1:latest"
LOCAL_MODE = True  # 設為 True 啟用完全本地模式

def setup_offline_environment():
    """設置完全離線環境"""
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['OLLAMA_HOST'] = 'localhost:11434'
    print("🏠 已設置完全本地化環境")

def test_ollama_connection():
    """測試本地Ollama連接"""
    try:
        print("🔍 測試本地 Ollama 連接...")
        models_response = ollama.list()
        
        model_names = []
        if hasattr(models_response, 'models'):
            for model in models_response.models:
                if hasattr(model, 'model'):
                    model_names.append(model.model)
                elif hasattr(model, 'name'):
                    model_names.append(model.name)
        elif isinstance(models_response, dict) and 'models' in models_response:
            for model in models_response['models']:
                if isinstance(model, dict) and 'name' in model:
                    model_names.append(model['name'])
                elif isinstance(model, str):
                    model_names.append(model)
        
        if model_names:
            print(f"✅ Ollama 連接成功！可用模型: {model_names}")
            
            # 檢查 Llama 3.1 是否存在
            llama_models = [m for m in model_names if 'llama3.1' in m.lower() or OLLAMA_MODEL in model_names]
            if llama_models or OLLAMA_MODEL in model_names:
                print(f"✅ 找到目標模型")
                return True, model_names
            else:
                print(f"⚠️ 未找到 {OLLAMA_MODEL}，將使用第一個可用模型")
                return True, model_names
        else:
            print("❌ 未找到任何模型")
            return False, []
        
    except Exception as e:
        print(f"❌ Ollama 連接失敗: {e}")
        print("💡 解決方案:")
        print("   1. 啟動 Ollama: ollama serve")
        print("   2. 下載模型: ollama pull llama3.1")
        return False, []

def call_ollama_local(prompt, model=OLLAMA_MODEL, max_retries=3):
    """調用本地Ollama，包含重試機制"""
    for attempt in range(max_retries):
        try:
            # 如果指定模型不存在，使用第一個可用模型
            actual_model = model
            if available_models and model not in available_models:
                actual_model = available_models[0]
                if attempt == 0:  # 只在第一次嘗試時顯示警告
                    print(f"⚠️ 模型 '{model}' 不存在，使用 '{actual_model}'")
            
            print(f"🤖 使用模型 '{actual_model}' 處理... (嘗試 {attempt + 1}/{max_retries})")
            
            response = ollama.generate(
                model=actual_model,
                prompt=prompt,
                options={
                    'num_ctx': 4096,
                    'temperature': 0.7,
                    'num_predict': 2000,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            
            if 'response' in response and response['response'].strip():
                print("✅ 本地 LLM 處理完成")
                return response['response'].strip()
            else:
                print(f"⚠️ 嘗試 {attempt + 1} - 獲得空回應")
                continue
                
        except Exception as e:
            print(f"❌ 嘗試 {attempt + 1} 失敗: {e}")
            if attempt == max_retries - 1:
                print("❌ 所有重試都失敗了")
                return None
            print("🔄 正在重試...")
    
    return None

# --- 本地嵌入模型載入 ---
def load_local_embedding_model():
    """嘗試載入本地嵌入模型"""
    if not LOCAL_MODE:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ SentenceTransformer 模型載入成功")
            return model, True
        except Exception as e:
            print(f"⚠️ SentenceTransformer 載入失敗: {e}")
            print("🔄 切換到簡單相似度模式")
            return None, False
    else:
        print("🏠 本地模式：使用簡單文本相似度算法")
        return None, False

# --- 簡單文本相似度算法 ---
def simple_cosine_similarity(text1, text2):
    """基於詞頻的簡單餘弦相似度"""
    def get_word_vector(text):
        words = re.findall(r'\w+', text.lower())
        return Counter(words)
    
    vec1 = get_word_vector(text1)
    vec2 = get_word_vector(text2)
    
    # 取得所有不重複的詞
    all_words = set(vec1.keys()) | set(vec2.keys())
    
    if not all_words:
        return 0.0
    
    # 建立向量
    v1 = np.array([vec1.get(word, 0) for word in all_words])
    v2 = np.array([vec2.get(word, 0) for word in all_words])
    
    # 計算餘弦相似度
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def get_local_embeddings(texts):
    """使用簡單方法生成文本嵌入"""
    if embedding_model_loaded and embedding_model:
        # 使用真實的嵌入模型
        return embedding_model.encode(texts, convert_to_numpy=True)
    else:
        # 使用簡單的詞頻向量
        embeddings = []
        all_texts_words = []
        
        # 收集所有詞彙
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            all_texts_words.extend(words)
        
        vocab = list(set(all_texts_words))
        vocab_size = len(vocab)
        
        if vocab_size == 0:
            return np.zeros((len(texts), 1))
        
        # 為每個文本建立向量
        for text in texts:
            word_counts = Counter(re.findall(r'\w+', text.lower()))
            vector = np.array([word_counts.get(word, 0) for word in vocab])
            embeddings.append(vector)
        
        return np.array(embeddings)

def simple_hierarchical_clustering(similarity_matrix, threshold):
    """簡單的階層式聚類實現"""
    n = similarity_matrix.shape[0]
    clusters = [[i] for i in range(n)]
    
    while len(clusters) > 1:
        max_sim = -1
        merge_i, merge_j = -1, -1
        
        # 找到最相似的兩個聚類
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # 計算聚類間的平均相似度
                sim_sum = 0
                count = 0
                for idx1 in clusters[i]:
                    for idx2 in clusters[j]:
                        sim_sum += similarity_matrix[idx1][idx2]
                        count += 1
                
                if count > 0:
                    avg_sim = sim_sum / count
                    if avg_sim > max_sim:
                        max_sim = avg_sim
                        merge_i, merge_j = i, j
        
        # 如果最大相似度低於閾值，停止聚類
        if max_sim < threshold:
            break
        
        # 合併兩個最相似的聚類
        if merge_i != -1 and merge_j != -1:
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
    
    # 產生聚類標籤
    labels = [0] * n
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = cluster_id
    
    return labels

# --- 基礎切塊函數 ---
def chunk_text_with_overlap(text: str, chunk_size: int, overlap_size: int) -> list[dict]:
    """基礎重疊切塊"""
    chunks = []
    text_length = len(text)
    start_index = 0
    chunk_id_counter = 0

    while start_index < text_length:
        end_index = min(start_index + chunk_size, text_length)
        chunk_content = text[start_index:end_index]
        chunks.append({
            "id": f"base_chunk_{chunk_id_counter}",
            "content": chunk_content
        })
        chunk_id_counter += 1

        if end_index == text_length:
            break

        start_index += (chunk_size - overlap_size)
        start_index = min(start_index, text_length - 1)
        
    return chunks

# --- 本地聚類函數 ---
def aggregate_chunks_by_similarity_local(
    base_chunks: list[dict],
    similarity_threshold: float
) -> list[dict]:
    """使用本地算法進行相似度聚類"""
    if not base_chunks:
        return []
    
    print("🔍 開始本地語義相似度分析...")
    
    base_chunk_contents = [chunk['content'] for chunk in base_chunks]
    
    # 生成嵌入向量
    print("📊 生成文本嵌入向量...")
    embeddings = get_local_embeddings(base_chunk_contents)
    
    # 計算相似度矩陣
    print("🧮 計算相似度矩陣...")
    n = len(base_chunks)
    similarity_matrix = np.zeros((n, n))
    
    if embedding_model_loaded and embedding_model:
        # 使用真實嵌入的餘弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
    else:
        # 使用簡單相似度
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim = simple_cosine_similarity(base_chunk_contents[i], base_chunk_contents[j])
                    similarity_matrix[i][j] = sim
                else:
                    similarity_matrix[i][j] = 1.0
    
    # 執行聚類
    print("🔗 執行階層式聚類...")
    cluster_labels = simple_hierarchical_clustering(similarity_matrix, similarity_threshold)
    
    # 將基礎區塊按聚類標籤分組
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(base_chunks[i])

    aggregated_chunks = []
    super_chunk_id_counter = 0

    # 只處理包含多個區塊的聚類
    for label, chunk_group in clusters.items():
        if len(chunk_group) > 1:
            content = "".join([chunk['content'] for chunk in chunk_group])
            original_ids = [chunk['id'] for chunk in chunk_group]
            
            aggregated_chunks.append({
                "id": f"super_chunk_{super_chunk_id_counter}",
                "content": content,
                "original_chunk_ids": original_ids
            })
            super_chunk_id_counter += 1
    
    print(f"✅ 聚類完成，生成 {len(aggregated_chunks)} 個聚合塊")
    return aggregated_chunks

# --- 本地文本精煉函數 ---
def refine_super_chunk_local(super_chunk_content: str) -> str:
    """使用本地 Ollama 精煉 Super Chunk 內容"""
    
    prompt = f"""請精煉以下中文文本，要求：
1. 去除重複和冗餘內容
2. 保持所有關鍵資訊
3. 使文本更流暢易讀
4. 維持原文的意思和語境

原始文本：
{super_chunk_content}

精煉後的文本："""
    
    try:
        refined_text = call_ollama_local(prompt)
        if refined_text and len(refined_text.strip()) > 20:
            print("✅ Super Chunk 精煉完成")
            return refined_text
        else:
            print("⚠️ LLM 回應異常，保留原文")
            return super_chunk_content
    except Exception as e:
        print(f"❌ Super Chunk 精煉失敗: {e}")
        return super_chunk_content

# --- 主程序 ---
def main():
    """主執行程序"""
    
    # 設置環境
    setup_offline_environment()
    
    # 測試 Ollama 連接
    global ollama_connected, available_models
    ollama_connected, available_models = test_ollama_connection()
    
    if not ollama_connected:
        print("❌ 無法連接到本地 Ollama，程序終止")
        return
    
    # 載入嵌入模型
    global embedding_model, embedding_model_loaded
    embedding_model, embedding_model_loaded = load_local_embedding_model()
    
    # 測試文本
    long_chinese_text = """
人工智能（AI）正在迅速改變我們的生活和工作方式。從智能手機中的語音助手到自動駕駛汽車，AI 的應用無處不在。AI 的核心目標是使機器能夠像人類一樣思考、學習和解決問題。

機器學習是AI的一個重要分支，它使計算機系統能夠從數據中學習而無需明確編程。深度學習是機器學習的子領域，它利用人工神經網絡，特別是多層次網絡，來處理複雜模式識別任務，例如圖像識別和自然語言處理。

自然語言處理（NLP）是AI的另一個關鍵領域，它專注於讓計算機理解、解釋和生成人類語言。語音識別、機器翻譯和情感分析都是NLP的應用。透過NLP，我們可以與計算機進行更自然的互動。

雖然AI帶來了巨大的潛力，但也伴隨著一些挑戰，例如倫理問題、隱私問題以及對就業市場的影響。如何平衡AI的發展與社會責任是我們需要共同面對的課題。未來，AI將繼續深入發展，影響我們生活的方方面面。

人工智能技術的快速發展為各個行業帶來了變革性影響。在醫療領域，AI協助醫生進行疾病診斷和治療方案制定。在金融業，AI用於風險評估和詐騙檢測。教育領域中，個性化學習系統能夠根據學生的需求提供量身定制的教學內容。

然而，隨著AI技術的普及，我們也必須關注其帶來的挑戰。數據隱私保護、算法偏見、就業結構變化等問題都需要我們認真思考和應對。只有在技術發展與倫理責任之間找到平衡，AI才能真正為人類社會帶來福祉。
"""
    
    print("=" * 60)
    print("🏠 完全本地化動態文本切塊系統")
    print("=" * 60)
    print(f"📄 原始文本長度: {len(long_chinese_text)} 字符")
    print("-" * 30)
    print(long_chinese_text)
    print("-" * 30)
    
    # 設置參數
    CHUNK_SIZE = 60
    OVERLAP_SIZE = 10
    SIMILARITY_THRESHOLD = 0.3
    
    print(f"\n📋 處理參數:")
    print(f"   - 塊大小: {CHUNK_SIZE}")
    print(f"   - 重疊大小: {OVERLAP_SIZE}")
    print(f"   - 相似度閾值: {SIMILARITY_THRESHOLD}")
    
    # 步驟 1: 基礎切塊
    print(f"\n🔪 步驟 1: 基礎重疊切塊")
    base_chunks = chunk_text_with_overlap(long_chinese_text, CHUNK_SIZE, OVERLAP_SIZE)
    
    print(f"✅ 生成 {len(base_chunks)} 個基礎塊:")
    for i, chunk in enumerate(base_chunks):
        print(f"   📦 {chunk['id']} (長度: {len(chunk['content'])})")
        print(f"      內容: {chunk['content'][:50]}{'...' if len(chunk['content']) > 50 else ''}")
    
    # 步驟 2: 語義聚類
    print(f"\n🔗 步驟 2: 語義相似度聚類")
    super_chunks = aggregate_chunks_by_similarity_local(base_chunks, SIMILARITY_THRESHOLD)
    
    if not super_chunks:
        print("⚠️ 未生成聚合塊，可能需要調整相似度閾值")
        print("🏁 程序結束")
        return
    
    print(f"✅ 生成 {len(super_chunks)} 個聚合塊:")
    
    # 步驟 3: LLM 精煉
    print(f"\n🤖 步驟 3: LLM 內容精煉")
    
    for i, super_chunk in enumerate(super_chunks):
        print(f"\n📦 處理聚合塊 {super_chunk['id']}")
        print(f"   📋 包含基礎塊: {super_chunk['original_chunk_ids']}")
        print(f"   📏 原始長度: {len(super_chunk['content'])} 字符")
        print(f"   📝 原始內容: {super_chunk['content'][:100]}{'...' if len(super_chunk['content']) > 100 else ''}")
        
        # 使用 LLM 精煉
        refined_content = refine_super_chunk_local(super_chunk['content'])
        
        print(f"\n   ✨ 精煉後長度: {len(refined_content)} 字符")
        print(f"   ✨ 精煉後內容:")
        print(f"      {refined_content}")
        print("   " + "=" * 50)
    
    # 總結
    print(f"\n📊 處理總結:")
    print(f"   📄 原始文本: {len(long_chinese_text)} 字符")
    print(f"   📦 基礎塊數: {len(base_chunks)} 個")
    print(f"   🔗 聚合塊數: {len(super_chunks)} 個")
    print(f"   🤖 使用模型: {OLLAMA_MODEL}")
    print(f"   🏠 運行模式: {'完全本地化' if LOCAL_MODE else '混合模式'}")
    print(f"   📊 嵌入方法: {'SentenceTransformer' if embedding_model_loaded else '簡單相似度'}")
    
    print(f"\n🎉 完全本地化處理完成！")

if __name__ == "__main__":
    main()