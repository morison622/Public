#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 17:33:53 2025

@author: morison
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 17:31:50 2025

@author: morison.su

本程式利用Google gemini LLM API 進行文本整理，請自行更改LLM API
"""


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer # 引入 SentenceTransformer

import ollama # 引入 ollama 庫

# --- Ollama / Llama 3.1 配置 ---
ollama_model_name = "llama3.1" # 請替換為您在 Ollama 中下載的 Llama 3.1 模型名稱
ollama_configured = False
try:
    # 嘗試檢查模型是否可用
    # 注意：ollama list 不會拋出異常，所以我們需要更積極的檢查方式
    # 一種方法是嘗試一個簡單的生成，或者直接假設如果ollama服務運行則模型可用
    # 更穩健的方法是調用ollama run，但在這裡我們只做一個基本的配置檢查
    # 這裡我們假設如果ollama服務在運行，那麼指定的模型是可用的，
    # 或者用戶會確保在運行此腳本之前模型已被下載。
    ollama.chat(model=ollama_model_name, messages=[{'role': 'user', 'content': 'hi'}], stream=False)
    ollama_configured = True
    print(f"Ollama 模型 '{ollama_model_name}' 配置成功。")
except Exception as e:
    print(f"警告：Ollama 模型 '{ollama_model_name}' 配置失敗。請確保 Ollama 服務正在運行且模型已下載。")
    print(f"錯誤信息: {e}")
    print("將無法執行 Super Chunk 的 LLM 整理步驟。")

if ollama_configured:
    # Ollama 不需要像 Gemini 那樣實例化一個模型對象，而是直接通過 ollama.chat() 或 ollama.generate() 調用
    local_llm_model = ollama_model_name 
else:
    local_llm_model = None

# --- 載入 SentenceTransformer 模型 ---
# 選擇一個適合中文的嵌入模型。
# 'paraphrase-multilingual-MiniLM-L12-v2' 是一個多語言模型，性能不錯且體積較小。
# 如果需要更好的性能或更大的模型，可以考慮其他選項。
try:
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("SentenceTransformer 嵌入模型加載成功。")
    embedding_model_loaded = True
except Exception as e:
    print(f"錯誤：無法加載 SentenceTransformer 嵌入模型: {e}")
    print("請確保您已安裝 'sentence-transformers' 庫，並檢查網路連接。")
    print("將無法執行語義相似度聚合。")
    embedding_model_loaded = False
    embedding_model = None


# 一篇較長的中文文章範例
long_chinese_text = """
人工智能（AI）正在迅速改變我們的生活和工作方式。從智能手機中的語音助手到自動駕駛汽車，AI 的應用無處不在。AI 的核心目標是使機器能夠像人類一樣思考、學習和解決問題。

機器學習是AI的一個重要分支，它使計算機系統能夠從數據中學習而無需明確編程。深度學習是機器學習的子領域，它利用人工神經網絡，特別是多層次網絡，來處理複雜模式識別任務，例如圖像識別和自然語言處理。

自然語言處理（NLP）是AI的另一個關鍵領域，它專注於讓計算機理解、解釋和生成人類語言。語音識別、機器翻譯和情感分析都是NLP的應用。透過NLP，我們可以與計算機進行更自然的互動。

雖然AI帶來了巨大的潛力，但也伴隨著一些挑戰，例如倫理問題、隱私問題以及對就業市場的影響。如何平衡AI的發展與社會責任是我們需要共同面對的課題。未來，AI將繼續深入發展，影響我們生活的方方面面。
"""

print("--- 原始長篇中文文章 ---")
print(long_chinese_text)
print("-" * 30)

### 步驟 1: 基礎重疊切塊函數 (無變動)

def chunk_text_with_overlap(text: str, chunk_size: int, overlap_size: int) -> list[dict]:
    """
    將長文本進行重疊切塊，並為每個區塊添加 ID。
    """
    chunks = []
    text_length = len(text)
    start_index = 0
    chunk_id_counter = 0

    while start_index < text_length:
        end_index = min(start_index + chunk_size, text_length)
        chunk_content = text[start_index:end_index]
        chunks.append({"id": f"base_chunk_{chunk_id_counter}", "content": chunk_content})
        chunk_id_counter += 1

        if end_index == text_length:
            break

        start_index += (chunk_size - overlap_size)
        start_index = min(start_index, text_length - 1) 
        
    return chunks



### **關鍵修改：使用真實嵌入模型**


# 使用真實的嵌入模型
def get_real_embeddings(texts: list[str], model) -> np.ndarray:
    """
    使用 SentenceTransformer 模型生成文本嵌入。

    Args:
        texts (list[str]): 待嵌入的文本列表。
        model: 已加載的 SentenceTransformer 模型實例。

    Returns:
        np.ndarray: 文本的嵌入向量陣列。
    """
    if model is None:
        raise ValueError("嵌入模型未加載或初始化失敗，無法生成嵌入。")
    
    # SentenceTransformer 的 encode 方法會直接返回 NumPy 陣列
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


def aggregate_chunks_by_similarity(
    base_chunks: list[dict], 
    embedding_model, # 傳入實際的嵌入模型
    similarity_threshold: float, 
    max_super_chunk_size: int = None, 
    min_super_chunk_size_required: int = 1, 
    max_gap_between_chunks: int = 0 
) -> list[dict]:
    """
    根據語義相似度將基礎文字塊聚合為更大的上下文文字塊，並限制最大大小。
    """
    if not base_chunks:
        return []
    
    if not embedding_model_loaded:
        print("警告：嵌入模型未加載，無法執行語義相似度聚合。")
        return []

    base_chunk_contents = [chunk['content'] for chunk in base_chunks]
    
    print("正在生成真實嵌入... (這可能需要一些時間，具體取決於模型大小和文本數量)")
    base_chunk_embeddings = get_real_embeddings(base_chunk_contents, embedding_model) # 調用真實嵌入模型
    print("真實嵌入生成完成。")

    aggregated_chunks = []
    
    merged_indices = set()
    super_chunk_id_counter = 0

    for i in range(len(base_chunks)):
        if i in merged_indices:
            continue

        current_super_chunk_content = base_chunks[i]['content']
        current_super_chunk_ids = [base_chunks[i]['id']]
        
        merged_indices.add(i)

        gap_count = 0
        for j in range(i + 1, len(base_chunks)):
            if j in merged_indices:
                continue

            if max_super_chunk_size is not None and len(current_super_chunk_ids) >= max_super_chunk_size:
                print(f"  [DEBUG] Super Chunk {super_chunk_id_counter} 達到最大限制 {max_super_chunk_size}，停止合併。")
                break 

            sim = cosine_similarity(
                base_chunk_embeddings[i].reshape(1, -1), 
                base_chunk_embeddings[j].reshape(1, -1)
            )[0][0]

            if sim >= similarity_threshold:
                current_super_chunk_content += base_chunks[j]['content']
                current_super_chunk_ids.append(base_chunks[j]['id'])
                merged_indices.add(j)
                gap_count = 0 
            else:
                gap_count += 1
                if gap_count > max_gap_between_chunks:
                    break 
        
        if len(current_super_chunk_ids) >= min_super_chunk_size_required:
            aggregated_chunks.append({
                "id": f"super_chunk_{super_chunk_id_counter}",
                "content": current_super_chunk_content,
                "original_chunk_ids": current_super_chunk_ids
            })
            super_chunk_id_counter += 1

    return aggregated_chunks


def summarize_or_refine_super_chunk(super_chunk_content: str, model_name: str) -> str:
    """
    使用本地 LLM (Llama 3.1) 整理或精煉合併後的 Super Chunk 內容。
    """
    prompt = f"""請精煉並整理以下文本內容，去除冗餘多餘詞彙與空格，使其更流暢和精確，同時保持所有原始關鍵資訊。
    
    文本內容:
    {super_chunk_content}
    
    精煉後的文本:
    """
    try:
        # 使用 ollama.chat 或 ollama.generate 調用本地模型
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            stream=False
        )
        return response['message']['content']
    except Exception as e:
        print(f"LLM 整理 Super Chunk 失敗: {e}")
        return super_chunk_content
    
    
    # 設定參數
CHUNK_SIZE = 60
OVERLAP_SIZE = 15
SIMILARITY_THRESHOLD = 0.7 # 語義相似度閾值 (需要根據模型和數據調整，多語言模型可能需要較低的閾值)

# 聚合 Super Chunk 的參數
MAX_SUPER_CHUNK_SIZE = 4 # 舉例：每個大塊最多包含 3 個基礎塊
MIN_SUPER_CHUNK_SIZE_REQUIRED = 2 # 每個大塊至少包含的基礎文字塊數量
MAX_GAP_BETWEEN_CHUNKS = 0 

print(f"\n--- 步驟 1: 基礎重疊切塊 (區塊大小: {CHUNK_SIZE}, 重疊大小: {OVERLAP_SIZE}) ---")
base_chunks = chunk_text_with_overlap(long_chinese_text, CHUNK_SIZE, OVERLAP_SIZE)

for i, chunk in enumerate(base_chunks):
    print(f"基礎區塊 {chunk['id']} (長度: {len(chunk['content'])} 字元):")
    print(chunk['content'])
    print("-" * 20)

print(f"\n--- 步驟 2 & 3: 語義相似度聚合 ---")
print(f"  語義相似度閾值: {SIMILARITY_THRESHOLD}")
print(f"  聚合大塊至多包含基礎塊數量: {MAX_SUPER_CHUNK_SIZE if MAX_SUPER_CHUNK_SIZE is not None else '無上限'}")
print(f"  聚合大塊至少包含基礎塊數量: {MIN_SUPER_CHUNK_SIZE_REQUIRED}")
print(f"  最大非相似塊間隔: {MAX_GAP_BETWEEN_CHUNKS}")

contextual_super_chunks = []
if embedding_model_loaded: # 只有在嵌入模型加載成功時才執行聚合
    contextual_super_chunks = aggregate_chunks_by_similarity(
        base_chunks, 
        embedding_model, # 傳入實際的嵌入模型
        SIMILARITY_THRESHOLD, 
        max_super_chunk_size=MAX_SUPER_CHUNK_SIZE, 
        min_super_chunk_size_required=MIN_SUPER_CHUNK_SIZE_REQUIRED, 
        max_gap_between_chunks=MAX_GAP_BETWEEN_CHUNKS
    )
else:
    print("跳過語義相似度聚合，因為嵌入模型加載失敗。")


if not contextual_super_chunks:
    print("沒有生成任何聚合大塊。請嘗試調整相似度閾值、基礎塊大小或 max_super_chunk_size。")
else:
    for i, super_chunk in enumerate(contextual_super_chunks):
        print(f"聚合大塊 {super_chunk['id']} (包含 {len(super_chunk['original_chunk_ids'])} 個基礎塊):")
        print(f"  基礎塊 ID: {super_chunk['original_chunk_ids']}")
        print(f"  內容 (原始拼接長度: {len(super_chunk['content'])} 字元):\n{super_chunk['content']}")
        
        # --- 步驟 4: 使用 LLM 整理 Super Chunk ---
        if ollama_configured and local_llm_model:
            print("\n  正在使用 LLM 整理 Super Chunk 內容...")
            refined_content = summarize_or_refine_super_chunk(super_chunk['content'], local_llm_model)
            print(f"  **LLM 整理後內容 (長度: {len(refined_content)} 字元):**\n{refined_content}")
        else:
            print("\n  **LLM 整理跳過：Ollama 未配置或初始化失敗。**")
        print("=" * 30)

print("\n--- 總結 ---")
print(f"原始文本長度: {len(long_chinese_text)} 字元")
print(f"共生成 {len(base_chunks)} 個基礎文字塊。")
print(f"共生成 {len(contextual_super_chunks)} 個聚合大塊。")
print("原始基礎文字塊 (base_chunks) 仍然存在，可以根據需求用於檢索。")
print("聚合大塊 (contextual_super_chunks) 經過 LLM 整理後，更適合作為提供上下文的單位。")