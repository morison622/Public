import requests
import json
import random
from typing import List, Dict

# 請將 YOUR_API_KEY_HERE 替換成你的 Google Gemini API Key
# 務必保護好你的 API Key，不要分享給他人。
GEMINI_API_KEY = "    "

def get_gemini_response(prompt: str, api_key: str) -> str:
    """
    呼叫 Gemini API 取得文本回應。
    """
    model_name = "gemini-2.5-flash-preview-05-20"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Gemini API 呼叫失敗: {e}")
        return "無法生成回應。"

def llm_as_challenger(question: str, rag_answer: str, retrieved_chunks: List[str]) -> str:
    """
    使用 LLM 扮演嚴厲使用者，對答案提出挑戰:挑戰答案中沒回答到原始問題的關鍵資訊。
    """
    prompt = f"""
    你現在是一位嚴厲的學術研究者，專注於找出資訊中的漏洞和不準確之處。
    你的任務是審查RAG系統給出的答案，並根據所提供的參考資料，先比較原始題目跟系統列出的結果，列出原始問題中但RAG結果中未提及的關鍵資訊。然後，針對個別遺漏的關鍵資訊，逐一提出簡潔、具體的後續問題來挑戰答案的完整性。

    以下是原始問題和RAG系統給出的答案，以及其參考的檢索資料：

    ---
    原始問題: {question}
    RAG答案: {rag_answer}
    參考資料: {retrieved_chunks}
    ---

    你的回饋必須是一個單獨的、針對原始問題中遺漏關鍵資訊的後續問題，例如：「你的答案沒有解釋...，你能補充嗎？」或「你能解釋一下...和...之間的關係嗎？」
    """
    return get_gemini_response(prompt, GEMINI_API_KEY)

def simple_rerank(query: str, chunks: List[str]) -> List[str]:
    """
    簡單的重排序邏輯：根據關鍵詞匹配來排序。
    在實際應用中，會使用更複雜的向量相似度模型。
    """
    def score_chunk(chunk: str, query: str) -> int:
        score = 0
        keywords = query.replace("?", "").replace("嗎", "").split()
        for keyword in keywords:
            if keyword.lower() in chunk.lower():
                score += 1
        return score

    sorted_chunks = sorted(chunks, key=lambda c: score_chunk(c, query), reverse=True)
    return sorted_chunks

def rag_generation(question: str, chunks: List[str]) -> str:
    """
    模擬 RAG 系統的生成階段，根據檢索到的短文生成答案。
    """
    prompt = f"""
    根據以下檢索到的資料，回答這個問題：
    
    原始問題: {question}
    
    檢索到的資料:
    {chunks}
    
    請用這些資料來生成一個全面、精準的答案。
    """
    return get_gemini_response(prompt, GEMINI_API_KEY)

if __name__ == "__main__":
    # 模擬檢索到的 10 個短文（chunkings）
    sample_chunks = [
        "檢索增強生成 (RAG) 結合了檢索模組和生成模組，可以讓大型語言模型 (LLM) 存取外部知識庫。這個模組主要目的是減少模型的幻覺問題。",
        "RAG 的工作流程首先是使用檢索器從龐大的文件資料庫中找到最相關的片段。",
        "在 RAG 系統中，生成器會利用檢索到的資訊來生成最終答案。生成器通常是一個預訓練好的 LLM，例如 GPT 或 PaLM。",
        "向量資料庫是實現 RAG 系統的關鍵技術之一，它能高效地儲存和查詢大量的文本嵌入向量。",
        "除了向量檢索，TF-IDF 或 BM25 等詞彙匹配方法也常用於 RAG 的檢索階段。",
        "RAG 的優勢在於其可解釋性和時效性，因為生成內容都有明確的來源依據。",
        "RAG 模型的挑戰之一是如何處理檢索結果的重排序 (re-ranking)，以確保最相關的資訊能被優先處理。",
        "語義分割 (Semantic Chunking) 是一種新的預處理技術，它會根據文章的語義完整性來切分文件，而不是固定大小。",
        "人工智慧的應用領域非常廣泛，涵蓋了自然語言處理、電腦視覺和機器學習。",
        "神經網路是深度學習的基礎，它模仿人腦的結構來處理複雜的模式識別任務。"
    ]

    # 初始問題
    initial_question = "RAG 的主要優點和工作原理是什麼？"

    current_question = initial_question
    current_answer = ""
    current_chunks = sample_chunks
    
    print("="*60)
    print("      AI 驅動的互動式重排序模擬 (十輪)")
    print("="*60)

    previous_top_chunks = None
    
    # 執行最多十輪互動，但會根據穩定性提前停止
    for i in range(1, 11):
        print(f"\n--- 第 {i} 輪互動 ---")
        print(f"當前使用者問題: {current_question}")
        
        # 步驟一：生成答案
        # 重排序
        reranked_chunks = simple_rerank(current_question, current_chunks)
        
        # 檢查停止條件
        current_top_chunks = [chunk for chunk in reranked_chunks[:3]]
        if previous_top_chunks and current_top_chunks == previous_top_chunks:
            print("\n→ 檢索結果已穩定，前三名與上一輪相同。提前結束互動。")
            break
            
        previous_top_chunks = current_top_chunks

        print("→ 系統已重新排序檢索結果。前三名：")
        for j, chunk in enumerate(reranked_chunks[:3], 1):
            print(f"   {j}. {chunk[:30]}...")
        answer = rag_generation(current_question, reranked_chunks)

        print(f"\n系統答案: {answer}")
        
        # 步驟二：LLM 扮演使用者提出挑戰
        challenger_response = llm_as_challenger(current_question, answer, current_chunks)
        print(f"\n嚴厲使用者提出的挑戰：{challenger_response}")
        
        # 步驟三：更新當前問題和檢索結果，準備下一輪
        current_question = challenger_response
        current_answer = answer
        current_chunks = sample_chunks.copy()

    print("\n" + "="*60)
    print("模擬結束。")
