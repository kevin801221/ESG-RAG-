# src/rag/retriever.py
import os
from typing import List, Dict, Any
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ESGRetriever:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 ESG 檢索器
        
        Args:
            config: 配置字典，包含 OpenAI API key 等設置
        """
        self.client = OpenAI(api_key=config['openai_api_key'])
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        獲取文本的向量嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            numpy array of embeddings
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return np.array([r.embedding for r in response.data])
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            raise

    def search(self, query: str, chunks: List[str], k: int = 3) -> List[Dict]:
        """
        搜索最相關的文本塊
        
        Args:
            query: 搜索查詢
            chunks: 文本塊列表
            k: 返回的結果數量
            
        Returns:
            包含相關文本和分數的字典列表
        """
        # 獲取查詢和文檔塊的嵌入
        query_embedding = self.get_embeddings([query])[0]
        chunk_embeddings = self.get_embeddings(chunks)
        
        # 計算相似度
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            chunk_embeddings
        )[0]
        
        # 獲取最相關的結果
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'text': chunks[idx],
                'score': float(similarities[idx])
            })
        
        return results

    async def extract_metric(
        self,
        metric_name: str,
        context: str
    ) -> Dict[str, str]:
        """
        從上下文中提取特定指標的值和來源
        
        Args:
            metric_name: ESG 指標名稱
            context: 相關文本上下文
            
        Returns:
            包含值和來源的字典
        """
        prompt = f"""請從以下文本中提取關於 "{metric_name}" 的資訊。
        
文本內容：
{context}

請以 JSON 格式返回以下資訊：
1. value: 指標的具體數值
2. source: 數據的來源位置（如：某頁、某表格、某章節等）
3. confidence: 提取的把握度（0-1之間的數值）

如果找不到相關資訊，請將 value 和 source 設為 null，confidence 設為 0。
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": "你是一個專門提取 ESG 報告數據的助手。請只返回要求的 JSON 格式資料，不要添加任何其他解釋。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error extracting metric: {e}")
            return {
                "value": None,
                "source": None,
                "confidence": 0
            }

    def process_metric(
        self,
        metric_name: str,
        document_chunks: List[str]
    ) -> Dict[str, Any]:
        """
        處理單個 ESG 指標
        
        Args:
            metric_name: 指標名稱
            document_chunks: 文檔分塊
            
        Returns:
            包含提取結果的字典
        """
        # 1. 搜索相關文本
        relevant_chunks = self.search(metric_name, document_chunks)
        
        # 2. 合併最相關的文本
        context = "\n".join(chunk['text'] for chunk in relevant_chunks)
        
        # 3. 提取指標數據
        result = self.extract_metric(metric_name, context)
        
        return result