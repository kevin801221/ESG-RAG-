# src/extractors/metric_extractor.py

from typing import List, Dict, Any
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
import yaml
from ..db.vector_store import KDBAIStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

class ESGMetricExtractor:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 ESG 指標提取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化 OpenAI 組件
        self.llm = OpenAI(model="gpt-4-turbo-preview")
        self.embed_model = OpenAIEmbedding()
        
        # 初始化向量存儲
        self.vector_store = KDBAIStore()
        
        # 載入預定義的指標
        self.metrics = self._load_metrics()
        
    def _load_metrics(self) -> Dict[str, Any]:
        """載入預定義的 ESG 指標定義"""
        try:
            metrics_path = Path("src/config/metrics_schema.yaml")
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading metrics schema: {e}")
            return {}
            
    async def process_document(
        self,
        chunks: List[str],
        document_id: str,
        metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """處理文檔並提取指標"""
        try:
            # 生成嵌入
            embeddings = await self.embed_model.aget_text_embedding_batch(chunks)
            
            # 準備元數據
            chunk_metadata = [
                {**metadata, 'chunk_index': i}
                for i in range(len(chunks))
            ]
            
            # 儲存到 KDB.ai
            self.vector_store.add_documents(
                texts=chunks,
                embeddings=embeddings,
                metadata=chunk_metadata,
                document_id=document_id
            )
            
            # 提取指標
            results = []
            for category, metrics in self.metrics.items():
                for metric_id, metric_info in metrics.items():
                    try:
                        # 為每個指標生成查詢嵌入
                        metric_name = metric_info['name']
                        query_text = f"尋找關於 {metric_name} 的資訊"
                        query_embedding = await self.embed_model.aget_text_embedding(query_text)
                        
                        # 搜索相關文本
                        relevant_chunks = self.vector_store.similarity_search(
                            query_embedding,
                            k=3
                        )
                        
                        # 合併相關文本
                        context = "\n".join(chunk['text'] for chunk in relevant_chunks)
                        
                        # 使用 LLM 提取具體數值
                        prompt = f"""
                        請從以下文本中提取關於 "{metric_name}" 的資訊。
                        
                        文本內容：
                        {context}
                        
                        請以 JSON 格式返回：
                        {{
                            "value": "指標的具體數值",
                            "source": "數據的來源位置",
                            "confidence": "提取的信心分數（0-1之間）"
                        }}
                        """
                        
                        response = await self.llm.acomplete(prompt)
                        result = json.loads(response.text)
                        
                        if result.get('value'):
                            results.append({
                                '項目': metric_name,
                                '數據': result['value'],
                                '資料來源': result['source'],
                                '指標類別': category,
                                '代碼': metric_id,
                                '信心分數': result['confidence']
                            })
                            
                    except Exception as e:
                        self.logger.error(f"Error processing metric {metric_name}: {e}")
                        continue
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.error(f"Error in document processing: {e}")
            raise