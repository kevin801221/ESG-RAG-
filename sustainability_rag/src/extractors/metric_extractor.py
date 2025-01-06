# # src/extractors/metric_extractor.py

# from typing import List, Dict, Any
# import pandas as pd
# import json
# import logging
# from pathlib import Path
# from datetime import datetime
# import yaml
# from ..db.vector_store import KDBAIStore
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.llms.openai import OpenAI

# class ESGMetricExtractor:
#     def __init__(self, config: Dict[str, Any]):
#         """
#         初始化 ESG 指標提取器
        
#         Args:
#             config: 配置字典
#         """
#         self.config = config
#         self.logger = logging.getLogger(__name__)
        
#         # 初始化 OpenAI 組件
#         self.llm = OpenAI(model="gpt-4-turbo-preview")
#         self.embed_model = OpenAIEmbedding()
        
#         # 初始化向量存儲
#         self.vector_store = KDBAIStore()
        
#         # 載入預定義的指標
#         self.metrics = self._load_metrics()
        
#     def _load_metrics(self) -> Dict[str, Any]:
#         """載入預定義的 ESG 指標定義"""
#         try:
#             metrics_path = Path("src/config/metrics_schema.yaml")
#             with open(metrics_path, 'r', encoding='utf-8') as f:
#                 return yaml.safe_load(f)
#         except Exception as e:
#             self.logger.error(f"Error loading metrics schema: {e}")
#             return {}
            
#     async def process_document(
#         self,
#         chunks: List[str],
#         document_id: str,
#         metadata: Dict[str, Any]
#     ) -> pd.DataFrame:
#         """處理文檔並提取指標"""
#         try:
#             # 生成嵌入
#             embeddings = await self.embed_model.aget_text_embedding_batch(chunks)
            
#             # 準備元數據
#             chunk_metadata = [
#                 {**metadata, 'chunk_index': i}
#                 for i in range(len(chunks))
#             ]
            
#             # 儲存到 KDB.ai
#             self.vector_store.add_documents(
#                 texts=chunks,
#                 embeddings=embeddings,
#                 metadata=chunk_metadata,
#                 document_id=document_id
#             )
            
#             # 提取指標
#             results = []
#             for category, metrics in self.metrics.items():
#                 for metric_id, metric_info in metrics.items():
#                     try:
#                         # 為每個指標生成查詢嵌入
#                         metric_name = metric_info['name']
#                         query_text = f"尋找關於 {metric_name} 的資訊"
#                         query_embedding = await self.embed_model.aget_text_embedding(query_text)
                        
#                         # 搜索相關文本
#                         relevant_chunks = self.vector_store.similarity_search(
#                             query_embedding,
#                             k=3
#                         )
                        
#                         # 合併相關文本
#                         context = "\n".join(chunk['text'] for chunk in relevant_chunks)
                        
#                         # 使用 LLM 提取具體數值
#                         prompt = f"""
#                         請從以下文本中提取關於 "{metric_name}" 的資訊。
                        
#                         文本內容：
#                         {context}
                        
#                         請以 JSON 格式返回：
#                         {{
#                             "value": "指標的具體數值",
#                             "source": "數據的來源位置",
#                             "confidence": "提取的信心分數（0-1之間）"
#                         }}
#                         """
                        
#                         response = await self.llm.acomplete(prompt)
#                         result = json.loads(response.text)
                        
#                         if result.get('value'):
#                             results.append({
#                                 '項目': metric_name,
#                                 '數據': result['value'],
#                                 '資料來源': result['source'],
#                                 '指標類別': category,
#                                 '代碼': metric_id,
#                                 '信心分數': result['confidence']
#                             })
                            
#                     except Exception as e:
#                         self.logger.error(f"Error processing metric {metric_name}: {e}")
#                         continue
            
#             return pd.DataFrame(results)
            
#         except Exception as e:
#             self.logger.error(f"Error in document processing: {e}")
#             raise

# src/extractors/metric_extractor.py

from typing import List, Dict, Any
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
import yaml
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.kdbai import KDBAIVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import kdbai_client as kdbai

class ESGMetricExtractor:
    def __init__(self, config: Dict[str, Any]):
        """初始化 ESG 指標提取器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化 OpenAI 組件
        self.llm = OpenAI(model="gpt-4-turbo-preview")
        self.embed_model = OpenAIEmbedding()
        
        # 初始化 KDB.AI 連接
        self.kdbai_session = self._init_kdbai()
        self.vector_store = self._init_vector_store()
        
        # 載入預定義的指標
        self.metrics = self._load_metrics()
        
    def _init_kdbai(self) -> kdbai.Session:
        """初始化 KDB.AI 連接"""
        try:
            endpoint = self.config.get('kdbai', {}).get('endpoint')
            api_key = self.config.get('kdbai', {}).get('api_key')
            
            if not endpoint or not api_key:
                raise ValueError("Missing KDBAI credentials in config")
                
            return kdbai.Session(api_key=api_key, endpoint=endpoint)
            
        except Exception as e:
            self.logger.error(f"初始化 KDB.AI 連接時出錯: {str(e)}")
            raise
            
    def _init_vector_store(self) -> KDBAIVectorStore:
        """初始化向量存儲"""
        try:
            # 定義 schema
            schema = [
                dict(name="document_id", type="str"),
                dict(name="text", type="str"),
                dict(name="embeddings", type="float32s"),
            ]
            
            # 定義索引
            index_flat = {
                "name": "flat",
                "type": "flat",
                "column": "embeddings",
                "params": {'dims': 1536, 'metric': 'L2'},
            }
            
            # 獲取資料庫
            db = self.kdbai_session.database("default")
            table_name = "esg_vectors"
            
            # 如果表已存在則刪除
            try:
                db.table(table_name).drop()
            except kdbai.KDBAIException:
                pass
                
            # 創建表
            table = db.create_table(
                table_name,
                schema,
                indexes=[index_flat]
            )
            
            return KDBAIVectorStore(table)
            
        except Exception as e:
            self.logger.error(f"初始化向量存儲時出錯: {str(e)}")
            raise
            
    def _load_metrics(self) -> Dict[str, Any]:
        """載入預定義的 ESG 指標定義"""
        try:
            metrics_path = Path("src/config/metrics_schema.yaml")
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"載入指標 schema 時出錯: {e}")
            return {}
            
    async def process_document(
        self,
        chunks: List[str],
        document_id: str,
        metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """處理文檔並提取指標"""
        try:
            # 創建存儲上下文
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # 創建文檔節點
            nodes = []
            for i, chunk in enumerate(chunks):
                node_metadata = {
                    **metadata,
                    'chunk_index': i,
                    'document_id': document_id
                }
                nodes.append({
                    'text': chunk,
                    'metadata': node_metadata
                })
            
            # 創建向量索引
            index = VectorStoreIndex.from_documents(
                nodes,
                storage_context=storage_context
            )
            
            # 提取指標
            results = []
            for category, metrics in self.metrics.items():
                for metric_id, metric_info in metrics.items():
                    try:
                        # 構建查詢
                        query = f"""
                        請從提供的文本中找出關於 "{metric_info['name']}" 的以下資訊:
                        1. 具體數值和單位
                        2. 資料來源 (頁碼或章節)
                        3. 如果有歷史數據,也請提供
                        
                        請以 JSON 格式返回:
                        {{
                            "value": "指標的具體數值",
                            "source": "數據的來源位置",
                            "confidence": "提取的信心分數 (0-1)",
                            "trend": "可選,與去年相比的趨勢"
                        }}
                        """
                        
                        # 執行查詢
                        response = await index.aquery(query)
                        
                        # 解析響應
                        try:
                            result = json.loads(response.response)
                            results.append({
                                '項目': metric_info['name'],
                                '數據': result['value'],
                                '資料來源': result['source'],
                                '指標類別': category,
                                '代碼': metric_id,
                                '信心分數': result['confidence'],
                                '趨勢': result.get('trend', '')
                            })
                        except json.JSONDecodeError:
                            self.logger.warning(f"解析響應時出錯: {response.response}")
                            continue
                            
                    except Exception as e:
                        self.logger.error(f"處理指標 {metric_info['name']} 時出錯: {e}")
                        continue
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.error(f"文檔處理時出錯: {str(e)}")
            raise
            
    def cleanup(self):
        """清理資源"""
        try:
            # 刪除向量表
            self.vector_store._table.drop()
        except Exception as e:
            self.logger.warning(f"清理向量存儲時出錯: {e}")