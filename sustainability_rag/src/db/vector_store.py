# src/db/vector_store.py

import os
from typing import List, Dict, Any
import kdbai_client as kdbai
import logging
from dotenv import load_dotenv

load_dotenv()

class KDBAIStore:
    def __init__(self):
        """初始化 KDB.ai 連接"""
        self.endpoint = os.getenv('KDBAI_ENDPOINT')
        self.api_key = os.getenv('KDBAI_API_KEY')
        self.logger = logging.getLogger(__name__)
        
        # 連接到 KDB.ai
        try:
            self.session = kdbai.Session(
                api_key=self.api_key,
                endpoint=self.endpoint
            )
            self.logger.info("Successfully connected to KDB.ai")
        except Exception as e:
            self.logger.error(f"Error connecting to KDB.ai: {e}")
            raise
            
        # 初始化資料庫和表格
        self._init_database()
        
    def _init_database(self):
        """初始化資料庫和表格結構"""
        try:
            # 創建或取得資料庫
            db_name = "esg_reports"
            try:
                self.db = self.session.create_database(db_name)
            except kdbai.KDBAIException:
                self.db = self.session.database(db_name)
                
            # 定義表格結構
            table_schema = [
                {"name": "document_id", "type": "str"},
                {"name": "chunk_id", "type": "str"},
                {"name": "text", "type": "str"},
                {"name": "metadata", "type": "str"},
                {"name": "embeddings", "type": "float64s"}
            ]
            
            # 定義向量索引
            indexes = [{
                'type': 'flat',  # 可以根據需求改為 'hnsw'
                'name': 'embedding_index',
                'column': 'embeddings',
                'params': {
                    'dims': 1536,  # OpenAI text-embedding-3-small 的維度
                    'metric': "CS"  # 餘弦相似度
                }
            }]
            
            # 創建或取得表格
            table_name = "document_chunks"
            try:
                self.table = self.db.create_table(
                    table=table_name,
                    schema=table_schema,
                    indexes=indexes
                )
            except kdbai.KDBAIException:
                self.table = self.db.table(table_name)
                
            self.logger.info("Database and table initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
            
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        document_id: str
    ):
        """添加文檔到向量存儲"""
        try:
            # 準備數據
            records = []
            for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
                chunk_id = f"{document_id}_{i}"
                records.append({
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": str(meta),
                    "embeddings": embedding
                })
            
            # 批量插入數據
            self.table.insert(records)
            self.logger.info(f"Added {len(records)} chunks to KDB.ai")
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise
            
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Dict = None
    ) -> List[Dict]:
        """
        執行相似度搜索
        
        Args:
            query_embedding: 查詢向量
            k: 返回的結果數量
            filter_dict: 過濾條件
            
        Returns:
            相似文檔列表
        """
        try:
            # 構建查詢
            search_params = {
                "vectors": {
                    "embedding_index": query_embedding
                },
                "n": k
            }
            
            # 如果有過濾條件，添加到查詢中
            if filter_dict:
                filters = []
                for key, value in filter_dict.items():
                    filters.append(("=", key, value))
                search_params["filter"] = filters
            
            # 執行搜索
            results = self.table.search(**search_params)
            
            # 格式化結果
            formatted_results = []
            for result in results[0].to_dict('records'):
                formatted_results.append({
                    'text': result['text'],
                    'metadata': eval(result['metadata']),
                    'score': float(result.get('_distance', 0))
                })
                
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            raise