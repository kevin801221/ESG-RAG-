# src/db/vector_store.py

import os
import logging
from typing import List, Dict, Any
import numpy as np
import kdbai_client as kdbai
from dotenv import load_dotenv

class KDBAIStore:
    def __init__(self):
        """初始化向量存儲"""
        load_dotenv()
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # 獲取環境變數
        self.endpoint = os.getenv('KDBAI_ENDPOINT')
        self.api_key = os.getenv('KDBAI_API_KEY')
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Missing KDBAI credentials in environment variables")
        
        # 初始化 KDB.AI 連接
        self.session = kdbai.Session(
            api_key=self.api_key,
            endpoint=self.endpoint
        )
        
        # 初始化或獲取向量表
        self.database = self._init_database()
        self.table = self._init_table()
        
    def _init_database(self) -> kdbai.Database:
        """初始化或獲取資料庫"""
        try:
            db_name = "esg_database"
            
            # 檢查是否已存在資料庫
            databases = self.session.databases()
            self.logger.debug(f"Available databases: {[db.name for db in databases]}")
            
            for db in databases:
                if db.name == db_name:
                    self.logger.info(f"Using existing database: {db_name}")
                    return db
            
            # 創建新資料庫
            self.logger.info(f"Creating new database: {db_name}")
            return self.session.create_database(db_name)  # 直接傳遞資料庫名稱作為第一個參數
            
        except Exception as e:
            self.logger.error(f"Error initializing KDB.AI database: {str(e)}")
            raise
            
    def _init_table(self) -> kdbai.Table:
        """初始化或獲取向量表"""
        try:
            table_name = "esg_vectors"
            
            # 檢查是否已存在表
            tables = self.database.tables
            self.logger.debug(f"Available tables: {[t.name for t in tables]}")
            
            for table in tables:
                if table.name == table_name:
                    self.logger.info(f"Using existing table: {table_name}")
                    return table
            
            # 創建新表
            self.logger.info(f"Creating new table: {table_name}")
            
            # 創建表格 schema
            schema = {
                'vector': {'type': 'vector', 'dimension': 1536},
                'document_id': {'type': 'string'},
                'text': {'type': 'string'},
                'metadata': {'type': 'json'}
            }
            
            # 創建表格
            return self.database.create_table(
                table_name,
                schema,
                {'similarity_metric': 'cosine'}
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing KDB.AI table: {str(e)}")
            raise

    def _validate_vector(self, vector: np.ndarray) -> np.ndarray:
        """驗證並格式化向量"""
        try:
            # 確保向量是 numpy array
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector)
            
            # 確保是 float32 類型
            vector = vector.astype(np.float32)
            
            # 檢查維度
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            elif vector.ndim > 2:
                raise ValueError(f"Vector has too many dimensions: {vector.ndim}")
                
            # 檢查向量長度
            if vector.shape[1] != 1536:
                raise ValueError(f"Vector dimension mismatch. Expected 1536, got {vector.shape[1]}")
                
            return vector
            
        except Exception as e:
            self.logger.error(f"Error validating vector: {str(e)}")
            raise
            
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[Any],
        metadata: List[Dict],
        document_id: str
    ) -> None:
        """添加文檔到向量存儲"""
        try:
            if not texts or not embeddings or not metadata:
                raise ValueError("Empty texts, embeddings, or metadata")
                
            if len(texts) != len(embeddings) or len(texts) != len(metadata):
                raise ValueError("Length mismatch between texts, embeddings, and metadata")
                
            # 處理每個向量
            for text, embedding, meta in zip(texts, embeddings, metadata):
                # 驗證並格式化向量
                vector = self._validate_vector(embedding)
                
                # 添加文檔ID到元數據
                meta['document_id'] = document_id
                meta['text'] = text
                
                # 添加到 KDB.AI
                self.table.add(
                    vectors=vector,
                    metadata=[meta]
                )
                
            self.logger.info(f"Successfully added {len(texts)} documents to KDB.AI")
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise
            
    async def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """執行相似度搜索"""
        try:
            # 確保查詢向量格式正確
            if isinstance(query, str):
                # 如果是文本，需要先獲取嵌入
                self.logger.error("Query must be a vector, not text")
                return []
                
            query_vector = self._validate_vector(query)
            
            # 執行搜索
            search_params = {
                "vectors": query_vector,
                "k": k,
                "include_vectors": False,
                "include_distances": True
            }
            
            results = self.table.search(**search_params)
            
            # 處理結果
            processed_results = []
            for i, (metadata, distance) in enumerate(zip(results.metadata, results.distances)):
                processed_results.append({
                    'text': metadata.get('text', ''),
                    'metadata': metadata,
                    'score': 1 - distance  # 將距離轉換為相似度分數
                })
                
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            return []  # 返回空列表而不是拋出異常
            
    def clear(self) -> None:
        """清除所有向量"""
        try:
            if hasattr(self.table, 'clear'):
                self.table.clear()
                self.logger.info("Successfully cleared all vectors")
            else:
                self.logger.warning("Table does not have clear method")
        except Exception as e:
            self.logger.error(f"Error clearing vectors: {str(e)}")
            raise

if __name__ == "__main__":
    # 設置日誌
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 測試代碼
    store = KDBAIStore()
    print("Successfully initialized KDBAIStore")