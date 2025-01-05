# src/pipeline/rag_pipeline.py

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ..data.document_loader import DocumentLoader
from ..processors.text_processor import TextProcessor
from ..processors.table_processor import TableProcessor
from ..embeddings.text_embedder import TextEmbedder
from ..retrieval.vector_store import VectorStore

class RAGPipeline:
    def __init__(
        self,
        config: Dict,
        device: Optional[str] = None
    ):
        self.config = config
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize components
        self.doc_loader = DocumentLoader()
        self.text_processor = TextProcessor()
        self.table_processor = TableProcessor()
        self.text_embedder = TextEmbedder()
        self.vector_store = VectorStore()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.logger = logging.getLogger(__name__)

    async def process_document(
        self,
        file_path: Path,
        metrics_schema: Dict
    ) -> Tuple[List[Document], Dict]:
        """
        處理文件並提取 ESG 指標數據
        """
        try:
            # 1. 加載文件
            raw_doc = await self.doc_loader.load(file_path)
            
            # 2. 分割文本
            chunks = self.text_splitter.split_documents(raw_doc)
            
            # 3. 處理文本和表格
            processed_chunks = []
            extracted_metrics = {}
            
            for chunk in chunks:
                # 處理文本
                if chunk.page_content.strip():
                    processed_text = await self.text_processor.process(
                        chunk.page_content,
                        metrics_schema
                    )
                    processed_chunks.append(processed_text)
                
                # 處理表格
                if hasattr(chunk, 'table_data') and chunk.table_data:
                    table_metrics = await self.table_processor.process(
                        chunk.table_data,
                        metrics_schema
                    )
                    extracted_metrics.update(table_metrics)
            
            # 4. 生成嵌入
            embeddings = await self.text_embedder.embed_documents(processed_chunks)
            
            # 5. 存儲向量
            self.vector_store.add_embeddings(embeddings, processed_chunks)
            
            return processed_chunks, extracted_metrics
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    async def query(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Document]:
        """
        查詢相關文檔片段
        """
        try:
            # 1. 生成查詢嵌入
            query_embedding = await self.text_embedder.embed_query(query)
            
            # 2. 檢索相關文檔
            results = self.vector_store.similarity_search(
                query_embedding,
                k=top_k
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during query: {str(e)}")
            raise

    def get_metrics_summary(self) -> Dict:
        """
        獲取已提取的指標摘要
        """
        return self.extracted_metrics