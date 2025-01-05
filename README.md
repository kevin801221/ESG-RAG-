```markdown
# ESG-RAG-AI 永續報告書轉換系統

一個基於多模態 RAG（Retrieval Augmented Generation）的 ESG 永續報告書分析系統，能夠自動化提取和分析 ESG 相關指標數據，並生成結構化的 Excel 輸出。

## 專案簡介

本系統利用大型語言模型（LLM）和多模態檢索增強生成（RAG）技術，專門處理企業永續報告中的 ESG 指標數據。系統能夠處理文本、表格和圖片等多種形式的內容，並通過先進的 RAG 技術提供精確的數據提取。

### 主要功能

- 🔍 **多模態內容處理**：支援文本、表格和圖片的解析與數據提取
- 📊 **自動化指標提取**：根據預定義的指標清單，自動從報告中提取對應數值
- 🤖 **智能數據識別**：使用 ImageBind 進行圖表解析和數據識別
- 📈 **結構化輸出**：將提取的數據整理成標準的 Excel 格式

### 技術架構

- 文件處理：LlamaParse
- 向量數據庫：KDB.AI
- 圖像處理：ImageBind
- 文本處理：OpenAI GPT-4
- 開發框架：PyTorch (MPS 加速)

## 系統要求

- Python 3.8+
- Apple Silicon Mac (支援 MPS 加速)
- OpenAI API Key
- KDB.AI 帳號和 API Key

## 專案結構

```
sustainability_rag/
├── .env                        # 環境變數配置
├── .gitignore                  # Git 忽略文件
├── README.md                   # 本文件
├── requirements.txt            # 依賴套件
├── app.py                      # Gradio 介面主程序
├── ESG_env/                    # 虛擬環境 (git 忽略)
│
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.yaml        # 基礎配置
│   │   ├── metrics_schema.yaml # ESG指標schema
│   │   └── config_loader.py   # 配置加載器
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── document_loader.py # 文檔加載器
│   │   ├── llamaparse_loader.py # LlamaParse 加載器
│   │   └── imagebind_loader.py  # ImageBind 圖片處理器
│   │
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── base_processor.py  # 基礎處理器
│   │   ├── text_processor.py  # 文本處理
│   │   ├── table_processor.py # 表格處理
│   │   └── chart_processor.py # 圖表處理
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── text_embedder.py   # 文本嵌入
│   │   └── multimodal_embedder.py # 多模態嵌入
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py    # 向量存儲
│   │   └── multimodal_retriever.py # 多模態檢索
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py    # RAG主流程
│   │
│   └── utils/
│       ├── __init__.py
│       └── device_utils.py     # 設備管理工具
│
└── tests/
    └── test_setup.py          # 環境設置測試
```

## 安裝步驟

1. **克隆專案**：
```bash
git clone https://github.com/yourusername/ESG-RAG-.git
cd ESG-RAG-/sustainability_rag
```

2. **創建虛擬環境**：
```bash
python -m venv ESG_env
source ESG_env/bin/activate  # macOS
```

3. **安裝依賴**：
```bash
pip install -r requirements.txt
```

4. **配置環境變數**：
創建 `.env` 文件並配置：
```env
OPENAI_API_KEY=your_openai_api_key
KDBAI_ENDPOINT=your_endpoint_url
KDBAI_API_KEY=your_api_key
USE_MPS=True
DEBUG=True
```

## 開發狀態

### 已完成功能
- [x] 基礎專案結構搭建
- [x] 環境配置管理
- [x] MPS 設備支援
- [x] 基本的 Gradio 介面

### 進行中功能
- [ ] LlamaParse 文檔解析
- [ ] ImageBind 圖表分析
- [ ] KDB.AI 向量數據庫整合
- [ ] 多模態檢索實現

## 性能優化建議

1. **MPS 加速**：
   - 確保 `USE_MPS=True`
   - 使用適當的批處理大小
   - 監控內存使用

2. **向量檢索優化**：
   - 調整檢索數量 k 值
   - 優化相似度閾值
   - 使用適當的向量索引類型

## 使用說明

### 運行測試
```bash
python tests/test_setup.py
```

### 啟動 Gradio 界面
```bash
python app.py
```

## 常見問題

1. **PDF解析問題**：
   - 確保 PDF 文件沒有加密
   - 檢查 PDF 文件編碼格式

2. **圖表識別問題**：
   - 確保圖片清晰度
   - 檢查圖片格式支援

3. **MPS 相關問題**：
   - 確認 macOS 版本
   - 檢查 PyTorch 版本兼容性

## Git 提交規範

提交信息格式：
```
<type>: <description>

[optional body]
```

Type 類別：
- feat: 新功能
- fix: 錯誤修復
- docs: 文檔更改
- style: 代碼格式調整
- refactor: 代碼重構
- test: 測試相關
- chore: 構建過程或輔助工具的變動

## 注意事項

1. 確保 `.env` 文件不被提交到版本控制
2. 運行測試前確保所有 API Keys 已正確配置
3. 對於 Apple Silicon Mac，確保 `USE_MPS=True` 以啟用硬體加速

## 授權協議

本專案採用 MIT 授權協議。

## 致謝

感謝以下開源專案的支持：
- OpenAI
- Meta ImageBind
- LlamaParse
- KDB.AI
- PyTorch

---
📝 **注意**：使用本系統時請確保符合相關數據保護法規和企業隱私政策。
```
