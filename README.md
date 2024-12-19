# ESG-RAG-AI轉換ESG永續報告書系統
製作一個可以透過RAG的力量自動生成ESG報告書轉Excel結構化重點生成的系統Application

## 詳細計畫：
# 多模態永續報告 RAG 系統

## 專案簡介

這是一個基於大型語言模型（LLM）的多模態檢索增強生成（RAG）系統，專門用於從企業永續報告中自動提取ESG相關指標數據。該系統能夠處理文本、表格和圖片等多種形式的內容，並通過 RAG 技術提供精確的指標數據提取。

### 主要功能

- 🔍 **多模態內容處理**：支援文本、表格和圖片的解析與數據提取
- 📊 **自動化指標提取**：根據預定義的指標清單，自動從報告中提取對應數值
- 🤖 **智能數據識別**：使用 GPT-4V 進行圖表解析和數據識別
- 📈 **結構化輸出**：將提取的數據整理成標準的 DataFrame 格式

### 技術架構

- 文件處理：Unstructured
- 向量數據庫：Chroma
- 嵌入模型：OpenAI Embeddings
- 語言模型：GPT-4 & GPT-4V
- 開發框架：LangChain

## 系統要求

- Python 3.8+
- OpenAI API Key
- 足夠的磁碟空間用於存儲向量數據庫

## 安裝步驟

1. clone專案：
```bash
git clone https://github.com/yourusername/sustainability-rag.git
cd sustainability-rag
```

2. 創建虛擬環境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. 安裝依賴：
```bash
pip install -r requirements.txt
```

4. 配置環境變數：
```bash
export OPENAI_API_KEY='your_api_key_here'
# or
set OPENAI_API_KEY='your_api_key_here'  # Windows
```

## 使用說明

### 1. 準備輸入文件

- **永續報告PDF文件**：確保PDF文件清晰可讀
- **指標定義Excel文件**：包含要提取的指標清單，格式如下：

```
| 指標名稱 | 指標類型 | 備註 |
|---------|---------|------|
| 員工總數 | 數值    | -    |
| ... | ... | ... |
```

### 2. 運行系統

```python
from src.main import main

# 運行系統
results_df = main(
    pdf_path="path/to/sustainability_report.pdf",
    excel_path="path/to/indicators.xlsx"
)

# 保存結果
results_df.to_csv('output.csv', index=False)
```

### 3. 輸出格式

系統輸出為 DataFrame 格式，包含以下列：
- 指標名稱
- 提取數值
- 數據來源（文本/表格/圖片）
- 提取置信度

## 專案結構

```
sustainability_rag/
│
├── src/                   # 源代碼
│   ├── config/           # 配置文件
│   ├── data/             # 數據處理
│   ├── extractors/       # 提取器
│   ├── embeddings/       # 嵌入模型
│   ├── retrieval/        # 檢索系統
│   ├── models/           # 模型封装
│   └── pipeline/         # RAG流程
│
├── tests/                # 測試文件
├── notebooks/            # 示例筆記本
├── requirements.txt      # 依賴清單
└── README.md            # 說明文檔
```

## 配置說明

在 `src/config/config.py` 中可以調整以下配置：

```python
class RAGConfig(BaseModel):
    chunk_size: int = 1000          # 文本分塊大小
    chunk_overlap: int = 200        # 分塊重疊長度
    text_embedding_model: str = "text-embedding-ada-002"
    vision_model: str = "gpt-4-vision-preview"
    ...
```

## 性能優化建議

1. **文本分塊**：
   - 根據文檔特點調整 chunk_size
   - 適當增加重疊區域以提高準確性

2. **向量檢索**：
   - 調整檢索數量 k 值
   - 優化相似度閾值

3. **模型選擇**：
   - 可根據需求選擇不同的嵌入模型
   - 平衡成本和效能

## 常見問題

1. **PDF解析失敗**
   - 確保PDF文件沒有加密
   - 檢查PDF文件編碼格式

2. **數據提取不準確**
   - 調整文本分塊大小
   - 優化提示詞模板
   - 增加檢索文檔數量

3. **圖片解析問題**
   - 確保圖片清晰度
   - 檢查圖片格式支援

## 開發計劃

- [ ] 添加批量處理功能
- [ ] 優化圖表識別準確率
- [ ] 添加 API 接口
- [ ] 實現並行處理
- [ ] 添加數據驗證功能

## 貢獻指南

1. Fork 專案
2. 創建功能分支
3. 提交更改
4. 發起 Pull Request

## 授權協議

本專案採用 MIT 授權協議。

## 聯絡方式

- 作者：
- Email：
- GitHub：

## 致謝

感謝以下開源專案的支持：
- LangChain
- Unstructured
- ChromaDB, KDB.ai
- OpenAI

---

📝 **注意**：使用本系統時請確保符合相關數據保護法規和企業隱私政策。
