# # app.py

# import os
# import gradio as gr
# import pandas as pd
# from pathlib import Path
# from datetime import datetime
# import logging
# from typing import List, Dict, Any
# from dotenv import load_dotenv
# import plotly.graph_objects as go

# # LlamaIndex 相關引入
# # from llama_index.indices import VectorStoreIndex
# from llama_index.core import Document, Settings
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_parse import LlamaParse
# import yaml

# # 載入環境變數
# load_dotenv()

# # 設置日誌
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # CSS 樣式
# custom_css = """
# .container {
#     margin: 0 auto;
#     padding: 2rem;
#     max-width: 1200px;
# }
# .stat-box {
#     padding: 1rem;
#     border-radius: 8px;
#     background-color: #f8f9fa;
#     margin: 0.5rem 0;
#     border: 1px solid #dee2e6;
# }
# .title-box {
#     text-align: center;
#     margin-bottom: 2rem;
#     padding: 2rem;
#     background-color: #f8f9fa;
#     border-radius: 10px;
# }
# .footer-box {
#     text-align: center;
#     margin-top: 2rem;
#     padding: 1rem;
#     background-color: #f8f9fa;
#     border-radius: 8px;
# }
# """

# async def process_report(
#     file: gr.File,
#     progress: gr.Progress
# ) -> tuple[str, pd.DataFrame, str, go.Figure]:
#     """處理上傳的永續報告"""
#     try:
#         progress(0, desc="開始處理文件...")
        
#         # 建立示例數據（後續替換為實際的處理邏輯）
#         example_data = {
#             "項目": ["員工培訓時數", "環境支出", "董事會diversity"],
#             "數據": ["85.4小時", "1000萬", "35%"],
#             "資料來源": ["第10頁", "第15頁", "第20頁"],
#             "指標類別": ["社會", "環境", "治理"],
#             "代碼": ["S1", "E1", "G1"],
#             "信心分數": [0.95, 0.88, 0.75]
#         }
        
#         results_df = pd.DataFrame(example_data)
        
#         progress(0.5, desc="生成分析報告...")
        
#         # 計算統計數據
#         high_confidence = len(results_df[results_df['信心分數'] >= 0.8])
#         mid_confidence = len(results_df[results_df['信心分數'].between(0.5, 0.8)])
#         low_confidence = len(results_df[results_df['信心分數'] < 0.5])
        
#         # 生成圓餅圖
#         fig = go.Figure(data=[go.Pie(
#             labels=['高信心指標', '中信心指標', '低信心指標'],
#             values=[high_confidence, mid_confidence, low_confidence],
#             hole=.3
#         )])
#         fig.update_layout(
#             title="指標信心分布",
#             height=400
#         )
        
#         # 生成報告
#         report = f"""### 📊 分析報告

# **文件資訊**
# - 檔名：{file.name}
# - 大小：{file.size / 1024 / 1024:.2f} MB

# **提取結果**
# - 找到指標數：{len(results_df)}
# - 高信心指標：{high_confidence}
# - 中信心指標：{mid_confidence}
# - 低信心指標：{low_confidence}

# 處理完成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
#         progress(1.0, desc="處理完成！")
        
#         return "處理成功", results_df, report, fig
        
#     except Exception as e:
#         error_msg = f"處理過程中發生錯誤: {str(e)}"
#         logger.error(error_msg, exc_info=True)
#         return error_msg, pd.DataFrame(), error_msg, go.Figure()

# # 創建 Gradio 介面
# with gr.Blocks(title="ESG 永續報告書分析系統", css=custom_css) as app:
#     # 標題區域
#     with gr.Column(elem_classes="title-box"):
#         gr.Markdown("""# 🌱 ESG 永續報告書分析系統
#         ### 智能化 ESG 指標擷取與分析平台""")
    
#     # 主要內容區域
#     with gr.Row():
#         # 左側：控制面板
#         with gr.Column(scale=1):
#             with gr.Column(elem_classes="stat-box"):
#                 gr.Markdown("### 📤 上傳文件")
#                 file_input = gr.File(
#                     label="支援格式：PDF、Word、Excel",
#                     file_types=[".pdf", ".docx", ".xlsx"],
#                     type="binary"
#                 )
#                 process_btn = gr.Button(
#                     "開始分析",
#                     variant="primary",
#                     size="lg"
#                 )
            
#             with gr.Column(elem_classes="stat-box"):
#                 gr.Markdown("### 📋 處理狀態")
#                 status_text = gr.Textbox(
#                     label="當前狀態",
#                     interactive=False,
#                     show_label=False
#                 )
        
#         # 右側：結果顯示
#         with gr.Column(scale=2):
#             with gr.Tabs():
#                 with gr.TabItem("分析結果"):
#                     with gr.Column(elem_classes="stat-box"):
#                         gr.Markdown("### 📊 指標數據")
#                         results_df = gr.DataFrame(
#                             headers=[
#                                 "項目",
#                                 "數據",
#                                 "資料來源",
#                                 "指標類別",
#                                 "代碼",
#                                 "信心分數"
#                             ]
#                         )
                
#                 with gr.TabItem("統計報告"):
#                     with gr.Row():
#                         with gr.Column():
#                             report_text = gr.Markdown()
#                         with gr.Column():
#                             plot = gr.Plot()
    
#     # 頁腳
#     with gr.Column(elem_classes="footer-box"):
#         gr.Markdown("""### 系統資訊
#         - 版本：1.0.0
#         - 更新時間：2024-01-05
#         - 技術支援：support@example.com""")
    
#     # 設置事件處理
#     process_btn.click(
#         fn=process_report,
#         inputs=[file_input],
#         outputs=[status_text, results_df, report_text, plot]
#     )

# if __name__ == "__main__":
#     # 確保必要的目錄存在
#     Path("uploads").mkdir(exist_ok=True)
#     Path("outputs").mkdir(exist_ok=True)
    
#     # 啟動 Gradio 介面
#     app.launch(
#         server_name="0.0.0.0",
#         server_port=int(os.getenv("PORT", 7860)),
#         show_error=True
#     )
    
# app.py

import os
import gradio as gr
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
import plotly.graph_objects as go

# 導入我們的處理器
from src.extractors.text_extractor import DocumentProcessor
from src.extractors.metric_extractor import ESGMetricExtractor
from src.config.config_loader import load_config

# 載入環境變數和配置
load_dotenv()
config = load_config()

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# [保留原有的 CSS 樣式定義...]

async def process_report(
    file: gr.File,
    progress: gr.Progress
) -> tuple[str, pd.DataFrame, str, go.Figure]:
    """處理上傳的永續報告"""
    try:
        progress(0, desc="初始化處理器...")
        
        # 初始化處理器
        doc_processor = DocumentProcessor(config)
        metric_extractor = ESGMetricExtractor(config)
        
        progress(0.2, desc="解析文件...")
        
        # 處理文檔
        file_path = Path(file.name)
        chunks, metadata = doc_processor.process_document(file_path)
        
        progress(0.4, desc="提取指標...")
        
        # 提取 ESG 指標
        results_df = await metric_extractor.process_document(
            chunks=chunks,
            document_id=file.name,
            metadata=metadata
        )
        
        progress(0.6, desc="生成分析報告...")
        
        # 計算統計數據
        high_confidence = len(results_df[results_df['信心分數'] >= 0.8])
        mid_confidence = len(results_df[results_df['信心分數'].between(0.5, 0.8)])
        low_confidence = len(results_df[results_df['信心分數'] < 0.5])
        
        # 生成圓餅圖
        fig = go.Figure(data=[go.Pie(
            labels=['高信心指標', '中信心指標', '低信心指標'],
            values=[high_confidence, mid_confidence, low_confidence],
            hole=.3
        )])
        fig.update_layout(
            title="指標信心分布",
            height=400
        )
        
        # 生成報告
        report = f"""### 📊 分析報告

**文件資訊**
- 檔名：{file.name}
- 大小：{file.size / 1024 / 1024:.2f} MB
- 處理的文本塊數：{metadata['total_chunks']}
- 識別的表格數：{metadata['tables_found']}

**提取結果**
- 找到指標數：{len(results_df)}
- 高信心指標：{high_confidence}
- 中信心指標：{mid_confidence}
- 低信心指標：{low_confidence}

**指標分類統計**
- 環境指標：{len(results_df[results_df['指標類別'] == '環境'])}
- 社會指標：{len(results_df[results_df['指標類別'] == '社會'])}
- 治理指標：{len(results_df[results_df['指標類別'] == '治理'])}

處理完成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        progress(0.8, desc="清理資源...")
        
        # 清理資源
        try:
            metric_extractor.cleanup()
        except Exception as e:
            logger.warning(f"清理資源時出錯: {e}")
            
        progress(1.0, desc="處理完成！")
        
        return "處理成功", results_df, report, fig
        
    except Exception as e:
        error_msg = f"處理過程中發生錯誤: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, pd.DataFrame(), error_msg, go.Figure()

# [保留原有的 Gradio 介面定義...]

if __name__ == "__main__":
    # 確保必要的目錄存在
    Path("uploads").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    
    # 確保配置文件存在
    if not Path("src/config/config.yaml").exists():
        logger.error("找不到配置文件！")
        exit(1)
        
    # 啟動 Gradio 介面
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_error=True
    )    