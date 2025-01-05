import os
import gradio as gr
from dotenv import load_dotenv
from src.pipeline.rag_pipeline import MultiModalRAGPipeline
from src.config.config_loader import load_config

# 加載環境變數
load_dotenv()

def initialize_pipeline():
    """初始化RAG Pipeline"""
    config = load_config()
    return MultiModalRAGPipeline(config)

def process_report(file_obj):
    """處理上傳的永續報告"""
    try:
        # 保存上傳的文件
        temp_path = "temp_report.pdf"
        with open(temp_path, "wb") as f:
            f.write(file_obj.read())
        
        # 處理文件
        pipeline = initialize_pipeline()
        results = pipeline.process_report(temp_path)
        
        # 清理臨時文件
        os.remove(temp_path)
        
        return f"報告處理完成! 共提取了 {len(results)} 個ESG指標"
    except Exception as e:
        return f"處理過程中發生錯誤: {str(e)}"

def query_metric(query_text):
    """查詢ESG指標"""
    try:
        pipeline = initialize_pipeline()
        results = pipeline.query_metrics(query_text)
        
        # 格式化輸出
        output = "查詢結果:\n"
        for result in results:
            output += f"\n指標: {result['metric_name']}\n"
            output += f"數值: {result['value']} {result['unit']}\n"
            output += f"來源: {result['source_type']}\n"
            output += f"位置: {result['location']}\n"
            output += "-" * 40 + "\n"
        
        return output
    except Exception as e:
        return f"查詢過程中發生錯誤: {str(e)}"

# 創建Gradio界面
with gr.Blocks(title="永續報告書分析系統") as demo:
    gr.Markdown("# 永續報告書多模態RAG分析系統")
    
    with gr.Tab("報告處理"):
        with gr.Row():
            file_input = gr.File(
                label="上傳永續報告PDF",
                file_types=[".pdf"]
            )
            process_output = gr.Textbox(
                label="處理結果",
                interactive=False
            )
        process_btn = gr.Button("處理報告")
        process_btn.click(
            fn=process_report,
            inputs=[file_input],
            outputs=[process_output]
        )
    
    with gr.Tab("指標查詢"):
        with gr.Row():
            query_input = gr.Textbox(
                label="輸入查詢（例如：員工平均學習時數）",
                placeholder="請輸入要查詢的ESG指標..."
            )
            query_output = gr.Textbox(
                label="查詢結果",
                interactive=False
            )
        query_btn = gr.Button("查詢")
        query_btn.click(
            fn=query_metric,
            inputs=[query_input],
            outputs=[query_output]
        )

if __name__ == "__main__":
    demo.launch(share=True)