# tests/test_setup.py

import os
import sys
import torch
from dotenv import load_dotenv
import openai
import kdbai_client as kdbai

def test_environment():
    """測試環境設置"""
    results = {
        "成功": [],
        "失敗": []
    }

    # 1. 測試 .env 載入
    try:
        load_dotenv()
        results["成功"].append("✅ .env 文件載入成功")
    except Exception as e:
        results["失敗"].append(f"❌ .env 文件載入失敗: {str(e)}")

    # 2. 測試 OpenAI API
    try:
        openai.api_key = os.getenv('OPENAI_API_KEY')
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=5
        )
        results["成功"].append("✅ OpenAI API 連接成功")
    except Exception as e:
        results["失敗"].append(f"❌ OpenAI API 連接失敗: {str(e)}")

    # 3. 測試 KDB.AI 連接
    try:
        KDBAI_ENDPOINT = os.getenv('KDBAI_ENDPOINT')
        KDBAI_API_KEY = os.getenv('KDBAI_API_KEY')
        session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)
        session.databases()  # 測試連接
        results["成功"].append("✅ KDB.AI 連接成功")
    except Exception as e:
        results["失敗"].append(f"❌ KDB.AI 連接失敗: {str(e)}")

    # 4. 測試 PyTorch MPS 可用性
    try:
        if torch.backends.mps.is_available():
            results["成功"].append("✅ MPS 可用")
            # 測試簡單的 tensor 運算
            tensor = torch.randn(3, 3).to('mps')
            tensor = tensor + tensor
            results["成功"].append("✅ MPS tensor 運算成功")
        else:
            results["失敗"].append("❌ MPS 不可用")
    except Exception as e:
        results["失敗"].append(f"❌ MPS 測試失敗: {str(e)}")

    # 5. 測試必要的目錄結構
    required_dirs = [
        'src/config',
        'src/data',
        'src/processors',
        'src/embeddings',
        'src/retrieval',
        'src/pipeline'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            results["成功"].append(f"✅ 目錄存在: {dir_path}")
        else:
            results["失敗"].append(f"❌ 目錄不存在: {dir_path}")

    return results

if __name__ == "__main__":
    print("開始測試環境設置...")
    results = test_environment()
    
    print("\n=== 測試結果 ===")
    print("\n✅ 成功項目:")
    for success in results["成功"]:
        print(success)
    
    print("\n❌ 失敗項目:")
    for failure in results["失敗"]:
        print(failure)

    if not results["失敗"]:
        print("\n🎉 所有測試都通過了！")
    else:
        print(f"\n⚠️ 有 {len(results['失敗'])} 項測試失敗")