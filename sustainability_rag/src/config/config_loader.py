# src/config/config_loader.py

import torch
import platform
import os
from pathlib import Path
import yaml
from typing import Dict, Optional

def determine_device() -> str:
    """確定要使用的計算設備，優先使用 MPS"""
    if torch.backends.mps.is_available() and os.getenv('USE_MPS', 'True').lower() == 'true':
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def load_yaml_config(config_path: Optional[Path] = None) -> Dict:
    """載入 YAML 配置檔案"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        return {}

def create_default_config() -> Dict:
    """創建默認配置"""
    device = determine_device()
    print(f"Using device: {device}") # 方便調試
    
    config = {
        'vector_store': {
            'dimensions': 1024,
            'metric': 'cosine',
            'index_type': 'flat'
        },
        'models': {
            'device': device,
            'imagebind': {
                'device': device,
                'half_precision': True,  # 使用 float16 以提高性能
                'use_mps': device == 'mps'
            },
            'llamaparse': {
                'parsing_instructions': """
                這是一份永續報告書，包含多個ESG指標。
                請特別注意：
                1. 數值型指標及其單位
                2. 表格中的數據
                3. 圖表展示的趨勢
                4. 各指標的所屬章節
                """
            },
            'embedding': {
                'model_name': 'text-embedding-3-small',
                'api_key': os.getenv('OPENAI_API_KEY')
            }
        },
        'kdbai': {
            'endpoint': os.getenv('KDBAI_ENDPOINT'),
            'api_key': os.getenv('KDBAI_API_KEY')
        },
        'optimization': {
            'use_mps': device == 'mps',
            'batch_size': 32,
            'half_precision': True
        }
    }
    
    return config

def load_config(config_path: Optional[Path] = None) -> Dict:
    """載入完整配置，合併默認配置和 YAML 配置"""
    # 首先載入默認配置
    config = create_default_config()
    
    # 然後載入 YAML 配置並合併
    yaml_config = load_yaml_config(config_path)
    
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    # 合併配置
    config = deep_update(config, yaml_config)
    
    return config

def load_metrics_schema(schema_path: Optional[Path] = None) -> Dict:
    """載入 ESG 指標 schema"""
    if schema_path is None:
        schema_path = Path(__file__).parent / "metrics_schema.yaml"
    
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load metrics schema: {e}")
        return {}