# src/config/config_loader.py

import torch
import platform
import os
from pathlib import Path
import yaml

def _determine_device():
    """確定要使用的計算設備，優先使用 MPS"""
    if torch.backends.mps.is_available() and os.getenv('USE_MPS', 'True').lower() == 'true':
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def _create_default_config():
    """創建默認配置"""
    device = _determine_device()
    print(f"Using device: {device}")  # 方便調試
    
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