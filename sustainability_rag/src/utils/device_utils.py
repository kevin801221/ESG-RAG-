# src/utils/device_utils.py

import torch

class DeviceManager:
    @staticmethod
    def get_device(tensor_or_module):
        """根據模型或張量的類型獲取合適的設備"""
        config_device = _determine_device()
        
        if hasattr(tensor_or_module, 'is_mps_available') and config_device == 'mps':
            return 'mps'
        elif hasattr(tensor_or_module, 'is_cuda_available') and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    @staticmethod
    def to_device(tensor_or_module, device=None):
        """將張量或模型移至合適的設備"""
        if device is None:
            device = DeviceManager.get_device(tensor_or_module)
            
        try:
            return tensor_or_module.to(device)
        except RuntimeError:
            print(f"Warning: Failed to move to {device}, falling back to CPU")
            return tensor_or_module.to('cpu')