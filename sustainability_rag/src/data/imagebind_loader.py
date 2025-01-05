# src/data/imagebind_loader.py

import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from src.utils.device_utils import DeviceManager
from PIL import Image
import numpy as np

class ImageBindLoader:
    def __init__(self, config):
        self.config = config
        self.device = config['models']['device']
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model = DeviceManager.to_device(self.model, self.device)
        
    def process_image(self, image):
        image = DeviceManager.to_device(image, self.device)
        with torch.no_grad():
            # 如果使用 MPS，某些操作可能需要特別處理
            if self.device == 'mps':
                # 某些操作可能需要臨時移到 CPU
                result = self.model(image.to('cpu')).to(self.device)
            else:
                result = self.model(image)
        return result

    def load_and_process_image(self, image_path):
        """從文件載入並處理圖片"""
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(
                [image_path], 
                self.device
            )
        }
        return self.process_image(inputs)

    def process_chart(self, image):
        """特別處理圖表"""
        embeddings = self.process_image(image)
        # 將嵌入轉換為numpy數組以便存儲
        return embeddings.cpu().numpy() if self.device != 'cpu' else embeddings.numpy()