import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = '/root/workspace/d4a82a7hri0c73cmluh0/HEFT-DETR-main/ultralytics/cfg/models/rt-detr/heft-detr.yaml''
    model = RTDETR('weights/best.pt') #
    result = model.val(data='/root/workspace/d4a82a7hri0c73cmluh0/HEFT-DETR-main/DatasetA/data.yaml',
                      split='test', 
                      imgsz=640,
                      batch=8,
                      save_json=True, 
                      project='runs/test',
                      name='exp_my',
                      )
    