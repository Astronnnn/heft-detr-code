import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import RTDETR

if __name__ == '__main__':
    # choose your yaml file
    model = RTDETR('/root/workspace/d4a82a7hri0c73cmluh0/HEFT-DETR-main/ultralytics/cfg/models/rt-detr/heft-detr.yaml')
    model.model.eval()
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640]) 
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()