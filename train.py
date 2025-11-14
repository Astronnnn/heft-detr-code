import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"    
warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('/root/workspace/d4a82a7hri0c73cmluh0/HEFT-DETR-main/ultralytics/cfg/models/rt-detr/heft-detr.yaml')
    model.train(data=
                '/root/workspace/d4a82a7hri0c73cmluh0/HEFT-DETR-main/DatasetA/data.yaml',
                # '/root/workspace/d4a82a7hri0c73cmluh0/HEFT-DETR-main/DatasetB/data.yaml',
                # '/root/workspace/d4a82a7hri0c73cmluh0/HEFT-DETR-main/dataset_spilt_NEU-DET/data.yaml',
                # '/root/workspace/d4a82a7hri0c73cmluh0/HEFT-DETR-main/dataset_visdrone/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8, 
                workers=4, 
                # resume='', # last.pt path
                project='/root/workspace/d4a82a7hri0c73cmluh0/HEFT-DETR-main/runs/train',
                name='exp_heft-detr',
                )