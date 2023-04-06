from yolo.engine.model import YOLO
from yolo.utils import DEFAULT_CFG
#from yolo.v8.train
import torch 
# Load a model
#model = YOLO("models\\detect\\v8_cfg\\yolov8n.yaml")  # build a new model from scratch
def train(cfg=DEFAULT_CFG, use_python=True):
    
    data = cfg.data   # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''
    args = dict(model=cfg.pretrained_wts, data=data, device=device)
    if use_python:
        from yolo.engine.model  import YOLO
        model=YOLO(cfg.model)  
        model.train(**args)
  

if __name__ == "__main__":
    train()
