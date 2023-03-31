from yolo.engine.model import YOLO
from yolo.utils import DEFAULT_CFG
import torch 
# Load a model
#model = YOLO("models\\detect\\v8_cfg\\yolov8n.yaml")  # build a new model from scratch
def train(cfg=DEFAULT_CFG, use_python=True):
    model ='yolov8n.pt'
  # import torch

    #model = YOLO(cfg.model)  # 创建模型实例
    #state_dict = YOLO(cfg.model).state_dict(model)

    print("Total number of layers:", len(state_dict))

    for name, param in state_dict.items():
           print(name, param.shape)

    #names = ckpt['model']
     # 打印字典的键和值
    #for key in names:
      #print(f"v {key}")
    if cfg.pretrained_wts:
       
        print("input_cfg pre_weight:",cfg.pretrained_wts)
       
        #rename_custom_weight(cfg.pretrained_wts, model)
    else:
         cfg.pretrained_wts=model
    data = cfg.data   # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''
    args = dict(model=cfg.pretrained_wts, data=data, device=device)
    if use_python:
        from yolo.engine.model  import YOLO
       
        model = YOLO(cfg.pretrained_wts)  # build a new model from scratch
        model.train(**args)
    else:
        trainer = DetectionTrainer(overrides=args)
        trainer.train()

if __name__ == "__main__":
    train()
