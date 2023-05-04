# Ultralytics YOLO 🚀, GPL-3.0 license

import torch

from yolo.data import ClassificationDataset, build_dataloader
from yolo.engine.validator import BaseValidator
from yolo.utils import DEFAULT_CFG
from yolo.utils.metrics import ClassifyMetrics
from yolo.utils.plotting import plot_images


class ClassificationValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, logger=None, args=None):
        super().__init__(dataloader, save_dir, pbar, logger, args)
        self.args.task = 'classify'
        self.metrics = ClassifyMetrics()

    def get_desc(self):
        return ('%22s' + '%11s' * 2) % ('classes', 'top1_acc', 'top5_acc')

    def init_metrics(self, model):
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        self.pred.append(preds.argsort(1, descending=True)[:, :5])
        self.targets.append(batch["cls"])

    def get_stats(self):
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict
    
    def build_dataset(self, img_path):
        dataset = ClassificationDataset(root=img_path, imgsz=self.args.imgsz, augment=False)
        return dataset
    
    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        pf = '%22s' + '%11.3g' * len(self.metrics.keys)  # print format
        self.logger.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(images=batch['img'],
                    batch_idx=torch.arange(len(batch['img'])),
                    cls=batch['cls'].squeeze(-1),
                    fname=self.save_dir / f'val_batch{ni}_labels.jpg',
                    names=self.names)

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(batch['img'],
                    batch_idx=torch.arange(len(batch['img'])),
                    cls=torch.argmax(preds, dim=1),
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names)  # pred
        
def val(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or "yolov8n-cls.pt"  # or "resnet18"
    data = cfg.data or "mnist160"

    args = dict(model=model, data=data)
    if use_python:
        from yolo.engine.model import YOLO
        YOLO(model).val(**args)
    else:
        validator = ClassificationValidator(args=args)
        validator(model=args['model'])


if __name__ == "__main__":
    val()
