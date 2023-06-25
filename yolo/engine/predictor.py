# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.
Usage - sources:
    $ yolo task=... mode=predict  model=s.pt --source 0                         # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ yolo task=... mode=predict --weights yolov8n.pt          # PyTorch
                                    yolov8n.torchscript        # TorchScript
                                    yolov8n.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                    yolov8n_openvino_model     # OpenVINO
                                    yolov8n.engine             # TensorRT
                                    yolov8n.mlmodel            # CoreML (macOS-only)
                                    yolov8n_saved_model        # TensorFlow SavedModel
                                    yolov8n.pb                 # TensorFlow GraphDef
                                    yolov8n.tflite             # TensorFlow Lite
                                    yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                                    yolov8n_paddle_model       # PaddlePaddle
    """
    
    
import platform
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from models.common import DetectMultiBackend as AutoBackend
from yolo.cfg import get_cfg
from yolo.data import load_inference_source
from yolo.data.augment import LetterBox, classify_transforms
from yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolo.utils.checks import check_imgsz, check_imshow
from yolo.utils.files import increment_path
from utils.torch_utils import select_device, smart_inference_mode

STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.
    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_setup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        annotator (Annotator): Annotator used for prediction.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f"{self.args.mode}"
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.bs = None
        self.imgsz = None
        self.device = None
        self.classes = self.args.classes
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)
        
    def pre_transform(self, im):
        """Pre-tranform input image before inference.
        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
        Return: A list of transformed imgs.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = same_shapes and self.model.pt
        return [LetterBox(self.imgsz, auto=auto, stride=self.model.stride)(image=x) for x in im]
    
    def preprocess(self, im):
        """Prepares input image before inference.
        Args:
            im (torch.Tensor | List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """

        not_tensor = not isinstance(im, torch.Tensor)  
        if not_tensor:    
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)
            
        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img
    
    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()
        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.data_path.stem)
        return log_string
    
    def postprocess(self, preds, img, orig_img):
        """Post-processes predictions for an image and returns them."""
        return preds
    def __call__(self, source=None, model=None, stream=False):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model)
        else:
            return list(self.stream_inference(source, model))  # merge list of Result into one
    def predict_cli(self, source=None, model=None):
        """Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode."""
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass
    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source, imgsz=self.imgsz, vid_stride=self.args.vid_stride)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')
        # Setup model
        if not self.model:
            self.setup_model(model)
            
        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)
        # Check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
            
        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True
            
         # Checks
        if self.source_type.tensor and (self.args.save or self.args.save_txt or self.args.show):
            LOGGER.warning("WARNING ⚠️ 'save', 'save_txt' and 'show' arguments not enabled for torch.Tensor inference.")

        self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
        self.run_callbacks('on_predict_start')
        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path[0]).stem,
                                       mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False
            # Preprocess
            with self.dt[0]:
                im = self.preprocess(im0s)
            # Inference
            with self.dt[1]:
                preds = self.model(im, augment=self.args.augment, visualize=visualize)
            # Postprocess
            with self.dt[2]:
                self.results = self.postprocess(preds, im, im0s)
            self.run_callbacks('on_predict_postprocess_end')
            # Visualize, save, write results
            n = len(im0s)
            for i in range(n):
                self.results[i].speed = {
                    'preprocess': self.dt[0].dt * 1E3 / n,
                    'inference': self.dt[1].dt * 1E3 / n,
                    'postprocess': self.dt[2].dt * 1E3 / n}
                p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                p = Path(p)
                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0))
                if self.args.show and self.plotted_img is not None:
                    self.show(p)
                if self.args.save and self.plotted_img is not None:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))
            self.run_callbacks('on_predict_batch_end')
            yield from self.results
            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f'{s}{self.dt[1].dt * 1E3:.1f}ms')
        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer
        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *im.shape[2:])}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
            
        self.run_callbacks('on_predict_end')

    def setup_model(self, model, verbose=True):
        device = select_device(self.args.device, verbose=verbose)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model,
                                 device=device,
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)
        self.device = device
        self.model.eval()

    def show(self, p):
        im0 = self.annotator.result()
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[4].startswith('image') else 1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):
        im0 = self.annotator.result()
        # save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)
