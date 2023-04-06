# YOLOv5 🚀 by Ultralytics, GPL-3.0 licenseorg
"""
Experimental modules
"""
import math
import contextlib
import numpy as np
import torch
from pathlib import Path 
import torch.nn as nn
from utils.downloads import attempt_download



import logging

LOGGER = logging.getLogger("yolo_research")
           
def guess_model_task(model):
    from models.yolo import V8_Detect ,V8_Segment,Classify,Pose 
    """
    Guess the task of a PyTorch model from its architecture or configuration.
    Args:
        model (nn.Module) or (dict): PyTorch model or model configuration in YAML format.
    Returns:
        str: Task of the model ('detect', 'segment', 'classify').
    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        # Guess from YAML dictionary
        m = cfg['head'][-1][-2].lower()  # output module name
        if m in ('classify', 'classifier', 'cls', 'fc'):
            return 'classify'
        if m == 'detect'or'v8_detect':
            return 'detect'
        if m == 'segment'or 'v8_segment':
            return 'segment'
        if m == 'pose':
            return 'pose'

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in 'model.args', 'model.model.args', 'model.model.model.args':
            with contextlib.suppress(Exception):
                return eval(x)['task']
        for x in 'model.yaml', 'model.model.yaml', 'model.model.model.yaml':
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, V8_Detect):
                return 'detect'
            elif isinstance(m, V8_Segment):
                return 'segment'
            elif isinstance(m, Classify):
                return 'classify'
            elif isinstance(m, Pose):
                return 'pose'

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if '-seg' in model.stem or 'segment' in model.parts:
            return 'segment'
        elif '-cls' in model.stem or 'classify' in model.parts:
            return 'classify'
        elif '-pose' in model.stem or 'pose' in model.parts:
            return 'pose'
        elif 'detect' in model.parts:
            return 'detect'

    # Unable to determine task from model
    LOGGER.warning("WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
                   "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.")
    return 'detect'  # assume detect

       
class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it
    catches the error, logs a warning message, and attempts to install the missing module via the check_requirements()
    function. After installation, the function again attempts to load the model using torch.load().
    Args:
        weight (str): The file path of the PyTorch model.
    Returns:
        The loaded PyTorch model.
    """
    from utils.general import check_requirements,check_suffix
    check_suffix(file=weight,suffix='.pt')
    print(weight)
    file = attempt_download(weight)  # search online if missing locally
    try:
        return torch.load(file, map_location='cpu'),file  # load
    except ModuleNotFoundError as e:
        if e.name == 'models':  # e.name is missing module name
            LOGGER.warning(f"WARNING ⚠️ {weight} requires {e.name}, which is not in ultralytics requirements."
                           f"\nAutoInstall will run now for {e.name} but this feature will be removed in the future."
                           f"\nRecommend fixes are to train a new model using updated ultraltyics package or to "
                           f"download updated models from https://github.com/ultralytics/assets/releases/tag/v0.0.0")
        #print("install miss module ")
        check_requirements(e.name)  # install missing module ,select no
        return torch.load(file, map_location='cpu'), file  # load



def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Detect, Model,Decoupled_Detect,ASFF_Detect,IDetect,IAuxDetect,IBin,Kpt_Detect,IKeypoint,Segment,V8_Detect,V8_Segment
    from models.common import Conv
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w),map_location='cpu')  # load ckpt
        #args = {**DEFAULT_CFG_DICT, **ckpt['train_args']}  # combine model and default args, preferring model args
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
       # ckpt.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
        ckpt.pt_path = weights  # attach *.pt file path to model
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        model.append(ckpt.fuse().eval() if fuse else ckpt.eval())  # fused or un-fused model in eval mode

    # Compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU,  Model,V8_Detect,V8_Segment):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if isinstance(t, (Detect,Decoupled_Detect,Kpt_Detect,IKeypoint, ASFF_Detect,IDetect,Segment,IBin, IAuxDetect,Segment,IBin, IAuxDetect))  and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is Conv:
            m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model  # return ensemble

def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    # Loads a single model weights
    from models.yolo import V8_Detect,V8_Segment
    from yolo.utils import DEFAULT_CFG_DICT,DEFAULT_CFG_KEYS
    #print("检查加载的权重:",weight)
    ckpt,weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **ckpt['train_args']}  # combine model and default args, preferring model args
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task= guess_model_task(model)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])

    model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU,V8_Detect, V8_Segment):
            m.inplace = inplace  # torch 1.7.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt

def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from yolo.utils import DEFAULT_CFG_DICT,DEFAULT_CFG_KEYS
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt,w =torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt['train_args']}  # combine model and default args, preferring model args
        model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
        model.pt_path = weights  # attach *.pt file path to model
        model.task=guess_model_task(model)
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval())  # model in eval mode

    # Module compatibility updates
    for m in ensemble.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment,V8_Detect,V8_Segment):
            m.inplace = inplace  # torch 1.7.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[torch.argmax(torch.tensor([m.stride.max() for m in ensemble])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in ensemble), f'Models have different class counts: {[m.nc for m in ensemble]}'
    return ensemble