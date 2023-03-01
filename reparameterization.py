#!/usr/bin/env python
# coding: utf-8

# # Reparameterization

# ## YOLOv7 reparameterization
"""                   
    python reparameterization.py  --cfg models/v7_cfg/deploy/yolov7.yaml  --weights yolov7_training.pt  --device 1  --name yolov7
"""
# In[ ]:


# import
import argparse
from copy import deepcopy
from models.yolo import Model
import torch
import sys
from utils.general import print_args
from utils.torch_utils import select_device, is_parallel
#config your class numbers and anchorss



def run(
        weights= 'yolov7_training.pt',  # weights path
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        name='yolov7',
        cfg="",
        save_file="models/v7_cfg/deploy/" ,
):
    
    #nc=80
    anchors=3
    device = select_device(device, batch_size=1)
    # model trained by cfg/training/*.yaml
    ckpt = torch.load(weights[0], map_location=device)
    # reparameterized model in cfg/deploy/*.yaml
    model = Model(cfg, ch=3, nc=80).to(device)
    #print(model)

    # copy intersect weights
    state_dict = ckpt['model'].float().state_dict()
    exclude = []
    intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(intersect_state_dict, strict=False)
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc

    if name=='yolov7':
        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
        
            model.state_dict()['model.105.m.0.weight'].data[i, :, :, :] *= state_dict['model.105.im.0.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.105.m.1.weight'].data[i, :, :, :] *= state_dict['model.105.im.1.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.105.m.2.weight'].data[i, :, :, :] *= state_dict['model.105.im.2.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.105.m.0.bias'].data += state_dict['model.105.m.0.weight'].mul(state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
        model.state_dict()['model.105.m.1.bias'].data += state_dict['model.105.m.1.weight'].mul(state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
        model.state_dict()['model.105.m.2.bias'].data += state_dict['model.105.m.2.weight'].mul(state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
        model.state_dict()['model.105.m.0.bias'].data *= state_dict['model.105.im.0.implicit'].data.squeeze()
        model.state_dict()['model.105.m.1.bias'].data *= state_dict['model.105.im.1.implicit'].data.squeeze()
        model.state_dict()['model.105.m.2.bias'].data *= state_dict['model.105.im.2.implicit'].data.squeeze()

    elif name=='yolov7x':
        # ## YOLOv7x reparameterization


        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.121.m.0.weight'].data[i, :, :, :] *= state_dict['model.121.im.0.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.121.m.1.weight'].data[i, :, :, :] *= state_dict['model.121.im.1.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.121.m.2.weight'].data[i, :, :, :] *= state_dict['model.121.im.2.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.121.m.0.bias'].data += state_dict['model.121.m.0.weight'].mul(state_dict['model.121.ia.0.implicit']).sum(1).squeeze()
        model.state_dict()['model.121.m.1.bias'].data += state_dict['model.121.m.1.weight'].mul(state_dict['model.121.ia.1.implicit']).sum(1).squeeze()
        model.state_dict()['model.121.m.2.bias'].data += state_dict['model.121.m.2.weight'].mul(state_dict['model.121.ia.2.implicit']).sum(1).squeeze()
        model.state_dict()['model.121.m.0.bias'].data *= state_dict['model.121.im.0.implicit'].data.squeeze()
        model.state_dict()['model.121.m.1.bias'].data *= state_dict['model.121.im.1.implicit'].data.squeeze()
        model.state_dict()['model.121.m.2.bias'].data *= state_dict['model.121.im.2.implicit'].data.squeeze()

    elif name=='yolov7-w6':
    # ## YOLOv7-W6 reparameterization
   
        idx = 118
        idx2 = 122

        # copy weights of lead head
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data -= model.state_dict()['model.{}.m.0.weight'.format(idx)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data -= model.state_dict()['model.{}.m.1.weight'.format(idx)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data -= model.state_dict()['model.{}.m.2.weight'.format(idx)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data -= model.state_dict()['model.{}.m.3.weight'.format(idx)].data
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data -= model.state_dict()['model.{}.m.0.bias'.format(idx)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data -= model.state_dict()['model.{}.m.1.bias'.format(idx)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data -= model.state_dict()['model.{}.m.2.bias'.format(idx)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data -= model.state_dict()['model.{}.m.3.bias'.format(idx)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.bias'.format(idx2)].data

    # reparametrized YOLOR
        for i in range(255):
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()

    elif name=='yolov7-e6':

        # ## YOLOv7-E6 reparameterization

        # In[ ]:

        idx = 140
        idx2 = 144

        # copy weights of lead head
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data -= model.state_dict()['model.{}.m.0.weight'.format(idx)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data -= model.state_dict()['model.{}.m.1.weight'.format(idx)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data -= model.state_dict()['model.{}.m.2.weight'.format(idx)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data -= model.state_dict()['model.{}.m.3.weight'.format(idx)].data
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data -= model.state_dict()['model.{}.m.0.bias'.format(idx)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data -= model.state_dict()['model.{}.m.1.bias'.format(idx)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data -= model.state_dict()['model.{}.m.2.bias'.format(idx)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data -= model.state_dict()['model.{}.m.3.bias'.format(idx)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.bias'.format(idx2)].data

        # reparametrized YOLOR
        for i in range(255):
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()

   
    elif name=='yolov7-d6':
        # ## YOLOv7-D6 reparameterization

        
        idx = 162
        idx2 = 166

        # copy weights of lead head
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data -= model.state_dict()['model.{}.m.0.weight'.format(idx)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data -= model.state_dict()['model.{}.m.1.weight'.format(idx)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data -= model.state_dict()['model.{}.m.2.weight'.format(idx)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data -= model.state_dict()['model.{}.m.3.weight'.format(idx)].data
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data -= model.state_dict()['model.{}.m.0.bias'.format(idx)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data -= model.state_dict()['model.{}.m.1.bias'.format(idx)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data -= model.state_dict()['model.{}.m.2.bias'.format(idx)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data -= model.state_dict()['model.{}.m.3.bias'.format(idx)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.bias'.format(idx2)].data

        # reparametrized YOLOR
        for i in range(255):
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()

   
    elif name=='yolov7-e6e':
        # ## YOLOv7-E6E reparameterization

        

        idx = 261
        idx2 = 265

        # copy weights of lead head
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data -= model.state_dict()['model.{}.m.0.weight'.format(idx)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data -= model.state_dict()['model.{}.m.1.weight'.format(idx)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data -= model.state_dict()['model.{}.m.2.weight'.format(idx)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data -= model.state_dict()['model.{}.m.3.weight'.format(idx)].data
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.3.weight'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data -= model.state_dict()['model.{}.m.0.bias'.format(idx)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data -= model.state_dict()['model.{}.m.1.bias'.format(idx)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data -= model.state_dict()['model.{}.m.2.bias'.format(idx)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data -= model.state_dict()['model.{}.m.3.bias'.format(idx)].data
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.bias'.format(idx2)].data
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.bias'.format(idx2)].data

        # reparametrized YOLOR
        for i in range(255):
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()
    else :
        print('check models name! ')

   
    # model to be saved
    ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'optimizer': None,
            'training_results': None,
            'epoch': -1}

    # save reparameterized model
    torch.save(ckpt, save_file+"/"+name+".pt")




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model path(s)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--name', type=str, help='model name ')
    parser.add_argument('--save_file', default='models/v7_cfg/deploy', help='save results path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    print('done')
    main(opt)



