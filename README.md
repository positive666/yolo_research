
<div align='left'>
  <img src=https://img.shields.io/github/stars/positive666/yolo_research.svg?style=social >
  <img src=https://img.shields.io/github/forks/positive666/yolo_research.svg?style=social >
  <img src=https://img.shields.io/github/watchers/positive666/yolo_research.svg?style=social>
</div> 


<!-- <div align="center">
<p>
   <a align="left" href="https://github.com/positive666/yolo_research" target="_blank">
   <img src="./data/images/yolov.jpg" width="87%"></a>
</p>
</div> -->



##  <div align="left">🚀 yolo_reserach PLUS High-level</div>

🚀🚀🚀招募：寻热爱算法研发的小伙伴共同开发和维护开源项目，添加IDEA，甚至多模态领域应用也可，有成果不建议自己带走，工程上我来维护，需要一定的算法研发背景和能力，联系可首页邮箱。
更新升级中.. Add latest V8 core yolov8解析：https://blog.csdn.net/weixin_44119362/article/details/129417459 ,（工作繁忙，会不断更新优化，有问题挂issue）

### <div align="left">⭐新闻板块【实时更新版块】</div>

	- 2023/4/6  从2021年到2023年，即使不做检测，依然坚持更新，搬砖不易，后续会有更多更新,但是目前先集成稳定各个功能为主：更新v8的pose模块，支持v8代码训练自定义的网络结构并加载权重训练；之前的问题作个简单解释：就是在安装源码环境后其实只是支持你解析官方的预训练权重，如果你用本项目的代码训练后,可以卸载掉源码环境，继续更新中
	- 2023/3/28 同步兼容最新的V8代码更新：目前V8依赖于pip install ultralytics,我在代码更新中也发现了该问题，虽然本项目做了分离，但是使用官方权重作为预训练权重去加载的前提下：仍然需要依赖中的ultralytics.nn文件夹，不然可能会报错，因为是这样的本项目改了模型层的参数名字，因为V8每层的名字是带“ultralytics.nn ”的，如果不安装这个部分代码，你torch打不开V8官方的权重，故目前两种解决办法：1.scratch 2.pip安装后打开将权重名字重构 3.代码目录修改 后续我会优化解决，不过目前项目中的工作太多了，故如果出现报错还是使用临时解决方案：pip install ultralytics,这样比较简单直接兼容，然后可以运行　python train_v8.py ，未解决的就是如果自定义机构可能无法直接加载官方的权重，汇后续解决！  
	- 2023/3/1  add v8 core:春节期间看了下V8，由于近半年项目比较多也是耽误了好久(原版本是将V8的所有功能全部融合到了V5的代码中，和V8命令一样，但是训练的时候发生了问题，排查发现问题发生在V5的数据读取处理，所以暂时使用V8的训练结构代码，也便于区分)，然后抓紧时间不停更新；
	- 2022/11/23 修复已知BUG，V7.0版本更新兼容，年底比较忙后续忙完业务会大更新
	- 2022/10/20 修复适配V7结构和额外任务引起的一些代码问题，实时更新V5的代码优化部分，添加了工具grad_cam在tools目录。
	- 2022/9/19 修复已知BUG，更新了实时的V5BUG修复和代码优化融合验证，核心检测、分类、分割的部分CI验证，关键点检测实测训练正常，基本功能整理完毕。
	- 2022/9/15 分类、检测、分割、关键点检测基本整合完毕，工程结构精简化中，关键点检测训练正常已经验证，分割待调试，火速迭代中
	- 分割代码结合V5和V7的代码进行了合并DEBUG调试，训练部分待验证，另外注意力层训练过程中，没法收敛或者NAN的情况，排除代码问题，需要在超参数YAML里，先对学习衰减率从改为0.2 ，比如GAM的注意力头部问题，训练周期加到400epoch+ 就可以。
	- 去年的decoupled结构虽然能提点，不过FLOPS增加的太多，目前用V5作者分支的解耦头替换，效果待验证。
	- 融合了代码做了部分的优化，这里看了下V7的代码优化较差，后续会集成精简版本的分类、分割、检测、POSE检测的结构，目前已经完成了一部分工作，更新频繁有问题欢迎反馈和提供实验结果。
<p>

关于这个项目的使用说明详请可参考下面博客：
[csdn持续更新2021-2022年](https://blog.csdn.net/weixin_44119362/article/details/126895964?spm=1001.2014.3001.5501) ,[csdn持续更新2023年](https://blog.csdn.net/weixin_44119362/article/details/129417459?spm=1001.2014.3001.5502)

<details >
<summary>当前 Project 结构说明</summary>


```
yolo_research
│   pose  
│   └─────   ## 关键点检测任务使用
│   ...    
│   models   ## 存储模型：算子定义和所有模型的yaml结构定义，包含yolov5\yolov7\yolov8  

    └─────   common.py   模型算子定义
             yolo.py     模型结构定义
│   └─────   cls         分类模型结构
│            pose        关键点模型结构
│            segment     分割模型结构
│            detect  v5u_cfg/v7_cfg/v8_cfg    检测模型结构..其余是V5版本以及一些改的参考示例      
│   ....
│   segment
│   └─────   ## 分割任务
|   classify
│   └─────   ## 分类任务
|   tracker
│   └─────   ## 跟踪任务 Fork V8
│   utils
│   └─────   #通用部分代码
|          .
|          .
|            segment   ##分割的数据处理操作部分
|   yolo
│   └─────   v8        ## yolov8 core ,主要包含训练部分和推理使用部分的相关代码
│             └───── .
|            cfg       ## default.yaml 设置所有V8相关参数
|            engine    ## 定义基类结构
|            utils
|            data
|               .
|               .
|       .
|       .    ##其余为检测核心代码和通用部分
```

</details>

# Feature 🚀 
    
     - 最新的yoloV5工程风格代码融合，支持自由定义搭配所有组件，加入V8部分,兼容了anchor-free的yolov8，针对High-level任务：完成先进的检测器、跟踪器、分类器、分割、关键点检测功能任务集成，逐步删除额外库依赖
     
     - 实时的v5代码更新改动&&v7等work的结构适配（每周同步yolov5的代码优化）
     
     - 早期集成的attention、self-attention维护和调试
     
     - 额外的网络结构和Tricks补充
    	
     - deepstream部署工程（仅限Linux平台:目前git上是21年开源的5.1版本，后续如果有空整理好说明上传6.xxx的版本）

<details >
<summary>关于模型修改和设计</summary>

     - 2021年在CSDN中介绍过一些范式示例包含注意力、自注意力层等机制早期引入了一些比较有热度的修改，其实在如今图像基础任务表现里，CNN和transformer并不没有明显差距，个人觉得作为学习积累就好。比如swinv1和v2等一些当时流行的论文网络组件，以及同样的NECK、HEAD、LOSS的添加，你可以参考github项目中的yaml结构示例去自己尝试修改模型，就是希望大家能够多思考多积累且自己动手实现，也是我当初文章的本意，而不是只限于一种范式或几种结构，如果遇到问题欢迎分享讨论，具体可以看博客中的[修改建议](https://editor.csdn.net/md/?articleId=126895964）

     - 对于自注意力机制的使用：很多人与CNN相结合使用得到精度提升，个人理解：原因不仅仅是长距离的依赖，早期我们使用固定权重的滤波器提取边缘再到CNN，CNN也许是对应着高通滤波，而self-attention对应于低通滤波，那么相当于对featuremap进行了一次平滑，这样从某种程度上可以解释互补之后的提升；而且transfromer是很难发生过拟合或者说不存在，同时由于增量爆炸和工程开发的现象，使得其并不好训练，但是动态特性确实更具泛化性，常规情况中优先考虑你训练数据集的拟合够不够好，你的模型是否能反映出数据之间的特征特异性，其次扩充构建相应的辅助分支加入特征属性描述。

</details>

[CSDN同步更新，主页可按兴趣原创文章点击](https://blog.csdn.net/weixin_44119362?type=blog)
(保证每周同步更新一次维护，不定期更新算法代码和引进结果实验！关于消融实验大多来自朋友的热心反馈，探究范式CNN和transformer，如何根据经验设计网络结构、LOSS改进、辅助训练分支、样本匹配....  欢迎提供实验数据和結果~)




<details open>
<summary>Install</summary>
Clone repo and install [requirements.txt](https://github.com/positive666/yolo_research/requirements.txt) 

```bash
git clone https://github.com/positive666/yolo_research  # clone
cd yolov5_research
pip install -r requirements.txt  # install
```

</details>

<details open>
<summary>YOLOV8 install in conda env  and  offical command</summary>

pip install ultralytics 

if you pip install ultralytics,you can run offical command 
```bash
yolo task=detect    mode=train   data=<data.yaml path>      model=yolov8n.pt        args...
          classify       predict        coco-128.yaml       yolov8n-cls.yaml  args...
          segment        val                                yolov8n-seg.yaml  args...
                         export                             yolov8n.pt        format=onnx  args...
```
ps: if your model=*.yaml -->scratch else use pretrained models
python command :


if use this repo ,you need set your data and model path in cfg/default.yaml

```bash
    
    python yolo\v8\detect\train.py  --<args>

```

推理部分和V5、V8的代码兼容
</details>  

<details>
<summary>Multi-GPU DistributedDataParallel </summary>
使用DistributedDataParallel，多个进程只进行倒数传播，每个GPU都进行一次梯度求导和参数更新，这比DataParallel的方式更高效，因为DataParalledl只有一个主GPU进行参数更新，所以需要各个子进程调用的GPU传递倒数到主GPU后，才会更新参数給各个GPU，所以这会比DistributedDataParallel每个GPU直接进行参数更新要慢很多。 –nproc_per_node: 作为GPU的使用数量节点数 –batch：总batch-size ,然后除以Node数量 ，平均给每个GPU。
```bash
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --weights yolov5s.pt --device 0,1
```
</details>  

<details>
<summary>Multi -machines && Multi-GPU </summary>
```bash
主机
python -m torch.distributed.run --nproc_per_node G --nnodes N --node_rank 0 --master_addr "192.168.1.1" --master_port 1234 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights ''
#多个副机
python -m torch.distributed.run --nproc_per_node G --nnodes N --node_rank R --master_addr "192.168.1.1" --master_port 1234 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights ''
```
–master_port：端口号
master_addr：主进程ip地址
G:每个机器的GPU数量
N:机器数量
R:子机器序号

</details>  


## <div align="center">目标检测篇</div>

<details>
<summary>Inference with detect.py</summary>


```bash
python detect.py --source 0  # webcam     --weights <your model weight>
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          'path/*.jpg'  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```
</details>  

yolov7 官方的訓練权重打包链接：https://pan.baidu.com/s/1UIYzEZqTPMUpWWBBczOcgA?pwd=v7v7(由于我删除了P6模型里的Reorg操作和Focus没本质区别，所以删掉需要重新训练，如果你想使用V7原始权重，你只需要在YAML里改回去，还有一种方式是遍历Reorg的权重把它替换掉) 提取码：v7v7

###  Train

see train.py args ,command as :


```bash
python train.py --data <your data yaml>  --cfg  <your model yaml> --weights <weights path>  --batch-size 128    --hyp   <hyps yaml>  --batch-size <numbers>  
```
<details open>
<summary>Notes</summary>

- "--swin_float"  for "SwinTransformer_Layer" layers,because of " @" not support  fp16,so you can use offical yolov7 “ Swinv2Block”
- "--aux_ota_loss" for aux- head only . Such "models/detect/v7_cfg/training/yolov7x6x.yaml, (P6-model) ,you can create aux -head models.		
- "ota_loss"  in hyps filse ,ota_loss default=0 
</details>   

<details>
<summary>Training commnd example </summary>

-  run yolov7-P5 model train and yolov5 seriese models ,scratch or fine ,your need a weights 

```bash 
python train.py  --data data/coco128.yaml  --cfg models/detect/yolov5s_decoupled.yaml   
```
```bash 
python train.py  --cfg  models/detect/v7_cfg/training/yolov7.yaml  --weights yolov7.pt  --data (custom datasets) --hyp data/hyps/hyp.scratch-v7.custom.yaml	
```
-  run yolov7-aux model train ,your model must P6-model !
```bash 
python train.py  --cfg  models/detect/v7_cfg/training/yolov7w6.yaml --imgsz 1280  --weights 'yolov7-w6_training.pt'  --data (custom datasets)  --aux_ota_loss  --hyp data/hyps/hyp.scratch-v7.custom.yaml
```
- After training/under yaml structure, your initial weight xxx. PT will become a trained yolov7xxx.pt , with specific references to reparameterized scripts. 
Then use the deploy model to load the weights of your training, change the index and structure to re-parameterize the model.

</details>

<details open>
<summary>re-parameterizetrained yolov7 model  </summary>

```bash 
   python reparameterization.py  --weights <yolov7.pt,yolov7e6e.pt.....>  --name <model name > --save_file   models/v7_cfg/deploy  --cfg <model.yaml>
```
</details>

/details>

<details >
<summary>Offical Model Zoo </summary>

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

</details>

##  <div align="center">关键点检测篇</div>

<details>
<summary>数据集构建</summary>

```
yolov5_research
│   pose  
│   └─────(key point detect code )
│   ...   
│
coco_kpts(your data yaml path name )
│   images
│   annotations/**.json
|   labels
│   └─────train2017
│       │       └───
|       |       └───
|       |       '
|       |       .
│       └─val2017
|               └───
|               └───
|               .
|               .
|    train2017.txt
|    val2017.txt

```
</details>

###  Inference
refernce v7 weights.
[yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

``` bash 
python pose/detect.py --weights pose/pose_weights/yolov7-w6-pose.pt  --source  data/images/bus.jpg   --kpt-label 
```
###  Train

[yolov7-w6-person.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-person.pt)

``` shell
    python pose/train.py --data  data/coco_kpts.yaml  --cfg  pose/cfg/yolov7-w6-pose.yaml weights yolov7-w6-person.pt --img 960  --kpt-label --hyp data/hyps/hyp.pose.yaml

```

##  <div align="center">分割篇</div>

###  Inference

``` bash 
python segment/predict.py --weights yolov5s-seg.pt --source 0                          
```

###  Train
``` bash 
python segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640  # from pretrained (recommended)
```


##  <div align="center">分类篇yolov5官方版本</div>
YOLOv5 [release v6.2](https://github.com/ultralytics/yolov5/releases) brings support for classification model training, validation, prediction and export! We've made training classifier models super simple. Click below to get started.

<details>
  <summary>Classification Checkpoints (click to expand)</summary>

<br>

We trained YOLOv5-cls classification models on ImageNet for 90 epochs using a 4xA100 instance, and we trained ResNet and EfficientNet models alongside with the same default training settings to compare. We exported all models to ONNX FP32 for CPU speed tests and to TensorRT FP16 for GPU speed tests. We ran all speed tests on Google [Colab Pro](https://colab.research.google.com/signup) for easy reproducibility.

| Offcial Model Zoo                                                                                  | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Training<br><sup>90 epochs<br>4xA100 (hours) | Speed<br><sup>ONNX CPU<br>(ms) | Speed<br><sup>TensorRT V100<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@224 (B) |
|----------------------------------------------------------------------------------------------------|-----------------------|------------------|------------------|----------------------------------------------|--------------------------------|-------------------------------------|--------------------|------------------------|
| [YOLOv5n-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n-cls.pt)         | 224                   | 64.6             | 85.4             | 7:59                                         | **3.3**                        | **0.5**                             | **2.5**            | **0.5**                |
| [YOLOv5s-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s-cls.pt)         | 224                   | 71.5             | 90.2             | 8:09                                         | 6.6                            | 0.6                                 | 5.4                | 1.4                    |
| [YOLOv5m-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m-cls.pt)         | 224                   | 75.9             | 92.9             | 10:06                                        | 15.5                           | 0.9                                 | 12.9               | 3.9                    |
| [YOLOv5l-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l-cls.pt)         | 224                   | 78.0             | 94.0             | 11:56                                        | 26.9                           | 1.4                                 | 26.5               | 8.5                    |
| [YOLOv5x-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x-cls.pt)         | 224                   | **79.0**         | **94.4**         | 15:04                                        | 54.3                           | 1.8                                 | 48.1               | 15.9                   |
|                                                                                                    |
| [ResNet18](https://github.com/ultralytics/yolov5/releases/download/v6.2/resnet18.pt)               | 224                   | 70.3             | 89.5             | **6:47**                                     | 11.2                           | 0.5                                 | 11.7               | 3.7                    |
| [ResNet34](https://github.com/ultralytics/yolov5/releases/download/v6.2/resnet34.pt)               | 224                   | 73.9             | 91.8             | 8:33                                         | 20.6                           | 0.9                                 | 21.8               | 7.4                    |
| [ResNet50](https://github.com/ultralytics/yolov5/releases/download/v6.2/resnet50.pt)               | 224                   | 76.8             | 93.4             | 11:10                                        | 23.4                           | 1.0                                 | 25.6               | 8.5                    |
| [ResNet101](https://github.com/ultralytics/yolov5/releases/download/v6.2/resnet101.pt)             | 224                   | 78.5             | 94.3             | 17:10                                        | 42.1                           | 1.9                                 | 44.5               | 15.9                   |
|                                                                                                    |
| [EfficientNet_b0](https://github.com/ultralytics/yolov5/releases/download/v6.2/efficientnet_b0.pt) | 224                   | 75.1             | 92.4             | 13:03                                        | 12.5                           | 1.3                                 | 5.3                | 1.0                    |
| [EfficientNet_b1](https://github.com/ultralytics/yolov5/releases/download/v6.2/efficientnet_b1.pt) | 224                   | 76.4             | 93.2             | 17:04                                        | 14.9                           | 1.6                                 | 7.8                | 1.5                    |
| [EfficientNet_b2](https://github.com/ultralytics/yolov5/releases/download/v6.2/efficientnet_b2.pt) | 224                   | 76.6             | 93.4             | 17:10                                        | 15.9                           | 1.6                                 | 9.1                | 1.7                    |
| [EfficientNet_b3](https://github.com/ultralytics/yolov5/releases/download/v6.2/efficientnet_b3.pt) | 224                   | 77.7             | 94.0             | 19:19                                        | 18.9                           | 1.9                                 | 12.2               | 2.4                    |

<details>
  <summary>Table Notes (click to expand)</summary>

- All checkpoints are trained to 90 epochs with SGD optimizer with `lr0=0.001` and `weight_decay=5e-5` at image size 224 and all default settings.<br>Runs logged to https://wandb.ai/glenn-jocher/YOLOv5-Classifier-v6-2
- **Accuracy** values are for single-model single-scale on [ImageNet-1k](https://www.image-net.org/index.php) dataset.<br>Reproduce by `python classify/val.py --data ../datasets/imagenet --img 224`
- **Speed** averaged over 100 inference images using a Google [Colab Pro](https://colab.research.google.com/signup) V100 High-RAM instance.<br>Reproduce by `python classify/val.py --data ../datasets/imagenet --img 224 --batch 1`
- **Export** to ONNX at FP32 and TensorRT at FP16 done with `export.py`. <br>Reproduce by `python export.py --weights yolov5s-cls.pt --include engine onnx --imgsz 224`
</details>
</details>

<details>
  <summary>Classification Usage Examples (click to expand)</summary>

### Train
YOLOv5 classification training supports auto-download of MNIST, Fashion-MNIST, CIFAR10, CIFAR100, Imagenette, Imagewoof, and ImageNet datasets with the `--data` argument. To start training on MNIST for example use `--data mnist`.

```bash
# Single-GPU
python classify/train.py --model yolov5s-cls.pt --data cifar100 --epochs 5 --img 224 --batch 128

# Multi-GPU DDP
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3
```

### Val
Validate YOLOv5m-cls accuracy on ImageNet-1k dataset:
```bash
bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate
```

### Predict
Use pretrained YOLOv5s-cls.pt to predict bus.jpg:
```bash
python classify/predict.py --weights yolov5s-cls.pt --data data/images/bus.jpg
```
```python
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s-cls.pt')  # load from PyTorch Hub
```

### Export
Export a group of trained YOLOv5s-cls, ResNet and EfficientNet models to ONNX and TensorRT:
```bash
python export.py --weights yolov5s-cls.pt resnet50.pt efficientnet_b0.pt --include onnx engine --img 224
```
</details>

##  <div align="center">历史更新</div>

<details>
<summary>更新记录</summary>
- 2020/9/15  High-level 集成待验证，目前姿态训练和检测已经调试完成。
- 2022/7/21  除关键点部分的V7代码以及V5代码风格优化合并更新，改善了重参数脚本的功能，详情看	reparameterization.py

- 2022/7/13  同步更新加入了yolov7的P6模型訓練部分代碼，p6是需要aux的所以需要添加Loss部分計算，代碼和CSDN持續更新中,由于我删除了P6模型里的Reorg操作其实就说FOcus，所以需要重新训练，如果你想使用V7原始权重，你只需要在YAML里改回去

- 2022/7/7   依旧全网首发 ：以目前仓库的魔改版V5为基准同步了YOLOV7的核心改动，代码风格是最新的，后续会持续完善优化，完美融合V7，后续博客争取更新第一时间！

- 2022/5/23  合并更新了YOLOV5仓库的最新版本，作者代码有点小问题就是数据集会重复下载，这部分我没就没合并他的更新，引入了新的算子，看来他也在探索实验

- 2022/3/26  测试下解耦训练结果/更新GAM注意力层代码：按照论文示意在大模型中使用分组卷积降低FLOPs,同步简单实验下，关于实验在闲暇之余都会慢慢完善的。
    以small模型，在Visdrone数据下的简单验证：
	|   Model     		 |   mAP@.5  | mAP@.5:95 | Parameters(M) | GFLOPs |
	| --------    		 |   ------  |  ------   | ------------- | ------ |
	| YOLOv5s     		 |   0.351    |  0.194   |     7.2       | 16.5   |
	| YOLOv5s+GAM 		 |   0.35    |  0.194    |     22.2      | 36.9   |
	| YOLOv5s_decoup     |   0.367   |  0.203    |     7.1       | 17.2   |
    | YOLOv5s_GAM_group   |  0.353  	|  0.192 	 |     11       | 21.4   |  （待进一步更新）


- 2022/3/26  1.修复了一些常规的问题BUG并合并了V5作者的最新代码更新，大概包含之前缺少了一些可学习参数和代码优化,如添加了swintransformerV2.0的相对位置编码加入优化器等。 2.目前看来GAM换用组卷积效果有待商榷，后续进一步整理消融实验总结。
- 2022/3/16  对上传的GAM注意力层进行了简单的实验，yolov5s+GAM在Visdrone数据集上的结果举例参考，后续的话其实难点在于轻量化，探究大模型的骨干估计只有大厂研究资源能有成本去做。
- 2022/3/5   近期会整理一些去年的实验数据/、使用swin2的骨干，超参数需要调试一下，首先要稍微减低学习率，（实测SGD）；也可以把SWIN层作为注意力插件训练，这个和以往的操作类似，不再赘述了 需要开启--swin_float   命令参数，因为点积不被cuda的half支持，而优化器的问题，那么问题基本就是较多的swin block 堆积导致的增量更新。同时伴随着GPU的开销。 
- 2022/3.1   （不完整更新,供参考，怕忙断更，所以先放出部分修改，目前还在动态调试中）按照SWintransformerV2.0 的改进点：修改了NORM层的位置/attention将dot换成scaled cosine self-attention，待更新的优化部分：1.序列窗口注意力计算，降低显存开销 2、训练优化器
- 2022/2.28  添加了一个Swintransformer的Backbone和yaml示意结构，很多人把SWIN还像之前做成注意力层，但是其实swin设计是为了摒弃CNN去和NLP一统，而且精髓在于控制计算复杂度，其实backbone的全替换也许更值得尝试 ，内存开销和结构设计待优化
- 2022/2.22  忙里抽闲：更新了今天的yolov5的工程修复，修改了解耦头的代码风格，直接yaml选择参考使用，服务器回滚了代码。SWIN算子在，YAML文件丢失了，找时间从新写一个再上传，太忙看更新可能优先GIT，等有空博客细致归纳下
- 2022/2.6   ASFF使用的BUG已经修复;近期更新Swintransformer代码，简单说明下程序上其实是两种改法：1.类似算子层的修改，这个比较简单 2、全部替换成Swintransformer，这个对于整个程序存在一定的代码修改地方，稍微复杂点。
- 2022/1.9   补充一些注意力算子GAM，原理后续CSDN说明，修复BUG
- 2021/11.3  合并最新的YOLOV5的改动， 替换了CSPBOTTLENNECK的LeakRELUw为SLIU，其余全是代码和工程规范修改
- 2021.10.25 修复BUG，恢复EIOU
- 2021.10.13 更新合并YOLOV5v6.0版本，改进点：第一时间的更新解析可参考[CSND博客](https://blog.csdn.net/weixin_44119362/article/details/120748319?spm=1001.2014.3001.5501)
- 2021.9.25  将自注意力位置编码设置成可选项，默认取消，CBAM不收敛——将激活函数改回Sigmoid
- 2021.6.25  添加BIFPN结构包含P5/P6层，增大开销但是对于一些任务是能够提点的
- 2021.6     Botnet transformer 算子块引入于Backbone底层
- 2021.2.10  全网首发的YOLOV5魔改，ASFF检测头封装加入、注意力机制CBAM、CooRD、等注意力算子引入，并介绍了通用修改方式


</details>

## <div align="center">高性能视频推理部署—待更新升级</div>
<details>
<summary>工程部署 Why Deepstream?</summary>

工程部署：该仓库只属于研究探索，但是工程部署讲究简单高效、故可以参考我的Deepstream SDK改的项目，集合了通用检测、人脸识别、OCR三个项目，高性能的部署AI框架开发逻辑，这个项目是我2021年整理并开源的，代码还未规范，但程序是没问题的。
 DS_5.1&&Tensorrt7+ ：https://github.com/positive666/Deepstream_Project

     1.英伟达提供的Deepstream &&Tensorrt，应用于流媒体处理，因为做过业务的都知道，推理性能不等于程序运行性能，核心除了模型的本身剪枝量化之外，涉及到了对数据输入的处理，这里的核心问题是如何提高GPU的利用率，那么最直接的就是GPU编解码.
     2.目前嵌入式部署可能大多采用剪枝通道压缩模型的流程，在结合一些框架去进行引擎推理，推荐Yolov5nano或者nanodetplus,(工程上主流是通道裁剪，替换如C3的BOLOCK，你可以在仔细比对YOLOV5的迭代。还有就是如何使用SGD炼丹的经验了)
     还有就是deepstream的普及，网上很多剪枝版本我也看了值得学习，但是工程不只在于学习，而在于成本和结果。
     3.x86和Jeston都可以部署，既有一站式解决方案，我觉得工程和研究应用是完全不同的操作思路，精简高效达到目的.deepstream全做了并完成降维打击 ，当然也需要一定的综合开发能力。

</details>

## C++ sdk的完整Deepstream5.1部署（内置C++嵌入的Kafka服务） 
  目前是5.1版本，近期更新6.0(主要区别在于Tensorrt7和Tensorrt8的源码区别导致的，部分6.0 SDK有变动)
  [Deepsteam YOLOV5 V5.0]https://github.com/positive666/Deepstream_Project/tree/main/Deepstream_Yolo 



## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)
* [https://github.com/ChristophReich1996/Swin-Transformer-V2](https://github.com/ChristophReich1996/Swin-Transformer-V2)
* https://github.com/positive666/Deepstream_Project

</details>