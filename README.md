## 基于yolov5的改进库(保证每周同步更新一次维护，不定期更新算法代码和引进,PS：注意层的使用大部分都适合少量数据集和特定业务目标数据集，通用数据集上不够好，关于消融实验太多，大多来自网友的热心反馈，自己这边有些业务的，但是提升不是很明显且没能在COOO这种数据集的提升几乎没有说服力，还在想把这些实验结果做个精简展示，等手头的事情忙完一个再考虑 欢迎提供实验数据~)

声明：这一年里有很多朋友用修改的idea方式以及注意力和设计结构去做Paper，不用咨询我意见：一切free！只是希望各位能把实验数据反馈给我，因为平时工作业务比较忙，实验的时间不够。也是借此目的来和大家一起学习交流。

   不改项目名，一切源于别人的工作，如何改进都是锦上添花和学习探索，只FORK，业余兴趣改动都基于Yolov5的官方项目：[参考官方](https://github.com/ultralytics/yolov5) 
   
***
最近更新：
- 2022/3/5  近期会整理一些去年的实验数据///使用swin2的骨干，超参数需要调试一下，首先要稍微减低学习率，使用adM,；也可以把SWIN层作为注意力插件训练，这个和以往的操作类似，不再赘述了 需要开启--swin_float   命令参数，因为点积不被cuda的half支持，而优化器的问题，目前看来是和多卡显存有关，需要降低模型复杂度，或者降低batch 
- 2022/3.1 （不完整更新,供参考，怕忙断更，所以先放出部分修改，目前还在动态调试中） 按照SWintransformerV2.0 的改进点：修改了NORM层的位置/attention将dot换成scaled cosine self-attention，待更新的优化部分：1.序列窗口注意力计算，降低显存开销 2、训练优化器
- 2022/2.28  添加了一个Swintransformer的Backbone和yaml示意结构，很多人把SWIN还像之前做成注意力层，但是其实swin设计是为了摒弃CNN去和NLP一统，而且精髓在于控制计算复杂度，其实backbone的全替换也许更值得尝试 ，内存开销和结构设计待优化
- 2022/2.22 忙里抽闲：更新了今天的yolov5的工程修复，修改了解耦头的代码风格，直接yaml选择参考使用，服务器回滚了代码。SWIN算子在，YAML文件丢失了，找时间从新写一个再上传，太忙看更新可能优先GIT，等有空博客细致归纳下
- 2022/2.6   ASFF使用的BUG已经修复;近期更新Swintransformer代码，简单说明下程序上其实是两种改法：1.类似算子层的修改，这个比较简单 2、全部替换成Swintreanformer，这个对于整个程序存在一定的代码修改地方，稍微复杂点。
- 2022/1.9   补充一些注意力算子GAM，原理后续CSDN说明，修复BUG
- 2021/11.3  合并最新的YOLOV5的改动， 替换了CSPBOTTLENNECK的LeakRELUw为SLIU，其余全是代码和工程规范修改
- 2021.10.25 修复BUG，恢复EIOU
- 2021.10.13 更新合并YOLOV5v6.0版本，改进点：第一时间的更新解析可参考[CSND博客](https://blog.csdn.net/weixin_44119362/article/details/120748319?spm=1001.2014.3001.5501)
- 2021.9.25  将自注意力位置编码设置成可选项，默认取消，CBAM不收敛《将激活函数改回Sigmoid
- 2021.6.25  添加BIFPN结构包含P5/P6层，增大开销但是对于一些任务是能够提点的
- 2021.6     Botnet transformer 算子块引入于Backbone底层
- 2021.2.10  全网首发的YOLOV5魔改，ASFF检测头封装加入、注意力机制CBAM、CooRD、等注意力算子引入，并介绍了通用修改方式
***


   
## C++ sdk的完整Deepstream5.1部署（内置C++嵌入的Kafka服务） 
  目前是5.1版本，近期更新6.0(主要区别在于Tensorrt7和Tensorrt8的源码区别导致的，部分6.0SDK有变动)
  [Deepsteam YOLOV5 V5.0]https://github.com/positive666/Deepstream_Project/tree/main/Deepstream_Yolo 

***
   
   不间断保持更新和汇总实验：算子引入，LOSS改进，针对网络结构进行不同的任务的最优结构汇总，样本匹配实验，业务拓展等等
   有针对于自己数据集或者公开数据集的效果请联系，目前有很多实验没做，平时工作繁忙，保证定期更新，也希望大家能一起探索最优结构和实验效果。
   
***





#Swintransformer2的修改 
参考：https://github.com/ChristophReich1996/Swin-Transformer-V2/blob/main/swin_transformer_v2/model_parts.py

# 业务拓展（可做人脸、文字等特定目标检测、分割）

人脸工作可参考： yolov5_face repo: [https://github.com/deepcam-cn/yolov5-face.git]
ocr 后面我会更新下自己的思路，看后面时间更新项目