## 基于yolov5的改进库(保证每周同步更新一次维护，不定期更新算法代码和引进)

预告：swintransformer的代码基本完成，两种修改方法，届时会结合博客一起发出，最近项目比较忙，看时间尽快更



   一切改动都基于YOLOV5的官方项目：[参考官方](https://github.com/ultralytics/yolov5) 
   
## C++ sdk的完整Deepstream5.1部署（内置libKafka服务），目前是5.1版本，近期更新6.0(主要区别在于Tensorrt7和Tensorrt8的源码区别导致的，部分6.0SDK有变动)
  [Deepsteam YOLOV5 V5.0]https://github.com/positive666/Deepstream_Project/tree/main/Deepstream_Yolo 

***
   
   不间断保持更新和汇总实验：算子引入，LOSS改进，针对网络结构进行不同的任务的最优结构汇总，样本匹配实验，业务拓展等等
   有针对于自己数据集或者公开数据集的效果请联系，目前有很多实验没做，平时工作繁忙，保证定期更新，也希望大家能一起探索最优结构和实验效果。
   
***


***
目前反馈的实验结果（待整理更新） 
***

***
最近更新：
- 2022/2.22 忙里抽闲：更新了今天的yolov5的工程修复，修改了解耦头的代码风格，直接yaml选择参考使用，服务器回滚了代码。SWIN算子在，YAML文件丢失了，找时间从新写一个再上传，太忙看更新可能优先GIT，等有空博客细致归纳下
- 2022/2.6（待更新完成）  近期更新SWINtransformer代码，简单说明下程序上其实是两种改法：1.类似算子层的修改，这个比较简单 2、全部替换成Swintreanformer，这个对于整个程序存在一定的代码修改地方，稍微复杂点。
- 2022/1.9   补充一些注意力算子GAM，原理后续CSDN说明，修复BUG
- 2021/11.3  合并最新的YOLOV5的改动， 替换了CSPBOTTLENNECK的LeakRELUw为SLIU，其余全是代码和工程规范修改
- 2021.10.25 修复BUG，恢复EIOU
- 2021.10.13 更新合并YOLOV5v6.0版本，第一时间的更新解析可参考[CSND博客](https://blog.csdn.net/weixin_44119362/article/details/120748319?spm=1001.2014.3001.5501)
- 2021.9.25  将自注意力位置编码设置成可选项，默认取消
- 2021.6.25  添加BIFPN结构包含P5/P6层，增大开销但是对于一些任务是能够提点的
- 2021.6     Botnet transformer 算子块引入于Backbone底层
- 2021.2.10  全网首发的YOLOV5魔改，ASFF检测头封装加入、注意力机制CBAM、CooRD、等注意力算法引入，并介绍了通用修改方法
***

# 业务拓展（可做人脸、文字等特定目标检测）

人脸工作可参考： yolov5_face repo: [https://github.com/deepcam-cn/yolov5-face.git]
ocr 后面我会更新下自己的思路，看后面时间更新项目