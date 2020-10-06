# Faster-RCNN

## 项目地址：

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/faster_rcnn

## 实验目的：

通过对实现代码的研读、理解与运行更深刻的理解Faster-RCNN的原理



## 实验使用环境：

* Python ：3.8.3
* torch：1.6.0+cu101        
* torchvision：0.7.0+cu101
* pycocotools（Linux: pip install pycocotools；Windows:pip install pycocotools-windows(不需要额外安装vs)）
* Ubuntu 16.04.6 LTS (GNU/Linux 4.12.9-041209-generic x86_64)

## 文件结构：

```
* ├── backbone: 特征提取网络，可以根据自己的要求选择，包括mobilenetv2,resnet50_fpn,vgg
* ├── network_files: Faster R-CNN网络（包括Fast R-CNN、RPN、ROI等模块）
* ├── train_utils: 训练验证相关模块（包括cocotools）
* ├── my_dataset.py: 自定义dataset用于读取VOC数据集
* ├── train_mobilenet.py: 以MobileNetV2做为backbone进行训练
* ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
* ├── train_multi_GPU.py: 针对使用多GPU的用户使用
* ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
* ├── pascal_voc_classes.json: pascal_voc标签文件
```

## 超参数设置

本实验使用mobilenetv2作为backbone



* 第一轮训练（冻结前置特征提取网络权重[backbone],训练rpn以及最终预测网络部分）：

  此阶段设置超参数：

  epochs = 5

  lr = 0.005

  momentum = 0.9

  weight_decay=0.0005

  train_data:batch_size = 8

  val_data:batch_size = 1

  不调整学习率

* 第二轮训练（解冻前置特征提取网络权重[backbone]，接着训练整个网络权重):

  此阶段设置超参数：

  epochs = 20

  lr = 0.005

  momentum = 0.9

  weight_decay=0.0005

  train_data:batch_size = 8

  val_data:batch_size = 1

  学习率下降因子gamma = 0.33 ，每五轮下降一次

anchor面积：sizes=((32, 64, 128, 256, 512) ,)

anchor缩放比：aspect_ratios=((0.5, 1.0, 2.0),)

输出特征矩阵尺寸：output_size=[7, 7]

采样率sampling_ratio = 2

## 数据集PASCAL VOC2012(下载后放入项目当前文件夹中)

- Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
- 如果需要使用Pascal VOC2012 test数据集请参考：https://pjreddie.com/projects/pascal-voc-dataset-mirror/

## 训练思路

![fasterRCNN](img\1.jpg)



原论文中采用分别训练RPN以及Fast R-CNN的方法

（1）利用ImageNet与训练分类模型初始化前置卷积网络层参数，并开始单独训练RPN网络参数

（2）固定RPN网络独有的卷积层以及全连接层参数，再利用ImageNet预训练分类模型初始化前置卷积网络参数，并利用RPN网络生成的目标建议框去训练Fast RCNN网络参数

（3)  固定利用Fast RCNN训练好的前置卷积网络层参数，去做微调RPN网络独有的卷积层以及全连接层参数。

（4）同样保持固定前置卷积网络层参数，去微调Fast RCNN网络的全连接参数。最后RPN网络与Fast RCNN网络共享前置卷积网络层参数，构成一个统一网络



但**此处采用RPN Loss + Fast R-CNN Loss的联合训练方法**，也符合pytorch官方给出的源码

## 训练方法

* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要训练mobilenetv2+fasterrcnn，直接使用train_mobilenet.py训练脚本
* 若要训练resnet50+fpn+fasterrcnn，直接使用train_resnet50_fpn.py训练脚本
* 若要使用多GPU训练，使用 "python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py" 指令,nproc_per_node参数为使用GPU数量

## Faster RCNN框架图

![fasterRCNN](img\fasterRCNN.png)



## 运行截图

![fasterRCNN](img\loss_and_lr.png)

![fasterRCNN](img\mAP.png)

