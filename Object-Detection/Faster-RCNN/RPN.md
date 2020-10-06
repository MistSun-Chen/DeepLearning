# RPN层

![fasterRCNN](img\fasterRCNN.png)

由以上整体框架可以看到，RPN结构以Backbone输出的特征图为输入，输出所有候选框

下面具体分析RPN的原理

# Anchor的生成

## **1、在特征图的每个3 x 3滑动窗口，计算中心点对应原始图像上的中心点**

![1](img\1.png)

计算方法：

假设:特征图中心点坐标（x,y) 特征图高度和宽度为(h,w),原图片的中心点坐标为（X，Y），原图的高度和宽度为（W，H）则：

X = int(W/w) * x 	 	原图中心点的横坐标对应 特征图中心点横坐标乘以步距（原图宽度除以特征图宽度）

Y = int(H/h) * y			原图中心点的纵坐标对应 特征图中心点纵坐标乘以步距（原图高度除以特征图高度）

## 2、根据给定的anchor尺度（面积）以及比例生成anchor

![1](img\2.png)



如上图，假定已经给定3种面积和3种尺度，则每个中心点应该生成9个anchor



现在以中心点坐标为(0,0),给定anchor面积为S，尺度为R来举例，则：

高度乘法因子：h_ratios = sqrt(R)	宽度乘法因子 w_ratios = 1.0 / h_ratios

面积为S的正方形边长为 l = sqrt(S)

anchor的高和宽分别为：

ws = w_ratios * l						hs = h_ratios * l

则anchor可以表示为 [ -ws , -hs , ws , hs]/2 ,即anchor左上的坐标和右下坐标

以坐标（0，0）生成的anchor可以当做模板

## 3、将anchor添加到原图

![2](img\3.png)



# 通过kernel_size = 1的卷积核计算预测分类的分数和矩形框的回归参数

![2](img\2.JPG)

![2](img\4.png)



将3x3的滑动窗口生成的特征向量分别用两个1x1 的卷积核进行分类与回归

分数的组成如上图所示

k个anchor生成2k个分类分数，分别代表前景的预测和背景的预测

k个anchor生成4k个回归分数，dx，dy分别代表对anchor中心点的调整，dw，dh分别代表对anchor宽度以及高度的调整



注：论文中使用2k个分类分数，损失计算应该用softmax cross entropy。代码实现时，我们仅使用k个分类分数，离1近的是前景，离0近的是背景，此处损失计算应使用binary cross entropy

下图所示是回归参数对矩形框的调整

![2](img\5.png)

# 筛选获得的Proposal

Proposal就是通过边界框回归器获得的预测矩形框信息

如果不筛选的话，矩形框过多且有很多重复部分无效。

如果采用特征金字塔，那么产生的特征图大于一，此时生成的Proposal将更多，**必须进行筛选，获取有效数据**

## 1、获取特征图上预测概率排前的anchor索引值

对于非特征金字塔结构，feature map一般只有一个，此时只需要torch.topk函数获取预测概率值排在前排的anchor索引即可，一般训练获取前2000个anchor索引，测试获取前1000个索引

对于特征金字塔结构，feature map有多个，此时需要分步处理，对每一个feature map都获取预测概率值排前的anchor索引

但多个feature map一般用一个tensor进行存储。要先进行feature map的划分，在获取概率派遣anchor索引值时，要注意该索引值是当前feature map内的索引值还是所有feature map下的索引值。

在整体索引值需要用offset偏移量进行调整

下图是划分feature map

![3](img\3.JPG)

## 2、细微调整与NMS处理

通过第一步获取的anchor索引信息可以获取其相应的bounding box信息

首先，先将不符合条件的bounding box剔除，如：

* **bounding box越界**
* **bounding box宽或高小于设定的最小值**

### NMS（non-maximum suppression）

此处的非极大值抑制操作其实也就是

直接采用官方的nms方法`torchvision.ops.nms`(*boxes*, *scores*, *iou_threshold*)

返回经过NMS操作的boxes的索引，且按分数降序排序

nms算法具体步骤如下

1. **由于我们已经有每个box是否包含物体（objectness）的分数，我们按照这个分数对box从高到低排序。**
2. **然后我们对排好序的每一个box，计算出剩下的box和它的IoU，对于剩下的box，当IoU超过某个阀值（比如0.7）就将他去掉（suppress）**



需要注意的一点就是，**如果采用特征金字塔结构，需要分别对每一层的特征层进行nms的操作，此时需要对每一层的特征图的anchor坐标进行一个偏移处理**，目的是为了区分每一层的boxes信息

具体做法如下：

**1、 获取当前层的boxes的坐标最大值max_coordinate**

**2 、 筛选Proposal的步骤一种我们将每个特征层给予了一个proposal mask，如若为第一层则有0标签，第二层是1标签.......我们用这个标签设该层标签为idx,则offset = idxs * (max_coordinate + 1)****

***

NMS操作过后，我们筛选其中排名靠前的boxes，在本实验中，training时筛选2000个，testing时是1000个



# RPN损失计算

## 划分正样本，背景以及废弃样本并计算每个anchor最匹配的ground truth

我们先对上一步获得的所有anchor对ground truth进行IOU操作

计算每个anchor匹配IOU最大的索引，但要区分正负样本与废弃样本（例如：如果iou<0.3索引置为-1，0.3<iou<0.7索引为-2）

对于正样本，计算每个anchor最契合的ground truth信息

对于正样本来说，一般大于阈值就认定为正样本，但对于极少数情况下出现的所有anchor与某ground truth相交的阈值都小于0.7的情况，我们选择最大的那个作为正样本

## 损失计算

### RPN网络损失计算

![4](img\4.JPG)

#### 1、分类损失

![6](img\6.png)

#### 2、回归损失

![7](img\7.png)