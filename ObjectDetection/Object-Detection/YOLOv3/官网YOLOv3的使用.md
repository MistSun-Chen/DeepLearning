# YOLOV3

#### 官网YOLOV3地址：https://pjreddie.com/darknet/yolo/

## 下载并编译yolov3

```shell
git clone https://github.com/pjreddie/darknet
cd darknet
make
```

## yoloV3测试

### 1、下载预训练文件

```shell
wget https://pjreddie.com/media/files/yolov3.weights
```

### 2、测试运行

```shell
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

其中cfg中保存yolov3的配置文件

可以看到yolov3.cfg文件保存了很多配置信息，包括学习率，batch，momentum以及darknet网络的网络结构，anchor的尺寸等也在这个文件中指定，显然其中的值是根据经验确定的。

当然还可以通过聚类的方式获取

### 3、cfg文件解析

[xxx]开始的行表示网络的一层，其后的内容为该层的参数配置，[net]为特殊的层，配置整个网络

#### 3.1、net层

```python
[net]
#Testing
#batch=1 
#subdivisions=1
#在测试的时候，设置batch=1,subdivisions=1
#Training
batch=64
subdivisions=32
#这里的batch与普遍意义上的batch不是一致的。
#训练的过程中将一次性加载16张图片进内存，然后分4次完成前向传播，每次4张。
#经过16张图片的前向传播以后，进行一次反向传播。
width=608
height=608
channels=3
#设置图片进入网络的宽、高和通道个数。
#由于YOLOv3的下采样一般是32倍，所以宽高必须能被32整除。
#多尺度训练选择为32的倍数最小320*320，最大608*608。
#长和宽越大，对小目标越好，但是占用显存也会高，需要权衡。
momentum=0.9
#动量参数影响着梯度下降到最优值得速度。
decay=0.0005
#权重衰减正则项，防止过拟合。
angle=0
#数据增强，设置旋转角度。
saturation = 1.5
#饱和度
exposure = 1.5
#曝光量
hue=.1
#色调

learning_rate=0.001
#学习率:刚开始训练时可以将学习率设置的高一点，而一定轮数之后，将其减小。
#在训练过程中，一般根据训练轮数设置动态变化的学习率。
burn_in=1000
max_batches = 500200
#最大batch,相当于epoch
policy=steps
#学习率调整的策略，有以下policy：
#constant, steps, exp, poly, step, sig, RANDOM，constant等方式
#调整学习率的policy，        
#有如下policy：constant, steps, exp, poly, step, sig, RANDOM。
#steps#比较好理解，按照steps来改变学习率。

steps=400000,450000
scales=.1,.1
#在达到40000、45000的时候将学习率乘以对应的scale
```

#### 3.2、convolutional层

```python
[convolutional]
batch_normalize=1
#是否做BN操作
filters=32
#特征图的数量
size=3
#卷积核的尺寸
stride=1
#卷积运算的步长
pad=1
#如果pad为0，padding由padding参数指定
#如果pad为1，padding大小为size/2，padding应该是对输入图像左边缘拓展的像素数量
activation=leaky
#激活函数的类型：logistic，loggy，relu，
#elu，relie，plse，hardtan，lhtan，
#linear，ramp，leaky,tanh,stair
#alexeyAB版添加了mish，swish,nrom_chan等新的激活函数
```

#### 3.3、上采样与下采样

通过线性插值实现上采样

```python
[upsample]
stride=2
```

通过卷积层参数进行下采样

```python
# Downsample

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky
```





#### 3.4、shorcut和route层

```python
[shortcut]
from=-3
activation=linear
#shortcut操作是类似ResNet的跨层连接，参数from是-3，
#意思是shortcut的输出是当前层与先前的倒数第三层相加而得到。
#通俗来讲就是add操作
[route]
layers = -1, 36
#当属性有两个值。以上面的route层为例，就是将上一层和第36层进行concate
#即沿深度的维度连接，这也要求feature map大小是一致的
[route]
layers = -4
#当属性只有一个值时，它会输出由该索引的网络层的特征图。
#本例子中就是提取从当前倒数第四个层输出

```

#### 3.5、yolo层

```python
[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear
#每一个[region/yolo]层前的最后一个卷积层中的filters = num(yolo层个数)*(classes+5),5的意义是5个坐标，代表论文中的tx,ty,tw,th,to(类似于confidence)


[yolo]
mask = 6,7,8
#若训练框mask的值是0,1,2就表示使用第一，第二和第三个anchor
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
#总共三个检测层，共计9个anchor
classes=80
#类别个数
num=9
#每个grid预测的BoundingBox num/yolo层个数
jitter=.3
#利用数据抖动产生更多数据，属于TTA（Test Time Augmentation）
ignore_thresh = .7
truth_thresh = 1
#ignore_thresh与truth_thresh是loss的相关参数。yolov3的loss函数包括coord Loss，objectness Loss，class Loss。每一个grid有多个预测框（其数量即为mask的数量），其中与ground truth的IOU最大的预测框，当其IOU小于ignore_thresh的时候，即作为负样本参与objectness Loss的计算，否则不参与该计算（不作为正样本）。而且，如果这个IOU大于truth_thresh，则作为样本参与coord Loss和class Loss的计算。ignore_thresh一般选取0.5-0.7之间的一个值，默认为0.5。truth_thresh默认为1

random=1
#random是多尺度训练使能参数。如果为1，那么每次10次迭代的图片大小在[320,608]区间内随机选择，步长为32。如果为0，图像尺寸为net定义的高宽。
```



### 4、使用自己的数据集重新训练YOLOv3

官网中给出了训练Pascal VOC Data的样例，我们仅需要做一些小修改即可

首先可以创建一个自己yolov3.cfg文件，net层可以修改学习率，batch_size,epoch等。

但最重要的是修改文件的yolo层及其上一层，以下是修改后信息

```python
[convolutional]
size=1
stride=1
pad=1
filters=30
#注意此处要修改，公式为filters = num(yolo层个数)*(classes+5)，具体见cfg文件解析
activation=linear


[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=5
#此处改成自己数据集的类别个数
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
```



然后我们查看cfg/voc.data文件

```config
classes= 5
train  = /home/guest/zsr/CWY/project/yolov3/darknet/data/data_all/train.txt
valid  = /home/guest/zsr/CWY/project/yolov3/darknet/data/data_all/test.txt
names = data/baggage.names
backup = backup

```

如上所示，我们将类别修改成自己的种类个数

train是训练数据文本的存储路径，train.txt存储的是训练样本的文件名

valid同上

backup是checkpoint的路径

接下来我们看看data/baggage.names

```
person
put
suitcase
handbag
backpack
```

如上所示，文件中的内容是类别对应的标签信息

如下是我的数据文件夹

```

* ├── JPEGImages: 存放所有数据图片的文件夹
* ├── labels: 以txt为格式存放所有JPEGImages文件夹中所有图片对应的标签信息，且文件名与图片文件名保持一致
* ├── test.txt: 所有测试用图片的路径
* ├── train.txt: 所有训练用图片的路径
```

让后获取预训练权重

```shell
wget https://pjreddie.com/media/files/darknet53.conv.74
```

开始训练

```shell
nohup ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74 -gpus 1,2 > mylog/log_1.txt 2>&1 &
```

结果保存在log_1.txt中

