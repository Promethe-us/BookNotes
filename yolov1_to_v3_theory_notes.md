

| publish time | paper |
|--|--|
| CVPR2016 | Yolov1 |
| CVPR2017 | Yolov2/Yolo9000 |
| CVPR2018 | Yolov3 |
| 2020.4 | Yolov4 |
| 2020.5 | Yolov5 |

### _______________________________________________________________________________________________________________
# Yolov1(Joseph Redmon) 20个类

- You Only Look Once: Unified, Real-Time Object Detection

## 1，预测阶段
#### Yolo是448*448*3 ---( 24层conv + 2层 FC)---> 7*7*30
![title](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fwww.pianshen.com%2Fimages%2F553%2F53717b612e7749ca7599e2aef41768a1.JPEG&refer=http%3A%2F%2Fwww.pianshen.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1643427206&t=6bccbce15a05e1e0754f418fb1c85844)

- 20分类
- s*s个gridcell(s=7)，每一个gridcell负责预测两个gridcell
- 每个gridcell的属性有(x,y,h,w,c) x,y是左上角坐标，c代表bondingbox的置信度
- c是置信度，后面20个是条件概率，真的概率=c*条件概率

![title](https://pic4.zhimg.com/80/v2-f9af0b8094b35f7c2ab2179efb6f4c8c_hd.jpg)

#### 预测阶段后处理 NMS(非极大值抑制)
- 7*7*30 --(后处理)--> 真正的框

![title](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fimg-blog.csdnimg.cn%2F20190823093521234.png%3Fx-oss-process%3Dimage%2Fwatermark%2Ctype_ZmFuZ3poZW5naGVpdGk%2Cshadow_10%2Ctext_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzU2OTY0Nw%3D%3D%2Csize_16%2Ccolor_FFFFFF%2Ct_70&refer=http%3A%2F%2Fimg-blog.csdnimg.cn&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1643430994&t=3e25923a1044fd77e21e6992f29a3c9f)

- 注意一下，在训练阶段是没有NMS的！

## 2，*训练阶段
- yolov1的损失函数包含5项
![title](https://img-blog.csdnimg.cn/5461070b083b4929820906a0e917ebc5.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZW50YW8zMjY=,size_16,color_FFFFFF,t_70)



### ________________________________________________________________________________________________________
# Yolov2 20个类

### [Joseph Redmon and Ali Farhadi. “YOLO9000: Better, Faster, Stronger” Computer Vision and Pattern Recognition (2017)]

![title](https://img1.doubanio.com/view/status/l/public/b6f669ee745528a.webp)

#### 相比于yolov1,yolov2中加入了一下新元素：
- (1) Batch Normalization
- (2) High Resolution Classifier
- (3) Anchor
- (4) Dimension Cluster
- (5) Direct location prediction
- (6) Fine-Grained Features
- (7) Multi-Scale Training

## 1,BatchNormalization(BN层) 批归一化
- 训练阶段:对于一个batch的数据，求均值和标准差和归一化，再把归一化后的值*gamma + beta (beta和gamma是可训练参数) 每个神经元都训练一对(beta,gamma)

- 测试阶段:gamma,beta均使用训练阶段基于全局求出来的均值方差作为(beta,gamma)
![title](http://tiebapic.baidu.com/forum/w%3D580/sign=f71389cefc389b5038ffe05ab537e5f1/4e9bd82a6059252d15e88ed4719b033b59b5b9c2.jpg)

### !!!BN和Dropout都能起到正则化的作用，但是不能一起用，一起用的话效果会变差

## 2，High Resolution Classifier
- yolov2是在大分辨率图片上训练出的backbone

## *3，Anchor机制   4，Dimension Clusters   5，Direct location prediction
- Yolov2把每个图片划分成了13*13个gridcell,每个gridcell有5个anchor
![title](https://img-blog.csdnimg.cn/bff62af7da484c589468d221e2b69fdd.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAY2hlbnRhbzMyNg==,size_16,color_FFFFFF,t_70,g_se,x_16)
- 我们分成奇数*奇数个gridcell是为了防止有四个gridcell抢中心点
### 与yolov1不同：Yolov2是448*448*3 ---(Darknet19)---> 13*13*125( 5*[5+20] )
![title](https://img-blog.csdnimg.cn/982c3b5d4db64a799cdeec99f85dfed0.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAY2hlbnRhbzMyNg==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 单阶段：输入图片，输出张量  目标检测结果就包含在张量之中
- 通过对label进行k-means聚类来选择anchor
- 为了避免anchor偏移偏出去，使用了sigmoid保证中心点永远落在当前anchor内
![title](https://pdf.cdn.readpaper.com/parsed/fetch_target/47ba62de1816c53ba5170ee4e7d75a60_3_Figure_3.png)


## yolov2's Loss Function
- yolov2的论文里并没有明确提到yolov2的损失函数是啥
![title](https://img-blog.csdnimg.cn/3f1033dc2973430ba08c7862d887cded.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAY2hlbnRhbzMyNg==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 6,Fine-Grained Features(细粒度特征)
- 有点像residual
![title](https://img-blog.csdnimg.cn/d8aec7ed28d149958941ca8ee153730c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAY2hlbnRhbzMyNg==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 7,Multiscale Training(多尺度训练)
![title](https://img-blog.csdnimg.cn/05a2bc50857645d08cc0324e310d024f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAY2hlbnRhbzMyNg==,size_20,color_FFFFFF,t_70,g_se,x_16)

- global有一个global average pooling是用来统一大小的
![title](http://tiebapic.baidu.com/forum/w%3D580/sign=ef87dfee49f3d7ca0cf63f7ec21ebe3c/6f7b8918367adab43d488393ced4b31c8601e42d.jpg)

### ________________________________________________________________________________________________________
# Yolov3

![title](https://pdf.cdn.readpaper.com/parsed/fetch_target/a4cd3405e81450bee1e442a1682580cc_0_Figure_1.png)

- yolov3一大特点就是快，在小目标上表现得也很好

### Backbone:Darknet-53 
#### 52个conv层 + 1个FCN 并且还加入了residual 
![title](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fpic4.zhimg.com%2Fv2-8959d1c826b11845c12c0fa8bd642ffb_b.jpg&refer=http%3A%2F%2Fpic4.zhimg.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1643977912&t=45e914245518714489d71b1880585671)

- 这是一个略差于resnet102的网络
![title](https://pic2.zhimg.com/v2-af7f12ef17655870f1c65b17878525f1_r.jpg)
##### 255 = （80 + 5） * 3
- 13 * 13的感受野是416/13 * 416/13，负责预测大物体
- 76 * 76的感受野是416/76 * 416/76，负责预测小物体
![title](https://images2.pianshen.com/296/4c/4c1cd6471a4ec7f8b9c4c7678e9c1810.png)
- 每一个gridcell有3个anchor(bounding box),每一个anchor对应5+80厚度(80分类)

- feature pyramid的作用和neck的差不多，就是融合特征

|特征图层|特征图大小|anchor尺寸|预设边界框数量|
|--|--|--|--|
|图层一|13×13|(116,90);(156,198);(373,326);|13×13×3|
|图层二|26×26|(30,61);(62,45);(59,119);|26×26×3|
|图层三|52×52|(10,13);(16,30);(33,23);|52×52×3|

### 输出的13,26,52是根据输入为416情况而定的，即下采样32，16，8倍数。
- （因此，如果输入是512，输出的就是16,32,64）
- 这代表我们可以输入32倍数的任意尺度的图像

### 损失函数
- 如果一个anchor与groundtruth IOU最大，那么这个anchor就是正样本，如果某个anchor的IOU小于某个阈值我们就认为是负样本。其余的我们忽略掉。
- 损失函数分为三部分：正样本坐标、正样本置信度和类别、正样本置信度
![title](https://img1.doubanio.com/view/status/l/public/b884feabafd49a9.webp)
### 训练过程
![title](https://img2.doubanio.com/view/status/l/public/55644e6df5521e1.webp)

#### AP指标
|指标|含义|
|--|--|
|TP|对groundtruth正确的检测(大于设定阈值就认为正确)|
|FP|对groundtruth错误的检测(小于设定阈值就认为正确)|

- http://t.zoukankan.com/itmorn-p-14193729.html

## yolov3论文精读---"YOLOv3: An Incremental Improvement"
- (1) OCO MAP的意思是 mAP@0.5-0.95 即分别0.5->0.95为阈值求mAP，最后求平均就是COCO mAP
- (2) 预测框分为三种情况，①正样本:与ground truth的IOU最大的样本。②IOU<0.5的样本。(3)忽略:IOU>0.5但是概率非最大的样本。
- (3) 每个ground truth只有一个anchor负责预测。
- (4) 对不负责预测ground truth的anchor，不计算其定位和分类损失函数，仅计算其置信度损失函数。
- (5) 每个预测框的每个类别逐一用逻辑回归输出概率，可能会出现多个类别输出高概率。

# 常见指标的介绍

|TP|$$Conf > P_{thresh}$$|$$IOU > IOU_{thresh}$$|
|--|--|--|
|FP|$$Conf > P_{thresh}$$|$$IOU < IOU_{thresh}$$|
|FN|$$Conf < P_{thresh}$$|$$IOU > IOU_{thresh}$$|
|TN|$$Conf < P_{thresh}$$|$$IOU < IOU_{thresh}$$|

|  |  |
|--|--|
|TP|猫所在的位置被预测大概率是猫|
|FP|猫所在的位置被预测小概率是猫|
|FN|没物体的位置被预测大概率是猫|
|TN|没物体的位置被预测小概率是猫|


- 对于每一个类别、TP,FP,FN,TN组成混淆矩阵(Confusion Matrix)
- Precision(查准率)有物体的位置类别被预测正确的概率
$$
Precision = \frac{TP}{TP+FP}
$$
- Recall(查全率、敏感性、召回率)
$$
Recall = \frac{TP}{TP+FN}
$$

|常用指标|解释|
|--|--|
|AP|$$P_{threshold} 从0到1变化对应的Precison围成的面积$$|
- 我们把框的范围在(32,32)以下的框叫小物体
- 我们把30FPS(frame per second)以上的目标检测叫做实时目标检测)


