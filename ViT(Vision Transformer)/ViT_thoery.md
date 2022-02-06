## vision transformer

在NLP领域的glue和super glue榜单中，transformer(bert)统治

在CV领域的imagenet榜单(图像分类)和coco(目标检测)榜单和ADE20K(语义分割)中，很多方法也借鉴了transformer

vit(vision transformer)

### 目录

> 1，注意力机制
>
> 2，注意力分数
>
> 3， 自注意力机制
>
> 4， 位置编码
>
> 5，Transformer



### 1，注意力机制 

- (书，报纸，杂志，草稿纸，红色杯子) 中，红色杯子是随意线索，但是当我想要读书时候，书就成了不随意线索。
- 卷积、全连接、池化层都只考虑不随意线索。
- Additive Attention :将等价于key和value合并起来后放入到一个隐藏大小为h,输出为1的MLP

$$
(x_{i},y_{i})代表(key,value)是候选的东西，x代表query你要查询的东西
$$

Nadaraya-Watson核回归：
平均池化相当于把候选标签求平均作为输出，更进一步·N-W回归则是

$$
使用核函数区分远近
f(x) = \sum_{i=1}^{n}softmax(-\frac{1}{2}(x-x_{i})^{2})y_{i}
$$

在Nadaraya-Watson核回归的基础上，引入可以学习的参数w(标量):
$$
f(x) = \sum_{i=1}^{n}softmax(-\frac{1}{2}(x-x_{i})^{2}w)y_{i}
$$

### 2，注意力分数

$$
f(x) = \sum_{i=1}^{n}\alpha(x,x_{i})y_{i}
$$

![title](http://tiebapic.baidu.com/forum/w%3D580/sign=5ee3a99e17ed2e73fce98624b700a16d/aeb9b9fd5266d01661bdd832ca2bd40734fa35e2.jpg)

当下有两种形式来表示key到query的距离，从而将key的value inference给query的value
$$
softmax(a(q,k_{i}))
$$

- 有2.1,2.2两种计算 α 的方式:

  ![title](http://tiebapic.baidu.com/forum/w%3D580/sign=83ca05348f1b0ef46ce89856edc451a1/1ad3c41349540923fa5c2e64cf58d109b3de4951.jpg)

#### 2.1 思路1:Additive Attention，

$$
(k,q) = v^{T}tanh(W_{k}k+W_{q}q)
$$

- 相当于将key和value合并起来放入到一个size=h,输出为1的MLP中 

#### 2.2 思路2:Scaled Dot-Product Attention(常用)

如query和key长度一致
$$
a(k,q) = \frac{<q,k_{i}>}{\sqrt{d} } 
$$

### 3,自注意力机制(self-attention)

- $$
  给定序列x_{1},...,x_{n} (x_{i}是d维向量)
  $$

- 

$$
自注意力池化层将x_{i}当作key,value,query来对序列抽取特征得到y_{1},...,y_{n}
$$

$$
y_{i} = f(x_{i},(x_{1},x_{1}),...,(x_{n},x_{n}))
$$



self-attention池化层就是将x(i)即当key,又当value，还当作query 来抽取特征

这也是叫做’自‘的原因，与RNN不同，自注意力机制没有记录位置的信息


  - ![title](http://tiebapic.baidu.com/forum/w%3D580/sign=a48f7a81b3246b607b0eb27cdbf91a35/247e8801a18b87d63df98fcd420828381e30fd0f.jpg)

​               self-attention用于处理整个Sequence的信息，FC用于处理某个位置的信息

self-attention就是判断整个sequence中哪个和当前位置的输入关系更密切

![title](http://tiebapic.baidu.com/forum/w%3D580/sign=5112dd1e5b178a82ce3c7fa8c602737f/6b619d45d688d43f54a0b620381ed21b0ff43bbd.jpg)
$$
计算完\alpha_{1,1},\alpha_{2,2},\alpha_{3,3},\alpha_{4,4}之后在进行softmax归一化
$$
![屏幕截图 2022-02-05 230846](http://tiebapic.baidu.com/forum/w%3D580/sign=e912dcdf35cf3bc7e800cde4e101babd/078d79c6a7efce1b130de818ea51f3deb58f65a4.jpg)
$$
b^{2} = \alpha_{2,1}^{'}\cdot v_{1}+\alpha_{2,2}^{'}\cdot v_{2}+\alpha_{2,3}^{'}\cdot v_{3}+\alpha_{2,4}^{'}\cdot v_{4}
$$

$$
写成矩阵乘法: \\Q=w^{q}I\\K=w^{k}I\\V=w^{v}I\\ A = K^{T}Q \\A^{’}=softmax(A) \\ O = VA^{’}
$$

q,k,v都是a(input)乘上可学矩阵W得到的，有一种Multi-head方法其实就是之前是a×W1=q --> a×W11=q1,a×W12=q2

### 4,位置编码

- 位置编码将位置信息注入到输入里

  - 假设设长度为n的序列(n×d)，使用位置编码矩阵P(n×d)来输出X+P作为自编码输入

  - P元素

  - $$
    P_{i,2j}=sin(\frac{i}{10000^{2j/d}}) \quad \\   
    P_{i,2j+1}=cos(\frac{i}{10000^{2j/d}})
    $$


- 每个位置都有一个e_{i}
- ![title](http://tiebapic.baidu.com/forum/w%3D580/sign=c8e54a8796ef76093c0b99971edda301/0f407d0e0cf3d7cabfda716caf1fbe096b63a95b.jpg)

### 5，Transformer

Transformer(https://arxiv.org/abs/1706.03762)

Bert(https://arxiv.org/abs/1810.04805)

都是self-attention的应用

图像中，我们把3,32,32的图看作是32*32个长度为3的向量

CNN就是self-attention的特例

在数据量比较多的情况下,self-attention很牛掰，vit弹性更好，需要更大量的数据。

![title](https://pic1.zhimg.com/80/v2-4b53b731a961ee467928619d14a5fd44_720w.jpg)

Feed Forward是基于位置的全连接，说白了就一普通的全连接:

- 将输入形状由(b,n,d)变为(bn,d)
- 作用两个全连接层
- 输出形状由(bn,d)变化回(b,n,d)
- 等价于两层核窗口为1的一维卷积层  

将编码器的输出作为解码中第i个Transformer块中多头注意力的key和value

- 预测第t+1输出时，解码器中输入前t个预测值，前t个预测值作为key和value，第t个预测值还作为query。 
- 编码器和解码器都有n个transforms块 