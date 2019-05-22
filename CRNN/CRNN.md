# CRNN

## 特点
- 端到端
- 任意长度，无需字符分割，无需水平方向归一化
- 无需lexicon，有没有lexicon 表现都很好
- 模型小而有效，对真实场景有事件意义

## 使用数据集
- IIIT-5K
- Street View Text 
- ICDIR03 ICDIR13

## 介绍
- 序列识别问题特点
    + 连续、不适合单独检测
    + 长度不一，变化幅度大（eg:"OK","Congratunation"）
- 因此，DCNN不能直接用
- 序列识别问题的一些方法
    + 先检测单个字符，再用DCNN识别
    + 当成分类问题，类别多，中文不可行
- RNN 解决序列问题
    + 一个好处是不需要每个元素的位置信息
    + 然而图片到图片特征是必要的，这导致不能端到端训练
- 传统方法不基于NN也很优秀
    + embed word images
    + mid_level feature 
    + 尽管他们很优秀，但比不过NN
- 介绍CRNN
    + 直接从图片开始学习
    + 信息表示和DCNN一样
    + 和RNN特性一样，可以处理序列
    + 不局限于长度，只需要高度归一化
    + 表现效果好

## 网络架构
- CNN
- RNN
- transciption

### 特征提取
- 卷积层和CNN模型中一样
    + 特征分为n列，第i个特征对应第i列，列长固定
- 特征和列的一一对应性
- 有的CNN提整体特征，我固定高度，对应变长序列

### 标签序列
- 在卷积层之上，双向lstm
    + 预测特征序列的标签分布序列
    + 三重好处
        + RNN描述上下文信息，信息有用，模糊标签好区分 eg:"il"
        + 可以反向传播学习，可以和RNN一起训练
        + RNN可以处理任意长度序列
-传统RNN和lstm
    + 传统RNN有隐层，之前的信息可以用来预测，但由于梯度消失，可以存储的信息长度有限
    + lstm 可以获取很长范围的信息
- 单双向
    + 单项只可以获取一侧信息
    + 双向更好，而且可以叠加变成多层，多层可以抽取高纬信息，在语音识别领域效果好
- 反向传播 BPTT
    + Map-to-Sequence 作为卷积层和递归层的桥梁

### 翻译
- 将RNN的预测转换成标签序列
    - lexicon-free
    - lexicon-based
#### label sequence 可能性
- CTC
    + 不需要知道每一个label对应的位置
    + 用负log似然估计，只需图片和标签序列，不需要字符位置
- 条件概率
    + yt表示概率分布 分布范围在字符加空白
    + CTC 先去重，再去空
    + 可以用forward-backword 算法求解
- lexicon-free 翻译
- lexicon-based 翻译
    + 关联一个lexicon D
    + 基本操作是找到D中条件概率最高的
    + 对于大的lexicon，很耗时
    + lexicon-free得到的结果和真正的结果编辑距离很小
    + 使用最邻近候选搜索 delta是搜索范围
    + 搜索时间 Bk-tree O(log(|D|))
    + 这样可以拓展到大lexicon
    + 在我们的方法中Bk-tree 离线构造 搜索时间取决于delta

### 网络训练
- 最小化 负log似然
- SGD 
    + 翻译层 前向后向传播
    + 递归层 BPTT
- ADADELTA 优化方法
    + 和动量方法相比，不需手动设置学习率
    + 和动量方法相比，更快收敛

## 实验
在情景文本识别和音符识别上做了实验
### 数据集
- 合成数据(Synth)训练一次，在真实测试数据集上测试，不调参
- 测试数据集
    + ICO3
    + IC13
    + IIIT5k
    + SVT
### 实现细节
- VGG-based 区别在于1x2池化，有更宽的宽度使得特征序列变长
- CNN和RNN在一起很难测，batch-normalization对于如此深度有用，加速作用
![table1](pic_resource\table1.PNG)
- 环境
    + Torch7/CUDA
    + transciption层 C++实现 
    + BK-tree C++实现
    + 2.5GHz Inter(R)Xeon(R)E5-2609CPU,64RAM
    + NVIDIA Tesla K40
    + ADADELTA p=0.9
    + 图片 100x32
    + 50小时收敛
    + 测试速度 0.16s/张
    + IC03上 BK-tree delta=3 0.53s/张

### 结果比较
![table2](pic_resource\table2.PNG)
![table3](pic_resource\table3.PNG)

### 音符识别
略

## 结论 
略