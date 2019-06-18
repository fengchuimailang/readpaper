# EAST

# 摘要
- 简单而强大的处理过程，快速和准确的文本检测。
- 任意方向矩形和四边形的单词或文本行，消除了不必要的中间步骤
- 集中精力设计损失函数和神经网络架构。
- 在ICDAR 2015数据集上，该算法在13.2fps、720p分辨率下，F-score达到0.7820。

# 介绍
- 它只有两个阶段。该管道使用完全卷积网络(FCN)模型
- 步骤1：该模型直接生成单词或文本行级别的预测
- 步骤2：生成的文本预测(可以是旋转矩形或四边形)被发送到非最大抑制以产生最终结果。
- 贡献
    + 提出了一种场景文本检测方法，由全卷积网络和LNMS两个阶段组成。FCN直接生成文本区域，不包括冗余和耗时的中间步骤。
    + 灵活地生成字级或行级预测，其几何形状根据特定的应用使用矩形旋转框或四边形。

# 相关工作
- 传统
    + SWT
    + MSER
    + 文本的局部对称性

- 今年
- Huang首先利用MSER发现候选项，然后利用深度卷积网络作为强分类器来剔除误报。
- Jaderberg以滑动窗口的方式扫描图像，并使用卷积神经网络模型为每个尺度生成密集的热图。
- Jaderberg同时使用CNN和ACF来搜索候选单词，并使用回归进一步细化它们。
- Tian开发了垂直锚，并构建了一个CNN-RNN联合模型来检测水平文本行。
- Zhang提出利用FCN生成热图，利用分量投影进行方向估计。这些方法在标准测试中取得了良好的性能。
- 它们大多由多个阶段和组件组成，如后过滤去除假阳性、候选聚合、行形成和分词等。大量的阶段和组件可能需要进行彻底的调优，导致性能不够理想，并增加了整个过程的处理时间。

![pipeline](pic_resource\pipeline.PNG)

# 方法 
- 预测通道有二
    + 阈值[0,1]的score map， 
    + 几何形状，score map分数代表几何形状置信度
- 两种几何形状，RBOX和QUAD，设计了不同的损失函数。
- 得分超过预定义阈值的几何图形被认为是有效的，NMS后的结果被认为最终输出。

## 网络设计
![structure](pic_resource\structure.PNG)
- 在设计用于文本检测的神经网络时，必须考虑几个因素。
    + 单词区域的大小差异很大：判断大单词的需要神经网络后期的特征此；预测小单词区域的精确几何形状则需要早期的低水平信息。
- HyperNet 满足需求，但是在大型feature map上合并大量通道会显著增加后期的计算开销。
- 使用U-shape。在保持上采样分支较小的同时，逐步合并特征图，计算成本比HyperNet小
- 模型可分解成三个部分
    + 特征提取器主干
    + 特征合并分支
    + 输出层
- 特征提取器主干可以是在ImageNet数据集上预先训练的卷积网络，具有交叉卷积和池化层。四层特征图，记为fi，是从茎中提取的，大小为1/32，1/16，1/8，1/4
分别输入4张图像。
- 采用两种 
    - PVANet
    - VGG16
- 特征合并公式如下图

![feature_mearging_formulation](pic_resource\feature_mearging_formulation.PNG)


- 输出层有不少1乘1卷积把32通道投影到目标通道数
    + 坐标水平 AABB 4通道
    + 有倾斜角的矩形 RBOX 5通道： 4通道 + 1角度
    + 四边形 多通道 

## 标签生成
### Score Map  

![lable_gen](pic_resource\lable_gen.PNG)
- 顺时针顺序
- 相对长度ri
![reference_length](pic_resource\reference_length.PNG)
- 缩小0.3的相对长度

### Geometry Map

- QUAD样式(例如ICDAR 2015)，我们首先生成一个旋转的矩形，它覆盖了面积最小的区域。然后对每个得分为正的像素，计算其到文本框4个边界的距离，并将其放入RBOX ground truth的4个通道中。
- 对于QUAD ground truth, 8通道几何图中每个得分为正的像素点的值为其坐标相对于四边形的四个顶点的偏移量

### 损失函数
！[loss](pic_resource\loss.PNG)
- 设拉姆达为1
![s_m_loss](pic_resource\s_m_loss.PNG)
- 目标对象分布的不均衡,父类比正类多，用类别平衡交叉熵，β参数计算如下
![beta](pic_resource\beta.PNG)
- AABB用 I0U
![AABB_loss](pic_resource\AABB_loss.PNG)
- RBOX 加个塞塔
![sita](pic_resource\sita.PNG)
![RBOX_loss](pic_resource\RBOX_loss.PNG)
- QUAD用smoothL1
![QUAD_loss_1](pic_resource\QUAD_loss_1.PNG)
![QUAD_loss](pic_resource\QUAD_loss.PNG)
![QUAD_loss_2](pic_resource\QUAD_loss_2.PNG)

## 训练
- ADAM 
- 加速计算 采样512x512，minibatch = 24 
- 学习率 开始 1e-3 每27300个minibatch 衰退十分之一，直到1e-5

## Locality-Aware NMS
![LNMS](pic_resource\LNMS.PNG)

# 效果
![result](pic_resource\result.PNG)
![FPS](pic_resource\FPS.PNG)

# 限制
- 长文本不太行
- 垂直文本不太行

# 总结展望
- 精度和效率上都明显优于以往的方法
- 未来可能的研究方向包括:
    + 调整几何公式，使其能够直接检测弯曲文本
    + 将检测器与文本识别器集成
    + 将该思想推广到一般对象检测。