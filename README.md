# SiameseNetwork


## 孪生网络简介
### 应用
一种相似性度量方法，可用于当类别数多，但每个类别的样本数量少的情况，如单样本分类。
### 提出原因
1. 传统的用于区分的分类方法是需要确切的知道每个样本属于哪个类，需要针对每个样本有确切的标签。而且相对来说标签的数量是不会太多的。当类别数量过多，每个类别的样本数量又相对较少的情况下，这些方法就不那么适用了。
2. siamese网络从数据中去学习一个相似性度量，用这个学习出来的度量去比较和匹配新的未知类别的样本。这个方法能被应用于那些类别数多或者整个训练样本无法用于之前方法训练的分类问题。
### 思想总结
1. 左右为两个相同的网络结构，共享权值，对输出进行了距离度量，可以是l1、l2等。
2. 区别于传统网络的输入特色在于输入一对样本，且样本对分为同属一类和不属于同一类两种。在示例代码具体体现在get_batch函数中，pairs[0]存放某一样本，pairs[1]的一半存放与pairs[0]相同的样本，一半存放不同的样本。同一类的标签为0，不同类标签为1。
3. 针对输入的样本对是否来自同一个类别设计了损失函数，损失函数形式有点类似交叉熵损失。最后使用获得的损失函数，使用梯度反传去更新两个网络共享的权值W。因为只有属于同一类和不属于同一类两种情况，套用二分类逻辑回归损失函数（binary_crossentropy)

## 改进方法
### 数据增强
#### 概念
所谓数据增强，就是采用在原有数据上随机增加抖动和扰动，从而生成新的训练样本，新样本的标签和原始数据相同。
#### 意义
* 深度学习参数多，模型复杂，需要大量的数据才能达到比较好的效果，而现实中有标注的数据往往是很珍贵的，所以想要利用好当前数据很有必要做数据增强。
* 增加数据扰动，提高模型泛化能力。
#### 方法
对于图片数据可采用平移、旋转、缩放、裁剪、切变、水平（垂直）翻转等方法。

### 算法调参
#### 学习率
在训练过程中，一般根据训练轮数设置动态变化的学习率。
* 刚开始训练时：学习率以 0.01 ~ 0.001 为宜。
* 一定轮数过后：逐渐减缓。
* 接近训练结束：学习速率的衰减应该在100倍以上。

Note： 
如果是 迁移学习 ，由于模型已在原始数据上收敛，此时应设置较小学习率 (≤10−4≤10−4) 在新数据上进行 微调 。

#### 批次和周期
batch size大小会决定最后的梯度，以及更新权重的频度。一个周期(epoch)指的是神经网络看一遍全部训练数据的过程。
* 尝试选取与训练数据同大小的batch size。  
* 尝试选取1作为batch size（在线学习（online learning））。  
* 尝试用格点搜索不同的小的batch size（8，16，32，…）。  
* 分别尝试训练少量周期和大量周期。  
* 考虑一个接近无穷的周期值(持续训练)，去记录到目前为止能得到的最佳的模型。

#### 正则化
正则化是一个避免模型在训练集上过拟合的好方法。
* 在输入、隐藏层和输出层间添加dropout方法
* 尝试其他更传统的神经网络正则化方法。例如:权重衰减（Weight decay）去惩罚大的权重、   激活约束（Activation constraint）去惩罚大的激活值
* 试验惩罚不同的方面，或者使用不同种类的惩罚/正则化（L1, L2, 或者二者同时）

#### 优化和损失
尝试不同的优化器，如随机梯度下降（sgd）、Adam、RMSprop等，并尝试调整优化器中相关参数。

#### 早停法
* 一旦训练过程中出现(验证集)性能开始下降，你可以停止训练与学习。这可以节省很多时间，而且甚至可以让你使用更详尽的重采样方法来评估你的模型的性能。
* 早停法是一种用来避免模型在训练数据上的过拟合的正则化方式，它需要你监测模型在训练集以及验证集上每一轮的效果。一旦验证集上的模型性能开始下降，训练就可以停止。
* 如果某个条件满足（衡量准确率的损失），你还可以设置检查点(Checkpointing)来储存模型，使得模型能够继续学习。检查点使你能够早停而非真正的停止训练，因此在最后，你将有一些模型可供选择。

#### 其他方法
* 分层学习率/梯度调整
* 贝叶斯超参数优化
* 调整网络拓扑结构
* ......

### 我所采用过的方法
* 用keras中ImageDataGenerator类对示例数据作数据增强

```
train_datagen = ImageDataGenerator(
                                  shear_range=0.2,
                                  zoom_range=0.3,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  rotation_range=30,
                                  horizontal_flip=True,
                                  fill_mode='nearest'
                                  )
                                  
(inputs,targets)=loader.get_batch(batch_size)
history = model.fit_generator(train_datagen.flow(inputs,targets,batch_size=batch_size),steps_per_epoch=len(inputs) / batch_size, verbose=0,epochs=1)
```
* 尝试SGD优化器

```
sgd = keras.optimizers.SGD(lr=0.002, decay=0.002 / epochs, momentum=0.9, nesterov=True)
```
* 尝试批量归一化

```
model.add(BatchNormalization())
```
* 尝试增大batch_size为32、64、128

#### 结果
* 采用百度AIStudio平台训练，训练4000次迭代，按示例代码运行val_acc最高有80.0%。
* 采用以上前三种方法，在刚开始训练的时候，准确率会下降到约20%，4000次迭代之后准确率相较原代码偏低2-5%左右。
* 在增大batch_size之后，val_acc有所提升，batch_size为64、128取得的效果最好。
* 数据存在的问题：示例代码迭代20000次，平均val_acc有86.5；而从原论文来看，大约要有90000次迭代才会收敛，所以只使用4000次迭代得到的val_acc会偏低很多。
* PS：某平台速度真的太慢了...

## 参考资料
* keras官方文档：<html>https://keras.io/</html>  
* 提高深度学习性能的四种方式：<html>http://www.sohu.com/a/115111584_494939</html>  
* Siamese Network理解：<html>https://blog.csdn.net/sxf1061926959/article/details/54836696</html>  
* 单样本学习（One shot learning）和孪生网络（Siamese Network）简介：<html>https://blog.csdn.net/weixin_41481113/article/details/88415546</html>
* 分层的学习率和自由的梯度：<html>https://kexue.fm/archives/6418</html>
* 优化器总结与比较：<html>https://blog.csdn.net/weixin_40170902/article/details/80092628</html>
* 常用激活函数（激励函数）理解与总结：<html>https://blog.csdn.net/tyhj_sf/article/details/79932893</html>
* 实现Keras搭建模型借助sklearn的网格搜索模块自动搜索最优超参数：<html>https://blog.csdn.net/qq_34514046/article/details/86531517</html>
* 孪生网络单样本学习其他例子：https://github.com/Goldesel23/Siamese-Networks-for-One-Shot-Learning
* L2正则：https://zhuanlan.zhihu.com/p/40814046
* ......


