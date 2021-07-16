# CV_Relation_Recognition
2020年 《计算机视觉》 编程作业

Data & Model 💁🏻 : https://drive.google.com/drive/folders/1e49dl0-9T8Z34YLfVVcdY_yIP5EwK2j0

##	处理框架

- 模型 ： 3层一维卷积 + 3层全连接层
<img width="711" alt="model" src="https://user-images.githubusercontent.com/44460142/85228574-13f48800-b41f-11ea-87b8-3af42484ba84.png">
- Concatenate 之后输入向量的总维度 ： [1 * 2211]
<img width="243" alt="concatenate" src="https://user-images.githubusercontent.com/44460142/85228575-148d1e80-b41f-11ea-8578-822b3f44411a.png">

##  关键技术
根据predicates.json里的关系类别，发现出现方向的关系（next to，under，on the top of，on the right of等）比较多了。所以输入的向量当中，添加了包括方向性的特征。计算方向性的特征是从主语物体的bounding box中心到宾语物体的bounding box中心的方向。首先计算两个点之间的角度之后换成到代表16个方向的One-hot encoding。但是生成了方向性的特征之后，发现大部分数据的方向是西方或者东方。

<img width="204" alt="example" src="https://user-images.githubusercontent.com/44460142/85228569-0f2fd400-b41f-11ea-93ce-f97f034d3e9e.png">

为了提高具体的辨别力，添加了还有一个特征包括主语和宾语bounding box的大小关联的信息。考虑了"sit next to", "stand next to"，"taller than"，"lying on"等的关系类别会跟主语和宾语bbox的大小和比率有相关，计算了主语和宾语之间的面积对比（主语的面积比宾语的面积几倍大）。而且添加了每个bbox的横纵比率信息（比如横向长度比纵向长度长两倍以上，横向长度和纵向长度是大概1:1像正方形等）。这种特征也表示为One-hot encoding

<code>

* 各Index的值为1的条件：

  Index[0] : 满足主语bbox的x长度 > 主语bbox的y长度*2

  Index[1] : 满足主语bbox的y长度*2 >主语bbox的x长度 > 主语bbox的y长度

  Index[2] : 满足主语bbox的x长度*2 >主语bbox的y长度 > 主语bbox的x长度

  Index[3] : 满足主语bbox的y长度 > 主语bbox的x长度*2

  Index[4] : 满足宾语bbox的x长度 > 宾语bbox的y长度*2

  Index[5] : 满足宾语bbox的y长度*2 >宾语bbox的x长度 > 宾语bbox的y长度

  Index[6] : 满足宾语bbox的x长度*2 >宾语bbox的y长度 > 宾语bbox的x长度

  Index[7] : 满足宾语bbox的y长度 > 宾语bbox的x长度*2

* 最后Index的值是主语和宾语bbox的面积比较值

  Index[8] : 主语bbox的面积 / 宾语bbox的面积

</code>

##  实验

* 评价指标 ： recall = 被正确识别的关系标签总数 / 标注的关系标签总数

一个测试样本包含：物体对<主语，宾语>和 k个关系类别（k不是固定值，k大于等于1）。在评估阶段，一个测试样本仅接受置信度最高的k个预测关系类别。
recall = 被正确识别的关系标签总数 / 标注的关系标签总数

例如：假设关系总数为10：[A,B,C,D,E,F,G,H,I,J]；一个测试样本的关系类别标签为[1,1,1,0,0,0,0,0,0,0]，表示其关系类别为[A,B,C]，k=3；模型为该样本预测得到的置信度向量为[0.9,0.8,0.1,0.0,0.0,0.0,0.5,0.2,0.0,0.0]，由于k=3，所以预测的关系标签为[1,1,0,0,0,0,1,0,0,0]，即[A,B,G]，其中正确的是[A,B]；recall = len([A,B]) / len([A,B,C])。


* 测试集 ： val_image , annotations_val.json

* Hyperparameter :  Learning_rate : 0.5, train_batch_size : 32, test_batch_size : 32
  
    


##### SI : 主语id

##### OI : 宾语id

##### H : 颜色直方图 + 梯度直方图特征

##### D : 方向特征

##### R : 大小和比率特征

<img width="674" alt="experiment" src="https://user-images.githubusercontent.com/44460142/85228573-135bf180-b41f-11ea-9e5c-084dabf2914f.png">

##  结论

1. 在输入向量中添加了颜色直方图和梯度直方图特征之后，recall率提高了一些。但是可以看到大小和比率特征对整个模型影响不大。为了达到更高的recall率, 今后打算使用图片里提取的特征来训练模型。从整个图片上只切掉每个主语和宾语部分之后，通过VGG-16模型会提取每个主语和宾语图片的特征。Bounding box关联的特征是在比较短的时间内可以进行提取特征和训练模型，但是可以看得出对提高recall率没有产生很大的影响。

