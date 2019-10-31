#face_rec   NAN 人脸识别复现（现只在YouTube Face 数据集上）   
===============

## 安装环境

>+ python版本：3.6   
>+ pytorch版本：0.4.1
>+ Jupyter(可选)


### 任务
~~~
1.    论文的复现，数据集的划分 (:heavy_check_mark: Accruacy:) 95.5% for Youtube Dataset
2.    光流数据加入  (:clock130:)
3.    额外数据集训练实验 (:clock430:)

~~~


### 文件介绍
+ Dataset.py：负责训练数据的读取
+ Inception.py：Googlenet的架构文件
+ Network.py：NAN网络结构以及一些子网络结构
+ ShowData.ipynb：展示数据和测试Dataset.py的jupyter文件
+ TrainCNN.ipynb:训练CNN网络的jupyter文件(可用来测试训练逻辑)
+ Train.ipynb/train_ver.py:训练网络的已经测试训练逻辑的jupyter文件
+ TrainVerificationTask.ipynb:训练NAN的聚合模块的jupyter文件
+ util.py: 用于计算人脸验证中的TPR,FPR率，用于绘制ROC曲线


### 版本
+ V2.5    更新时间:2019-9-27 （设置聚合模块训练帧数为100帧，达到论文复现效果）
+ V2.0    更新时间:2019-9-26 （加入NAN模型的训练，ROC曲线的绘制，精度的计算）
+ V1.5    更新时间:2019-9-25 （加入CNN模型的单独训练）
+ V1.0    更新时间:2019-9-22 （改动大部分代码,初始化库）


### 数据集：
[Youtube Face](http://www.cs.tau.ac.il/~wolf/ytfaces/)<br>
http://www.cslab.openu.ac.il/download/wolftau/YouTubeFaces.tar.gz
http://www.cslab.openu.ac.il/download/wolftau/frame_images_DB.tar.gz



### GuardSKill
