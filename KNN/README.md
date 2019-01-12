KNN（K-Nearest Neighors) K近邻算法实现
=====================================
使用MATLAB实现K近邻算法，一共三个实例。  
1. 鸢尾花数据集分类(Iris Data Set)  
2. 海伦约会对象分类  
3. 手写数字集（Minist）分类

## 算法简介
算法核心思想是如果一个样本 (test sample) 在特征空间的K个最相邻的样本中 (train sample) 的大多数属于某一类别，则该样本属于这个类别。最近邻使用距离衡量，其中距离有：
1. 欧几里得距离 Eucliden Distance 
2. 曼哈顿距离 Manhattan Distance
3. 切比雪夫距离 Chebyshev distance
不同距离的使用视不同的数据集以及需求而定。
---
## 鸢尾花数据集 Iris Data Set
### 数据集简介
该数据集共有150个样本，每个样本有4类特征 *(feature)* ，分别是：
1. 花萼长度  *Sepal Length*
2. 花萼宽度  *Sepal Width*
3. 花瓣长度  *Petal Length*
4. 花瓣宽度  *Petal Width*  

依据四个不同的特征，150个样本被分为三类  
1. *Iris Setosa* 
2. *Iris Versicolour*
3. *Iris Virginica*

其中前两类是线性可分离，后两类不可线性分离

### 数据准备
各类别取40各样本，一共取120各样本作为已分类数据。剩下的30个样本视为为分类样本。*KNN*任务就是将这30个样本进行分类。

### 基本流程
计算待分类样本与已分类各个样本的欧几里得距离  *Eucliden Distance* , 得到 120 * 30 的距离矩阵，计算每个样本点到同一类别的平均距离 *Mean Distance* , 待分类样本距离哪一类别平均距离最小，即可判定样本属于哪一类。

---
## 海伦约会对象数据集
### **数据简介**
该数据集来源于 [https://github.com/apachecn/AiLearning](https://github.com/apachecn/AiLearning), 以文本文件 [dateingTestSet.txt](https://github.com/HouCQUPT/BasicML/blob/master/KNN/datingTestSet.txt)存在。  
该数据集共有1000个样本，每个样本有3个特征，分别是：
1. 每年获得的飞行常客里程数
2. 玩视频游戏所耗时间百分比
3. 每周消费的冰淇淋公升数

依上述特征1000个样本被分为三类,分别为
1. 不喜欢的人
2. 魅力一般的人
3. 极具魅力的人

3-D散点图如下：  
![3-D三点图](https://github.com/HouCQUPT/BasicML/blob/master/img/HalenDate_3D.jpg?raw=true)

### **数据准备**
1000个样本，随机选择100个作为测试数据，使用*MATLAB函数：randperm(1000,100)* 实现随机选取， 剩下的900个作为已分类数据。由于该数据集中的三个特征数值相差较大，因此进行归一化 *Normalization* ,归一化公式为：

![d](https://latex.codecogs.com/gif.latex?y=\frac{x-min}{max&space;-&space;min})  

### **基本流程**
计算测试数据到已分类数据每个样本之间的欧几里得距离，形成一个900 * 100 的距离矩阵，按列降序排列该矩阵。取排序好的前 *K* 行，通过排序生成的序列 *Index* 统计前 *K* 出现的个类别个数。出现类别最多的既是该待测样本所属的类别。

### **结果**
*K* 取5时正确在*90%*附近，偶尔下降到*85%*

---
## **Minist 手写数字数据集**
### **数据准备**
将每个32 * 32 的图片转换为 1024 维向量，使用欧式距离计算测试样本与已分类样本的距离。  
### **结果**
由于样本量较大，使用遍历方法比较耗时。在 *K = 5* 以及 *K = 10* 时，测试结果均达到*97%*.




2019-01-13 1:00
