# ***Logistic Regression 笔记***
## **环境**
python3.6
>numpy  
>pandas  
>scikit-learn  


---
## ***Binomial Logistic Regression 数学推导***
有 *Binomial Logistic Regression Classifier*  
![img](https://latex.codecogs.com/gif.latex?P(Y=1|x)=\frac{exp(\vec{X}\cdot&space;\vec{W}&plus;b)}{1&plus;exp(\vec{X}\cdot&space;\vec{W}&plus;b)})  


![img](https://latex.codecogs.com/gif.latex?P(Y=0|x)=\frac{1}{1&plus;exp(\vec{X}\cdot&space;\vec{W}&plus;b)})  

*W* 为权重向量， *b* 为偏置，可以发现 *P(Y=0 | x)* *P(Y=1 | x)* 相加等于1，通过概率大小判断所属类别。训练的本质为找到一个合适的 *W* 权重向量，是的似然函数最大，或者损失函数最小

有似然函数

![img](https://latex.codecogs.com/gif.latex?Likelihood(w)=\prod&space;^{N}_{i=1}[\pi{(x_i)}]^{y_i}[1-\pi{(x_i)}]^{1-y_{i}})  

对数似然函数

![img](https://latex.codecogs.com/gif.latex?Likelihood(w)=\sum^N_{i=1}[y_iw\cdot&space;x-log(1&plus;exp(w\cdot&space;x))])  

使用梯度上升法 *(Ggradient ascent)* 训练权重向量 *W*.

推导得到梯度

![img](https://latex.codecogs.com/gif.latex?\bigtriangledown&space;L(w)=\sum^N_{i=1}\vec{X_i}[y_i-\frac{exp(X_i\cdot&space;W)}{1&plus;exp(X_i\cdot&space;W)}])  

利用梯度上升法可以更新 
![img](https://latex.codecogs.com/gif.latex?W^*=W&space;&plus;&space;\eta\:W)  多次迭代可以得到似然函数的最大值。


---
## 程序实现
1. 准备数据，30% 为测试数据集
2. 梯度上升法训练数据，设置迭代次数和步长
3. 验证测试数据集


---
## 实例应用
### 鸢尾花数据集分类
### 心脏病数据集分类
数据处理部分参考[github](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3)
