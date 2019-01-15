# ***Perceptron*** 笔记
## 算法简介
对于输入空间<img src="https://latex.codecogs.com/png.latex?\vec{X}=\subseteq&space;R^n" title="\vec{X}=\subseteq R^n" />, 输出<img src="https://latex.codecogs.com/gif.latex?\vec{Y}=\left&space;\{&space;&plus;1,-1&space;\right&space;\}" title="\vec{Y}=\left \{ +1,-1 \right \}" />, 用以下模型分类
  
  <img src="https://latex.codecogs.com/gif.latex?f(x)=sign(x\cdot&space;w&plus;b)" title="f(x)=sign(x\cdot w+b)" />   

 其中<img src="https://latex.codecogs.com/gif.latex?w\subseteq&space;R^{n}" title="w\subseteq R^{n}" />为权重向量*weight vector*,  *x* 为输入空间的实例, <img src="https://latex.codecogs.com/gif.latex?b&space;\subseteq&space;R" title="b \subseteq R" />为偏置*bias*. 函数 *sgin(x)* 定义如下:  
<img src="https://latex.codecogs.com/gif.latex?sign(x)=&space;\left\{\begin{matrix}&space;&plus;1&space;&{x\geqslant&space;0}&space;&&space;\\&space;-1&space;&{x<&space;0}&space;&&space;\end{matrix}\right." title="sign(x)= \left\{\begin{matrix} +1 &{x\geqslant 0} & \\ -1 &{x< 0} & \end{matrix}\right." />

可知感知机为线性分类器 *Linear Classcifier*.  
当  <img src="https://latex.codecogs.com/png.latex?x&space;\cdot&space;w&plus;b&space;=0" title="x \cdot w+b =0" />  可以视为分离 <img src="https://latex.codecogs.com/png.latex?&plus;1,-1" title="+1,-1" /> 两类的超平面 *Separating Hyerplain*. 在二维输入空间中，*Separating Hyperplane*为一条直线，如下图所示![img](../img/perceptron_1.jpg).

<img src="https://latex.codecogs.com/png.latex?\vec{w}" title="\vec{w}" />  为直线 <img src="https://latex.codecogs.com/png.latex?x\cdot&space;w&plus;b" title="x\cdot w+b" /> 的法向量.  
在高维空间，<img src="https://latex.codecogs.com/png.latex?x\cdot&space;w&plus;b&space;=&space;0" title="x\cdot w+b = 0" /> 为分离超平面 *Separating Hyperplane* 的法向量. 以下为证明过程:  

---
设点 *A、B* 为分离超平面任意的两点，有 <img src="https://latex.codecogs.com/png.latex?A\cdot&space;w&plus;b=0" title="A\cdot w+b=0" /> 及 <img src="https://latex.codecogs.com/png.latex?B\cdot&space;w&space;&plus;b=0" title="B\cdot w +b=0" /> , 两式做差，则有 <img src="https://latex.codecogs.com/png.latex?\vec{AB}\cdot&space;w=0" title="\vec{AB}\cdot w=0" /> , 可得出结论 *w* 为分离超平面的法向量。 证毕！  

---
假设训练数据集为线性可分数据集 *(Linearly Separable Data-set)*, 则感知机的任务是找到一个超平面将正负实例点完全正确的分离. 为此我们需要找到一个函数对分类情况进行衡量，且这个函数是关于参数 *w* 和 *b* 的连续可导函数，目的是为了对这个函数优化. 可定义下面函数：  

<img src="https://latex.codecogs.com/png.latex?L(w,b)&space;=&space;-\sum_{X_{i}\in&space;M}y_{i}(w\cdot&space;x&plus;b)" title="L(w,b) = -\sum_{X_{i}\in M}y_{i}(w\cdot x+b)" />  

其中 *M* 为误分类点集合, 具体推导过程见《统计学习方法》一书。下面使用梯度下降法优化 *Loss* 函数, 参数更新方式为：

<img src="https://latex.codecogs.com/png.latex?w^*=w&plus;\eta&space;y_{i}x_{i}" title="w^*=w+\eta y_{i}x_{i}" />\
<img src="https://latex.codecogs.com/png.latex?b^*=b&plus;\eta&space;y_{i}" title="b^*=b+\eta y_{i}" /> 
<img src="https://latex.codecogs.com/png.latex?\eta" title="\eta" /> 为学习率 *(Learning Rate)*.
  
2019-01-15 13:11 

