# 支持向量机笔记 NOTE

学习资料：《统计学习方法》

支持向量机 *(Support Vector Machines)* 是一个二分类模型，支持向量机的学习方法有三类

1. 线性可分支持向量机 *(linear support vector machine in linearly separabel case)*

2. 线性支持向量机 *(linear support vector machine)*

3. 非线性支持向量机 *(non-lineat suuport vector machin)*

主要区别在于 hold place

## 数学推导

对于支持向量机，有超平面 $W^T \cdot X + b=0$，其中 $W\subseteq R^n, X\subseteq R^n$ ,决策函数
$$
y_i = sign(W^T \cdot X_i+b)
$$

对于线性可分数据集，存在无数个超平面区分正负实例点。但是支持向量机使用间隔最大方法，使得在线性可分数据集上一定存在且唯一的超平面区分正负实例（非线性数据集后面讨论）。为展开讨论最大间隔法，下面简要介绍一些必要的数学结论。

### 知识准备

有超平面
$$
W^T\cdot X +b = 0
$$
>命题1. 证明 $W$ 为超平面的法向量.

在超平面上任取两点 $A,B$ , 注意此处为任意两点，而非实例点。由上可知，必有
$$
W^T \cdot A+b=0\\
W^T \cdot B+b=0
$$
上述两式做差，有
$$
W^T(A-B)=0
$$
记点 $A-B$ 有 $\vec{C}$, 由于 $A,B$为超平面上任意两点，各可知超平面上的任意直线均垂直与 $W$, 得证.

>命题2. 求超平面上一点 P 到超平面的距离


记 *A* 为超平面内一点，则有向量$\vec{AP}$, 据此计算向量 $\vec{AP}$ 与法向量 $\vec{W}$的夹角 $cos<\vec{AP},\vec{W}>=\frac{\vec{AP}\cdot \vec{W}}{| \vec{AP} ||\vec{W}|}$, 已知夹角，可以计算点 P 到超平面上的一点 *d*
$$
d=cos<\vec{AP},\vec{W}>|\vec{AP}|=\frac{\vec{AP}\cdot\vec{W}}{\vec{|W|}}
$$
此处距离有方向，与法向量同侧的点距离为正，与法向量异侧的点为负。

> 命题3. 求几何间隔  

对于给定训练集 $T=\{(x_1,y_1),(x_2,y_2)...(x_N,y_N)\}$，其中 $x_i$为 *n* 维向量。注意此处与上面 $y_i$的区别，此处$y_i\subseteq \{+1,-1\}$，规定超平面关于训练点 $(x_i,y_i)$的几何间隔如下：
$$
\gamma_i=y_i(\frac{w}{||w||}\cdot x_i+\frac{b}{||w||})
$$
注意这里$w\subseteq R^n$, 为权重向量，且与前个命题的距离公式相同，除了多了个$y_i$.  
定义训练集T中所有样本点的最小最小值为
$$
\gamma=\underset{i=1,2,...N}{min} \:\:\:\gamma _i \tag{1}
$$

### 间隔最大化模型建立
支持向量机的基本方法就是求解几何间隔最大的分离超平面，且这个超平面可以正确的划分训练数据集，可以知道，对于线性数据集而言，存在无数个分离超平面可以正确划分训练集中的数据，但是几何间隔最大的超平面是唯一的。在《统计学习方法》中证明了几何间隔最大化的存在性和唯一性。这里建立硬间隔的数学最优化模型
$$
\underset{w,b}{max}\;\;\gamma  \\
s.t.\quad y_i(\frac{w\cdot x_i}{||w||}+\frac{b}{||w||}) \geqslant\gamma 
$$
目标函数 *max*很好理解，对约束条件 *s.t* 解释如下：约束条件的目的是使得线性可分训练集的全部样本均可分类正确，如果没有约束条件，目标条件只能保证找到一个最大的几何间隔超平面，不一定有唯一性且意味着这个超平面不能正确分类。下面逐步对上述模型进行推导. 将 *(1)* 式代入上述优化模型中.有 
$$
\underset{w,b}{max}\;\;\underset{i=1,...N}{min}\;\;y_i(\frac{w}{||w||}\cdot x_i+\frac{b}{||w||}) \\
s.t.\quad y_i(\frac{w\cdot x_i}{||w||}+\frac{b}{||w||}) \geqslant {min}\;\;y_i(\frac{w}{||w||}\cdot x_i+\frac{b}{||w||})
$$
观察到在上述变化后的优化模型中, *min*与 *||w||* 无关，可以将 *||w||* 移出 *min* 操作。有以下变换
$$
\underset{w,b}{max}\;\;\frac{1}{||w||}\underset{i=1,...N}{min}\;\;y_i(w\cdot x_i+b) \\
s.t.\quad y_i(\frac{w\cdot x_i}{||w||}+\frac{b}{||w||}) \geqslant \frac{1}{||w||}\underset{i=1,...N}{min}\;\;y_i(w\cdot x_i+b)
$$
约束条件约去$\frac{1}{||w||}$.这里留意一下 $y_i(w\cdot x_i+b)$, 如果我们成比例的同时放大或者缩小 *(w,b)*, $y_i(w\cdot x_i+b)$也会成比例的放大或者缩小，但是这对目标函数的 *min* 不构成影响，因此 $\underset{i=1,...N}{min}\;\;y_i(w\cdot x_i+b)$ 我们可以取任意值，为了方便计算，我们对 $\underset{i=1,...N}{min}\;\;y_i(w\cdot x_i+b)$ 取值1.综上，优化模型有
$$
\underset{w,b}{max}\frac{1}{||w||} \\
s.t.\quad y_i(w\cdot x_i+b) \geqslant 1
$$
由于 $\underset{w,b}{max}\frac{1}{||w||}$ 等价与 $\underset{w,b}{min}\frac{1}{2}||w||^2$, 故此优化问题转变为：
$$
\underset{w,b}{min}\frac{1}{2}\;||w||^2\\
s.t.\quad y_i(w\cdot x_i+b) \geqslant 1
$$