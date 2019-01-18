# ***朴素贝叶斯 Naive Bayes***
## ***Python 实现Iris分类***
python3.6  
依赖库
>numpy  
>pandas  
>scikit-learn  

使用 *Gauss Naive Bayes Classifier* 分类，与 *scikit-learn* 的 *Gaussian Navie Bayes* 进行对比，分类效果完全一致

## ***Gauss Naive Bayes 理论推导***
有 *Naive Bayes Classifier:*  

![img](https://latex.codecogs.com/gif.latex?y=arg\;\underset{c_k}{max}P(Y=c_k)P(X=x|Y=c_k))  

先验概率
![img](https://latex.codecogs.com/gif.latex?P(Y=c_k))  的计算与 *Naive Bayes* 无异，下面讨论条件概率的计算. *Gauss Naive Bayes* 假设每一维服从正态分布 *Gaussian Distribution* , 因此需要计算出在每一类别下的每一维的均值与标准差.  
条件概率有下列公式计算

![img](https://latex.codecogs.com/gif.latex?P(x^{(i)}|y_k)=\frac{1}{\sqrt{2\pi}\sigma_{ki}}e^{-\frac{({x^{(i)}-u_{ki}})^2}{2\sigma_{ki}^2}})


![img](https://latex.codecogs.com/gif.latex?\sigma_{ki}\:u_{ki}) 
指第 *k* 下第 *j* 维的均值与标准差.
### **步骤**
1. 准备数据, 获得 *Iris* 数据的前两类
2. 计算先验概率

```python
    def prior_p(self, trainlabel):
        """
        Be used to calculating prior probability
        :param trainlabel:   train label set
        :return:        prior probability, list
        """
        type_list = [0, 0]
        for i in range(len(trainlabel)):
            if trainlabel[i] == 0:
                type_list[0] = type_list[0] + 1
            else:
                type_list[1] = type_list[1] + 1
        self.probability = [ele / len(trainlabel) for ele in type_list]
```
3. 构造两个矩阵，分别存储 *0/1* 类的训练数据
```python
        i_zero = 0
        i_one = 0
        for ii in range(len(trainlabel)):
            if trainlabel[ii] == 0:
                type_zero_matrix[i_zero][:] = traindata[ii][:]
                i_zero += 1
            else:
                type_one_matrix[i_one][:] = traindata[ii][:]
                i_one += 1
```
5. 构造四个向量，分别存储 *0/1* 类的各维度的均值和方差

```python
        self.type_zero_mean = np.mean(type_zero_matrix, axis=0)
        self.type_zero_std = np.std(type_zero_matrix, axis=0)
        self.type_one_mean = np.mean(type_one_matrix, axis=0)
        self.type_one_std = np.std(type_one_matrix, axis=0)
```
6. 至此，训练过程结束，依照 *Gauss Naive Classifier* 对数据进行分类
```python
        for ii in range(4):
            type_zero_feature_p[ii] = self.gauss_p(input[ii], self.type_zero_mean[ii], self.type_zero_std[ii])
            type_one_feature_p[ii] = self.gauss_p(input[ii], self.type_one_mean[ii], self.type_one_std[ii])

        type_zero_p = self.probability[0] * reduce(lambda x, y: x*y, type_zero_feature_p)
        type_one_p = self.probability[1] * reduce(lambda x, y: x*y, type_one_feature_p)
``` 


2019-01-18 10：13
