"""
Binomial Logistic Regression Model
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class BLogisticReg:
    """
    Binomial Logistic Regression Classifier
    """
    def __init__(self, iterations, rate):
        self.max_iter = iterations
        self.stu_rate = rate
        self.weight = np.random.random([4, 1])

    def model_train(self, input_data, input_label):
        """
        the purpose of this function is training model by gradient ascent
        :param input_data:      train data
                                type: np.ndarray
        :param input_label:     train label
                                type: np.ndarray
        :return: weight vector
                 typo: np.ndarray
        """
        m, n = np.shape(input_data)     # m is number of train instance, n is number of dimension
        _iter = 0
        bias = np.ones([m, 1])
        weight = np.random.random([n + 1, 1])
        input_data = np.concatenate((input_data, bias), axis=1)
        while _iter < self.max_iter:    # iteration
            _iter += 1
            deri_weight_mat = np.zeros([m, n+1])
            for i in range(m):
                exp_wx = np.exp(np.dot(input_data[i], weight))
                deri_weight_mat[i][:] = input_data[i] * (input_label[i] - (exp_wx / (1+exp_wx)))
            deri_w = sum(deri_weight_mat)

            # update weight vector
            a = deri_w.reshape(deri_w.shape[0], 1)      # transpose one-dimensional vector
            weight = weight + self.stu_rate * a
            self.weight = weight
        print("权重向量：")
        print(self.weight)

    def predict(self, data, label):
        """
        :param data:        test data set
                            type: np.ndarray
        :param label:       label of data set
                            type: np.ndarray
        :return:
        """
        data = np.concatenate((data, [1]))
        exp_wx = np.exp(np.dot(data, self.weight))
        p1 = exp_wx / (1+exp_wx)
        p0 = 1/(1+exp_wx)
        if p0 > p1:
            print("预测\t{0}\t类\t实际\t{1}\t类\n".format(0, int(label)))
            return 0
        else:
            print("预测\t{0}\t类\t实际\t{1}\t类\n".format(1, int(label)))
            return 1


if __name__ == '__main__':
    # loading data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    irisdata = np.array(df.iloc[:100, :])  # instance of type 1 and type 2
    x = irisdata[:, :-1]  # instance feature
    irislabel = irisdata[:, -1]  # type

    # splitting data set into 2 data sets, test data and training set
    data_train, data_test, label_train, label_test = train_test_split(x, irislabel, test_size=0.3)

    model = BLogisticReg(100, 0.01)
    model.model_train(data_train, label_train)
    error_count = 0
    for i in range(len(label_test)):
        flower_species = model.predict(data_test[i], label_test[i])
        if flower_species != int(label_test[i]):
            error_count += 1
    print("分类准确率：{0}%".format((len(label_test)-error_count)/len(label_test)*100))
