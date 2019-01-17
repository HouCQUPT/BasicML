import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math

from functools import reduce


# Loading Iris Data-set
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
data = np.array(df.iloc[:100, :])       # instance of type 1 and type 2
x = data[:, :-1]                        # instance feature
label = data[:, -1]                     # type

# splitting data set into 2 data sets, test data and training set
x_train, x_test, label_train, label_test = train_test_split(x, label, test_size=0.3)


class GaussNB:
    """
    Gauss Naive Bayes
    """
    def __init__(self):
        self.model = None
        self.type_zero_mean = np.ndarray([1, 4])
        self.type_zero_std, self.type_one_mean, self.type_one_std = np.ndarray([1, 4]), np.ndarray([1, 4]), np.ndarray([1, 4])
        self.probability = [0, 0]        # Prior Probability

    @staticmethod
    def gauss_p(value, mean_value, std):
        """
        :param value:       input argument, type: float
        :param mean_value:  mean value
        :param std:         standard deviation
        :return:            gauss probability value
        """
        exponent = math.exp(-(math.pow(value - mean_value, 2) / (2 * math.pow(std, 2))))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

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

    def conditional_p(self, traindata, trainlabel):
        """
        the function is used to calculate mean value and standard deviation of each feature in each type
        :param traindata:   np.ndarray
        :param trainlabel:  np.ndarray
        :return:    mean_value_matrix and stdev_matrix
        """
        count = 0       # store number of type 1
        for ii in range(len(trainlabel)):
            if trainlabel[ii] == 0:
                count += 1

        # constructing two matrix
        type_zero_matrix = np.ndarray([count, 4])     # store all feature of type zero
        type_one_matrix = np.ndarray([len(trainlabel) - count, 4])    # store all feature of type one

        # getting matrix
        i_zero = 0
        i_one = 0
        for ii in range(len(trainlabel)):
            if trainlabel[ii] == 0:
                type_zero_matrix[i_zero][:] = traindata[ii][:]
                i_zero += 1
            else:
                type_one_matrix[i_one][:] = traindata[ii][:]
                i_one += 1

        # calculating mean value and standard deviation
        self.type_zero_mean = np.mean(type_zero_matrix, axis=0)
        self.type_zero_std = np.std(type_zero_matrix, axis=0)
        self.type_one_mean = np.mean(type_one_matrix, axis=0)
        self.type_one_std = np.std(type_one_matrix, axis=0)

    def forecast(self, input):
        """
        Be used to forecast type by Gauss Naive Bayes Classifier
        :param: np.ndarray
        :return: 0 or 1
        """
        type_zero_feature_p = [0, 0, 0, 0]
        type_one_feature_p = [0, 0, 0, 0]
        for ii in range(4):
            type_zero_feature_p[ii] = self.gauss_p(input[ii], self.type_zero_mean[ii], self.type_zero_std[ii])
            type_one_feature_p[ii] = self.gauss_p(input[ii], self.type_one_mean[ii], self.type_one_std[ii])

        type_zero_p = self.probability[0] * reduce(lambda x, y: x*y, type_zero_feature_p)
        type_one_p = self.probability[1] * reduce(lambda x, y: x*y, type_one_feature_p)
        if type_zero_p > type_one_p:
            return print("预测类别：0")
        else:
            return print("预测类别：1")


if __name__ == '__main__':
    model = GaussNB()
    model.conditional_p(x_train, label_train)
    model.prior_p(label_train)
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    clf.fit(x_train, label_train)
    for ii in range(len(label_test)):
        model.forecast(x_test[ii])
        print("S-K预测:", int(clf.predict([x_test[ii]])[0]))
        print("-------------------------------------------------")
