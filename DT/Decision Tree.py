import numpy as np
import pandas as pd

credit_info = [['青年', '否', '否', '一般', '否'],
            ['青年', '否', '否', '好', '否'],
            ['青年', '是', '否', '好', '是'],
            ['青年', '是', '是', '一般', '是'],
            ['青年', '否', '否', '一般', '否'],
            ['中年', '否', '否', '一般', '否'],
            ['中年', '否', '否', '好', '否'],
            ['中年', '是', '是', '好', '是'],
            ['中年', '否', '是', '非常好', '是'],
            ['中年', '否', '是', '非常好', '是'],
            ['老年', '否', '是', '非常好', '是'],
            ['老年', '否', '是', '好', '是'],
            ['老年', '是', '否', '好', '是'],
            ['老年', '是', '否', '非常好', '是'],
            ['老年', '否', '否', '一般', '否'],
            ]
credit_label = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
train_data = pd.DataFrame(credit_info, columns=credit_label)


class Entropy:
    def __init__(self, data, label, attribute_list):
        """
        :type data: np.array
        :type label: np.array
        :type attribute_list: list, list of fields described feature
        """
        self.data = data
        self.label = label
        self.attribute = attribute_list

    @staticmethod
    def empirical(data, label):
        """
        统计学习方法 P61 5.7
        :return: empirical entropy, float
        """
        data = data
        label = label
        sample_num, _ = np.shape(data)
        label_count = {}
        for i in range(sample_num):
            if label[i] not in label_count:
                label_count[label[i]] = 0
            label_count[label[i]] += 1
        h_d = 0
        for label_k in label_count:
            temp = label_count[label_k] / sample_num
            h_d += temp * np.log2(temp)
        return -1 * h_d

    def gain(self, attribute_field):
        """
        calculate empirical conditional entropy of each data-set feature.
        :param attribute_field: describe field
        :type attribute_field: str
        :return: float, information gain
        """
        sample_num, feature_m = np.shape(self.data)
        subset = {}
        index_field = -1
        for i in range(feature_m):
            if attribute_field == self.attribute[i]:
                index_field = i

        for i in range(sample_num):
            if self.data[i][index_field] not in subset:
                subset[self.data[i][index_field]] = [i]
            else:
                subset[self.data[i][index_field]].append(i)
        condition_entropy = 0
        for i in subset:
            index_subset = subset[i]
            subset_data = self.data[index_subset][:]
            subset_label = self.label[index_subset][:]
            condition_entropy += (len(index_subset)/15) * self.empirical(subset_data, subset_label)
        empirical = self.empirical(self.data, self.label)
        return empirical - condition_entropy


if __name__ == '__main__':
    label = train_data['类别'].values
    data = train_data.drop(['类别'], axis=1).values
    clf = Entropy(data, label, credit_label[0:-1])
    print(clf.gain(credit_label[0]))

