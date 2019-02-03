import numpy as np
import pandas as pd


class DT:
    def __init__(self, data_frame, epsilon):
        """
        :param data_frame:  数据框架，包含类别，特征信息
        :type data_frame:   pandas df
        :param epsilon:     阈值
        :type epsilon:      float
        """
        self.df = data_frame
        self.e = epsilon

    @staticmethod
    def __hd(data_frame):
        """
        计算经验熵，返回经验熵值
        :param data_frame: 数据框架
        :type data_frame:   pandas df
        """
        df = data_frame
        label = df['类别'].values
        label_dir = {}
        for i in range(len(label)):
            label_dir[label[i]] = list(label).count(label[i])

        __TOTAL = len(label)
        __SUM = 0
        for i in label_dir:
            __SUM -= (label_dir[i] / __TOTAL) * np.log2((label_dir[i] / __TOTAL))       # 统计学习方法 P62 5.7
        return __SUM

    def __gain(self, data_frame, feature):
        """
        计算信息增益
        :param feature:     需要计算信息增益的特征
        :type  feature:    str
        :type data_frame:   pandas pd
        :return:
        """
        df = data_frame
        # 依特征feature可能的取值划分子集
        _DATA_TOTAL = len(df[feature])
        _index_dir = {}     # 键：特征 feature 可能的取值， 值：特征对应取值的索引
        for i in range(_DATA_TOTAL):
            _index_dir[df[feature][i]] = []
        for i in range(_DATA_TOTAL):
            _index_dir[df[feature][i]].append(i)
        _SUBSET = -1
        _conditional = 0
        # 计算特征的条件熵
        for i in _index_dir:
            _SUBSET = len(_index_dir[i])        # 子集实例个数
            _conditional += (_SUBSET / _DATA_TOTAL) * self.__hd(df.iloc[_index_dir[i], :])    # 统计学习方法 P62 5.8
        return self.__hd(df) - _conditional           # 统计学习方法 P62 5.9

    def __split(self, data_frame, feature):
        """
        依照 feature 划分数据集，返回字典，键： feature 可能的取值。值：在数据集的索引
        :param data_frame:
        :type data_frame: pandas data frame
        :param feature:
        :type feature:  str
        :return: dir
        """
        df = data_frame
        __DIR = {}
        for i in range(len(df[feature])):
            __DIR[df[feature][i]] = []
        for i in range(len(df[feature])):
            __DIR[df[feature][i]].append(i)     # 可能取值的索引
        return __DIR

    def fit(self, data_frame):
        """

        :return:
        """
        df = data_frame
        label = df.iloc[:, -1]      # 类别
        if len(set(label)) == 1:         # ID3 统计学习方法 P63    step 1
            decision_info = "decision：" + str(label[0])
            return decision_info
        if len(df.columns) == 1:    # step 2
            __dir = {}
            # 初始化字典
            for i in set(label):
                __dir[i] = 0
            for i in range(len(label)):
                if __dir[label[i]] == 0:
                    __dir[label[i]] = 1
                else:
                    __dir[label[i]] += 1
            decision_info = "decision：" + str(label[0])
            return decision_info

        # step 3
        __Ag = None
        __max_g = -1
        for i in df.columns[0:-1]:      # 不包含“类别”
            if self.__gain(df, i) > __max_g:
                __max_g = self.__gain(df, i)
                __Ag = i
        # step 4
        if __max_g < self.e:
            __dir = {}
            # 初始化字典
            for i in set(label):
                __dir[i] = 0
            for i in range(len(label)):
                if __dir[label[i]] == 0:
                    __dir[label[i]] = 1
                else:
                    __dir[label[i]] += 1
            decision_info = "decision：" + str(label[0])
            return decision_info

        # step 5
        new_df = df.drop([__Ag], axis=1)
        __dir = self.__split(df, __Ag)
        __Ag = '属性:'+__Ag
        my = {__Ag: {}}
        for i in __dir:
            _new_df = new_df.iloc[__dir[i], :]
            _new_df = _new_df.reset_index(drop=True)
            i = '值:' + i
            my[__Ag][i] = self.fit(_new_df)
        return my


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
df = pd.DataFrame(data=credit_info, columns=credit_label)
clf = DT(df, 0.01)
mytree = clf.fit(df)
print(mytree)


