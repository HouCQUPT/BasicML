import pandas as pd


class DATA:
    def __init__(self):
        self.data = None
        self.label = None

    def normaldata(self):
        origin_data = pd.read_csv("Heart.csv")
        normal = [1, 4, 5, 8, 10, 12, 11]   # 标准化处理
        one_hot = [3, 7, 13]                # one_hot编码
        binary = [14]                       # 原始类别为1的依然为1类，原始为2的变为0类
        key = origin_data.keys()
        for i in range(1, len(key)):
            if i in normal:
                origin_data[key[i-1]] = (origin_data[key[i-1]]-origin_data[key[i-1]].mean()) / origin_data[key[i-1]].std()
            elif i in one_hot:
                temp = pd.get_dummies(origin_data[key[i-1]], prefix=[key[i-1]])
                origin_data = pd.concat([origin_data, temp], axis=1)
            elif i in binary:
                origin_data[key[i-1]] = [1 if inum == 2 else 0 for inum in origin_data[key[i-1]]]

            data_label = origin_data.values
            self.data = data_label[:, :-1]
            self.label = data_label[:, -1]


if __name__ == '__main__':
    A = DATA()
    A.normaldata()
    B = A.data
    print(B)
