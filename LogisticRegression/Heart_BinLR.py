from Heart_data import DATA
from sklearn.model_selection import train_test_split
from BinLR import BLogisticReg

# load data and label
heart = DATA()
heart.normaldata()
heart_label = heart.label
heart_data = heart.data

x_train, x_test, y_train, y_test = train_test_split(heart_data, heart_label, test_size=0.3)
LogisticR = BLogisticReg(1000, 0.005)
LogisticR.model_train(x_train, y_train, 0.5)
error = 0

for i in range(len(y_test)):
    heart_p = LogisticR.predict(x_test[i], y_test[i])
    if heart_p != int(y_test[i]):
        error += 1
print("分类准确率：{0}%".format((len(y_test) - error) / len(y_test) * 100))
