# SGD
import pandas as pd
import numpy as np

# from sklearn.metrics import classification_report

data_1 = pd.read_csv('C:/Users/Bin/Desktop/titanic_train.csv')
data = data_1.drop(['Name', 'Ticket'], axis=1)  # data_1 中删除两列 'Name' 和 'Ticket'

data['Age'].fillna(data['Age'].mean(), inplace=True)
# data['Cabin'] = data['Cabin'].notnull().astype(int)舱位转换为0和1，想怎么转自己看
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  # 缺失值用该列的平均值填充，怎么填充自己看
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

np.random.seed(1)  # 设置随机数生成器的种子
indices = np.random.permutation(len(data))
train_size = int(len(data) * 0.7)
train_indices, test_indices = indices[:train_size], indices[train_size:]
x_train, x_test = data.iloc[train_indices, 2:], data.iloc[test_indices, 2:]
y_train, y_test = data.iloc[train_indices, 1], data.iloc[test_indices, 1]

x_train = np.hstack((x_train.values, np.ones((len(x_train), 1))))  # 最后一列添加全为 1 的一列，以便引入偏置项
w = np.zeros((x_train.shape[1], 1))  # 权重向量 w 长度为扩展后的特征矩阵的列数
lr =  # 确定学习率
epochs =  # 确定迭代次数
alpha =  # 确定正则化系数
for i in range(epochs):
    # 每次迭代从训练集中随机选择一个样本进行处理,计算使用了交叉熵损失函数的导数和正则化项的导数
    idx = np.random.randint(len(x_train))
    y_pred = 1 / (1 + np.exp(-np.dot(x_train[idx], w)))
    grad = x_train[idx].reshape(-1, 1) * (y_pred - y_train.iloc[idx]) + alpha * np.sign(w)

    # 更新权重向量 w
    w -= lr * grad

    # 计算当前损失函数的值,通过交叉熵损失函数计算损失值 loss
    y_pred_all = 1 / (1 + np.exp(-np.dot(x_train[idx], w)))
    loss = -np.mean(y_train.iloc[idx] * np.log(y_pred_all) + (1 - y_train.iloc[idx]) * np.log(1 - y_pred_all))
    print(f"Epoch {i + 1}: Loss = {loss:.4f}")

# 使用训练得到的权重向量 w 对测试集特征进行预测，并通过比较预测结果和真实标签计算准确率
x_test = np.hstack((x_test.values, np.ones((len(x_test), 1))))  # 添加全为 1 的一列以匹配训练集的特征矩阵
y_pred = np.round(1 / (1 + np.exp(-np.dot(x_test, w))))  # 预测值
acc = np.mean(y_pred.ravel() == y_test)
print(acc)
