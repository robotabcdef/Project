# randomforest
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as snsa

data_train = pd.read_csv(r'\train.csv')
data_test = pd.read_csv(r'\test.csv')  # 位置自己设置

combine = [data_train, data_test]
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

print(dataset.head())
guess_ages = np.zeros((2, 3))  # 全零矩阵
for dataset in combine:  # 遍历数据集中的每个组合，计算非空年龄值的中位数
    for i in range(0, 2):  # 遍历数据集，将空值的年龄根据性别和舱位等级补充上
        for j in range(0, 3):
            age_guess = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()
            guess = age_guess.median()  # 算出该类别下的平均年龄
            guess_ages[i, j] = int(guess / 0.5 + 0.5) * 0.5  # 这个是为了保证平均年龄更加准确
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[
                i, j]  # 将空白的年龄值按类别补上

    dataset['Age'] = dataset['Age'].astype(int)
data_train.head()

data_train['AgeBand'] = pd.cut(data_train['Age'], 5)
data_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',
                                                                                            ascending=True)  # 年龄分为5个组，并计算每个组中的存活率,按照年龄组排序
for dataset in combine:  # 将年龄小于等于16岁的乘客分为组0，年龄在16到32岁的分为组1，以此类推
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age']
data_train.head()

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

data_train.head()

# 分测试集与训练集
X_train = data_train.drop("Survived", axis=1)
Y_train = data_train["Survived"]
X_test = data_test.drop("PassengerId", axis=1).copy()
# 进行随机森林算法
random_forest = RandomForestClassifier(n_estimators=100)  # 创建一个随机森林分类器对象，设定树的数量为100
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)  # yuce
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest  # 预测准确率