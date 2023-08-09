import pandas as pd
import numpy as np
import matplotlib

titanic = pd.read_csv('titanic_train.csv')

titanic_5 = titanic.copy()
#将缺失值删去
numeric_cols = titanic_5.select_dtypes(include=['float64']).columns
#定义欧氏距离
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
#定义一个knn函数
def knn_impute(titanic, idx, k):
    non_missing = titanic.dropna()
    distances = non_missing.apply(lambda x: distance(x, titanic.loc[idx]))
    nearest = non_missing.iloc[distances.argsort()[:k]]
    impute_value = nearest.mean()
    return impute_value
#利用索引逐一用knn替换缺失值
for col in numeric_cols:
    missing_idx = titanic_5[col][titanic_5[col].isnull()].index
    for idx in missing_idx:
        impute_value = knn_impute(titanic_5[col], idx, k=5)
        titanic_5[col][idx] = impute_value
titanic_5.to_csv('mode5.csv', index=False)
#检验新生成的文件是否还存在缺失值并简单分析新数据
mode5_read = pd.read_csv("mode5.csv")
mode5_read.isnull().sum()
print(mode5_read.describe())