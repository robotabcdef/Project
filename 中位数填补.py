import pandas as pd
import numpy as np
import matplotlib

titanic = pd.read_csv('titanic_train.csv')

titanic_4 = titanic.copy()
titanic_4.fillna(0, inplace=True)
titanic_4.to_csv('mode4.csv', index=False)
#检验新生成的文件是否还存在缺失值并简单分析新数据
mode4_read = pd.read_csv("mode4.csv")
mode4_read.isnull().sum()
print(mode4_read.describe())