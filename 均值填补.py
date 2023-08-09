import pandas as pd
import numpy as np
import matplotlib

titanic = pd.read_csv('titanic_train.csv')

titanic_3 = titanic.copy()
#由于median()函数只能用于数值型数据，所以此处要做一个drop处理
titanic_3_drop = titanic_3.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked'], axis=1)
median = titanic_3_drop.median()
print(median)
titanic_3.fillna(median,inplace = True)
titanic_3.to_csv("mode3.csv",index= False)
#检验新生成的文件是否还存在缺失值并简单分析新数据
mode3_read = pd.read_csv("mode3.csv")
mode3_read.isnull().sum()
print(mode3_read.describe())