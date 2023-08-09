import pandas as pd
import numpy as np
import matplotlib

titanic = pd.read_csv('titanic_train.csv')

titanic_2 = titanic.copy()
#由于mean()函数只能用于数值型数据，所以此处要做一个drop处理
titanic_2_drop = titanic_2.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked'], axis=1)
mean = titanic_2_drop.mean()
#print(mean)
titanic_2.fillna(mean,inplace = True )
titanic_2.to_csv("mode2.csv",index= False)
#检验新生成的文件是否还存在缺失值并简单分析新数据
mode2_read = pd.read_csv("mode2.csv")
mode2_read.isnull().sum()
print(mode2_read.describe())