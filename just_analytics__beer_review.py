# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:23:50 2018

@author: mudit
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
path = "C:/Users/mudit/Desktop/Data Analytics/Just_Analytics_Test/Beer_Review/"
dataset = pd.read_csv(path + 'Beer Review.csv')

#1.brewery that produces strongest beer by ABV%
max = max(dataset["beer/ABV"])
brewery =dataset.loc[dataset["beer/ABV"].idxmax(),'beer/brewerId']

#2.top 3 to recommend
top3 = dataset.loc[dataset['review/overall'].nlargest(3).index,'beer/name']

#3. 
cols = ['review/appearance','review/aroma','review/palate', 'review/taste', 'review/overall']
dataset = dataset.dropna(subset=cols, thresh=2)




pd_dtypes = dataset.dtypes
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
col = ['review/appearance','review/aroma','review/palate', 'review/taste']
dataset2 = dataset[col].astype(str)
df2 = dataset2.apply(LabelEncoder().fit_transform, axis=0)
pd_dtypes2 = df2.dtypes
X_train =df2[col]
y_train = labelencoder_X.fit_transform(dataset['review/overall'].astype(str))

#RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg)
rfe = rfe.fit(X_train, y_train)
print(rfe.support_)
print(rfe.ranking_)

#check P-value
import statsmodels.api as sm
logit_model=sm.Logit(y_train, X_train)
result=logit_model.fit()
print(result.summary2())


#4. beerstyle add columns and create new column
dataset['total_val'] = dataset['review/aroma'] + dataset['review/appearance']
c = dataset.loc[(dataset['total_val'] >= 8),'beer/style']
c.describe()