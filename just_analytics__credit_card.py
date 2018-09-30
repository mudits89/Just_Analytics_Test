# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:09:53 2018

@author: mudit
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read Train & Test datasets
path = "C:/Users/mudit/Desktop/Data Analytics/Just_Analytics_Test/Credit_card/"
train_set = pd.read_excel(path + 'Credit Card training set.xls', header = 0 )
test_set = pd.read_excel(path + 'Credit Card Testing Set.xls', header = 0 )

# retrieve balance pending
def create_balance(dataset):
    for col_pos in range(1,6):
        print(col_pos)
        dataset['balance_amt'+str(col_pos)] = dataset['BILL_AMT'+str((col_pos+1))]-dataset['PAY_AMT'+str(col_pos)]
        print(dataset[['BILL_AMT'+ str(col_pos+1), 'PAY_AMT' + str(col_pos), 'balance_amt' + str(col_pos) ]].head())
    return dataset

train_set = create_balance(dataset=train_set)
test_set = create_balance(dataset=test_set)

#retrieve average balance pending for last 5 months
def create_average(dataset1):
    col = dataset1.loc[:, 'balance_amt1':'balance_amt5']
    dataset1['avg_bal'] = col.mean(axis = 1)
    return dataset1

train_set = create_average(dataset1 = train_set)
test_set = create_average(dataset1 =  test_set)

#check if limit has been crossed 
def check_limit(dataset2):
    dataset2['limit_crossed'] = dataset2['LIMIT_BAL'] < dataset2['BILL_AMT1']
    return dataset2

train_set = check_limit(dataset2 = train_set)
test_set = check_limit(dataset2 = test_set)

cols = ['LIMIT_BAL','SEX','EDUCATION', 'MARRIAGE', 'AGE','PAY_2','PAY_3',
        'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
        'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4',
        'PAY_AMT5', 'PAY_AMT6', 'balance_amt1','balance_amt2','balance_amt3',
        'balance_amt4', 'balance_amt5', 'avg_bal']
X_train = train_set[cols]
y_train = train_set['default payment next month']
X_test = test_set[cols]

#Oversampling
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE


sm = SMOTE(random_state=12, ratio = 1.0)

# 1. Further split X_train & y_train into 2 sets - train & test

# take 5% / 300 rows sample from defaulters & non-defaulters
y_train__interim_default = y_train[y_train==1].sample(frac=0.10, random_state=1)
y_train__interim_no_default = y_train[y_train==0].sample(n=len(y_train__interim_default), random_state=1)

# combine the 5% sample of defaulters & equivalent # of non-defaulters into the y_train__test dataset
y_train__test = pd.concat([y_train__interim_default, y_train__interim_no_default])
# all remaining rows in y_train become the actual training data set
y_train__train = y_train[~y_train.index.isin(y_train__test.index)]

# replicate the above for the x_train data set
x_train__train = X_train.loc[X_train.index.isin(y_train__train.index)]
x_train__test = X_train.loc[X_train.index.isin(y_train__test.index)]



x_ovr_sampld_train, y_ovr_sampld_train= sm.fit_sample(x_train__train, y_train__train)
print (y_train.value_counts(), np.bincount(y_ovr_sampld_train))

clf_rf = RandomForestClassifier(n_estimators=2500, random_state=12)
clf_rf.fit(x_ovr_sampld_train, y_ovr_sampld_train)


clf_rf.score(x_train__test, y_train__test)
recall_score(y_train__test, clf_rf.predict(x_train__test))

y_test__rand_forest = clf_rf.predict(X_test)
y_test_prob__rand_forest = clf_rf.predict_log_proba(X_train)

#logit
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(x_train__train, y_train__train)
y_pred = logreg.predict(x_train__test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'
      .format(logreg.score(x_train__test,y_train__test)))
#build confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_train__test, y_pred)
print(confusion_matrix)

#number of occurences of each class
from sklearn.metrics import classification_report
print(classification_report(y_train__test, y_pred))

y_test__logit = logreg.predict(X_test)
y_test_prob__logit = logreg.predict_proba(X_test)




from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(x_train__train, y_train__train)


y_pred_Decision_Tree = clf_gini.predict(x_train__test)
print ("Accuracy is ", accuracy_score(y_train__test,y_pred_Decision_Tree)*100)
y_test_Decision_Tree = clf_gini.predict(X_test)

