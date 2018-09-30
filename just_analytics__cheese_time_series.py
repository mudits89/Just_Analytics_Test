# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 10:14:29 2018

@author: mudit
"""

import pandas as pd
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima


## read in dataset
pth = 'C:/Users/mudit/Desktop/Data Analytics/Just_Analytics_Test/Time_Series_Analysis_Forecasting/'
df = pd.read_csv(pth + 'Cheese_Production_Data.txt', header=0)
df.info()


## do data pre-processing
df['Month_num'] = df['Month.Count'].apply(lambda row: row%12 if row%12!=0 else 12)
df['yr_mnth'] = df['Year'].map(str) + '__' + df['Month'].map(str)
df['date'] = df.apply(lambda row: datetime(row['Year'], row['Month_num'], 1), axis=1)


## find the months with missing data
yr = [str(el) for el in range(1995, 2014)]
mnth = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for year in yr:
    for mnt in mnth:
        if str(str(year)+'__'+str(mnt)) not in df['yr_mnth'].unique():
            print(str(year)+'__'+mnt)


## Train & Test Split
df_2__train = df.loc[df.Year<2012, ['Cheese1.Prod', 'Cheese2.Prod', 'Cheese3.Prod', 'date']]
df_2__test = df.loc[~df.index.isin(df_2__train.index), ['Cheese1.Prod', 'Cheese2.Prod', 'Cheese3.Prod', 'date']]

df_2__train.index = df_2__train['date']
df_2__test.index = df_2__test['date']

df_2__train = df_2__train[[k for k in df_2__train.columns if k!='date']]
df_2__test = df_2__test[[k for k in df_2__test.columns if k!='date']]

col_names = ['Cheese1.Prod', 'Cheese2.Prod', 'Cheese3.Prod']


## plot the raw / unprocessed Cheese production trend charts
stationary_lst = []
for col in col_names:
    print("\n\nProcessing " + col)
    df_2__train[col].plot(figsize=(24,20), title= 'Monthly Cheese Production', fontsize=14)
    df_2__test[col].plot(figsize=(24,20), title= 'Monthly Cheese Production', fontsize=14)
    plt.xlabel('Year', fontsize=20);
    plt.ylabel(col, fontsize=20);
    plt.legend([col+'_train', col+'_test'])
    plt.savefig(pth + col.replace('.','_') +'__raw_overall_trend.png', format='png')
    plt.show()
    seasonal_decompose(x=df_2__train[col], freq=12).plot()
    result = sm.tsa.stattools.adfuller(df_2__train[col])
    if result[1]<=0.05:
        stationary_lst.append('stationary')
    else:
        stationary_lst.append('not_stationary')
    print(result)
    plt.xlabel(col+' [Year]', fontsize=10);
    plt.savefig(pth + col.replace('.', '_') + '__seasonal_decomposition.png', format='png')
    plt.show()

## overall Production trend - to help visualize Cheese' production levels
df_2__train.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
plt.savefig(pth + 'overall_raw_trends__train.png', format='png')
plt.show()

df_2__test.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
plt.savefig(pth + 'overall_raw_trends__test.png', format='png')
plt.show()


### Predict the 3 trends differently:
## Cheese1.Prod --> exhibits small variations & relatively stable, yet decreasing production trends
## Cheese2.Prod --> exhibits large variations & almost recurring / constant avg (yearly) production trends
## Cheese3.Prod --> exhibits small variations & almost annually increasing production trends

## Understand the variation, generally, for each Production line
df_2__train['Cheese3.Prod'].diff(periods=3).plot(figsize=(24,20),  title= 'Periodic Cheese Variation Production', linewidth=5, fontsize=20)
df_2__test['Cheese3.Prod'].diff(periods=3).plot(figsize=(24,20), title= 'Periodic Cheese Variation Production', linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.show()


# rolling average - to help remove seasonality & determine actual trend in each Cheese' production levels
df_2__train.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
plt.savefig(pth + 'non-seasonal_overall_rolling_avg.png', format='png')
plt.show()

df_2__train.corr()
df_2__train.describe()


df_2__train[['Cheese2.Prod']].diff(periods=6).plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
plt.show()


## Predictions

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
from math import sqrt

y3_pred__ses = df_2__test.copy()
fit2 = SimpleExpSmoothing(np.asarray(df_2__train['Cheese3.Prod'])).fit(smoothing_level=0.6,optimized=True)
y3_pred__ses['SES'] = fit2.forecast(len(df_2__test))
plt.figure(figsize=(24,20))
plt.plot(df_2__train['Cheese3.Prod'], label='Train')
plt.plot(df_2__test['Cheese3.Prod'], label='Test')
plt.plot(y3_pred__ses['SES'], label='SES')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(df_2__test['Cheese3.Prod'], y3_pred__ses.SES))
print(rms)


# Predict using Simple Exponential Smoothing
ses_predicted_df = pd.DataFrame()
ses_rme_lst = []
for col in col_names:
    print("\n\nProcessing SES on " + col)
    y3_pred__ses = df_2__test.copy()
    fit2 = SimpleExpSmoothing(np.asarray(df_2__train[col])).fit(smoothing_level=0.6, optimized=True)
    fin_predictions = fit2.forecast(len(df_2__test)+12)
    y3_pred__ses['SES_'+col] = fin_predictions[:len(df_2__test)]
    plt.figure(figsize=(24, 20))
    plt.plot(df_2__train[col], label='Train')
    plt.plot(df_2__test[col], label='Test')
    plt.plot(y3_pred__ses['SES_'+col], label='SES_'+col)
    plt.legend(loc='best')
    plt.show()
    rms = sqrt(mean_squared_error(df_2__test[col], y3_pred__ses['SES_' + col]))
    print("\tRMSE = " + str(rms))
    print("\t%error = " + str(rms/df_2__test[col].mean()*100))
    ses_rme_lst.append(rms)
    ses_predicted_df['SES_'+col] = pd.Series(fin_predictions[len(df_2__test):])

# Predict using Auto ARIMA
aa_predicted_df = pd.DataFrame()
aa_rme_lst = []
for col in col_names:
    print("\n\nProcessing SES on " + col)
    y3_pred__auto_arima = df_2__test.copy()
    stepwise_model = auto_arima(df_2__train[col], start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True,
                            d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    print(stepwise_model.aic())
    stepwise_model.fit(df_2__train[col])
    fin_predictions = stepwise_model.predict(n_periods=len(df_2__test) + 12)
    y3_pred__auto_arima['AA_'+col] = fin_predictions[:len(df_2__test)]
    rms = sqrt(mean_squared_error(df_2__test[col], y3_pred__auto_arima['AA_' + col]))
    print("\tRMSE = " + str(rms))
    print("\t%error = " + str(rms/df_2__test[col].mean()*100))
    aa_rme_lst.append(rms)
    aa_predicted_df['AA_'+col] = pd.Series(fin_predictions[len(df_2__test):])


# select the prediction output (from SES / AA) with the lower RMSE
final_predicted = pd.DataFrame()
for col_pos in range(len(col_names)):
    if ses_rme_lst[col_pos]<aa_rme_lst[col_pos]:
        final_predicted[col_names[col_pos]] = ses_predicted_df['SES_'+col_names[col_pos]]
    else:
        final_predicted[col_names[col_pos]] = aa_predicted_df['AA_' + col_names[col_pos]]

final_predicted.to_csv(pth + 'final_predicted__cheese_production.csv', header = True)
