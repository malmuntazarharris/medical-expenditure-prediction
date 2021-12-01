# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:38:17 2021

@author: Malcolm
"""

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('C:/Users/Malcolm/Documents/MedicalExpenditure/data/meps_data_2019_new_feats.csv')

# preprocessing

# train test split
X = df.drop('total_expenditure', axis =1)
y = df.total_expenditure.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)
print("The average MAE for Linear Regression is " + str(np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))))

# random forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
print("The average MAE for Random Forest is " + str(np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))))

# gradient boosting
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
print("The average MAE for XGBoost is " + str(np.mean(cross_val_score(gb,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))))


