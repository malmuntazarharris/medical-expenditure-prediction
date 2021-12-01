# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:38:17 2021

@author: Malcolm
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
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

# xgboost
xgb = GradientBoostingRegressor()
xgb.fit(X_train, y_train)
print("The average MAE for XGBoost is " + str(np.mean(cross_val_score(xgb,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))))

# tune models
parameters = {
    "n_estimators": list(range(60, 71, 5)) + list(range(71, 81)) + list(range(85, 106, 5)),
    "max_depth": list(range(2, 8)),
    "min_samples_split": list(range(2, 8)),
    "min_samples_leaf": list(range(2, 8)),
    "random_state": [123]
}
reg = GridSearchCV(xgb, parameters, n_jobs = -1)

%%time
reg.fit(X_train, y_train)

# parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')} # This tunes the random forest model with different parameters and finds the ideal amount of estimators

# gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
# gs.fit(X_train,y_train)

# test ensembles