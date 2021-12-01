# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:04:35 2021

@author: Malcolm
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pickle

df = pd.read_csv('C:/Users/Malcolm/Documents/MedicalExpenditure/data/meps_data_2019_new_feats.csv')

# train test split
X = df.drop('total_expenditure', axis =1)
y = df.total_expenditure.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)

parameters = {
    "n_estimators": list(range(60, 71, 5)) + list(range(71, 81)) + list(range(85, 106, 5)),
    "max_depth": list(range(2, 8)),
    "min_samples_split": list(range(2, 8)),
    "min_samples_leaf": list(range(2, 8)),
} 
reg = GridSearchCV(gb, parameters, scoring='neg_mean_absolute_error', n_jobs = 3, verbose=2, cv=3)
reg.fit(X_train, y_train)

reg.best_score_  # -4094.3398028604183
reg.best_params_ # {'max_depth': 6,
#  'min_samples_leaf': 7,
#  'min_samples_split': 2,
#  'n_estimators': 72}

pickle.dump(reg, open("MEPS_xgb_model_non_processed_v1.pickle", "wb"))