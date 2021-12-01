# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:32:13 2021

@author: Malcolm
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

df = pd.read_csv('C:/Users/Malcolm/Documents/MedicalExpenditure/data/meps_data_2019_new_feats.csv')
cv = pickle.load(open('C:/Users/Malcolm/Documents/MedicalExpenditure/xgboost/pkl_objects/gridcv_result.pickle', 'rb'))
cv.best_params_

# train test split
X = df.drop('total_expenditure', axis =1)
y = df.total_expenditure.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

gb = GradientBoostingRegressor(max_depth = 6,min_samples_leaf = 7,min_samples_split = 2,n_estimators = 72)
gb.fit(X_train, y_train)

pickle.dump(gb, open("C:/Users/Malcolm/Documents/MedicalExpenditure/xgboost/pkl_objects/xgboost_model.pickle", "wb"))