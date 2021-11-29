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

df = pd.read_csv('C:/Users/Malcolm/Documents/MedicalExpenditure/data/meps_data_2019_new_feats.csv')

# train test split
X = df.drop('total_expenditure', axis =1)
y = df.total_expenditure.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)
# random forest
# decision tree
# tune models
# test ensembles