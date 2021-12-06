# -*- coding: utf-8 -*-
"""
This file builds a model with limited features to restrict the amount of information the web app will need
to make a prediction
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pickle

df = pd.read_csv("../data/meps_data_2019_new_feats.csv")

# choose features included in web app
data = df[["age","race","sex","other_language_spoken_at_home","born_in_usa",
                "marriage_status","military_status","total_income", "employment_status",
                "insurance_status","highest_education","number_of_visits","diag_amt","total_expenditure"]]

# remove inapplicable values (values < 0) for categorical variables

# medical expenditure is strongly skewed to the right so it needs to be transformed for our model
val = data["total_expenditure"].values 
arr = np.array([0 if v == 0 else np.log(v) for v in val])
data["total_expenditure"] = arr

# one hot encoding of our categorical columns