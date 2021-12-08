# -*- coding: utf-8 -*-
"""
This file builds a model with limited features to restrict the amount of information the web app will need
to make a prediction
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

web_app_features = ["age","race","ethnicity","sex","other_language_spoken_at_home","born_in_usa",
                "marriage_status","military_status","total_income", "employment_status",
                "insurance_status","highest_education","number_of_visits","diag_amt","total_expenditure"]

def normalize_target(df, target_column):
    # medical expenditure is strongly positively skewed so best practice is to normalized  it for our model
    vals = data[target_column].values 
    return np.array([0 if v == 0 else np.log(v) for v in vals])

def readin_and_split(path, target_column, features):
    df = pd.read_csv(path)
    df[target_column] = normalize_target(df, target_column)
    X = df.drop(target_column)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=0)
    
df = pd.read_csv("../data/meps_data_2019_new_feats.csv")

"""Delete below block and make more functional"""

# choose features included in web app
data = df[["age","race","ethnicity","sex","other_language_spoken_at_home","born_in_usa",
                "marriage_status","military_status","total_income", "employment_status",
                "insurance_status","highest_education","number_of_visits","diag_amt","total_expenditure"]]

cat_cols = ["race","sex","other_language_spoken_at_home","born_in_usa","marriage_status",
            "military_status","employment_status","insurance_status","highest_education"]
num_cols = [f for f in data.columns if f not in cat_cols]

# remove employment status == 2
data = data[data.employment_status != 2]

# MEPs encode inapplicable responses as negative integers per the documentation
# This block remove inapplicable values (values that are < 0)
data = data[data['age'] >= 0] 
data = data[df['marriage_status'] >= 0]
data = data[df['total_income'] >= 0]

data = data[(data[["other_language_spoken_at_home","born_in_usa",
                "marriage_status","military_status","employment_status",
                "insurance_status","highest_education"]] >= -1).all(1)] # remove numbers lower than -1 for other categorical variables

# scale numeric features
scaler = StandardScaler()
scaler.fit_transform(data[num_cols])
pickle.dump(scaler, open("../xgboost/pkl_objects/MEPS_xgb_model_v2_scaler.pickle", "wb")) # pickle to use in backend

# one hot encoding of our categorical columns
encoder = OneHotEncoder(sparse = False, handle_unknown = "ignore")
encoder.fit_transform(data[cat_cols])

def main():
    X_train, X_test, y_train, y_test = readin_and_split("../data/meps_data_2019_new_feats.csv", "medical_expenditure",  web_app_features)