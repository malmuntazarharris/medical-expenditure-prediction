# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:57:42 2021

@author: Malcolm
"""

from flask import Flask
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/")
def hello_world():
    test_np_input = np.ones(54).reshape(1, -1)
    model = pickle.load(open('C:/Users/Malcolm/Documents/MedicalExpenditure/xgboost/pkl_objects/xgboost_model.pickle', 'rb'))
    preds = model.predict(test_np_input)
    preds_as_str = str(preds)
    return preds_as_str