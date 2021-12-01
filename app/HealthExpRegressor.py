# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:25:58 2021

@author: Malcolm
"""

import pickle

class HealthExpRegressor:
    reg = None
    
    def __init__(self):
        self.reg = pickle.load(open('MedicalExpenditure/xgboostclassifier/xgboost_model.pickle', 'rb'))
    
    def predict(self, arr)
        