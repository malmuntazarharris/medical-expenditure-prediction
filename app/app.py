"""
Resources: 
ML Webapp example: https://github.com/krishnaik06/Deployment-flask/blob/master/app.py
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import math
import sklearn
import pickle

app = Flask(__name__)
pipeline = pickle.load(open('xgboost/pkl_objects/MEPS_xgb_model_pipeline_v2.pickle', 'rb'))

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_values = np.array([int(x) for x in request.form.values()])
    form_df = pd.DataFrame([form_values], columns = list(request.form.keys()))
    prediction = math.e ** (pipeline.predict(form_df)) # target value for the model was transformed using log(x), this converts them to the original representation
    pred_str = "${:,.2f}".format(float(prediction))

    return render_template('result.html', prediction_text='Predicted Medical Expenditure per year is {}'.format(pred_str))

if __name__ == '__main__':
   app.run(debug=True)