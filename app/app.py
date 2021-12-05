"""
Resources: 
ML Webapp example: https://github.com/krishnaik06/Deployment-flask/blob/master/app.py
"""

from flask import Flask, render_template
import numpy as np
import pickle

app = Flask(__name__)
# model = pickle.load(open('C:/Users/Malcolm/Documents/MedicalExpenditure/xgboost/pkl_objects/xgboost_model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict',methods=['POST'])
# def predict():
#     '''
#     Renders results in html
#     '''
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)

#     output = round(prediction[0], 2)

#     return render_template('index.html', prediction_text='Predicted Medical Expenditure per year is $ {}'.format(output))

if __name__ == '__main__':
   app.run(debug=True)