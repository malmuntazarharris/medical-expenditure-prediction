from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

def load_models():
    file_name = 'C:/Users/Malcolm/Documents/MedicalExpenditure/xgboost/pkl_objects/xgboost_model.pickle'
    with open(file_name, 'rb') as pickled:
       data = pickle.load(pickled)
       model = data['model']
    return model

if __name__ == '__main__':
   app.run(debug=True)