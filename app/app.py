from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

posts = [
    {
        'author': 'Corey Schafer',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 21, 2018'
    }
]

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', posts=posts)

@app.route('/about')
def about():
    return render_template('about.html')    

def load_models():
    file_name = 'C:/Users/Malcolm/Documents/MedicalExpenditure/xgboost/pkl_objects/xgboost_model.pickle'
    with open(file_name, 'rb') as pickled:
       data = pickle.load(pickled)
       model = data['model']
    return model

if __name__ == '__main__':
   app.run(debug=True)