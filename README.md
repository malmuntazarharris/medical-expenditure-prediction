# Medical Expenditure Prediction Web App: Project Overview

https://medical-expenditure-prediction.herokuapp.com/

This project is a machine learning model deployed to a Flask web app that users can use to predict the amount of spending an individual will spend out-of-pocket on healthcare a year. The steps I followed for this project are as follows:

* Researched for a database with health expenditure and found the MEPS database
* Loaded data in Juptyer Notebook, reduced the amount of columns and renamed the columns for ease of use
* Explored dataset and find any trends or correlations with the features
* Feature engineered columns to total number of visits, diseases and physical limitations for each row
* Ran a baseline comparison between three models and found gradient boost regressor had the best results
* Trained an initial XGBoost model and optimized with GridSearchCV
* Researched and added additional preprocessing steps including normalizing the target value, scaling the numeric values, and one hot encoding
* Trained and test a final version of the XGBoost model pipeline with increased accuracy
* Created an ML pipeline with the final version of the model and pickled the pipeline for use on the web app
* Built a client facing API using flask
* Built simple frontend with HTML and CSS
* Deployed app on Heroku

## Code and Resources Used 
**Python Version:** Python 3.9.7

**Web Dev Tools**: HTML5, CSS, Heroku

**Python Modules:** scikit-learn, pandas, numpy, pickle, matplotlib, seaborn, flask

**Survey Data:** https://www.meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-216

**ML Flask Example:** https://github.com/krishnaik06/Deployment-flask

**Form Frontend Example:** https://www.geeksforgeeks.org/build-a-survey-form-using-html-and-css/

**Flask Heroku Deployment Tutorial:** https://www.section.io/engineering-education/integrate-ml-to-flask-api-and-deploy-to-heroku/ 

## Data Cleaning

[Data Cleaning Notebook](https://github.com/malmuntazarharris/medical-expenditure-prediction/blob/master/xgboost/datacleaning.ipynb)

The original dataset has 1447 features so I reduced to 53 features by removing any features directly related to spending and selecting any features related to demographic, socioeconomic status or health status. The selected features were then renamed to be more descriptive.

## EDA

[Exploratory Data Analysis Notebook](https://github.com/malmuntazarharris/medical-expenditure-prediction/blob/master/xgboost/eda_feature_engineering.ipynb)

After data cleaning, I viewed the distributions of the data and produced several plots in order to gain some insight of the data.

![download](https://user-images.githubusercontent.com/29358953/146485915-3e42495b-4064-4c2f-b1d8-887b94831342.png)
![download](https://user-images.githubusercontent.com/29358953/146486051-3b057f74-88c5-4047-8a43-6a2055c124a6.png)
![download](https://user-images.githubusercontent.com/29358953/146486122-3db94dbb-fd7f-430f-99a7-109f7949b3a3.png)

## Feature Engineering

Utilizing the data for medical visits, medical condition diagnoses, and physical limitations, I featured engineered three additional columns.

## Model Building: Initial Comparison

I tested three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret. Out of the three models I tried, XGBoost perfomed the best:

| Model             | MAE                |
|-------------------|--------------------|
| Linear Regression | -4480.291433183665 |
| Random Forest     | -4343.886650547299 |
| XGBoost           | -4214.298763882473 |

## Model Building: Initial Build
