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

![Model Comparison File](https://github.com/malmuntazarharris/medical-expenditure-prediction/blob/master/xgboost/model_comparison.py)

I tested three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret. Out of the three models I tried, XGBoost perfomed the best:

| Model             | MAE                |
|-------------------|--------------------|
| Linear Regression | -4480.291433183665 |
| Random Forest     | -4343.886650547299 |
| XGBoost           | -4214.298763882473 |

## Model Building: Initial Build

![Initial Build and Tuning File](https://github.com/malmuntazarharris/medical-expenditure-prediction/blob/master/xgboost/gridcv_xgboost.py)

![Initial Build Pickle Dump File](https://github.com/malmuntazarharris/medical-expenditure-prediction/blob/master/xgboost/xgboost_dump.py)

I trained an XGBoost model and used GridSearchCV to turn the parameters. I then dumped them into a pickle object.

## Model Building: Model Version 2

![Final Model Build Notebook](https://github.com/malmuntazarharris/medical-expenditure-prediction/blob/master/xgboost/xgboost_model_v2_building.ipynb)

I wasn't satisfied with the model's performance and I thought it used an unwieldy amount of features to ask a user in a survey-based web application. I opted to reduce the dataframe to 15 features and use my feature engineered columns to avoid requesting detailed medical information. Furthermore, after some research, I chose to scale the numeric features, one hot encode the categorical features and normalize the target variable by taking the log(x) of value. After extensive tuning using GridSearchCV, I put together a pipeline and pickled it for use in the web app.

![image](https://user-images.githubusercontent.com/29358953/146582465-cf531aac-6842-4482-8bb5-234266932d08.png)

## Productionization

![App Folder (Contains Flask App, HTML and CSS)](https://github.com/malmuntazarharris/medical-expenditure-prediction/tree/master/app)

![image](https://user-images.githubusercontent.com/29358953/146584813-0df24910-d313-4cf9-820b-0b7686e14323.png)
![image](https://user-images.githubusercontent.com/29358953/146584940-0699ac49-d271-4493-a11b-bc72ef240f7d.png)

In order to make a viable product with this project, I found several tutorials (referenced at the top) to help me put this model on a web server. I built a Flask backend and put together a form using HTML and CSS. With the website made, I deployed the project onto heroku where any user can use it. 

## Takeaways

* In my initial planning phase for ML projects, I should keep in my the constraints I have. I had to limit the amount of features I used to trained my model due to the limitations of the web app and what I thought was appropriate to collect. It would have saved time if this was realized earlier on. However, this was good practice in case I ever find similar constraints in my future work.
* I thoroughly enjoyed the process of putting together a website and deploy. It was satisfying putting together a final useable product and it makes me excited to learn more about web dev in the future
* I learned several ways to make my models more accurate and further research into how each model's accuracy depends on how I preprocessing my data will be illuminating and will improve the quality of my future work.

