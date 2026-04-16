# Boston Housing Price Predictor

A full stack machine learning web application that predicts median house prices based on neighborhood features using a trained regression model.

Live application:
https://neelabho.pythonanywhere.com


## Project Overview

This project trains a regression model on the Boston Housing dataset and deploys it as an interactive web application.

Users provide thirteen neighborhood features such as crime rate, number of rooms, and air pollution level. The application returns a predicted median house price along with model confidence and feature influence insights.


## Key Features

- Real time prediction without page reload
- Model comparison across multiple algorithms
- Pre trained model for fast inference
- Clean and responsive user interface
- Approximate feature importance display
- Deployment ready architecture


## Machine Learning Details

Dataset: Boston Housing Dataset  https://www.kaggle.com/datasets/arunjathari/bostonhousepricedata
Number of rows: 506  
Number of features: 13  
Target variable: MEDV (median value in thousands of dollars)

Models evaluated:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Selected model:
Gradient Boosting Regressor

Performance:
- R squared score: approximately 0.89
- Root Mean Squared Error: approximately 3.2 (thousand dollars)

Gradient Boosting was selected because it iteratively improves prediction accuracy by correcting previous errors, resulting in better performance than the other models.


## Technology Stack

Machine Learning:
- scikit learn

Backend:
- Flask

Frontend:
- HTML
- CSS
- JavaScript

Data handling:
- pandas
- numpy

Model persistence:
- joblib

Deployment:
- PythonAnywhere

Version control:
- Git and GitHub


## Project Structure

boston-housing/
|
|-- app.py
|-- train_model.py
|-- model.pkl
|-- scaler.pkl
|-- requirements.txt
|
|-- templates/
|   |-- index.html
|
|-- static/
|   |-- style.css
|
|-- eda_notebook.ipynb


## Application Workflow

1. The user opens the web application.
2. The frontend collects input values from the form.
3. A POST request is sent to the Flask backend.
4. The backend loads the trained model and scaler.
5. Input data is scaled and passed to the model.
6. The prediction is returned as JSON.
7. The frontend updates the result dynamically without reloading the page.


## Setup Instructions

1. Clone the repository

git clone https://github.com/neelabho/boston-housing-ml_60.git
cd boston-housing


2. Create a virtual environment

python -m venv venv


3. Activate the virtual environment

Windows:
venv\Scripts\activate

Mac or Linux:
source venv/bin/activate


4. Install dependencies

pip install -r requirements.txt


5. Train the model

python train_model.py

This step generates model.pkl and scaler.pkl


6. Run the application

python app.py

Open a browser and go to:
http://127.0.0.1:5000


## Input Features

CRIM: Per capita crime rate  
ZN: Residential land zoning percentage  
INDUS: Non retail business area percentage  
CHAS: Proximity to Charles River (0 or 1)  
NOX: Nitric oxide concentration  
RM: Average number of rooms per dwelling  
AGE: Proportion of older housing units  
DIS: Distance to employment centers  
RAD: Accessibility to highways  
TAX: Property tax rate  
PTRATIO: Pupil teacher ratio  
B: Demographic index  
LSTAT: Percentage of lower income population  


## Important Notes

- The model must be trained locally before running the application
- Do not retrain the model on the deployment server
- The scaler must be used during inference to ensure correct predictions
- Input feature order must match the training data


## Common Errors and Solutions

Model file not found:
Run train_model.py before starting the application

Incorrect predictions:
Ensure the correct scaler is loaded

Server error:
Check Flask logs for detailed error messages

Feature mismatch:
Verify that all thirteen inputs are provided in the correct order


## Deployment

The application is deployed on PythonAnywhere.

Live URL:
https://neelabho.pythonanywhere.com


## Author

Neelabho Chakraborty & Aakriti Thakur
