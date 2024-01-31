import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score,precision_score,recall_score
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xg

diabetes_dataset = pd.read_csv('diabetes_sl.csv')

#replacing 0 values with mean values if data is normally distributed and with median values if its a skewed distribution 
diabetes_dataset['Glucose']=diabetes_dataset['Glucose'].replace(0,diabetes_dataset['Glucose'].mean())#normal distribution
diabetes_dataset['BloodPressure']=diabetes_dataset['BloodPressure'].replace(0,diabetes_dataset['BloodPressure'].mean())#normal distribution
diabetes_dataset['SkinThickness']=diabetes_dataset['SkinThickness'].replace(0,diabetes_dataset['SkinThickness'].median())#skewed distribution
diabetes_dataset['Insulin']=diabetes_dataset['Insulin'].replace(0,diabetes_dataset['Insulin'].median())#skewed distribution
diabetes_dataset['BMI']=diabetes_dataset['BMI'].replace(0,diabetes_dataset['BMI'].median())#skewed distribution

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

import mlflow

# Define the experiment name
experiment_name = "diabetes_prediction_experiment"

# Check if the experiment exists
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    # If the experiment does not exist, create it
    mlflow.create_experiment(experiment_name)

# Set the experiment
mlflow.set_experiment(experiment_name)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import mlflow

# Assuming you have X_train, Y_train, X_test, Y_test defined

# Define hyperparameters grid for each classifier
param_grid = {
    'LogisticRegression': {'C': [0.1, 1.0, 10.0]},
    'RandomForestClassifier': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'SVC': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
    'GradientBoostingClassifier': {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.01]}
}

# List of classifiers
al_list = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC(),
    GradientBoostingClassifier()
]

for classifier in al_list:
    # Retrieve the hyperparameters grid for the current classifier
    params = param_grid[classifier._class.name_]

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(classifier, params, scoring='accuracy', cv=3)
    grid_search.fit(X_train, Y_train)

    # Get the best estimator and its parameters
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Use the best estimator to make predictions
    y_train_pred = best_estimator.predict(X_train)
    y_test_pred = best_estimator.predict(X_test)

    # Calculate metrics
    accuracy_train = accuracy_score(Y_train, y_train_pred)
    accuracy_test = accuracy_score(Y_test, y_test_pred)
    recall_train = recall_score(Y_train, y_train_pred)
    recall_test = recall_score(Y_test, y_test_pred)
    f1_train = f1_score(Y_train, y_train_pred)
    f1_test = f1_score(Y_test, y_test_pred)

    # Print metrics
    print(f'Metrics for {classifier._class.name_}:')
    print(f'  Best Parameters: {best_params}')
    print(f'  Accuracy (Train): {accuracy_train:.4f}')
    print(f'  Accuracy (Test):  {accuracy_test:.4f}')
    print(f'  Recall (Train):   {recall_train:.4f}')
    print(f'  Recall (Test):    {recall_test:.4f}')
    print(f'  F1 Score (Train): {f1_train:.4f}')
    print(f'  F1 Score (Test):  {f1_test:.4f}')

    # MLflow logging
    with mlflow.start_run(run_name=str(classifier._class.name_)):
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy_train", accuracy_train)
        mlflow.log_metric("accuracy_test", accuracy_test)
        mlflow.log_metric("recall_train", recall_train)
        mlflow.log_metric("recall_test", recall_test)
        mlflow.log_metric("f1_train", f1_train)
        mlflow.log_metric("f1_test", f1_test)

        
