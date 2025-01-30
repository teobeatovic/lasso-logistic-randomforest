# Regression and Classification Analysis

# Overview
This project leverages advanced machine learning techniques, including Lasso Regression, Logistic Regression, and Random Forest Classification, to extract insights from user input datasets, build predictive models, and evaluate performance, detailing this through various prediction plots. It encompasses a robust data preprocessing pipeline with feature scaling, categorical encoding, and missing value handling. Model evaluation is conducted using key performance metrics such as Mean Squared Error (MSE), R² Score, Bayesian Information Criterion (BIC), Accuracy, and Classification Reports.

# Features
* Regression Analysis: Uses Lasso Regression to predict continuous target variables and evaluates model performance using BIC, MSE, and R².
* Classification Analysis: Implements Logistic Regression and Random Forest Classifier for classification tasks.
* Preprocessing Pipeline:
  * Scales numerical features using StandardScaler.
  * Encodes categorical variables using OneHotEncoder.
  * Splits data into training and test sets.
* Model Persistence: Saves trained models (.pkl files) for future use.
* Performance Evaluation: Prints accuracy scores, confusion matrices, and classification reports.

# How to Run
1. Prepare Your Dataset:
* Place your dataset in the data/ folder.
* Ensure it is in CSV format.
* Update the script with the correct file path, e.g., data/your_dataset.csv.
2. Define Your Features and Target Variables:
* Open regression_classification.py.
* Modify the feature and target column names:
  * regression_features = ["YourFeature1", "YourFeature2"]
  * regression_target = "YourTargetColumn"
  * classification_features = ["YourFeature1", "YourFeature2"]
  * classification_target = "YourTargetColumn"
3. Run the Python script (regression_classification.py).
4. Review Outputs:
* Regression and classification results will be printed in the terminal.
* Saved models will be stored in .pkl files for later use.

# Output
* Regression Results:
  * Mean Squared Error (MSE)
  * R² Score
  * Bayesian Information Criterion (BIC)
  * Model saved as lasso_regression_model.pkl
* Classification Results:
  * Accuracy for Logistic Regression and Random Forest
  * Confusion matrix and classification report
  * Models saved as logistic_regression_model.pkl and random_forest_model.pkl

# Dependencies
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* joblib

