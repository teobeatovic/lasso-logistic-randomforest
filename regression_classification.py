import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from math import log

class ModelTrainer:
    """
    A class to handle regression and classification tasks using scikit-learn.
    """
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
    
    def load_data(self, file_path):
        """Loads the dataset from the given file path."""
        return pd.read_csv(file_path)
    
    def preprocess_data(self, features, target):
        """Handles missing values, scales numerical data, and encodes categorical variables."""
        X = self.data[features]
        y = self.data[target]
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor, X, y
    
    def regression_analysis(self, features, target):
        """Performs Lasso regression with BIC evaluation."""
        preprocessor, X, y = self.preprocess_data(features, target)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Lasso(alpha=0.1))])
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Compute Bayesian Information Criterion (BIC)
        n = X_test.shape[0]
        k = X_test.shape[1]
        bic = n * log(mse) + k * log(n)
        
        print("--- Regression Results ---")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.2f}")
        print(f"BIC Score: {bic:.2f}")
        
        joblib.dump(model, 'lasso_regression_model.pkl')
        return model, X_test, y_test, y_pred
    
    def classification_analysis(self, features, target):
        """Performs logistic regression and random forest classification."""
        preprocessor, X, y = self.preprocess_data(features, target)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Logistic Regression
        log_reg_model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])
        log_reg_model.fit(X_train, y_train)
        y_pred_log = log_reg_model.predict(X_test)
        accuracy_log = accuracy_score(y_test, y_pred_log)
        print("--- Logistic Regression Results ---")
        print(f"Accuracy: {accuracy_log:.2f}")
        print(classification_report(y_test, y_pred_log))
        joblib.dump(log_reg_model, 'logistic_regression_model.pkl')
        
        # Random Forest Classifier
        rf_model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        print("--- Random Forest Classification Results ---")
        print(f"Accuracy: {accuracy_rf:.2f}")
        print(classification_report(y_test, y_pred_rf))
        joblib.dump(rf_model, 'random_forest_model.pkl')
        
        return log_reg_model, rf_model, X_test, y_test, y_pred_log, y_pred_rf
    
if __name__ == "__main__":
    trainer = ModelTrainer("data/input_dataset.csv")
    
    regression_features = ["feature1", "feature2"]
    regression_target = "target_regression"
    classification_features = ["feature1", "feature2"]
    classification_target = "target_classification"
    
    # Perform regression analysis
    trainer.regression_analysis(regression_features, regression_target)
    
    # Perform classification analysis with both Logistic Regression and Random Forest
    trainer.classification_analysis(classification_features, classification_target)
