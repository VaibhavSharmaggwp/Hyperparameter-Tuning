import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

#Loading Dataset
data = load_breast_cancer()
X = data.data
y = data.target

#Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store model results
results = {}

# Function to evaluate model and store metrics

def evaluate_model(model, model_name, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    print(f"\n{model_name} Performance:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# 1. Logistic Regression with GridSearchCV
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

lr = LogisticRegression(random_state=42)
grid_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
grid_lr.fit(X_train_scaled, y_train)
print("Best Logistic Regression Parameters:", grid_lr.best_params_)
evaluate_model(grid_lr.best_estimator_, "Logistic Regression", X_test_scaled, y_test)

# 2. Support Vector Machine with RandomizedSearchCV
param_dist_svm = {
    'C': np.logspace(-3, 3, 100),
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 0.01]
}


svm = SVC(random_state=42)
random_svm = RandomizedSearchCV(svm, param_dist_svm, n_iter=20, cv=5, scoring='f1', n_jobs=-1, random_state=42)
random_svm.fit(X_train_scaled, y_train)
print("Best SVM Parameters:", random_svm.best_params_)
evaluate_model(random_svm.best_estimator_, "SVM", X_test_scaled, y_test)


# 3. Random Forest with GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
print("Best Random Forest Parameters:", grid_rf.best_params_)
evaluate_model(grid_rf.best_estimator_, "Random Forest", X_test_scaled, y_test)


# Analyze results to select the best model
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)


# Select the best model based on F1-Score
best_model = results_df['F1-Score'].idxmax()
print(f"\nBest Performing Model: {best_model}")
print(f"Best Model Metrics:\n{results_df.loc[best_model]}")


