MLflow Project for Heart Disease Classification
================================================

This MLflow project trains a Random Forest classifier on the Heart Disease dataset
with hyperparameter tuning and MLflow tracking.

Usage
-----
    mlflow run . -P max_depth=10 -P n_estimators=100 --env-manager=local

Parameters
----------
- max_depth: Maximum depth of the random forest (default: 10)
- n_estimators: Number of trees in the forest (default: 100)
- data_path: Path to preprocessed data directory (default: heart_disease_preprocessing)
