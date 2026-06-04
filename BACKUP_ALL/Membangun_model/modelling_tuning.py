"""
modelling_tuning.py
Model training dengan hyperparameter tuning dan manual logging di MLflow untuk Heart Disease Classification.
"""

import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib
import json


def load_preprocessed(data_dir='heart_disease_preprocessing'):
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('Heart Disease Tuning')

    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed()

    # Hyperparameter grid for Random Forest
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
    }

    with mlflow.start_run(run_name='RandomForest_Tuning'):
        rf = RandomForestClassifier(random_state=42)

        grid_search = GridSearchCV(
            rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Manual Logging
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric('best_cv_accuracy', grid_search.best_score_)
        mlflow.log_metric('test_accuracy', accuracy_score(y_test, y_pred))
        mlflow.log_metric('test_f1_weighted', f1_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric('test_precision_weighted', precision_score(y_test, y_pred, average='weighted', zero_division=0))
        mlflow.log_metric('test_recall_weighted', recall_score(y_test, y_pred, average='weighted', zero_division=0))

        # Log model
        mlflow.sklearn.log_model(best_model, 'model')
        
        # Save locally
        joblib.dump(best_model, 'best_rf_model.pkl')

        # Log classification report artifact
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = 'classification_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        print(f'RF Best params: {grid_search.best_params_}')
        print(f'RF Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')

    # Hyperparameter grid for Gradient Boosting
    param_grid_gb = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
    }

    with mlflow.start_run(run_name='GradientBoosting_Tuning'):
        gb = GradientBoostingClassifier(random_state=42)

        grid_search_gb = GridSearchCV(
            gb, param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid_search_gb.fit(X_train, y_train)

        best_gb = grid_search_gb.best_estimator_
        y_pred_gb = best_gb.predict(X_test)

        # Manual Logging
        mlflow.log_params(grid_search_gb.best_params_)
        mlflow.log_metric('best_cv_accuracy', grid_search_gb.best_score_)
        mlflow.log_metric('test_accuracy', accuracy_score(y_test, y_pred_gb))
        mlflow.log_metric('test_f1_weighted', f1_score(y_test, y_pred_gb, average='weighted'))
        mlflow.log_metric('test_precision_weighted', precision_score(y_test, y_pred_gb, average='weighted', zero_division=0))
        mlflow.log_metric('test_recall_weighted', recall_score(y_test, y_pred_gb, average='weighted', zero_division=0))

        # Log model
        mlflow.sklearn.log_model(best_gb, 'model')
        
        # Save locally
        joblib.dump(best_gb, 'best_gb_model.pkl')

        print(f'GB Best params: {grid_search_gb.best_params_}')
        print(f'GB Test Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}')

    print('\nTuning selesai. Cek MLflow UI di http://127.0.0.1:5000')


if __name__ == '__main__':
    main()
