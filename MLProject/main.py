"""
modelling.py
Model training dengan MLflow autolog untuk Heart Disease Classification.
"""

import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def load_preprocessed(data_dir='heart_disease_preprocessing'):
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True)
    return X_train, X_test, y_train, y_test


def main():
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('Heart Disease Classification')

    X_train, X_test, y_train, y_test = load_preprocessed()

    with mlflow.start_run(run_name='RandomForest_Basic'):
        mlflow.autolog()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {acc:.4f}')

if __name__ == '__main__':
    main()
