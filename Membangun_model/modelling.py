"""
modelling.py
Model training dengan MLflow autolog untuk Heart Disease Classification.
"""

import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def load_preprocessed(data_dir='heart_disease_preprocessing'):
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    # Set tracking URI and experiment
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('Heart Disease Classification')

    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed()

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Using autolog for basic criteria
            mlflow.autolog()

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            mlflow.log_metric('test_accuracy', acc)
            mlflow.log_metric('test_f1_weighted', f1)

            print(f'{name}: Accuracy={acc:.4f}, F1={f1:.4f}')

    print('\nSemua model selesai dilatih. Cek MLflow UI di http://127.0.0.1:5000')


if __name__ == '__main__':
    main()
