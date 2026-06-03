"""
MLProject entry point: Heart Disease Classification
"""

import os
import argparse
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--data_path', type=str, default='heart_disease_preprocessing')
    args = parser.parse_args()

    # In MLProject, we usually don't set tracking URI explicitly 
    # as it's handled by environment variables or CLI flags
    # But for local consistency with other scripts:
    # mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('Heart Disease MLProject')

    X_train = np.load(os.path.join(args.data_path, 'X_train.npy'))
    X_test = np.load(os.path.join(args.data_path, 'X_test.npy'))
    y_train = np.load(os.path.join(args.data_path, 'y_train.npy'))
    y_test = np.load(os.path.join(args.data_path, 'y_test.npy'))

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_param('max_depth', args.max_depth)
        mlflow.log_param('n_estimators', args.n_estimators)
        mlflow.log_metric('test_accuracy', acc)
        mlflow.log_metric('test_f1_weighted', f1)
        mlflow.sklearn.log_model(model, 'model')

        print(f'Accuracy: {acc:.4f}, F1: {f1:.4f}')


if __name__ == '__main__':
    main()
