import os, argparse, numpy as np, mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="preprocessing/thyroid_preprocessing")
    args = parser.parse_args()
    X_train = np.load(os.path.join(args.data_path, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_path, "y_train.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(args.data_path, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_path, "y_test.npy"), allow_pickle=True)
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
if __name__ == "__main__": main()
