import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model_train")

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets/")

def train_model(X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        rfc = RandomForestClassifier(n_estimators=200)
        rfc.fit(X_train, y_train)
        pred_rfc = rfc.predict(X_test)
        mlflow.log_artifact(local_path="/home/artem/Projects/mlops-task3/scripts/model_train.py",
                            artifact_path="model_train")
        mlflow.end_run()
    accuracy = accuracy_score(y_test, pred_rfc)
    # print()
    # print(os.path.abspath(mlflow.get_artifact_uri()))
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    X_train = np.load(os.path.join(BASE_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(BASE_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(BASE_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(BASE_DIR, 'y_test.npy'))

    train_model(X_train, X_test, y_train, y_test)