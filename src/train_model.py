import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_models(X_train, y_train):
    os.makedirs("models", exist_ok=True)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=100, random_state=42,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
    }

    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_res, y_res)
        joblib.dump(model, f"models/{name}.pkl")
        print(f"✅ {name} saved!")
        trained[name] = model

    return trained

if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    X_train = train.drop('Class', axis=1)
    y_train = train['Class']
    train_models(X_train, y_train)
    print(" All models trained and saved!")