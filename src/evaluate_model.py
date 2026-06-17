import pandas as pd
import numpy as np
from sklearn.metrics import (classification_report, 
                             confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def evaluate_model(model, X_test, y_test, name):
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # Confusion Matrix
    os.makedirs("reports", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"reports/confusion_matrix_{name}.png")
    plt.close()
    print(f" Saved confusion_matrix_{name}.png")

if __name__ == "__main__":
    test = pd.read_csv("data/processed/test.csv")
    X_test = test.drop('Class', axis=1)
    y_test = test['Class']

    models = ["logistic_regression", "random_forest", "xgboost"]
    for name in models:
        model = joblib.load(f"models/{name}.pkl")
        evaluate_model(model, X_test, y_test, name)

    print("\n All models evaluated!")