import pandas as pd
import numpy as np
import joblib
import os

def load_best_model(model_name="random_forest"):
    model_path = f"models/{model_name}.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)
    return model

def predict_transaction(model, transaction_data):
    df = pd.DataFrame([transaction_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if prediction == 1:
        print(f"  >> FRAUD DETECTED! Probability: {probability:.4f}")
    else:
        print(f"  >> Transaction is NORMAL. Fraud probability: {probability:.4f}")

    return prediction, probability

if __name__ == "__main__":
    print("Loading test data...")
    
    if not os.path.exists("data/processed/test.csv"):
        raise FileNotFoundError("test.csv not found in data/processed/")
    
    test = pd.read_csv("data/processed/test.csv")

    # Choose model: "random_forest", "xgboost", or "logistic_regression"
    model = load_best_model("random_forest")

    print("\n--- Sample Predictions (first 5) ---\n")
    for i in range(5):
        row = test.iloc[i].drop("Class").to_dict()
        actual = test.iloc[i]["Class"]
        print(f"Transaction #{i+1}:")
        pred, prob = predict_transaction(model, row)
        print(f"  Actual:    {'FRAUD' if actual == 1 else 'NORMAL'}\n")