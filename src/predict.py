import pandas as pd
import joblib
import os

# loading the best model we saved after evaluation
def load_model():
    model = joblib.load("models/best_model.pkl")
    return model

def predict(model, transaction):
    df = pd.DataFrame([transaction])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability

if __name__ == "__main__":
    print("Loading model...")
    model = load_model()

    # loading test data to try some predictions
    test = pd.read_csv("data/processed/test.csv")
    X_test = test.drop("Class", axis=1)
    y_test = test["Class"]

    print("\n--- Sample Predictions ---\n")

    # testing on 5 random transactions
    for i in range(5):
        row = X_test.iloc[i].to_dict()
        actual = y_test.iloc[i]
        pred, prob = predict(model, row)

        print(f"Transaction {i+1}:")
        print(f"  Actual  : {'FRAUD' if actual == 1 else 'NORMAL'}")
        print(f"  Predicted: {'FRAUD' if pred == 1 else 'NORMAL'}")
        print(f"  Fraud Probability: {round(prob * 100, 2)}%")
        print()

    print("Done!")