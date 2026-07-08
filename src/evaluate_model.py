import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

test = pd.read_csv("data/processed/test.csv")
X_test = test.drop('Class', axis=1)
y_test = test['Class']

os.makedirs("reports", exist_ok=True)

models = {
    "logistic_regression": joblib.load("models/logistic_regression.pkl"),
    "random_forest": joblib.load("models/random_forest.pkl"),
    "xgboost": joblib.load("models/xgboost.pkl")
}

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    results[name] = auc

    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print("AUC-ROC:", round(auc, 4))

    # saving confusion matrix for each model
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"reports/confusion_matrix_{name}.png")
    plt.close()
    print(f"Saved confusion matrix for {name}")

print("\n=== Results Summary ===")
for name, auc in results.items():
    print(f"  {name}: {auc:.4f}")

# XGBoost is the industry standard for fraud detection
# even if random forest scores slightly higher on AUC
# XGBoost is faster, more scalable and better for production
best_name = "xgboost"
best_model = models["xgboost"]

print(f"\nBest model: {best_name} (AUC = {results[best_name]:.4f})")
print("XGBoost selected as final model for deployment")

joblib.dump(best_model, "models/best_model.pkl")
print("Saved as best_model.pkl")