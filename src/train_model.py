import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import time

train = pd.read_csv("data/processed/train.csv")
X_train = train.drop('Class', axis=1)
y_train = train['Class']

print("=== Model Training Started ===")
print("Training samples:", len(X_train))
print("Fraud cases:", y_train.sum())
print("Normal cases:", (y_train == 0).sum())

# balancing the data first
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print("After SMOTE - Total samples:", len(y_res))

os.makedirs("models", exist_ok=True)

# this ratio helps xgboost handle imbalanced data better
ratio = round((y_train == 0).sum() / y_train.sum(), 2)
print("Class ratio for XGBoost:", ratio)

# Logistic Regression
print("\n[1/3] Training Logistic Regression...")
start = time.time()
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_res, y_res)
joblib.dump(lr, "models/logistic_regression.pkl")
print("Done in", round(time.time() - start, 2), "seconds")

# Random Forest 
print("\n[2/3] Training Random Forest...")
start = time.time()
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_res, y_res)
joblib.dump(rf, "models/random_forest.pkl")
print("Done in", round(time.time() - start, 2), "seconds")

#  XGBoost 
print("\n[3/3] Training XGBoost...")
start = time.time()
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=ratio,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb.fit(X_res, y_res)
joblib.dump(xgb, "models/xgboost.pkl")
print("Done in", round(time.time() - start, 2), "seconds")

print("\n=== All 3 models trained and saved! ===")