import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import joblib
import os

def apply_smote(X_train, y_train):
    print("Applying SMOTE to handle class imbalance...")
    print(f"Before SMOTE - Fraud: {y_train.sum()}, Normal: {(y_train==0).sum()}")
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Fraud: {y_resampled.sum()}, Normal: {(y_resampled==0).sum()}")
    print(f"New training shape: {X_resampled.shape}")
    return X_resampled, y_resampled

if __name__ == "__main__":
    print("Loading processed data...")
    train = pd.read_csv("data/processed/train.csv")
    X_train = train.drop('Class', axis=1)
    y_train = train['Class']
    X_resampled, y_resampled = apply_smote(X_train, y_train)
    print(" Feature engineering done!")