import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()}")
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time'] = scaler.fit_transform(df[['Time']])
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test):
    print("Saving processed data...")
    os.makedirs("data/processed", exist_ok=True)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    print("Saved train.csv and test.csv!")

if __name__ == "__main__":
    df = load_data("data/raw/creditcard.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    save_processed_data(X_train, X_test, y_train, y_test)