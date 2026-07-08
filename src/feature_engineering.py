import pandas as pd
from imblearn.over_sampling import SMOTE

# loading the training data we saved earlier
train = pd.read_csv("data/processed/train.csv")

X_train = train.drop('Class', axis=1)
y_train = train['Class']

print("Before SMOTE:")
print("  Fraud cases  :", y_train.sum())
print("  Normal cases :", (y_train == 0).sum())
print("  Fraud %      :", round(y_train.sum() / len(y_train) * 100, 4), "%")

# the dataset is heavily imbalanced (only 0.17% fraud)
# SMOTE creates synthetic fraud samples so the model learns fraud patterns better
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print("  Fraud cases  :", y_resampled.sum())
print("  Normal cases :", (y_resampled == 0).sum())
print("  Total samples:", len(y_resampled))
print("  New shape    :", X_resampled.shape)

print("\nSMOTE done. Data is now balanced and ready for training!")