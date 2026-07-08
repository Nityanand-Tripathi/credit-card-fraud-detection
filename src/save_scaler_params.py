# that means predict.py can't reproduce the exact same scaling at inference time
# this script fixes that by saving the scaler stats as a tiny json file
# small enough to commit to git, unlike the 150MB raw csv
#
# IMPORTANT: main.py expects this file to look like:
#     { "mean": [<30 numbers>], "scale": [<30 numbers>] }
# in the exact feature order [Time, V1..V28, Amount].
# V1-V28 are already PCA components (pre-centered/scaled by the dataset's
# original authors), so we leave them as mean=0, scale=1 (untouched).
# Only Time and Amount are raw columns that need standardizing.

import pandas as pd
import json
import os

df = pd.read_csv("data/raw/creditcard.csv")

feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

mean = [0.0] * 30
scale = [1.0] * 30

mean[0] = float(df["Time"].mean())
scale[0] = float(df["Time"].std())
mean[29] = float(df["Amount"].mean())
scale[29] = float(df["Amount"].std())

scaler_params = {
    "feature_order": feature_order,
    "mean": mean,
    "scale": scale,
}

os.makedirs("models", exist_ok=True)
with open("models/scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)

print("saved scaler params (mean/scale, 30 features, only Time & Amount scaled)")