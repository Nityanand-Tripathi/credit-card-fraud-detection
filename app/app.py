"""
app.py — Credit Card Fraud Detection web app.

Built from scratch to replace a broken app.py, but designed to load your
EXISTING trained artifacts rather than retrain anything:

    models/best_model.pkl
    models/logistic_regression.pkl
    models/random_forest.pkl
    models/xgboost.pkl
    models/scaler_params.json

------------------------------------------------------------------------
ASSUMPTIONS (documented so you can fix in 2 minutes if your setup differs)
------------------------------------------------------------------------
1. Dataset is the classic Kaggle "Credit Card Fraud Detection" dataset:
   columns Time, V1..V28, Amount, Class. This app expects exactly that
   feature order: [Time, V1, V2, ..., V28, Amount] = 30 features.

2. scaler_params.json is assumed to look like:
       { "mean": [<30 numbers>], "scale": [<30 numbers>] }
   i.e. a serialized sklearn StandardScaler (mean_ and scale_ attributes),
   fit on ALL 30 columns (Time, V1-V28, Amount) OR just [Time, Amount].
   The code below tries the 30-feature case first and falls back to a
   2-feature [Time, Amount] scaler automatically — see `apply_scaler()`.
   If your real file uses different keys, edit `apply_scaler()` only.

3. Each .pkl is a plain pickle of a scikit-learn-compatible estimator
   (has .predict() and .predict_proba()). xgboost.pkl is assumed to be
   an xgboost.XGBClassifier pickled the same way — if it was saved with
   `booster.save_model()` instead, see the try/except in `load_models()`.

4. "best_model.pkl" is used by default; the UI lets you switch between
   all 4 files so you can compare them without touching code.

If any assumption is wrong, the app still starts — it will show a clear
error message in the UI (via flash-style JSON response) instead of a
silent crash, and logs the real exception to logs/app.log.
"""

import os
import json
import pickle
import logging
import traceback
from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, jsonify

# --------------------------------------------------------------------------
# App + logging setup
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ccfd")

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]  # 30 features

MODEL_FILES = {
    "best_model": "best_model.pkl",
    "logistic_regression": "logistic_regression.pkl",
    "random_forest": "random_forest.pkl",
    "xgboost": "xgboost.pkl",
}

# --------------------------------------------------------------------------
# Load models + scaler once at startup (not per-request — much faster)
# --------------------------------------------------------------------------
_models = {}
_scaler = None
_load_errors = {}


def load_models():
    """Load every .pkl we can find. Missing/broken files are skipped, not fatal."""
    for key, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            _load_errors[key] = f"File not found: {path}"
            log.warning(_load_errors[key])
            continue
        try:
            with open(path, "rb") as f:
                _models[key] = pickle.load(f)
            log.info(f"Loaded model '{key}' from {filename}")
        except Exception as e:
            _load_errors[key] = f"Failed to unpickle {filename}: {e}"
            log.error(_load_errors[key])


def load_scaler():
    """
    Load scaler_params.json and build a callable that scales a raw
    30-length feature vector. Handles two common shapes:
      A) scaler fit on all 30 columns -> mean/scale each length 30
      B) scaler fit on just [Time, Amount] -> mean/scale each length 2
    """
    global _scaler
    path = os.path.join(MODELS_DIR, "scaler_params.json")
    if not os.path.exists(path):
        _load_errors["scaler"] = f"scaler_params.json not found at {path}"
        log.warning(_load_errors["scaler"])
        return

    try:
        with open(path, "r") as f:
            params = json.load(f)
        mean = np.array(params["mean"], dtype=float)
        scale = np.array(params["scale"], dtype=float)
        # Guard against zero-scale (division by zero) entries
        scale[scale == 0] = 1.0
        _scaler = {"mean": mean, "scale": scale, "mode": "full" if len(mean) == 30 else "time_amount"}
        log.info(f"Loaded scaler ({_scaler['mode']} mode, {len(mean)} params)")
    except Exception as e:
        _load_errors["scaler"] = f"Failed to parse scaler_params.json: {e}"
        log.error(_load_errors["scaler"])


def apply_scaler(vector):
    """
    Apply the loaded scaler to a raw 30-length feature vector
    [Time, V1..V28, Amount]. Falls back to returning the vector
    unscaled (with a logged warning) if no scaler loaded.
    """
    if _scaler is None:
        log.warning("No scaler loaded — using raw unscaled features.")
        return vector

    v = vector.copy()
    if _scaler["mode"] == "full":
        v = (v - _scaler["mean"]) / _scaler["scale"]
    else:
        # time_amount mode: only Time (index 0) and Amount (index 29) are scaled
        v[0] = (v[0] - _scaler["mean"][0]) / _scaler["scale"][0]
        v[29] = (v[29] - _scaler["mean"][1]) / _scaler["scale"][1]
    return v


load_models()
load_scaler()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    """Main page: prediction form + model status."""
    available_models = list(_models.keys())
    return render_template(
        "index.html",
        available_models=available_models,
        load_errors=_load_errors,
        feature_order=FEATURE_ORDER,
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
        {
          "model": "best_model",
          "time": 0,
          "amount": 84.12,
          "v_features": [<28 numbers>]   // optional, defaults to zeros
        }

    Returns JSON:
        { "ok": true, "prediction": 0|1, "probability": 0.0-1.0, "model_used": "..." }
      or
        { "ok": false, "error": "human readable message" }
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}

        model_key = payload.get("model", "best_model")
        if model_key not in _models:
            return jsonify({
                "ok": False,
                "error": f"Model '{model_key}' is not loaded. "
                         f"Available: {list(_models.keys()) or 'none — check logs/app.log'}"
            }), 400

        try:
            time_val = float(payload.get("time", 0))
            amount_val = float(payload.get("amount", 0))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Time and Amount must be numbers."}), 400

        if amount_val < 0:
            return jsonify({"ok": False, "error": "Amount cannot be negative."}), 400

        v_features = payload.get("v_features")
        if v_features is None:
            v_features = [0.0] * 28
        if len(v_features) != 28:
            return jsonify({
                "ok": False,
                "error": f"Expected 28 V-features, received {len(v_features)}."
            }), 400
        try:
            v_features = [float(x) for x in v_features]
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "All V-features must be numbers."}), 400

        raw_vector = np.array([time_val] + v_features + [amount_val], dtype=float)
        scaled_vector = apply_scaler(raw_vector).reshape(1, -1)

        model = _models[model_key]
        prediction = int(model.predict(scaled_vector)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled_vector)[0]
            # class 1 = fraud, by dataset convention
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])

        log.info(
            f"Prediction made | model={model_key} amount={amount_val} "
            f"time={time_val} -> pred={prediction} prob={probability}"
        )

        return jsonify({
            "ok": True,
            "prediction": prediction,
            "probability": probability,
            "model_used": model_key,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    except Exception as e:
        log.error(f"Unhandled error in /api/predict: {e}\n{traceback.format_exc()}")
        return jsonify({
            "ok": False,
            "error": "Internal error while scoring the transaction. Check logs/app.log."
        }), 500


@app.route("/api/health")
def health():
    """Simple health check — also reports which models/scaler loaded OK."""
    return jsonify({
        "ok": True,
        "models_loaded": list(_models.keys()),
        "scaler_loaded": _scaler is not None,
        "load_errors": _load_errors,
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"ok": False, "error": "Route not found."}), 404


@app.errorhandler(500)
def server_error(e):
    log.error(f"500 error: {e}")
    return jsonify({"ok": False, "error": "Internal server error."}), 500


if __name__ == "__main__":
    # debug=True is fine for local dev; turn off before any real deployment.
    app.run(host="127.0.0.1", port=5000, debug=True)