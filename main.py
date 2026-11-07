import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load .env from the same folder as main.py (works no matter where uvicorn is started from).
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

app = FastAPI(title="Smart Waste Bin ML API", version="1.0.0")

MONGODB_URI = (os.getenv("MONGODB_URI") or "").strip()
MONGODB_DB = os.getenv("MONGODB_DB", "test")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "readings")
BIN_ID = os.getenv("BIN_ID", "BIN-01")
HORIZON_MINUTES = int((os.getenv("HORIZON_MINUTES") or "5").strip())

_model: Optional[RandomForestRegressor] = None
_model_info = {"trained": False}


def get_collection():
    if not MONGODB_URI:
        raise HTTPException(status_code=503, detail="MONGODB_URI environment variable is not configured")
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=7000)
        client.admin.command("ping")
        return client[MONGODB_DB][MONGODB_COLLECTION]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MongoDB connection failed: {str(exc)}")


def load_readings():
    collection = get_collection()
    docs = list(collection.find(
        {"binId": BIN_ID},
        {"_id": 0, "binId": 1, "distanceCm": 1, "fillLevel": 1, "createdAt": 1}
    ).sort("createdAt", 1).limit(5000))
    if not docs:
        raise HTTPException(status_code=404, detail=f"No readings found for {BIN_ID}")

    df = pd.DataFrame(docs)
    required = ["distanceCm", "fillLevel", "createdAt"]
    if any(col not in df.columns for col in required):
        raise HTTPException(status_code=422, detail="Readings must contain distanceCm, fillLevel and createdAt")

    df["distanceCm"] = pd.to_numeric(df["distanceCm"], errors="coerce")
    df["fillLevel"] = pd.to_numeric(df["fillLevel"], errors="coerce")
    df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True, errors="coerce")
    df = df.dropna(subset=required)
    df = df[(df.distanceCm >= 0) & (df.fillLevel >= 0) & (df.fillLevel <= 100)]
    df = df.sort_values("createdAt").drop_duplicates(subset=["createdAt", "fillLevel", "distanceCm"])
    if len(df) < 10:
        raise HTTPException(status_code=422, detail="Not enough valid readings; collect at least 10 first")
    return df.reset_index(drop=True)


def make_training_data(df):
    # Each sample uses recent readings to predict the fill level about 5 minutes later.
    rows = []
    times = df["createdAt"].tolist()
    fills = df["fillLevel"].to_numpy(dtype=float)
    distances = df["distanceCm"].to_numpy(dtype=float)
    horizon = pd.Timedelta(minutes=HORIZON_MINUTES)

    for i in range(5, len(df) - 1):
        target_idx = None
        target_time = times[i] + horizon
        for j in range(i + 1, len(df)):
            delta = times[j] - target_time
            if delta >= pd.Timedelta(0):
                if delta <= pd.Timedelta(minutes=2):
                    target_idx = j
                break
        if target_idx is None:
            continue
        elapsed = max((times[i] - times[i - 5]).total_seconds() / 60.0, 0.01)
        slope = (fills[i] - fills[i - 5]) / elapsed
        rows.append([
            fills[i], distances[i], fills[i - 1], fills[i - 3], fills[i - 5], slope
        ] + [fills[target_idx]])

    if len(rows) < 20:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(rows)} training sequences found. Need at least 20; collect readings over a longer period."
        )
    values = np.asarray(rows, dtype=float)
    return values[:, :-1], values[:, -1]


@app.get("/")
def root():
    return {"service": "Smart Waste Bin ML API", "status": "running"}


@app.get("/health")
def health():
    return {"success": True, "service": "ml-api", "model_trained": _model is not None}


@app.post("/train")
def train():
    global _model, _model_info
    df = load_readings()
    X, y = make_training_data(df)

    split = max(int(len(X) * 0.8), 1)
    if len(X) - split < 1:
        raise HTTPException(status_code=422, detail="Not enough samples for train/test evaluation")

    model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    model.fit(X[:split], y[:split])
    predictions = model.predict(X[split:])
    mae = float(mean_absolute_error(y[split:], predictions))

    # Refit on all available samples after evaluation.
    model.fit(X, y)
    _model = model
    _model_info = {
        "trained": True,
        "binId": BIN_ID,
        "readings_used": int(len(df)),
        "training_sequences": int(len(X)),
        "horizon_minutes": HORIZON_MINUTES,
        "test_mae_percentage_points": round(mae, 2),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "note": "Experimental estimate; reliability depends on collecting realistic readings over time."
    }
    return {"success": True, "model": _model_info}


@app.get("/predict")
def predict():
    if _model is None:
        raise HTTPException(status_code=409, detail="Model is not trained yet. Call POST /train first.")
    df = load_readings()
    if len(df) < 6:
        raise HTTPException(status_code=422, detail="Need at least 6 recent readings")
    tail = df.tail(6).reset_index(drop=True)
    fills = tail["fillLevel"].to_numpy(dtype=float)
    distances = tail["distanceCm"].to_numpy(dtype=float)
    elapsed = max((tail.createdAt.iloc[-1] - tail.createdAt.iloc[0]).total_seconds() / 60.0, 0.01)
    slope = (fills[-1] - fills[0]) / elapsed
    features = [[fills[-1], distances[-1], fills[-2], fills[-4], fills[-6], slope]]
    predicted = float(np.clip(_model.predict(features)[0], 0, 100))
    return {
        "success": True,
        "binId": BIN_ID,
        "current_fill_level": round(float(fills[-1]), 2),
        "predicted_fill_level": round(predicted, 2),
        "prediction_horizon_minutes": HORIZON_MINUTES,
        "model_test_mae_percentage_points": _model_info.get("test_mae_percentage_points"),
        "model": _model_info,
        "warning": "Experimental prediction, not a guaranteed forecast."
    }


@app.get("/model-info")
def model_info():
    return _model_info