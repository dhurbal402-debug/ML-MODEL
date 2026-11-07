# Smart Waste Bin ML API

A separate FastAPI service that reads bin measurements from the existing MongoDB collection, cleans them, trains a Random Forest model, and predicts the fill level approximately five minutes ahead.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set environment variables (do not commit credentials):

```powershell
$env:MONGODB_URI="your MongoDB Atlas connection string"
$env:MONGODB_DB="your_database_name"
$env:MONGODB_COLLECTION="readings"
$env:BIN_ID="BIN-01"
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## API

- `GET /health` — service health
- `POST /train` — fetch and clean MongoDB readings, create time-based training samples, evaluate on the latest 20%, and train the model
- `GET /predict` — predict fill level after `HORIZON_MINUTES` (default 5)
- `GET /model-info` — training metrics and metadata

## Render deployment

Create a new **Web Service** from this folder/repository. Runtime: Python. Build command: `pip install -r requirements.txt`. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.

Add environment variables in Render: `MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION`, `BIN_ID`, and optionally `HORIZON_MINUTES`.

After deployment, call `POST https://YOUR-ML-SERVICE.onrender.com/train` once, then test `GET /predict`.

## Important data limitation

The current prototype dataset contains a short period of readings and some abrupt fill-level resets. The model is experimental; its test MAE should be reported and the model should be retrained with longer, realistic measurements before making strong accuracy claims. Fill level itself is calculated from distance, so the prediction target here is a future fill level, not a second estimate of the current fill level.
