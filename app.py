# app.py
"""
SaaS Churn Prediction API (FastAPI)

- Loads the trained model & fitted preprocessor ON STARTUP (via FastAPI lifespan)
- Exposes:
    GET  /health          -> basic sanity/uptime check
    GET  /model_info      -> shows training metrics (from model/metrics.json)
    POST /predict         -> predict for a single customer
    POST /predict/batch   -> predict for many customers
- Input is RAW customer fields (no engineered features needed). The service
  applies the same feature engineering & preprocessing used during training.

Notes:
- Uses Pydantic v2: BaseModel.model_dump() (dict() is deprecated).
- The predictor is lazy-loaded when the app starts, not at import time.
- If you see 404s on "/" or "/favicon.ico", we redirect "/" to /docs
  and return 204 for "/favicon.ico" to keep logs quiet.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Literal

import json
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, conint, confloat
from starlette.middleware.cors import CORSMiddleware

# Our reusable inference helper (loads artifacts, preprocesses, predicts)
from src.predict import Predictor

APP_TITLE = "SaaS Churn Prediction API"
ARTIFACTS_DIR = Path("model")  # expects artifacts from `python -m src.train`

# Will hold the Predictor instance after startup
predictor: Optional[Predictor] = None


# ------------------------------------------------------------------------------
# Lifespan: modern alternative to @app.on_event("startup") / ("shutdown")
# This runs once when the app boots, and once when it shuts down.
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = Predictor(artifacts_dir=ARTIFACTS_DIR)  # load model + preprocessor here
    try:
        yield
    finally:
        # optional cleanup (free handles, etc.)
        predictor = None


# Create the app with lifespan handler
app = FastAPI(title=APP_TITLE, version="0.1.0", lifespan=lifespan)

# (Optional) Allow local tools/frontends to call the API easily.
# Tighten CORS in real environments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Pydantic request/response schemas
# (Using constrained types for basic validation; v2 supports conint/confloat)
# ------------------------------------------------------------------------------
class CustomerRecord(BaseModel):
    # Optional ID for traceability in responses
    customer_id: Optional[str] = None

    # Required raw fields (match your original CSV)
    company_size: Literal["Small", "Medium", "Large"]
    subscription_plan: Literal["Basic", "Pro", "Enterprise"]
    months_active: conint(ge=1)
    monthly_revenue: confloat(ge=0)
    days_since_last_login: conint(ge=0)
    monthly_logins: conint(ge=0)
    features_used: conint(ge=1)
    support_tickets: conint(ge=0)
    satisfaction_score: confloat(ge=1, le=5)
    payment_failures: conint(ge=0)


class Prediction(BaseModel):
    # Mirrors the output DataFrame from Predictor
    customer_id: Optional[str] = None
    churn_probability: float
    churn_pred: int


# ------------------------------------------------------------------------------
# Convenience routes
# ------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    """Redirect humans to the interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Silence default browser favicon requests."""
    return Response(status_code=204)


@app.get("/health")
def health():
    """Basic liveness check."""
    ok = predictor is not None
    return {"status": "ok" if ok else "booting", "model_dir": str(ARTIFACTS_DIR)}


@app.get("/model_info")
def model_info():
    """
    Return training-time metrics and info saved by src/train.py.
    Useful for quick smoke checks after redeploys.
    """
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    try:
        return json.loads(metrics_path.read_text())
    except Exception as e:
        return {"error": f"Could not load metrics.json: {e}"}


# ------------------------------------------------------------------------------
# Inference routes
# ------------------------------------------------------------------------------
@app.post("/predict", response_model=Prediction)
def predict_single(
    record: CustomerRecord = Body(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold for churn_pred"),
):
    """
    Predict churn for ONE customer. Uses the same preprocessing as training.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    # Pydantic v2: use model_dump() instead of dict()
    df_out = predictor.predict(record.model_dump(), threshold=threshold)
    row = df_out.iloc[0].to_dict()
    return Prediction(**row)


@app.post("/predict/batch", response_model=List[Prediction])
def predict_batch(
    records: List[CustomerRecord] = Body(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold for churn_pred"),
):
    """
    Predict churn for MANY customers at once.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    payload = [r.model_dump() for r in records]
    df_out = predictor.predict(payload, threshold=threshold)
    return [Prediction(**rec) for rec in df_out.to_dict(orient="records")]
