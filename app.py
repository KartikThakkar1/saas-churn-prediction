# A FastAPI app to serve churn predictions on raw inputs

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Literal

import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, conint, confloat
from starlette.middleware.cors import CORSMiddleware

# reusing the production-ready predictor (loads model + preprocessor + feature list)
from src.predict import Predictor

APP_TITLE = "SaaS Churn Prediction API"
ARTIFACTS_DIR = Path("model")  # expects model/ from `python -m src.train`

# -------- Pydantic schemas (request/response) --------

class CustomerRecord(BaseModel):
    # optional
    customer_id: Optional[str] = None

    # required raw fields (match your CSV)
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
    customer_id: Optional[str] = None
    churn_probability: float
    churn_pred: int

# -------- app initialization --------

app = FastAPI(title=APP_TITLE, version="0.1.0")

# (optional) allow local tools or frontends to call the API
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],  # tighten later
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)

# load artifacts once on startup
predictor = Predictor(artifacts_dir=ARTIFACTS_DIR)

# -------- routes --------

@app.get("/health")
def health():
    return {"status": "ok", "model_dir": str(ARTIFACTS_DIR)}

@app.get("/model_info")
def model_info():
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    try:
        info = json.loads(metrics_path.read_text())
    except Exception:
        info = {"error": "metrics.json not found; did you run training?"}
    return info

@app.post("/predict", response_model=Prediction)
def predict_single(record: CustomerRecord = Body(...), threshold: float = Query(0.5, ge=0.0, le=1.0)):
    try:
        # was: record.dict()
        df_out = predictor.predict(record.model_dump(), threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    row = df_out.iloc[0].to_dict()
    return Prediction(**row)


@app.post("/predict/batch", response_model=List[Prediction])
def predict_batch(records: List[CustomerRecord] = Body(...), threshold: float = Query(0.5, ge=0.0, le=1.0)):
    try:
        # was: [r.dict() for r in records]
        payload = [r.model_dump() for r in records]
        df_out = predictor.predict(payload, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return [Prediction(**rec) for rec in df_out.to_dict(orient="records")]
