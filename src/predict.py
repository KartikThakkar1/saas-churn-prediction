"""
predict.py

load the trained churn model + fitted preprocessor and run predictions on *raw* inputs.
raw schema matches the original csv (no engineered columns required).

required raw fields per record:
  - company_size: {"Small","Medium","Large"}
  - subscription_plan: {"Basic","Pro","Enterprise"}
  - months_active: int >= 1
  - monthly_revenue: float/int
  - days_since_last_login: int >= 0
  - monthly_logins: int >= 0
  - features_used: int >= 1
  - support_tickets: int >= 0
  - satisfaction_score: float in [1,5]
  - payment_failures: int >= 0

optional:
  - customer_id: str

notes:
  - we engineer all derived features internally via src.data_prep.preprocess_data()
  - the fitted OneHotEncoder in the saved preprocessor uses handle_unknown="ignore",
    so unseen categories are safely ignored at inference.
"""


from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Union

import json
import joblib
import numpy as np
import pandas as pd

# i import only the wrapper that builds *all* engineered columns exactly as in training
from src.data_prep import preprocess_data


# ---------- paths & artifact loading ----------

def _project_root() -> Path:
    # resolve repo root assuming this file lives in src/
    return Path(__file__).resolve().parents[1]


def load_artifacts(artifacts_dir: Union[str, Path] = None):
    """
    load model, preprocessor, and the original feature column list
    that the preprocessor expects (pre-transform, i.e., raw *input* columns).

    returns:
        model, preprocessor, feature_cols (list[str])
    """
    root = _project_root()
    adir = Path(artifacts_dir) if artifacts_dir else (root / "model")

    model = joblib.load(adir / "model.pkl")
    preprocessor = joblib.load(adir / "preprocessor.pkl")

    with open(adir / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    return model, preprocessor, feature_cols


# ---------- payload handling ----------

RAW_REQUIRED = [
    "company_size",
    "subscription_plan",
    "months_active",
    "monthly_revenue",
    "days_since_last_login",
    "monthly_logins",
    "features_used",
    "support_tickets",
    "satisfaction_score",
    "payment_failures",
]

RAW_OPTIONAL = ["customer_id"]


def _to_dataframe(payload: Union[Dict, List[Dict]]) -> pd.DataFrame:
    """
    accept a single dict or a list of dicts and return a dataframe.
    """
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    elif isinstance(payload, list) and all(isinstance(r, dict) for r in payload):
        return pd.DataFrame(payload)
    else:
        raise ValueError("payload must be a dict or a list[dict]")


def _validate_required_columns(df_raw: pd.DataFrame):
    """
    ensuring all required raw fields are present.
    i fail fast with a clear error if anything is missing.
    """
    missing = [c for c in RAW_REQUIRED if c not in df_raw.columns]
    if missing:
        raise ValueError(f"missing required input fields: {missing}")


# ---------- core predictor ----------

class Predictor:
    """
    minimal reusable predictor:
      - loads artifacts once
      - engineers features with preprocess_data()
      - applies fitted preprocessor
      - returns probabilities and hard labels
    """

    def __init__(self, artifacts_dir: Union[str, Path] = None):
        self.model, self.preprocessor, self.feature_cols = load_artifacts(artifacts_dir)

    def _prepare_X(self, df_raw: pd.DataFrame) -> np.ndarray:
        # engineer the same features we used in training
        df_eng = preprocess_data(df_raw)

        # drop target if someone sends it by mistake
        if "churned" in df_eng.columns:
            df_eng = df_eng.drop(columns=["churned"])

        # ensure every expected input column exists after engineering
        missing_inputs = [c for c in self.feature_cols if c not in df_eng.columns]
        if missing_inputs:
            raise ValueError(
                f"engineered dataframe is missing columns expected by the preprocessor: {missing_inputs}\n"
                f"did the raw schema change?"
            )

        # transform with the fitted column transformer
        X = self.preprocessor.transform(df_eng[self.feature_cols])
        return X

    def predict_proba(self, payload: Union[Dict, List[Dict]]) -> pd.DataFrame:
        """
        return a dataframe with probabilities (and keep any supplied customer_id).
        """
        df_raw = _to_dataframe(payload)
        _validate_required_columns(df_raw)

        X = self._prepare_X(df_raw)
        proba = self.model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)

        out = pd.DataFrame({
            "churn_probability": proba,
            "churn_pred": preds,
        })

        # if caller provided ids, include them for traceability
        if "customer_id" in df_raw.columns:
            out.insert(0, "customer_id", df_raw["customer_id"].values)

        return out

    def predict(self, payload: Union[Dict, List[Dict]], threshold: float = 0.5) -> pd.DataFrame:
        """
        like predict_proba but allows a custom threshold for the positive class.
        """
        df = self.predict_proba(payload)
        df["churn_pred"] = (df["churn_probability"] >= threshold).astype(int)
        return df


# ---------- simple cli for ad-hoc checks ----------

def _example_record() -> Dict:
    # a minimal, valid single-record example
    return {
        "customer_id": "CUST_99999",
        "company_size": "Small",
        "subscription_plan": "Basic",
        "months_active": 5,
        "monthly_revenue": 29,
        "days_since_last_login": 9,
        "monthly_logins": 6,
        "features_used": 3,
        "support_tickets": 1,
        "satisfaction_score": 2.8,
        "payment_failures": 0,
    }


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="run churn predictions on raw input json")
    parser.add_argument("--input_json", type=str, default=None, help="path to a json file (dict or list[dict])")
    parser.add_argument("--artifacts_dir", type=str, default=None, help="custom artifacts dir (default: ./model)")
    parser.add_argument("--threshold", type=float, default=0.5, help="decision threshold for churn_pred")
    args = parser.parse_args()

    predictor = Predictor(artifacts_dir=args.artifacts_dir)

    if args.input_json:
        with open(args.input_json, "r") as f:
            payload = json.load(f)
    else:
        # if no file provided, i run the built-in example
        payload = _example_record()

    df_out = predictor.predict(payload, threshold=args.threshold)
    # print as csv to stdout for quick inspection
    df_out.to_csv(sys.stdout, index=False)