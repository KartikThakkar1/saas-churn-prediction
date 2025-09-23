# src/train.py

"""
train.py

train churn model(s) using the reusable preprocessing pipeline.
saves:
  - model/model.pkl
  - model/preprocessor.pkl
  - model/feature_columns.json  (input columns expected by preprocessor)
  - model/output_feature_names.json (post-transform columns; helpful for debugging)
  - model/metrics.json
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# local imports
from src.data_prep import preprocess_for_model

ARTIFACTS_DIR = Path("model")
RAW_DATA = Path("data") / "raw" / "saas_churn_data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def get_models():
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=5000, class_weight="balanced", solver="saga", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    return models

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    return metrics

def main():
    # 1) load raw data
    df_raw = pd.read_csv(RAW_DATA)

    # 2) preprocess for model (engineer features + build/fit preprocessor)
    X, y, preprocessor, feature_cols, out_feature_names = preprocess_for_model(df_raw)

    # 3) split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 4) train and evaluate multiple models
    models = get_models()
    results = {}
    best_name, best_model, best_auc = None, None, -np.inf

    for name, model in models.items():
        print(f"training {name}...")
        model.fit(X_train, y_train)
        m = evaluate(model, X_test, y_test)
        results[name] = m
        print(f"  metrics: {m}")
        if m["roc_auc"] > best_auc:
            best_name, best_model, best_auc = name, model, m["roc_auc"]

    print(f"\nbest model: {best_name} (roc_auc={best_auc:.3f})")

    # 5) save artifacts
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    joblib.dump(best_model, ARTIFACTS_DIR / "model.pkl")
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")

    with open(ARTIFACTS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(ARTIFACTS_DIR / "output_feature_names.json", "w") as f:
        json.dump(out_feature_names, f, indent=2)

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(
            {
                "results_by_model": results,
                "chosen_model": best_name,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            },
            f,
            indent=2,
        )

    print("artifacts saved to ./model")

if __name__ == "__main__":
    main()
