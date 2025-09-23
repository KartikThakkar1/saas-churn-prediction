"""
data_prep.py

Reusable data preprocessing and feature engineering functions
for SaaS churn prediction.
"""

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer engagement-related features.

    Features:
    - login_engagement_score: scaled inverse of days since last login
    - feature_adoption_rate: percent of features used
    - login_frequency_category: bin monthly logins into categories
    - login_recency_category: bin days since last login into recency buckets
    """
    df = df.copy()
    max_days = df['days_since_last_login'].max()

    df['login_engagement_score'] = 100 - (df['days_since_last_login'] / max_days * 100)
    df['login_engagement_score'] = df['login_engagement_score'].clip(0, 100)

    df['feature_adoption_rate'] = (df['features_used'] / 10 * 100).clip(0, 100)

    df['login_frequency_category'] = df['monthly_logins'].apply(
        lambda x: 'high' if x >= 20 else 'medium' if x >= 10 else 'low' if x >= 5 else 'very_low'
    )

    df['login_recency_category'] = df['days_since_last_login'].apply(
        lambda x: 'very_recent' if x <= 1 else 'recent' if x <= 3 else 'moderate' if x <= 7 else 'old' if x <= 14 else 'very_old'
    )

    return df


def create_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create churn risk-related features.


    Features created :
      - payment_risk_score (0–100): How risky the account looks from a billing perspective. Higher = more failed payments => higher churn risk.
      - satisfaction_risk (0–100): How risky the account looks from a sentiment perspective. Higher = lower satisfaction => higher churn risk.
      - support_intensity (>= 0): How frequently this customer needs help relative to tenure. Higher = more recurring friction => higher churn risk.
      - overall_risk_score (0–100): A single, interpretable churn risk index combining billing, satisfaction, and support intensity (weighted blend). Higher = higher overall churn risk.
      - high_risk_customer (0/1): Operational flag for very risky accounts (overall_risk_score > 60).

    """
    df = df.copy()

    df['payment_risk_score'] = (df['payment_failures'] * 25).clip(0, 100)
    df['satisfaction_risk'] = (5 - df['satisfaction_score']) * 25
    df['support_intensity'] = df['support_tickets'] / df['months_active']
    df['support_intensity'] = df['support_intensity'].fillna(0)

    df['overall_risk_score'] = (
        df['payment_risk_score'] * 0.3 +
        df['satisfaction_risk'] * 0.4 +
        (df['support_intensity'] * 20).clip(0, 100) * 0.3
    )

    df['high_risk_customer'] = (df['overall_risk_score'] > 60).astype(int)

    return df


def create_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer lifecycle features.

    Features:
      - tenure_category {'new','growing','mature','veteran'}: Customer lifecycle stage.
                   new (≤3m): onboarding risk,
                   growing (≤12m): adoption shaping,
                   mature (≤24m): stabilized usage,
                   veteran (>24m): entrenched accounts.
      - revenue_per_feature (>= 0): Value extraction efficiency (revenue per distinct feature used).
                   Lower values can indicate under-utilization risk.
      - customer_value_segment {'low_value','medium_value','high_value'}: Revenue tier used for retention and prioritization strategies.
      - activity_score (0–100): Overall product engagement level, blending frequency (logins) and breadth (features used) into one interpretable KPI.
                    Higher = more active customer.
    """
    df = df.copy()

    df['tenure_category'] = df['months_active'].apply(
        lambda x: 'new' if x <= 3 else 'growing' if x <= 12 else 'mature' if x <= 24 else 'veteran'
    )

    df['revenue_per_feature'] = (df['monthly_revenue'] / df['features_used']).fillna(0)

    df['customer_value_segment'] = df['monthly_revenue'].apply(
        lambda x: 'high_value' if x >= 250 else 'medium_value' if x >= 100 else 'low_value'
    )

    max_logins = df['monthly_logins'].max()
    max_features = df['features_used'].max()
    login_score = df['monthly_logins'] / max_logins if max_logins > 0 else 0
    feature_score = df['features_used'] / max_features if max_features > 0 else 0
    df['activity_score'] = (login_score * 0.6 + feature_score * 0.4) * 100

    return df


def apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transforms, flags, and binnings to selected features.
    """
    df = df.copy()

    # log transforms
    df['months_active_log'] = np.log1p(df['months_active'])
    df['monthly_revenue_log'] = np.log1p(df['monthly_revenue'])
    df['days_since_last_login_log'] = np.log1p(df['days_since_last_login'])
    df['monthly_logins_log'] = np.log1p(df['monthly_logins'])
    df['support_intensity_log'] = np.log1p(df['support_intensity'])
    df['revenue_per_feature_log'] = np.log1p(df['revenue_per_feature'])

    # binary flag
    df['had_payment_failure'] = (df['payment_failures'] > 0).astype(int)

    # satisfaction bins
    def bin_satisfaction(score):
        if score < 3:
            return 'low'
        elif score < 4:
            return 'medium'
        else:
            return 'high'

    df['satisfaction_level'] = df['satisfaction_score'].apply(bin_satisfaction)

    return df


def convert_booleans_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all boolean columns to integers (0/1).
    """
    df = df.copy()
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline: engagement, risk, lifecycle, transformations, boolean fix.
    """
    df = create_engagement_features(df)
    df = create_risk_features(df)
    df = create_lifecycle_features(df)
    df = apply_transformations(df)
    df = convert_booleans_to_int(df)
    return df



def get_feature_lists_for_model() -> dict:
    """
    return the column groups expected by the model preprocessor.
    note: these are the engineered/transformed columns from our pipeline.
    """
    categorical = [
        'company_size',
        'subscription_plan',
        'login_frequency_category',
        'login_recency_category',
        'tenure_category',
        'customer_value_segment',
        'satisfaction_level',  # created in apply_transformations
    ]

    # binary features we do not scale
    binary = [
        'had_payment_failure',   # created in apply_transformations
        'high_risk_customer',    # created in create_risk_features
    ]

    # numeric features we do scale (engineered + raw numerics we kept)
    numeric = [
        'months_active_log',
        'monthly_revenue_log',
        'days_since_last_login_log',
        'monthly_logins_log',
        'features_used',
        'support_tickets',
        'satisfaction_score',
        'login_engagement_score',
        'feature_adoption_rate',
        'payment_risk_score',
        'satisfaction_risk',
        'support_intensity_log',
        'overall_risk_score',
        'revenue_per_feature_log',
        'activity_score',
    ]

    return {"categorical": categorical, "binary": binary, "numeric": numeric}



def build_preprocessor(categorical_cols, numeric_cols, binary_cols):
    """
    create a ColumnTransformer that:
      - one-hot encodes categoricals (handle_unknown='ignore')
      - standard-scales numeric cols
      - passes binary cols through unchanged
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("bin", "passthrough", binary_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def fit_transform_preprocessor(df: pd.DataFrame, preprocessor: ColumnTransformer, feature_cols: list) -> np.ndarray:
    """
    fit the preprocessor on df[feature_cols] and return the transformed matrix.
    """
    X = preprocessor.fit_transform(df[feature_cols])
    return X

def transform_with_preprocessor(df: pd.DataFrame, preprocessor: ColumnTransformer, feature_cols: list) -> np.ndarray:
    """
    transform df[feature_cols] using an already-fitted preprocessor.
    """
    X = preprocessor.transform(df[feature_cols])
    return X

def get_output_feature_names(preprocessor: ColumnTransformer, feature_cols: list) -> list:
    """
    get the expanded feature names after the column transformer (for saving).
    """
    # sklearn >=1.0 supports get_feature_names_out
    try:
        names = preprocessor.get_feature_names_out(feature_cols).tolist()
    except Exception:
        # fallback: best-effort names
        names = []
    return names


def preprocess_for_model(df: pd.DataFrame):
    """
    full preprocessing for training:
      1) engineer features (engagement, risk, lifecycle)
      2) apply transformations (logs, flags, bins)
      3) convert booleans to ints
      4) build and fit a preprocessor (scale numeric, ohe categorical, passthrough binary)
      5) return X, y, preprocessor, and column metadata
    """
    df = preprocess_data(df)  # wrapper: engagement -> risk -> lifecycle -> transforms -> boolean fix

    cols = get_feature_lists_for_model()
    feature_cols = cols["numeric"] + cols["binary"] + cols["categorical"]

    preprocessor = build_preprocessor(
        categorical_cols=cols["categorical"],
        numeric_cols=cols["numeric"],
        binary_cols=cols["binary"],
    )

    X = fit_transform_preprocessor(df, preprocessor, feature_cols)
    y = df['churned'].astype(int).values

    out_feature_names = get_output_feature_names(preprocessor, feature_cols)

    return X, y, preprocessor, feature_cols, out_feature_names
