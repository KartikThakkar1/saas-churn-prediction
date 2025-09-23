import numpy as np
import pandas as pd

from src.data_prep import preprocess_data, get_feature_lists_for_model, build_preprocessor

def test_preprocess_timeline_runs(sample_payload):

    df_raw = pd.DataFrame([sample_payload])

    # 1. engineering new features and performing transformations on required features. 

    df_engineer_transform = preprocess_data(df_raw)   # this returns a dataframe with all the engineered+transformed features

    # 2. building a preprocessor (the one that i made during the training using ColumnTransformer)
    cols = get_feature_lists_for_model()
    feature_columns = cols["numeric"] + cols["binary"] + cols["categorical"]
    preprocessor = build_preprocessor(categorical_cols=cols["categorical"],numeric_cols=cols["numeric"],binary_cols=cols["binary"])

    # 3 fitting the preprocessor using the fit_transform method, this gives out a numeric matrix X
    X = preprocessor.fit_transform(df_engineer_transform[feature_columns])

    # assertions 
    assert X.shape[0] == 1 # the input i.e. only 1 row should lead to an output matrix with only 1 row

    assert X.shape[1] >=1 # the output has at least 1 feature

    assert np.isfinite(X).all()  # there should be no NaNs/inf in the output matrix X
    