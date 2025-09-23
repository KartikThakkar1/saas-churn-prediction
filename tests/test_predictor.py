from src.predict import Predictor

def test_predictor_end_to_end(sample_payload):
    p = Predictor()                  # loads model/ + preprocessor/
    out = p.predict(sample_payload)  # runs preprocessing + prediction

    # Assertions
    assert "churn_probability" in out.columns
    assert "churn_pred" in out.columns

    prob = float(out.loc[0, "churn_probability"])
    pred = int(out.loc[0, "churn_pred"])
    assert 0.0 <= prob <= 1.0
    assert pred in (0, 1)