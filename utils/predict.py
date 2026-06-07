"""
utils/predict.py
Helper to load best model and run inference on new inputs.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from utils.preprocess import engineer_features, encode_categoricals, get_feature_list


MODEL_PATH = Path("models/best_model.pkl")
ENCODER_PATH = Path("models/encoders.pkl")


def load_model():
    """Load the saved best model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    return joblib.load(MODEL_PATH)


def load_encoders():
    """Load the saved label encoders."""
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Encoders not found at {ENCODER_PATH}. Run train.py first."
        )
    return joblib.load(ENCODER_PATH)


def predict_single(input_dict: dict) -> dict:
    """
    Predict disposable income for a single user.

    Parameters
    ----------
    input_dict : dict
        Keys: Income, Age, Dependents, Occupation, City_Tier,
              Rent, Loan_Repayment, Insurance, Groceries, Transport,
              Eating_Out, Entertainment, Utilities, Healthcare,
              Education, Miscellaneous

    Returns
    -------
    dict with keys: predicted_disposable_income, expense_ratio, savings_tip
    """
    model = load_model()
    encoders = load_encoders()

    # Build DataFrame from input
    df = pd.DataFrame([input_dict])

    # Feature engineering
    df = engineer_features(df)

    # Encode categoricals using saved encoders
    for col, le in encoders.items():
        df[col] = le.transform(df[col].astype(str))

    # Select features in correct order
    feature_list = get_feature_list()
    X = df[feature_list]

    # Predict
    prediction = float(model.predict(X)[0])

    # Extra insights
    total_expense = df["Total_Expense"].iloc[0]
    income = input_dict["Income"]
    expense_ratio = round(total_expense / income * 100, 1) if income > 0 else 0
    savings_suggestion = max(0, prediction * 0.20)

    return {
        "predicted_disposable_income": round(prediction, 2),
        "total_expense": round(float(total_expense), 2),
        "expense_ratio_pct": expense_ratio,
        "suggested_savings": round(savings_suggestion, 2),
    }


if __name__ == "__main__":
    # Quick test
    sample = {
        "Income": 44637,
        "Age": 49,
        "Dependents": 0,
        "Occupation": "Self_Employed",
        "City_Tier": "Tier_1",
        "Rent": 13391,
        "Loan_Repayment": 0,
        "Insurance": 2206,
        "Groceries": 6658,
        "Transport": 2636,
        "Eating_Out": 1651,
        "Entertainment": 1536,
        "Utilities": 2911,
        "Healthcare": 1546,
        "Education": 0,
        "Miscellaneous": 831,
    }
    result = predict_single(sample)
    print("\n📊 Prediction Result:")
    for k, v in result.items():
        print(f"  {k}: ₹{v:,.2f}" if "income" in k or "expense" in k or "savings" in k
              else f"  {k}: {v}")
