"""
utils/preprocess.py
Reusable preprocessing functions for Smart Finance Forecaster.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ── Columns ────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Income", "Age", "Dependents",
    "Occupation", "City_Tier",
    "Rent", "Loan_Repayment", "Insurance",
    "Groceries", "Transport", "Eating_Out",
    "Entertainment", "Utilities", "Healthcare",
    "Education", "Miscellaneous",
]

TARGET_COL = "Disposable_Income"

CATEGORICAL_COLS = ["Occupation", "City_Tier"]

EXPENSE_COLS = [
    "Rent", "Loan_Repayment", "Insurance",
    "Groceries", "Transport", "Eating_Out",
    "Entertainment", "Utilities", "Healthcare",
    "Education", "Miscellaneous",
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw Kaggle CSV."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing target; fill feature NaNs with median."""
    df = df.dropna(subset=[TARGET_COL]).copy()
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def remove_outliers(df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
    """Remove rows where target is outside 1.5 × IQR."""
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    removed = (~mask).sum()
    print(f"Removed {removed} outlier rows from '{col}'")
    return df[mask].copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived financial features."""
    df = df.copy()

    # Total monthly expense
    df["Total_Expense"] = df[EXPENSE_COLS].sum(axis=1)

    # What fraction of income goes to expenses
    df["Expense_Ratio"] = df["Total_Expense"] / df["Income"].replace(0, np.nan)

    # What fraction of income is disposable
    # What fraction of income is disposable (only during training, not inference)
    if TARGET_COL in df.columns:
        df["Disposable_Ratio"] = df[TARGET_COL] / df["Income"].replace(0, np.nan)
    else:
        df["Disposable_Ratio"] = np.nan
        # Fixed vs variable expense split
        df["Fixed_Expense"] = df[["Rent", "Loan_Repayment", "Insurance"]].sum(axis=1)
        df["Variable_Expense"] = df["Total_Expense"] - df["Fixed_Expense"]

    # Expense burden per dependent (avoid divide-by-zero)
    df["Expense_Per_Dependent"] = df["Total_Expense"] / (df["Dependents"] + 1)

    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns. Returns df + encoder dict."""
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def get_feature_list() -> list[str]:
    """Return the full list of features used at training time."""
    base = FEATURE_COLS.copy()
    engineered = [
        "Total_Expense", "Expense_Ratio", "Disposable_Ratio",
        "Fixed_Expense", "Variable_Expense", "Expense_Per_Dependent",
    ]
    return base + engineered


def prepare_dataset(filepath: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Full pipeline: load → clean → engineer → encode.
    Returns (X, y, encoders).
    """
    df = load_data(filepath)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)

    feature_list = get_feature_list()
    X = df[feature_list]
    y = df[TARGET_COL]

    print(f"Final dataset: {X.shape[0]:,} rows, {X.shape[1]} features")
    return X, y, encoders
