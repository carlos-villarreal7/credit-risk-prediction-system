from typing import List

import pandas as pd


TARGET_STATUSES = ["Fully Paid", "Charged Off"]
REQUIRED_COLUMNS: List[str] = [
    "loan_status",
    "annual_inc",
    "dti",
    "int_rate",
    "grade",
    "term",
    "loan_amnt",
    "installment",
    "purpose",
    "home_ownership",
    "verification_status",
]


def _normalize_percentage(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce")


def _normalize_term(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter valid outcomes and build a model-ready base dataset."""
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[df["loan_status"].isin(TARGET_STATUSES)].copy()
    df["default"] = (df["loan_status"] == "Charged Off").astype(int)

    df["int_rate"] = _normalize_percentage(df["int_rate"])
    df["term"] = _normalize_term(df["term"])

    numeric_cols = ["annual_inc", "dti", "int_rate", "term", "loan_amnt", "installment"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only columns needed in modeling and interpretation.
    selected = [
        "default",
        "annual_inc",
        "dti",
        "int_rate",
        "term",
        "loan_amnt",
        "installment",
        "grade",
        "purpose",
        "home_ownership",
        "verification_status",
    ]
    df = df[selected].dropna(subset=["default", "annual_inc", "dti", "int_rate", "grade"])

    return df.reset_index(drop=True)
