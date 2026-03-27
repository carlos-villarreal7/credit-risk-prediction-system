import pandas as pd

from src.preprocessing import preprocess_data


def test_preprocess_data_creates_target() -> None:
    df = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Current"],
            "annual_inc": [50000, 70000, 65000],
            "dti": [10.0, 20.0, 15.0],
            "int_rate": ["10.5%", "15.2%", "9.0%"],
            "grade": ["B", "D", "A"],
            "term": ["36 months", "60 months", "36 months"],
            "loan_amnt": [10000, 12000, 9000],
            "installment": [320.5, 300.1, 290.0],
            "purpose": ["debt_consolidation", "credit_card", "car"],
            "home_ownership": ["RENT", "MORTGAGE", "OWN"],
            "verification_status": ["Verified", "Source Verified", "Verified"],
        }
    )

    out = preprocess_data(df)

    assert out.shape[0] == 2
    assert "default" in out.columns
    assert set(out["default"].unique()) == {0, 1}
