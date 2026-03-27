import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictive features and one-hot encode categoricals."""
    engineered = df.copy()

    engineered["log_annual_inc"] = np.log1p(engineered["annual_inc"].clip(lower=0))
    engineered["dti_sq"] = engineered["dti"] ** 2
    engineered["int_rate_x_dti"] = engineered["int_rate"] * engineered["dti"]

    categorical_cols = ["grade", "purpose", "home_ownership", "verification_status"]
    present_cats = [c for c in categorical_cols if c in engineered.columns]
    engineered = pd.get_dummies(engineered, columns=present_cats, drop_first=True)

    return engineered
