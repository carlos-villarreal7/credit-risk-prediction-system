from typing import Iterable

import joblib
import pandas as pd


def load_model(model_path: str):
    return joblib.load(model_path)


def segment_risk(probability: float) -> str:
    if probability < 0.20:
        return "Low"
    if probability < 0.50:
        return "Medium"
    return "High"


def predict_default_probability(model, feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Generate probability and risk segment for each loan application."""
    probabilities = model.predict_proba(feature_frame)[:, 1]

    scored = feature_frame.copy()
    scored["default_probability"] = probabilities
    scored["risk_segment"] = [segment_risk(p) for p in probabilities]
    return scored


def segment_probabilities(probabilities: Iterable[float]) -> list[str]:
    return [segment_risk(p) for p in probabilities]
