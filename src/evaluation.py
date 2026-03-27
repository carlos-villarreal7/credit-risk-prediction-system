import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def _extract_feature_importance(estimator: Any, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    else:
        importance = np.zeros(len(feature_names))

    fi = pd.DataFrame({"feature": feature_names, "importance": importance})
    return fi.sort_values("importance", ascending=False).reset_index(drop=True)


def evaluate_model(model_artifact: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate predictive quality and attach interpretation artifacts."""
    estimator = model_artifact["estimator"]
    X_test = model_artifact["X_test"]
    y_test = model_artifact["y_test"]

    y_prob = estimator.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    feature_importance = _extract_feature_importance(estimator, model_artifact["feature_names"])

    model_artifact["evaluation"] = {
        "metrics": metrics,
        "feature_importance": feature_importance,
    }
    return model_artifact["evaluation"]


def save_results(model_artifact: Dict[str, Any], output_dir: str = "results") -> None:
    """Persist model, metrics, and feature importance outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    evaluation = model_artifact.get("evaluation")
    if evaluation is None:
        raise ValueError("Call evaluate_model before save_results.")

    joblib.dump(model_artifact["estimator"], output_path / "credit_risk_model.joblib")

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(evaluation["metrics"], f, indent=2)

    evaluation["feature_importance"].to_csv(output_path / "feature_importance.csv", index=False)
