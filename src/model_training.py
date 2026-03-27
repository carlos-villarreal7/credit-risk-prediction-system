from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def _build_model() -> Any:
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=RANDOM_STATE,
        )
    except Exception:
        try:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                verbose=False,
                random_seed=RANDOM_STATE,
            )
        except Exception:
            return GradientBoostingClassifier(random_state=RANDOM_STATE)


def train_model(df: pd.DataFrame, target_col: str = "default") -> Dict[str, Any]:
    """Train a gradient-boosted model and return full modeling artifact."""
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = _build_model()
    model.fit(X_train, y_train)

    return {
        "estimator": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
    }
