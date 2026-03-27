from typing import Optional

from src.data_loader import load_data
from src.evaluation import evaluate_model, save_results
from src.feature_engineering import engineer_features
from src.model_training import train_model
from src.preprocessing import preprocess_data


def run_pipeline(data_path: Optional[str] = None):
    """End-to-end training and evaluation pipeline."""
    df = load_data(data_path)
    df = preprocess_data(df)
    df = engineer_features(df)
    model = train_model(df)
    evaluate_model(model, df)
    save_results(model)
    return model


if __name__ == "__main__":
    artifact = run_pipeline()
    auc = artifact["evaluation"]["metrics"]["auc"]
    print(f"Pipeline finished successfully. Test AUC: {auc:.4f}")
