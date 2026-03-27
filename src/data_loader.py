from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "loan.csv"


def load_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load Lending Club loan-level data from CSV."""
    path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Input dataset was not found at {path}. Place loan.csv in the data directory "
            "or pass a custom data_path."
        )

    return pd.read_csv(path, low_memory=False)
