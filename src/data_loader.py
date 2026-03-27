from pathlib import Path
from typing import Optional

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FULL_DATA_PATH = _DATA_DIR / "loan.csv"
SAMPLE_DATA_PATH = _DATA_DIR / "sample_loans.csv"


def load_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load Lending Club loan-level data from CSV.

    Resolution order (when *data_path* is not supplied):
    1. ``data/loan.csv``  — full Lending Club dataset (not committed to git).
    2. ``data/sample_loans.csv`` — 300-row demo dataset included in the repo.

    Pass *data_path* explicitly to override both defaults.
    """
    if data_path:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}.")
        return pd.read_csv(path, low_memory=False)

    if FULL_DATA_PATH.exists():
        return pd.read_csv(FULL_DATA_PATH, low_memory=False)

    if SAMPLE_DATA_PATH.exists():
        print(
            "[data_loader] Full dataset not found — running in demo mode with "
            f"sample_loans.csv ({SAMPLE_DATA_PATH})."
        )
        return pd.read_csv(SAMPLE_DATA_PATH, low_memory=False)

    raise FileNotFoundError(
        "No dataset found. Place loan.csv (full) or sample_loans.csv (demo) "
        f"in {_DATA_DIR}."
    )
