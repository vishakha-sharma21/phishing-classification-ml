"""Dataset acquisition and loading for the UCI Phishing Websites dataset (id=327).

The dataset is downloaded once via ``ucimlrepo`` and saved to ``data/raw/phishing.csv``.
All later code loads the local copy so experiments run offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN, UCI_DATASET_ID


def download_dataset(path: Path = RAW_DATA_PATH, overwrite: bool = False) -> Path:
    """Fetch the UCI dataset and save features + target as a single CSV.

    Args:
        path: Destination CSV path.
        overwrite: Re-download even if the file already exists.

    Returns:
        The path of the saved CSV.
    """
    if path.exists() and not overwrite:
        return path
    from ucimlrepo import fetch_ucirepo  # imported lazily: only needed online

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    features = dataset.data.features
    targets = dataset.data.targets
    if targets.shape[1] != 1:
        raise ValueError(f"Expected one target column, got {list(targets.columns)}")
    df = pd.concat([features, targets], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the locally saved raw dataset.

    Raises:
        FileNotFoundError: If the dataset has not been downloaded yet.
        ValueError: If the target column is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data_loading` to download it."
        )
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {path}")
    return df


if __name__ == "__main__":
    saved = download_dataset()
    print(f"Saved dataset to {saved}")
