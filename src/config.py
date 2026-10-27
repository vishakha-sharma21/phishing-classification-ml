"""Central configuration: paths, seeds and label conventions."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "phishing.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

UCI_DATASET_ID = 327
TARGET_COLUMN = "result"
RANDOM_SEED = 42

# Original UCI convention: -1 = phishing, +1 = legitimate.
# Project convention: 1 = phishing (positive class), 0 = legitimate.
LABEL_MAPPING = {-1: 1, 1: 0}
CLASS_NAMES = {1: "phishing", 0: "legitimate"}

EXPECTED_FEATURE_VALUES = {-1, 0, 1}
