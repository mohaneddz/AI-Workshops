from __future__ import annotations

from app.config import DATA_FILE
from app.services.training import train_all_models


if __name__ == "__main__":
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Missing required file: data/data_preprocessed.csv\n"
            "Run tasks/01_Preprocessing.ipynb first and save the output as data/data_preprocessed.csv."
        )

    print("Training all models...")
    train_all_models()
    print("\nAll models trained successfully.")
