from __future__ import annotations

from app.services.model import train_and_save_models


if __name__ == "__main__":
    meta = train_and_save_models()
    print("Models trained successfully")
    print(meta)
