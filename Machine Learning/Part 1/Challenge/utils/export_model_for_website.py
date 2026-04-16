r2 = r2_score(y_test_actual, y_pred)
"""Compatibility wrapper for the new FastAPI training flow.

Run this file if you want to regenerate the default model bundle without
touching the notebook workflow.
"""

from app.services.training import train_default_model


if __name__ == "__main__":
    bundle = train_default_model()
    metrics = bundle.get("metrics", {})
    print("Model bundle exported successfully")
    print(f"R2: {metrics.get('r2_score')}")
    print(f"MAE: {metrics.get('mae_dzd')}")
