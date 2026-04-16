# Part 2 Challenge - Global Network Anomaly Monitor

This MVP simulates a live global network and supports:
- realtime anomaly visualization
- 3 selectable anomaly models (`z_score`, `isolation_forest`, `lof`)
- ingest API with request-level classification metrics

## Run flow

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Train model (required before first app run)

```bash
python train.py
```

3. Start server

```bash
python -m app.main
```

4. Open dashboard

- http://127.0.0.1:1240

5. (Optional) Run probabilistic traffic generator

```bash
python connections.py --model isolation_forest --count 300 --bad-prob 0.25
```

This sends both `good_request` and `bad_request` payloads to `POST /api/ingest`.
Bad requests use safe attack-like signatures for simulation only.

## Metrics captured

The server tracks:
- `total_requests`
- `passed_requests`
- `anomalies_classified` (bad requests correctly flagged)
- `unclassified` (bad requests that passed)
- `good_connections_misclassified` (good requests incorrectly flagged)

Available via:
- `GET /api/metrics`
- `GET /api/metrics?model=<model_key>`
- `POST /api/metrics/reset`

## Files

- `data/connections.csv`: challenge dataset (from Kaggle credit-card features)
- `models/`: saved model artifacts
- `app/`: FastAPI backend + realtime frontend
- `connections.py`: probabilistic traffic simulator for ingest endpoint
