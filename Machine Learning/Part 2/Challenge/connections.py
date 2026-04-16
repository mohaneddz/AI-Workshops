from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests

FEATURE_COLUMNS = ["time_s", "amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]

SAFE_ATTACK_TEMPLATES = [
    ("sql_injection_probe", "id=1' OR '1'='1' -- [SIMULATION_ONLY]"),
    ("xss_probe", "<script>console.log('simulation')</script>"),
    ("path_traversal_probe", "../../etc/passwd [SIMULATION_ONLY]"),
    ("cmd_injection_probe", "ping 127.0.0.1 && echo SIMULATION_ONLY"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic simulator for /api/ingest")
    parser.add_argument("--server", default="http://127.0.0.1:1240", help="Base server URL")
    parser.add_argument("--count", type=int, default=200, help="Total requests to send; <=0 means infinite")
    parser.add_argument("--interval", type=float, default=0.25, help="Delay between requests in seconds")
    parser.add_argument("--bad-prob", type=float, default=0.22, help="Probability of generating bad_request")
    parser.add_argument("--model", default="isolation_forest", help="Model key: z_score | isolation_forest | lof")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report-every", type=int, default=20, help="How often to print metrics snapshot")
    parser.add_argument("--data-file", default=str(Path("data") / "connections.csv"), help="Dataset used to sample feature baselines")
    return parser.parse_args()


def _sample_features(df: pd.DataFrame, rng: np.random.Generator) -> Dict[str, float]:
    row = df.iloc[int(rng.integers(0, len(df)))]
    return {col: float(row[col]) for col in FEATURE_COLUMNS}


def _make_good_features(base: Dict[str, float], rng: np.random.Generator, std_map: Dict[str, float]) -> Dict[str, float]:
    features = dict(base)
    for col in FEATURE_COLUMNS:
        noise = float(rng.normal(0.0, 0.15 * std_map[col]))
        features[col] += noise
    return features


def _make_bad_features(base: Dict[str, float], rng: np.random.Generator, std_map: Dict[str, float]) -> Dict[str, float]:
    features = dict(base)
    features["amount"] = max(0.0, features["amount"] * float(rng.uniform(8.0, 30.0)))
    spikes = ["V2", "V3", "V4", "V7", "V8"]
    for col in spikes:
        direction = 1.0 if rng.random() >= 0.5 else -1.0
        features[col] += direction * float(rng.uniform(3.0, 7.0) * std_map[col])
    features["time_s"] += float(rng.uniform(5000, 300000))
    return features


def _build_payload(
    *,
    is_bad: bool,
    features: Dict[str, float],
    model_key: str,
    rng: np.random.Generator,
) -> Dict[str, object]:
    if is_bad:
        attack_type, payload_text = SAFE_ATTACK_TEMPLATES[int(rng.integers(0, len(SAFE_ATTACK_TEMPLATES)))]
        actual_type = "bad_request"
    else:
        attack_type = "none"
        payload_text = "normal-client-request"
        actual_type = "good_request"

    return {
        "actual_type": actual_type,
        "attack_type": attack_type,
        "payload": payload_text,
        "model_key": model_key,
        "features": features,
    }


def _print_metrics(session: requests.Session, base_url: str, model_key: str) -> None:
    try:
        resp = session.get(f"{base_url}/api/metrics", params={"model": model_key}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            print("metrics-error:", data)
            return
        m = data["metrics"]
        print(
            "[metrics]",
            f"total={m['total_requests']}",
            f"passed={m['passed_requests']}",
            f"anomalies_classified={m['anomalies_classified']}",
            f"unclassified={m['unclassified']}",
            f"good_misclassified={m['good_connections_misclassified']}",
        )
    except Exception as exc:
        print("metrics-fetch-failed:", exc)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    base_url = args.server.rstrip("/")
    ingest_url = f"{base_url}/api/ingest"

    df = pd.read_csv(args.data_file)
    std_map = {col: float(df[col].std(ddof=0)) if float(df[col].std(ddof=0)) > 1e-12 else 1.0 for col in FEATURE_COLUMNS}

    session = requests.Session()
    sent = 0
    success = 0

    print(
        f"Starting traffic simulation -> {ingest_url} | model={args.model} | "
        f"bad_prob={args.bad_prob:.2f} | interval={args.interval:.2f}s"
    )

    while args.count <= 0 or sent < args.count:
        sent += 1
        is_bad = bool(rng.random() < args.bad_prob)

        base_features = _sample_features(df, rng)
        if is_bad:
            features = _make_bad_features(base_features, rng, std_map)
        else:
            features = _make_good_features(base_features, rng, std_map)

        body = _build_payload(is_bad=is_bad, features=features, model_key=args.model, rng=rng)

        try:
            resp = session.post(ingest_url, json=body, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            success += 1
            print(
                f"[{sent}] actual={body['actual_type']:<12} "
                f"pred={'anomaly' if result['predicted_anomaly'] else 'normal':<8} "
                f"score={result['score']:.4f} thr={result['threshold']:.4f} "
                f"attack={body['attack_type']}"
            )
        except Exception as exc:
            print(f"[{sent}] send-failed: {exc}")

        if sent % max(args.report_every, 1) == 0:
            _print_metrics(session, base_url, args.model)

        time.sleep(max(args.interval, 0.0))

    print(f"Completed. sent={sent}, success={success}")
    _print_metrics(session, base_url, args.model)


if __name__ == "__main__":
    main()
