from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PERIOD_TO_HOUR = {"morning": 8, "afternoon": 14, "evening": 18, "night": 23}


def _road_lookup(graph) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for u, v, data in graph.edges(data=True):
        lookup[f"{u}-{v}"] = data
        lookup[f"{v}-{u}"] = data
    return lookup


def make_training_frame(graph, traffic_patterns: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    road_data = _road_lookup(graph)
    for pattern in traffic_patterns:
        road_key = str(pattern.get("road"))
        data = road_data.get(road_key)
        if not data:
            continue
        for period, hour in PERIOD_TO_HOUR.items():
            rows.append({
                "road": road_key,
                "period": period,
                "hour": hour,
                "distance": float(data.get("distance", 0)),
                "capacity": float(data.get("capacity", 0)),
                "condition": float(data.get("condition", 10)),
                "is_proposed": int(data.get("road_type") == "proposed"),
                "congestion_ratio": float(pattern.get(period, 0)) / max(float(data.get("capacity", 1)), 1.0),
                "traffic_count": float(pattern.get(period, 0)),
            })
    return pd.DataFrame(rows)


def train_congestion_forecaster(graph, traffic_patterns: List[Dict[str, Any]], random_state: int = 42) -> Tuple[RandomForestRegressor, Dict[str, float], pd.DataFrame]:
    """Train Random Forest regression to forecast traffic_count from road/time features."""
    df = make_training_frame(graph, traffic_patterns)
    if df.empty:
        raise ValueError("No traffic data available for training.")

    features = ["hour", "distance", "capacity", "condition", "is_proposed", "congestion_ratio"]
    X = df[features]
    y = df["traffic_count"]

    # Dataset is small, so keep deterministic split and guard against very tiny data.
    if len(df) >= 12:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = RandomForestRegressor(n_estimators=150, random_state=random_state, min_samples_leaf=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": round(float(mean_absolute_error(y_test, predictions)), 3),
        "R2": round(float(r2_score(y_test, predictions)), 3) if len(set(y_test)) > 1 else 1.0,
        "training_rows": int(len(df)),
    }
    return model, metrics, df


def predict_congestion(model: RandomForestRegressor, graph, road_id: str, hour: int, traffic_patterns: List[Dict[str, Any]]) -> float:
    period = "morning" if 6 <= hour <= 11 else "afternoon" if 12 <= hour <= 15 else "evening" if 16 <= hour <= 21 else "night"
    lookup = _road_lookup(graph)
    road_data = lookup.get(road_id)
    if not road_data:
        raise ValueError(f"Road {road_id!r} was not found.")
    pattern = next((p for p in traffic_patterns if p.get("road") == road_id or p.get("road") == "-".join(reversed(road_id.split("-")))), None)
    traffic_value = float(pattern.get(period, 0)) if pattern else float(road_data.get("avg_traffic", 0))
    row = pd.DataFrame([{
        "hour": int(hour) % 24,
        "distance": float(road_data.get("distance", 0)),
        "capacity": float(road_data.get("capacity", 0)),
        "condition": float(road_data.get("condition", 10)),
        "is_proposed": int(road_data.get("road_type") == "proposed"),
        "congestion_ratio": traffic_value / max(float(road_data.get("capacity", 1)), 1.0),
    }])
    return round(float(model.predict(row)[0]), 2)
