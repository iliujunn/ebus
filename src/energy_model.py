#!/usr/bin/env python3
"""Train and use the trip-level energy consumption prediction model."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "energy_samples.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "models" / "energy_model.pkl"
METRICS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "energy_model_metrics.json"
PREDICTIONS_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "energy_predictions.csv"

FEATURE_COLUMNS = [
    "distance_km",
    "avg_slope",
    "congestion_index",
    "avg_speed",
    "passenger_load",
    "temperature",
    "peak_hour",
]
TARGET_COLUMN = "energy_kwh"
IDENTIFIER_COLUMNS = ["trip_id", "route_id", "start_time", "end_time"]
OPTIONAL_LABEL_COLUMNS = ["energy_kwh_clean", "noise_kwh", "noise_ratio", "noise_profile"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnergyModelConfig:
    """Configuration used for a reproducible energy model experiment."""

    random_seed: int = 42
    test_size: float = 0.2
    hidden_layer_sizes: tuple[int, int] = (64, 32)
    max_iter: int = 300
    learning_rate_init: float = 0.001
    early_stopping: bool = True
    validation_fraction: float = 0.1


def set_random_seed(seed: int) -> None:
    """Set random seeds for deterministic data splits and model initialization."""

    random.seed(seed)
    np.random.seed(seed)


def load_energy_samples(input_path: Path = INPUT_DATA_PATH) -> pd.DataFrame:
    """Load the processed energy samples and validate required fields."""

    logger.info("Reading energy samples from %s", input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Energy sample file not found: {input_path}. Run `make data-derived` first."
        )

    data = pd.read_csv(input_path)
    validate_energy_samples(data)
    return data


def validate_energy_samples(data: pd.DataFrame) -> None:
    """Validate the schema and numeric ranges required by the energy model."""

    required_columns = set(IDENTIFIER_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"energy_samples.csv is missing required columns: {missing_columns}")

    if data.empty:
        raise ValueError("energy_samples.csv is empty; model training requires samples.")

    null_counts = data[list(required_columns)].isna().sum()
    null_columns = null_counts[null_counts > 0].to_dict()
    if null_columns:
        raise ValueError(f"energy_samples.csv contains missing values: {null_columns}")

    numeric_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    range_checks = {
        "distance_km": data["distance_km"] > 0,
        "congestion_index": data["congestion_index"].between(0, 1),
        "passenger_load": data["passenger_load"].between(0, 1),
        "peak_hour": data["peak_hour"].isin([0, 1]),
        "energy_kwh": data["energy_kwh"] > 0,
    }
    failed_checks = [name for name, mask in range_checks.items() if not bool(mask.all())]
    if failed_checks:
        raise ValueError(f"energy_samples.csv has invalid numeric ranges: {failed_checks}")

    for column in ["start_time", "end_time"]:
        if not data[column].map(is_valid_gtfs_time).all():
            raise ValueError(
                f"energy_samples.csv contains invalid {column} values; expected GTFS HH:MM:SS time."
            )


def is_valid_gtfs_time(value: Any) -> bool:
    """Return whether a value is a valid GTFS time string.

    GTFS allows hours beyond 23 for service after midnight, so strict datetime
    parsing would reject valid values such as 24:00:00.
    """

    if not isinstance(value, str):
        return False
    match = re.fullmatch(r"(\d{2}):([0-5]\d):([0-5]\d)", value)
    if match is None:
        return False
    return int(match.group(1)) <= 47


def build_energy_model(config: EnergyModelConfig) -> Pipeline:
    """Build a standardized MLP regression pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                MLPRegressor(
                    hidden_layer_sizes=config.hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    learning_rate_init=config.learning_rate_init,
                    max_iter=config.max_iter,
                    early_stopping=config.early_stopping,
                    validation_fraction=config.validation_fraction,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics."""

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_model(model: Pipeline, config: EnergyModelConfig, metrics: dict[str, Any]) -> None:
    """Persist the trained model and metadata for downstream schedulers."""

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "config": asdict(config),
        "metrics": metrics,
    }
    joblib.dump(model_payload, MODEL_OUTPUT_PATH)
    logger.info("Saved model to %s", MODEL_OUTPUT_PATH)


def save_metrics(metrics: dict[str, Any]) -> None:
    """Write experiment metrics as a traceable JSON artifact."""

    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    logger.info("Saved metrics to %s", METRICS_OUTPUT_PATH)


def save_predictions(data: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Save one row per trip with actual and predicted energy consumption."""

    optional_columns = [column for column in OPTIONAL_LABEL_COLUMNS if column in data.columns]
    prediction_columns = IDENTIFIER_COLUMNS + FEATURE_COLUMNS + optional_columns + [TARGET_COLUMN]
    prediction_data = data[prediction_columns].copy()
    prediction_data["predicted_energy_kwh"] = model.predict(data[FEATURE_COLUMNS])
    prediction_data["prediction_error_kwh"] = (
        prediction_data["predicted_energy_kwh"] - prediction_data[TARGET_COLUMN]
    )
    prediction_data.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    logger.info("Saved predictions to %s", PREDICTIONS_OUTPUT_PATH)
    return prediction_data


def train_energy_model(config: EnergyModelConfig | None = None) -> Pipeline:
    """
    Train the energy consumption prediction model.

    The saved artifact is a scikit-learn Pipeline, so normalization is applied
    consistently during later batch inference.
    """

    resolved_config = config or EnergyModelConfig()
    set_random_seed(resolved_config.random_seed)
    start_time = time.perf_counter()

    logger.info("Start training energy model")
    logger.info("Training config: %s", asdict(resolved_config))
    data = load_energy_samples()

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS],
        data[TARGET_COLUMN],
        test_size=resolved_config.test_size,
        random_state=resolved_config.random_seed,
    )

    model = build_energy_model(resolved_config)
    model.fit(x_train, y_train)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    train_metrics = evaluate_predictions(y_train.to_numpy(), train_predictions)
    test_metrics = evaluate_predictions(y_test.to_numpy(), test_predictions)

    regressor = model.named_steps["regressor"]
    runtime_seconds = time.perf_counter() - start_time
    metrics: dict[str, Any] = {
        "model_type": "StandardScaler + MLPRegressor",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "sample_count": int(len(data)),
        "train_sample_count": int(len(x_train)),
        "test_sample_count": int(len(x_test)),
        "random_seed": resolved_config.random_seed,
        "test_size": resolved_config.test_size,
        "train_loss": float(regressor.loss_),
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "train_r2": train_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "runtime_seconds": float(runtime_seconds),
        "input_path": str(INPUT_DATA_PATH.relative_to(PROJECT_ROOT)),
        "model_output_path": str(MODEL_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "metrics_output_path": str(METRICS_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "predictions_output_path": str(PREDICTIONS_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "config": asdict(resolved_config),
    }

    save_model(model, resolved_config, metrics)
    save_metrics(metrics)
    save_predictions(data, model)

    logger.info(
        "Energy model finished: train_loss=%.6f, test_mae=%.4f, test_rmse=%.4f, test_r2=%.4f, runtime=%.2fs",
        metrics["train_loss"],
        metrics["test_mae"],
        metrics["test_rmse"],
        metrics["test_r2"],
        runtime_seconds,
    )
    return model


def load_trained_energy_model(model_path: Path = MODEL_OUTPUT_PATH) -> dict[str, Any]:
    """Load the saved energy model payload."""

    if not model_path.exists():
        raise FileNotFoundError(f"Trained energy model not found: {model_path}. Run `make train` first.")
    return joblib.load(model_path)


def predict_energy(
    features: pd.DataFrame | list[dict[str, float]],
    model_path: Path = MODEL_OUTPUT_PATH,
) -> np.ndarray:
    """Predict energy consumption in kWh for one or more feature rows."""

    payload = load_trained_energy_model(model_path)
    model: Pipeline = payload["model"]
    feature_columns: list[str] = payload["feature_columns"]
    feature_data = pd.DataFrame(features)

    missing_columns = sorted(set(feature_columns) - set(feature_data.columns))
    if missing_columns:
        raise ValueError(f"Missing feature columns for prediction: {missing_columns}")

    return model.predict(feature_data[feature_columns])


def main() -> None:
    """CLI entrypoint used by `python -m src.energy_model` and `make train`."""

    train_energy_model()


if __name__ == "__main__":
    main()
