#!/usr/bin/env python3
"""Build a unified evaluation summary for JSQ and GA scheduling outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSQ_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "jsq_metrics_full.json"
DEFAULT_GA_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "ga_metrics_full.json"
DEFAULT_JSQ_SCHEDULE_PATH = PROJECT_ROOT / "outputs" / "schedules" / "jsq_schedule_full.csv"
DEFAULT_GA_SCHEDULE_PATH = PROJECT_ROOT / "outputs" / "schedules" / "ga_schedule_full.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "evaluation_summary_full.json"
GRID_EMISSION_FACTOR_KGCO2_PER_KWH = 0.55

KEY_METRIC_NAMES = [
    "trip_count",
    "completed_trip_count",
    "unserved_trip_count",
    "completion_rate",
    "vehicle_count",
    "charging_event_count",
    "total_predicted_energy_kwh",
    "total_charged_energy_kwh",
    "grid_emission_factor_kgco2_per_kwh",
    "total_charging_co2_kg",
    "total_charging_cost",
    "average_wait_time_min",
    "max_wait_time_min",
    "average_charge_duration_min",
    "min_final_soc",
    "max_final_soc",
    "runtime_seconds",
    "fitness_score",
]
COMPARISON_EPSILON = 1e-6


def project_relative(path: Path) -> str:
    """Return a stable project-relative path when possible."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    """Read one metrics JSON file."""

    if not path.exists():
        raise FileNotFoundError(f"Required metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Metrics file must contain a JSON object: {path}")
    return data


def to_float(value: Any, default: float = 0.0) -> float:
    """Convert CSV scalar text to float, returning a default for blank values."""

    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_truthy(value: Any) -> bool:
    """Interpret schedule boolean fields saved by pandas CSV output."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def average(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty numeric list."""

    return sum(values) / len(values) if values else 0.0


def summarize_schedule(path: Path) -> dict[str, Any]:
    """Read a schedule CSV and recalculate comparable aggregate metrics."""

    if not path.exists():
        raise FileNotFoundError(f"Required schedule file not found: {path}")

    trip_count = 0
    completed_trip_count = 0
    charging_event_count = 0
    total_predicted_energy_kwh = 0.0
    total_charged_energy_kwh = 0.0
    total_charging_cost = 0.0
    vehicle_ids: set[str] = set()
    wait_times: list[float] = []
    charge_durations: list[float] = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            trip_count += 1
            status = str(row.get("status", "")).strip().lower()
            if status == "completed":
                completed_trip_count += 1
                total_predicted_energy_kwh += to_float(row.get("predicted_energy_kwh"))
                bus_id = str(row.get("bus_id", "")).strip()
                if bus_id:
                    vehicle_ids.add(bus_id)

            if is_truthy(row.get("charge_requested")):
                charging_event_count += 1
                total_charged_energy_kwh += to_float(row.get("charged_energy_kwh"))
                total_charging_cost += to_float(row.get("charging_cost"))
                wait_times.append(to_float(row.get("wait_time_min")))
                charge_durations.append(to_float(row.get("charge_duration_min")))

    unserved_trip_count = trip_count - completed_trip_count
    return {
        "path": project_relative(path),
        "trip_count": trip_count,
        "completed_trip_count": completed_trip_count,
        "unserved_trip_count": unserved_trip_count,
        "completion_rate": completed_trip_count / trip_count if trip_count else 0.0,
        "vehicle_count": len(vehicle_ids),
        "charging_event_count": charging_event_count,
        "total_predicted_energy_kwh": total_predicted_energy_kwh,
        "total_charged_energy_kwh": total_charged_energy_kwh,
        "grid_emission_factor_kgco2_per_kwh": GRID_EMISSION_FACTOR_KGCO2_PER_KWH,
        "total_charging_co2_kg": total_charged_energy_kwh * GRID_EMISSION_FACTOR_KGCO2_PER_KWH,
        "total_charging_cost": total_charging_cost,
        "average_wait_time_min": average(wait_times),
        "max_wait_time_min": max(wait_times) if wait_times else 0.0,
        "average_charge_duration_min": average(charge_durations),
    }


def extract_key_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep the headline fields used in the side-by-side comparison."""

    return {name: metrics[name] for name in KEY_METRIC_NAMES if name in metrics}


def scenario_name(metrics: dict[str, Any]) -> str:
    """Resolve the scenario name across JSQ and GA metrics schemas."""

    if "scenario_name" in metrics:
        return str(metrics["scenario_name"])
    config = metrics.get("config", {})
    if isinstance(config, dict) and "scenario_name" in config:
        return str(config["scenario_name"])
    return "unknown"


def metric_delta(ga_value: Any, jsq_value: Any) -> dict[str, float]:
    """Calculate absolute and relative GA-minus-JSQ deltas for one metric."""

    ga_number = float(ga_value)
    jsq_number = float(jsq_value)
    delta = ga_number - jsq_number
    return {
        "jsq": jsq_number,
        "ga": ga_number,
        "delta": delta,
        "delta_pct_vs_jsq": delta / jsq_number if jsq_number else 0.0,
    }


def build_delta_metrics(jsq_metrics: dict[str, Any], ga_metrics: dict[str, Any]) -> dict[str, Any]:
    """Build per-metric numeric deltas for fields shared by both algorithms."""

    deltas: dict[str, Any] = {}
    for name in KEY_METRIC_NAMES:
        if name not in jsq_metrics or name not in ga_metrics:
            continue
        if isinstance(jsq_metrics[name], (int, float)) and isinstance(ga_metrics[name], (int, float)):
            deltas[name] = metric_delta(ga_metrics[name], jsq_metrics[name])
    return deltas


def build_outcome_summary(jsq_metrics: dict[str, Any], ga_metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize the most presentation-friendly comparison fields."""

    completed_delta = int(ga_metrics["completed_trip_count"] - jsq_metrics["completed_trip_count"])
    unserved_delta = int(ga_metrics["unserved_trip_count"] - jsq_metrics["unserved_trip_count"])
    cost_delta = float(ga_metrics["total_charging_cost"] - jsq_metrics["total_charging_cost"])
    wait_delta = float(ga_metrics["average_wait_time_min"] - jsq_metrics["average_wait_time_min"])
    co2_delta = float(ga_metrics.get("total_charging_co2_kg", 0.0) - jsq_metrics.get("total_charging_co2_kg", 0.0))
    return {
        "completed_trip_delta": completed_delta,
        "unserved_trip_delta": unserved_delta,
        "completion_rate_delta": float(ga_metrics["completion_rate"] - jsq_metrics["completion_rate"]),
        "charging_cost_delta": cost_delta,
        "charging_cost_delta_pct": cost_delta / jsq_metrics["total_charging_cost"]
        if jsq_metrics["total_charging_cost"]
        else 0.0,
        "charging_co2_delta_kg": co2_delta,
        "charging_co2_delta_pct": co2_delta / jsq_metrics["total_charging_co2_kg"]
        if jsq_metrics.get("total_charging_co2_kg")
        else 0.0,
        "average_wait_time_delta_min": wait_delta,
        "charging_event_delta": int(ga_metrics["charging_event_count"] - jsq_metrics["charging_event_count"]),
        "ga_completed_more_trips": completed_delta > 0,
        "ga_cost_lower": cost_delta < -COMPARISON_EPSILON,
        "ga_average_wait_lower": wait_delta < -COMPARISON_EPSILON,
        "directly_comparable": True,
    }


def validate_schedules_against_metrics(
    metrics: dict[str, Any],
    schedule_summary: dict[str, Any],
) -> dict[str, Any]:
    """Report whether schedule-derived aggregates match the saved metrics."""

    checks: dict[str, Any] = {}
    for name in [
        "trip_count",
        "completed_trip_count",
        "unserved_trip_count",
        "charging_event_count",
        "total_charged_energy_kwh",
        "total_charging_co2_kg",
        "total_charging_cost",
    ]:
        metric_value = metrics.get(name)
        schedule_value = schedule_summary.get(name)
        matches = abs(float(metric_value) - float(schedule_value)) < COMPARISON_EPSILON if metric_value is not None else False
        checks[name] = {
            "metrics": metric_value,
            "schedule": schedule_value,
            "matches": matches,
        }
    return checks


def build_evaluation_summary(
    jsq_metrics_path: Path,
    ga_metrics_path: Path,
    jsq_schedule_path: Path,
    ga_schedule_path: Path,
) -> dict[str, Any]:
    """Build the unified JSQ-vs-GA evaluation payload."""

    jsq_metrics = load_json(jsq_metrics_path)
    ga_metrics = load_json(ga_metrics_path)
    jsq_schedule = summarize_schedule(jsq_schedule_path)
    ga_schedule = summarize_schedule(ga_schedule_path)

    jsq_inputs = jsq_metrics.get("input_paths", {})
    ga_inputs = ga_metrics.get("input_paths", {})
    same_trip_input = jsq_inputs.get("trips") == ga_inputs.get("trips")
    same_vehicle_input = jsq_inputs.get("vehicles") == ga_inputs.get("vehicles")
    same_station_input = jsq_inputs.get("stations") == ga_inputs.get("stations")
    same_trip_count = jsq_metrics.get("trip_count") == ga_metrics.get("trip_count")
    same_vehicle_count = jsq_metrics.get("vehicle_count") == ga_metrics.get("vehicle_count")
    directly_comparable = bool(
        same_trip_input
        and same_vehicle_input
        and same_station_input
        and same_trip_count
        and same_vehicle_count
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "jsq_metrics": project_relative(jsq_metrics_path),
            "ga_metrics": project_relative(ga_metrics_path),
            "jsq_schedule": project_relative(jsq_schedule_path),
            "ga_schedule": project_relative(ga_schedule_path),
        },
        "scenario": {
            "jsq": scenario_name(jsq_metrics),
            "ga": scenario_name(ga_metrics),
            "same_trip_input": same_trip_input,
            "same_vehicle_input": same_vehicle_input,
            "same_station_input": same_station_input,
            "same_trip_count": same_trip_count,
            "same_vehicle_count": same_vehicle_count,
            "directly_comparable": directly_comparable,
        },
        "algorithms": {
            "JSQ": extract_key_metrics(jsq_metrics),
            "GA": extract_key_metrics(ga_metrics),
        },
        "schedule_recalculation": {
            "JSQ": jsq_schedule,
            "GA": ga_schedule,
        },
        "schedule_validation": {
            "JSQ": validate_schedules_against_metrics(jsq_metrics, jsq_schedule),
            "GA": validate_schedules_against_metrics(ga_metrics, ga_schedule),
        },
        "comparison": {
            "ga_minus_jsq": build_delta_metrics(jsq_metrics, ga_metrics),
        },
    }

    if directly_comparable:
        summary["comparison"]["headline"] = build_outcome_summary(jsq_metrics, ga_metrics)
    else:
        summary["comparison"]["headline"] = {
            "directly_comparable": False,
            "reason": "JSQ and GA artifacts use different trip, vehicle, or station inputs.",
        }
    return summary


def save_summary(summary: dict[str, Any], output_path: Path) -> None:
    """Write the evaluation summary JSON artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation generation."""

    parser = argparse.ArgumentParser(description="Build JSQ-vs-GA evaluation summary.")
    parser.add_argument("--jsq-metrics", type=Path, default=DEFAULT_JSQ_METRICS_PATH)
    parser.add_argument("--ga-metrics", type=Path, default=DEFAULT_GA_METRICS_PATH)
    parser.add_argument("--jsq-schedule", type=Path, default=DEFAULT_JSQ_SCHEDULE_PATH)
    parser.add_argument("--ga-schedule", type=Path, default=DEFAULT_GA_SCHEDULE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint used by `python -m src.evaluation` and `make evaluate`."""

    args = parse_args()
    summary = build_evaluation_summary(
        jsq_metrics_path=args.jsq_metrics,
        ga_metrics_path=args.ga_metrics,
        jsq_schedule_path=args.jsq_schedule,
        ga_schedule_path=args.ga_schedule,
    )
    save_summary(summary, args.output)
    print(f"Saved evaluation summary to {project_relative(args.output)}")


if __name__ == "__main__":
    main()
