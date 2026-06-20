#!/usr/bin/env python3
"""Build a cross-scenario JSQ/GA comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ["current", "planned", "full"]
ALGORITHMS = ["JSQ", "GA"]
SUMMARY_FIELDS = [
    "scenario",
    "algorithm",
    "trip_count",
    "vehicle_count",
    "completion_rate",
    "total_charging_cost",
    "average_wait_time_min",
    "max_wait_time_min",
    "charging_event_count",
    "total_charged_energy_kwh",
    "total_charging_co2_kg",
    "min_final_soc",
    "runtime_seconds",
]


def metrics_path(algorithm: str, scenario: str) -> Path:
    """Return the expected metrics path for one algorithm/scenario pair."""

    prefix = algorithm.lower()
    return PROJECT_ROOT / "outputs" / "metrics" / f"{prefix}_metrics_{scenario}.json"


def load_metrics(path: Path) -> dict[str, Any]:
    """Read one metrics JSON artifact."""

    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Metrics file must contain a JSON object: {path}")
    return data


def build_row(scenario: str, algorithm: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract the headline fields used in the penetration summary table."""

    row = {"scenario": scenario, "algorithm": algorithm}
    for field in SUMMARY_FIELDS:
        if field in {"scenario", "algorithm"}:
            continue
        row[field] = metrics.get(field, "")
    return row


def build_summary(scenarios: list[str]) -> list[dict[str, Any]]:
    """Build the six-row current/planned/full by JSQ/GA comparison table."""

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for algorithm in ALGORITHMS:
            metrics = load_metrics(metrics_path(algorithm, scenario))
            rows.append(build_row(scenario, algorithm, metrics))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write summary rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write summary rows to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump({"rows": rows}, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Build current/planned/full JSQ-vs-GA comparison table.")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "penetration_scenario_summary.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "penetration_scenario_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    rows = build_summary(args.scenarios)
    write_csv(args.output_csv, rows)
    write_json(args.output_json, rows)
    print(f"Wrote scenario summary to {args.output_csv} and {args.output_json}")


if __name__ == "__main__":
    main()
