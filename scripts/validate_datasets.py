#!/usr/bin/env python3
"""Validate that processed datasets are mutually compatible."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


DATA_DIR = Path("data/processed")


def load_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / f"{name}.csv").open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    trips = load_csv("trips_hk_gtfs_full")
    vehicles = load_csv("vehicles")
    stations = load_csv("stations")
    prices = load_csv("prices")
    weather = load_csv("weather_hourly")
    energy = load_csv("energy_samples")
    paths = load_csv("path_candidates")

    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    trip_ids = {row["trip_id"] for row in trips}
    route_ids = {row["route_id"] for row in trips}
    energy_trip_ids = {row["trip_id"] for row in energy}
    path_trip_ids = {row["trip_id"] for row in paths}

    check(len(trip_ids) == len(trips), "trips_hk_gtfs_full.csv has duplicate trip_id values")
    check(len(energy_trip_ids) == len(energy), "energy_samples.csv has duplicate trip_id values")
    check(energy_trip_ids == trip_ids, "energy_samples.csv trip_id values do not match trips_hk_gtfs_full.csv")
    check(path_trip_ids <= trip_ids, "path_candidates.csv has trip_id values not found in trips_hk_gtfs_full.csv")
    check(
        all(count == 3 for count in Counter(row["trip_id"] for row in paths).values()),
        "each trip must have exactly 3 candidate paths",
    )
    check(
        all(row["assigned_route"] in route_ids for row in vehicles),
        "vehicles.csv has assigned_route values not found in trips_hk_gtfs_full.csv",
    )
    check(
        all(
            0 <= float(row["min_soc"]) < float(row["initial_soc"]) <= float(row["max_soc"]) <= 1
            and float(row["battery_capacity"]) > 0
            for row in vehicles
        ),
        "vehicles.csv has invalid SOC or battery capacity values",
    )
    check(
        all(
            int(row["fast_chargers"]) >= 0
            and int(row["slow_chargers"]) >= 0
            and float(row["fast_power"]) > float(row["slow_power"]) > 0
            for row in stations
        ),
        "stations.csv has invalid charger counts or power values",
    )
    check(
        all(
            float(row["price_per_kwh"]) > 0
            and re.match(r"^\d{2}:\d{2}$", row["start_time"])
            and re.match(r"^\d{2}:\d{2}$", row["end_time"])
            for row in prices
        ),
        "prices.csv has invalid prices or time formats",
    )
    check(
        sorted(int(row["hour"]) for row in weather) == list(range(24)),
        "weather_hourly.csv must contain one row for each hour 0..23",
    )
    check(
        all(-10 <= float(row["temperature"]) <= 45 and 0 <= float(row["humidity"]) <= 100 for row in weather),
        "weather_hourly.csv has out-of-range weather values",
    )
    check(all(float(row["distance_km"]) > 0 for row in trips), "trips_hk_gtfs_full.csv has non-positive distances")
    check(
        all(
            float(row["distance_km"]) > 0
            and float(row["energy_kwh"]) > 0
            and 0 <= float(row["congestion_index"]) <= 1
            and 0 <= float(row["passenger_load"]) <= 1
            for row in energy
        ),
        "energy_samples.csv has invalid feature or energy values",
    )
    check(
        all(
            float(row["distance_km"]) > 0
            and float(row["energy_kwh"]) > 0
            and abs(float(row["carbon_kgco2"]) - float(row["energy_kwh"]) * 0.55) <= 0.02
            for row in paths
        ),
        "path_candidates.csv has invalid distance, energy, or carbon values",
    )

    if errors:
        print("Dataset validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Dataset validation passed.")
    print(
        "Rows:",
        {
            "trips": len(trips),
            "vehicles": len(vehicles),
            "stations": len(stations),
            "prices": len(prices),
            "weather": len(weather),
            "energy": len(energy),
            "paths": len(paths),
        },
    )


if __name__ == "__main__":
    main()
