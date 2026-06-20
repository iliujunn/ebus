#!/usr/bin/env python3
"""Build experiment-ready datasets from GTFS plus public/assumed inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from collect_hk_gtfs import clean_stop_name, parse_time, read_csv


GTFS_URL = "https://static.data.gov.hk/td/pt-headway-sc/gtfs.zip"
HKO_CURRENT_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en"
RAW_DIR = Path("data/raw")
GTFS_DIR = RAW_DIR / "hk_gtfs"
WEATHER_RAW = RAW_DIR / "hko_current_weather.json"
PROCESSED_DIR = Path("data/processed")

BASE_ENERGY_NOISE_RATIO = 0.08
BASE_ENERGY_NOISE_KWH = 1.5
OUTLIER_PROBABILITY = 0.03
FULL_FLEET_COUNT = 5870

PENETRATION_SCENARIOS = {
    "current": {
        "fleet_count": 150,
        "trip_ratio": 150 / FULL_FLEET_COUNT,
        "station_count": 6,
        "fast_chargers": 4,
        "slow_chargers": 4,
        "trip_output": "trips_current.csv",
        "vehicle_output": "vehicles_current_150.csv",
        "station_output": "stations_current.csv",
        "bus_prefix": "CURBUS",
        "station_prefix": "CURCS",
        "label": "Current pilot-stage e-bus penetration scenario",
    },
    "planned": {
        "fleet_count": 750,
        "trip_ratio": 750 / FULL_FLEET_COUNT,
        "station_count": 20,
        "fast_chargers": 8,
        "slow_chargers": 6,
        "trip_output": "trips_planned.csv",
        "vehicle_output": "vehicles_planned_750.csv",
        "station_output": "stations_planned.csv",
        "bus_prefix": "PLANBUS",
        "station_prefix": "PLANCS",
        "label": "Planned near-term e-bus expansion scenario",
    },
    "full": {
        "fleet_count": FULL_FLEET_COUNT,
        "trip_ratio": 1.0,
        "station_count": 80,
        "fast_chargers": 12,
        "slow_chargers": 8,
        "trip_output": "trips_full_coverage.csv",
        "vehicle_output": "vehicles_full_5870.csv",
        "station_output": "stations_full_80hubs.csv",
        "bus_prefix": "FULLBUS",
        "station_prefix": "FULLCS",
        "label": "Full e-bus coverage scenario",
    },
}


def stable_random(*parts: str) -> random.Random:
    seed = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(seed[:16], 16))


def download_if_needed(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def ensure_gtfs() -> Path:
    zip_path = GTFS_DIR / "gtfs.zip"
    extract_dir = GTFS_DIR / "extracted"
    download_if_needed(GTFS_URL, zip_path)
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    return extract_dir


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_trips(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_vehicles(trips: list[dict[str, str]], output: Path, bus_count: int = 20) -> None:
    route_counts = Counter(row["route_id"] for row in trips)
    top_routes = [route for route, _ in route_counts.most_common(max(1, min(8, len(route_counts))))]
    rows = []
    for i in range(bus_count):
        route = top_routes[i % len(top_routes)]
        rnd = stable_random("vehicle", str(i), route)
        rows.append(
            {
                "bus_id": f"BUS{i + 1:03d}",
                "battery_capacity": 300,
                "initial_soc": round(rnd.uniform(0.72, 0.88), 3),
                "max_soc": 0.9,
                "min_soc": 0.2,
                "assigned_route": route,
                "source": "Constructed from project assumptions; route assignment from DATA.GOV.HK GTFS",
            }
        )
    write_csv(
        output,
        ["bus_id", "battery_capacity", "initial_soc", "max_soc", "min_soc", "assigned_route", "source"],
        rows,
    )


def terminal_stop_counts(gtfs_dir: Path, bus_route_ids: set[str]) -> Counter[str]:
    trips = read_csv(gtfs_dir / "trips.txt")
    trip_route = {row["trip_id"]: row["route_id"] for row in trips if row["route_id"] in bus_route_ids}
    by_trip: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(gtfs_dir / "stop_times.txt"):
        if row["trip_id"] in trip_route:
            by_trip[row["trip_id"]].append(row)

    counts: Counter[str] = Counter()
    for rows in by_trip.values():
        rows.sort(key=lambda row: int(row.get("stop_sequence") or 0))
        if rows:
            counts[rows[0]["stop_id"]] += 1
            counts[rows[-1]["stop_id"]] += 1
    return counts


def build_stations(gtfs_dir: Path, trips: list[dict[str, str]], output: Path, station_count: int = 6) -> None:
    bus_route_ids = {row["route_id"] for row in trips}
    stops = {row["stop_id"]: row for row in read_csv(gtfs_dir / "stops.txt")}
    terminal_counts = terminal_stop_counts(gtfs_dir, bus_route_ids)
    rows = []
    for idx, (stop_id, _) in enumerate(terminal_counts.most_common(station_count)):
        stop = stops[stop_id]
        name = clean_stop_name(stop["stop_name"])
        rows.append(
            {
                "station_id": f"CS{idx + 1:02d}",
                "station_name": f"{name} Charging Hub",
                "location": name,
                "stop_id": stop_id,
                "lat": stop.get("stop_lat", ""),
                "lon": stop.get("stop_lon", ""),
                "fast_chargers": 4 if idx == 0 else 2 if idx < 3 else 1,
                "slow_chargers": 6 if idx == 0 else 4 if idx < 3 else 2,
                "fast_power": 120,
                "slow_power": 40,
                "source": "Derived from high-frequency GTFS bus termini; charger counts are project assumptions",
            }
        )
    write_csv(
        output,
        [
            "station_id",
            "station_name",
            "location",
            "stop_id",
            "lat",
            "lon",
            "fast_chargers",
            "slow_chargers",
            "fast_power",
            "slow_power",
            "source",
        ],
        rows,
    )


def select_route_cluster_trips(trips: list[dict[str, str]], target_ratio: float) -> list[dict[str, str]]:
    """Select high-frequency route clusters until the target trip share is reached."""

    if target_ratio >= 1.0:
        return list(trips)

    target_count = max(1, math.ceil(len(trips) * target_ratio))
    route_counts = Counter(row["route_id"] for row in trips)
    selected_routes: set[str] = set()
    selected_count = 0
    for route_id, count in sorted(route_counts.items(), key=lambda item: (-item[1], item[0])):
        if selected_routes and abs(target_count - selected_count) <= abs(target_count - (selected_count + count)):
            break
        selected_routes.add(route_id)
        selected_count += count
        if selected_count == target_count:
            break
    return [row for row in trips if row["route_id"] in selected_routes]


def build_penetration_vehicles(
    trips: list[dict[str, str]],
    output: Path,
    bus_count: int,
    bus_prefix: str,
    scenario_label: str,
) -> None:
    """Build a deterministic fleet assigned to the selected route cluster."""

    route_counts = Counter(row["route_id"] for row in trips)
    routes = [route for route, _ in sorted(route_counts.items(), key=lambda item: (-item[1], item[0]))]
    vehicle_counts = apportion_vehicles_to_routes(route_counts, bus_count)
    assigned_routes = [route for route in routes for _ in range(vehicle_counts[route])]

    rows = []
    for index, route in enumerate(assigned_routes):
        rnd = stable_random("penetration-vehicle", scenario_label, str(index), route)
        rows.append(
            {
                "bus_id": f"{bus_prefix}{index + 1:05d}",
                "battery_capacity": 300,
                "initial_soc": round(rnd.uniform(0.72, 0.88), 3),
                "max_soc": 0.9,
                "min_soc": 0.2,
                "assigned_route": route,
                "source": f"{scenario_label}; route assignment weighted by selected GTFS trip frequency",
            }
        )
    write_csv(
        output,
        ["bus_id", "battery_capacity", "initial_soc", "max_soc", "min_soc", "assigned_route", "source"],
        rows,
    )


def apportion_vehicles_to_routes(route_counts: Counter[str], bus_count: int) -> dict[str, int]:
    """Allocate vehicles by trip frequency while covering every route when possible."""

    routes = [route for route, _ in sorted(route_counts.items(), key=lambda item: (-item[1], item[0]))]
    if bus_count < len(routes):
        return {route: 1 for route in routes[:bus_count]}

    allocation = {route: 1 for route in routes}
    remaining = bus_count - len(routes)
    if remaining == 0:
        return allocation

    total_trips = sum(route_counts.values())
    quotas = {route: remaining * route_counts[route] / total_trips for route in routes}
    for route, quota in quotas.items():
        extra = math.floor(quota)
        allocation[route] += extra
        remaining -= extra

    remainders = sorted(
        ((quota - math.floor(quota), route) for route, quota in quotas.items()),
        key=lambda item: (-item[0], item[1]),
    )
    for _, route in remainders[:remaining]:
        allocation[route] += 1
    return allocation


def build_penetration_stations(
    gtfs_dir: Path,
    trips: list[dict[str, str]],
    output: Path,
    station_count: int,
    station_prefix: str,
    scenario_label: str,
    fast_chargers: int,
    slow_chargers: int,
) -> None:
    """Build charging hubs near the busiest termini in a selected route cluster."""

    bus_route_ids = {row["route_id"] for row in trips}
    stops = {row["stop_id"]: row for row in read_csv(gtfs_dir / "stops.txt")}
    terminal_counts = terminal_stop_counts(gtfs_dir, bus_route_ids)
    terminal_items = terminal_counts.most_common()
    if not terminal_items:
        raise ValueError("Cannot build penetration stations without terminal stop candidates.")

    rows = []
    for idx in range(station_count):
        stop_id = terminal_items[idx % len(terminal_items)][0]
        stop = stops[stop_id]
        name = clean_stop_name(stop["stop_name"])
        rows.append(
            {
                "station_id": f"{station_prefix}{idx + 1:03d}",
                "station_name": f"{name} {scenario_label} Charging Hub",
                "location": name,
                "stop_id": stop_id,
                "lat": stop.get("stop_lat", ""),
                "lon": stop.get("stop_lon", ""),
                "fast_chargers": fast_chargers,
                "slow_chargers": slow_chargers,
                "fast_power": 120,
                "slow_power": 40,
                "source": f"{scenario_label}; hubs selected from high-frequency GTFS termini",
            }
        )
    write_csv(
        output,
        [
            "station_id",
            "station_name",
            "location",
            "stop_id",
            "lat",
            "lon",
            "fast_chargers",
            "slow_chargers",
            "fast_power",
            "slow_power",
            "source",
        ],
        rows,
    )


def build_penetration_scenarios(gtfs_dir: Path, trips: list[dict[str, str]]) -> None:
    """Build nested current/planned/full e-bus penetration scenario inputs."""

    trip_fields = list(trips[0].keys()) if trips else []
    for scenario_name, config in PENETRATION_SCENARIOS.items():
        scenario_trips = select_route_cluster_trips(trips, float(config["trip_ratio"]))
        scenario_label = str(config["label"])
        write_csv(PROCESSED_DIR / str(config["trip_output"]), trip_fields, scenario_trips)
        build_penetration_vehicles(
            scenario_trips,
            PROCESSED_DIR / str(config["vehicle_output"]),
            int(config["fleet_count"]),
            str(config["bus_prefix"]),
            scenario_label,
        )
        build_penetration_stations(
            gtfs_dir,
            scenario_trips,
            PROCESSED_DIR / str(config["station_output"]),
            int(config["station_count"]),
            str(config["station_prefix"]),
            scenario_label,
            int(config["fast_chargers"]),
            int(config["slow_chargers"]),
        )
        print(
            f"Built {scenario_name}: {len(scenario_trips)} trips, "
            f"{config['fleet_count']} vehicles, {config['station_count']} hubs"
        )


def build_prices(output: Path) -> None:
    rows = [
        {
            "start_time": "00:00",
            "end_time": "16:00",
            "period_type": "off_peak",
            "price_per_kwh": 0.913,
            "currency": "HKD",
            "source": "CLP EV Residential ToU Tariff: off-peak 23:00-16:00 at 91.3 cents/unit",
        },
        {
            "start_time": "16:00",
            "end_time": "23:00",
            "period_type": "on_peak",
            "price_per_kwh": 1.343,
            "currency": "HKD",
            "source": "CLP EV Residential ToU Tariff: on-peak 16:00-23:00 at 134.3 cents/unit",
        },
        {
            "start_time": "23:00",
            "end_time": "24:00",
            "period_type": "off_peak",
            "price_per_kwh": 0.913,
            "currency": "HKD",
            "source": "CLP EV Residential ToU Tariff: off-peak 23:00-16:00 at 91.3 cents/unit",
        },
    ]
    write_csv(output, ["start_time", "end_time", "period_type", "price_per_kwh", "currency", "source"], rows)


def fetch_weather() -> dict:
    download_if_needed(HKO_CURRENT_URL, WEATHER_RAW)
    with WEATHER_RAW.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_weather(output: Path) -> dict[int, float]:
    data = fetch_weather()
    temps = [item["value"] for item in data.get("temperature", {}).get("data", []) if "value" in item]
    humidity = data.get("humidity", {}).get("data", [{}])[0].get("value", 75)
    base_temp = sum(temps) / len(temps) if temps else 25.0
    rows = []
    hourly: dict[int, float] = {}
    for hour in range(24):
        # Daily profile: cooler pre-dawn, warmest mid-afternoon.
        temp = base_temp + 3.0 * math.sin((hour - 8) / 24 * 2 * math.pi)
        hourly[hour] = temp
        rows.append(
            {
                "date": "2026-06-18",
                "hour": hour,
                "temperature": round(temp, 1),
                "humidity": humidity,
                "source": "Hong Kong Observatory current weather API; hourly profile constructed for one-day simulation",
            }
        )
    write_csv(output, ["date", "hour", "temperature", "humidity", "source"], rows)
    return hourly


def duration_minutes(start: str, end: str) -> float:
    start_s = parse_time(start)
    end_s = parse_time(end)
    if end_s < start_s:
        end_s += 24 * 3600
    return max(1.0, (end_s - start_s) / 60)


def trip_features(row: dict[str, str], hourly_temp: dict[int, float]) -> dict[str, object]:
    rnd = stable_random("energy", row["trip_id"])
    distance = float(row["distance_km"] or 0)
    if distance <= 0:
        distance = rnd.uniform(4.0, 18.0)
    start_hour = parse_time(row["start_time"]) // 3600 % 24
    minutes = duration_minutes(row["start_time"], row["end_time"])
    avg_speed = max(8.0, min(45.0, distance / (minutes / 60)))
    peak_hour = 1 if 7 <= start_hour < 10 or 17 <= start_hour < 20 else 0
    congestion = min(1.0, max(0.05, (0.62 if peak_hour else 0.28) + rnd.uniform(-0.12, 0.18)))
    passenger_load = min(1.0, max(0.1, (0.78 if peak_hour else 0.42) + rnd.uniform(-0.18, 0.18)))
    avg_slope = rnd.uniform(-0.025, 0.06)
    positive_slope = max(0.0, avg_slope)
    temperature = hourly_temp.get(start_hour, 25.0)
    temperature_penalty = max(0.0, abs(temperature - 22.0) - 3.0) * 0.08 * distance
    clean_energy = (
        1.1 * distance
        + 0.25 * positive_slope * 100 * distance
        + 0.4 * congestion * distance
        + 0.2 * passenger_load * distance
        + temperature_penalty
    )
    clean_energy = max(0.8, clean_energy)
    noisy_energy, noise_kwh, noise_ratio, noise_profile = apply_energy_noise(
        clean_energy,
        congestion,
        passenger_load,
        temperature,
        peak_hour,
        avg_speed,
        rnd,
    )
    return {
        "trip_id": row["trip_id"],
        "route_id": row["route_id"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "distance_km": round(distance, 2),
        "avg_slope": round(avg_slope, 4),
        "congestion_index": round(congestion, 3),
        "avg_speed": round(avg_speed, 2),
        "passenger_load": round(passenger_load, 3),
        "temperature": round(temperature, 1),
        "peak_hour": peak_hour,
        "energy_kwh_clean": round(clean_energy, 2),
        "energy_kwh": round(noisy_energy, 2),
        "noise_kwh": round(noise_kwh, 2),
        "noise_ratio": round(noise_ratio, 4),
        "noise_profile": noise_profile,
        "source": "GTFS distance/timing + HKO temperature + calibrated simulation with Gaussian and scenario noise",
    }


def apply_energy_noise(
    clean_energy_kwh: float,
    congestion: float,
    passenger_load: float,
    temperature: float,
    peak_hour: int,
    avg_speed: float,
    rnd: random.Random,
) -> tuple[float, float, float, str]:
    """Apply reproducible Gaussian and scenario-dependent noise to energy labels."""

    scenario_ratio = BASE_ENERGY_NOISE_RATIO
    profile_parts = ["gaussian"]
    if peak_hour:
        scenario_ratio += 0.05
        profile_parts.append("peak_hour")
    if congestion >= 0.65:
        scenario_ratio += 0.06
        profile_parts.append("high_congestion")
    if passenger_load >= 0.75:
        scenario_ratio += 0.03
        profile_parts.append("high_load")
    if abs(temperature - 22.0) >= 4.0:
        scenario_ratio += 0.04
        profile_parts.append("temperature_stress")
    if avg_speed <= 12.0:
        scenario_ratio += 0.03
        profile_parts.append("low_speed")

    additive_noise = rnd.gauss(0.0, BASE_ENERGY_NOISE_KWH)
    proportional_noise = clean_energy_kwh * rnd.gauss(0.0, scenario_ratio)
    outlier_noise = 0.0
    if rnd.random() < OUTLIER_PROBABILITY:
        outlier_noise = clean_energy_kwh * rnd.gauss(0.0, 0.35)
        profile_parts.append("rare_outlier")

    noise_kwh = additive_noise + proportional_noise + outlier_noise
    noisy_energy = max(0.8, clean_energy_kwh + noise_kwh)
    actual_noise_kwh = noisy_energy - clean_energy_kwh
    return (
        noisy_energy,
        actual_noise_kwh,
        actual_noise_kwh / max(clean_energy_kwh, 1e-9),
        "+".join(profile_parts),
    )


def build_energy_samples(trips: list[dict[str, str]], hourly_temp: dict[int, float], output: Path) -> None:
    rows = [trip_features(row, hourly_temp) for row in trips]
    write_csv(
        output,
        [
            "trip_id",
            "route_id",
            "start_time",
            "end_time",
            "distance_km",
            "avg_slope",
            "congestion_index",
            "avg_speed",
            "passenger_load",
            "temperature",
            "peak_hour",
            "energy_kwh_clean",
            "energy_kwh",
            "noise_kwh",
            "noise_ratio",
            "noise_profile",
            "source",
        ],
        rows,
    )


def build_path_candidates(trips: list[dict[str, str]], hourly_temp: dict[int, float], output: Path) -> None:
    rows = []
    for trip in trips:
        base = trip_features(trip, hourly_temp)
        distance = float(base["distance_km"])
        for path_id, multiplier, slope_delta, congestion_delta, label in [
            ("A", 0.94, 0.018, 0.02, "shorter_with_more_slope"),
            ("B", 1.00, -0.006, -0.08, "balanced_low_congestion"),
            ("C", 1.08, -0.012, 0.10, "longer_but_flatter"),
        ]:
            candidate = dict(trip)
            candidate["distance_km"] = f"{distance * multiplier:.2f}"
            feat = trip_features(candidate, hourly_temp)
            feat["avg_slope"] = round(float(base["avg_slope"]) + slope_delta, 4)
            feat["congestion_index"] = round(min(1.0, max(0.05, float(base["congestion_index"]) + congestion_delta)), 3)
            feat["path_id"] = f"{trip['trip_id']}-{path_id}"
            feat["path_type"] = label
            feat["carbon_kgco2"] = round(float(feat["energy_kwh"]) * 0.55, 2)
            feat["source"] = "Candidate paths constructed from GTFS trip geometry proxies"
            rows.append(feat)
    write_csv(
        output,
        [
            "path_id",
            "trip_id",
            "route_id",
            "start_time",
            "end_time",
            "path_type",
            "distance_km",
            "avg_slope",
            "congestion_index",
            "avg_speed",
            "passenger_load",
            "temperature",
            "peak_hour",
            "energy_kwh_clean",
            "energy_kwh",
            "noise_kwh",
            "noise_ratio",
            "noise_profile",
            "carbon_kgco2",
            "source",
        ],
        rows,
    )


def build_all() -> None:
    gtfs_dir = ensure_gtfs()
    trips_path = PROCESSED_DIR / "trips_hk_gtfs_full.csv"
    if not trips_path.exists():
        raise SystemExit("Missing data/processed/trips_hk_gtfs_full.csv. Run scripts/collect_hk_gtfs.py first.")
    trips = load_trips(trips_path)
    hourly_temp = build_weather(PROCESSED_DIR / "weather_hourly.csv")
    build_vehicles(trips, PROCESSED_DIR / "vehicles.csv")
    build_stations(gtfs_dir, trips, PROCESSED_DIR / "stations.csv")
    build_penetration_scenarios(gtfs_dir, trips)
    build_prices(PROCESSED_DIR / "prices.csv")
    build_energy_samples(trips, hourly_temp, PROCESSED_DIR / "energy_samples.csv")
    build_path_candidates(trips, hourly_temp, PROCESSED_DIR / "path_candidates.csv")
    print("Built project datasets in data/processed")


if __name__ == "__main__":
    build_all()
