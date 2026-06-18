#!/usr/bin/env python3
"""Run the JSQ baseline charging scheduler."""

from __future__ import annotations

import json
import logging
import argparse
import hashlib
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIPS_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "trips_hk_gtfs_full.csv"
SMALL_TRIPS_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "trips_hk_gtfs.csv"
VEHICLES_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "vehicles.csv"
HK_SCALE_VEHICLES_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "vehicles_hk_scale_5870.csv"
STATIONS_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "stations.csv"
HK_SCALE_STATIONS_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "stations_hk_scale_80hubs.csv"
PRICES_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "prices.csv"
ENERGY_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "energy_predictions.csv"
SCHEDULE_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "schedules" / "jsq_schedule.csv"
METRICS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "jsq_metrics.json"
SMALL_SCHEDULE_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "schedules" / "jsq_schedule_small_demo.csv"
SMALL_METRICS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "jsq_metrics_small_demo.json"
HK_SCALE_SCHEDULE_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "schedules" / "jsq_schedule_hk_scale.csv"
HK_SCALE_METRICS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "jsq_metrics_hk_scale.json"

MINUTES_PER_DAY = 24 * 60
HK_SCALE_BUS_COUNT = 5870
HK_SCALE_HUB_COUNT = 80


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JSQConfig:
    """Configuration for a deterministic JSQ baseline simulation."""

    charge_trigger_soc: float = 0.30
    target_soc: float = 0.90
    charger_travel_time_min: float = 0.0


@dataclass(frozen=True)
class SchedulerPaths:
    """Input and output paths for a named JSQ scenario."""

    scenario_name: str
    trips_input_path: Path
    vehicles_input_path: Path
    stations_input_path: Path
    prices_input_path: Path
    energy_input_path: Path
    schedule_output_path: Path
    metrics_output_path: Path
    description: str


@dataclass
class VehicleState:
    """Mutable vehicle state used by the scheduling simulation."""

    bus_id: str
    battery_capacity_kwh: float
    current_soc: float
    max_soc: float
    min_soc: float
    assigned_route: str
    available_time_min: float = 0.0


@dataclass
class ChargerQueue:
    """One FCFS charging queue belonging to a physical charger."""

    station_id: str
    charger_id: str
    charger_type: str
    power_kw: float
    available_time_min: float = 0.0


def parse_time_to_minutes(value: Any) -> int:
    """Parse HH:MM or GTFS HH:MM:SS text into minutes after service-day midnight."""

    if not isinstance(value, str):
        raise ValueError(f"Invalid time value: {value!r}")
    match = re.fullmatch(r"(\d{2}):([0-5]\d)(?::([0-5]\d))?", value)
    if match is None:
        raise ValueError(f"Invalid time format: {value!r}")
    hours = int(match.group(1))
    minutes = int(match.group(2))
    return hours * 60 + minutes


def format_minutes(value: float) -> str:
    """Format absolute service minutes as HH:MM, preserving hours beyond 24."""

    rounded_minutes = int(round(value))
    hours, minutes = divmod(rounded_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def build_scheduler_paths(scenario_name: str) -> SchedulerPaths:
    """Resolve file paths for a supported JSQ experiment scenario."""

    if scenario_name == "small_demo":
        return SchedulerPaths(
            scenario_name=scenario_name,
            trips_input_path=SMALL_TRIPS_INPUT_PATH,
            vehicles_input_path=VEHICLES_INPUT_PATH,
            stations_input_path=STATIONS_INPUT_PATH,
            prices_input_path=PRICES_INPUT_PATH,
            energy_input_path=ENERGY_INPUT_PATH,
            schedule_output_path=SMALL_SCHEDULE_OUTPUT_PATH,
            metrics_output_path=SMALL_METRICS_OUTPUT_PATH,
            description="20-bus small demo on the sampled GTFS routes",
        )
    if scenario_name == "hk_scale":
        return SchedulerPaths(
            scenario_name=scenario_name,
            trips_input_path=TRIPS_INPUT_PATH,
            vehicles_input_path=HK_SCALE_VEHICLES_INPUT_PATH,
            stations_input_path=HK_SCALE_STATIONS_INPUT_PATH,
            prices_input_path=PRICES_INPUT_PATH,
            energy_input_path=ENERGY_INPUT_PATH,
            schedule_output_path=HK_SCALE_SCHEDULE_OUTPUT_PATH,
            metrics_output_path=HK_SCALE_METRICS_OUTPUT_PATH,
            description="Hong Kong scale full GTFS scenario with synthetic fleet and charging hubs",
        )
    if scenario_name == "default":
        return SchedulerPaths(
            scenario_name=scenario_name,
            trips_input_path=TRIPS_INPUT_PATH,
            vehicles_input_path=VEHICLES_INPUT_PATH,
            stations_input_path=STATIONS_INPUT_PATH,
            prices_input_path=PRICES_INPUT_PATH,
            energy_input_path=ENERGY_INPUT_PATH,
            schedule_output_path=SCHEDULE_OUTPUT_PATH,
            metrics_output_path=METRICS_OUTPUT_PATH,
            description="Legacy 20-bus full-GTFS stress scenario",
        )
    raise ValueError(f"Unsupported JSQ scenario: {scenario_name}")


def ensure_scenario_inputs(paths: SchedulerPaths) -> None:
    """Create deterministic synthetic scale inputs required by a scenario."""

    if paths.scenario_name == "hk_scale":
        ensure_hk_scale_vehicles(paths.vehicles_input_path)
        ensure_hk_scale_stations(paths.stations_input_path)


def ensure_hk_scale_vehicles(output_path: Path) -> None:
    """Generate a Hong Kong scale synthetic vehicle table when absent."""

    if output_path.exists():
        return
    if not TRIPS_INPUT_PATH.exists():
        raise FileNotFoundError(f"Cannot build HK scale vehicles before trips exist: {TRIPS_INPUT_PATH}")

    trips = pd.read_csv(TRIPS_INPUT_PATH, dtype={"route_id": str})
    route_counts = trips["route_id"].value_counts()
    routes = route_counts.index.tolist()
    weights = route_counts.to_numpy(dtype=float)
    weights = weights / weights.sum()

    rng = random.Random(42)
    assigned_routes = rng.choices(routes, weights=weights, k=HK_SCALE_BUS_COUNT)
    rows: list[dict[str, Any]] = []
    for index, route_id in enumerate(assigned_routes):
        seed = hashlib.sha256(f"hk-scale-fleet|{index}|{route_id}".encode("utf-8")).hexdigest()
        local_rng = random.Random(int(seed[:16], 16))
        rows.append(
            {
                "bus_id": f"HKBUS{index + 1:05d}",
                "battery_capacity": 300,
                "initial_soc": round(local_rng.uniform(0.72, 0.88), 3),
                "max_soc": 0.9,
                "min_soc": 0.2,
                "assigned_route": route_id,
                "source": (
                    "Hong Kong scale synthetic fleet; count set to 5870 licensed franchised buses reference "
                    "scale, route assignment weighted by GTFS trip frequency"
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info("Generated HK scale vehicles at %s", output_path)


def ensure_hk_scale_stations(output_path: Path) -> None:
    """Generate a scaled charging station table when absent."""

    if output_path.exists():
        return
    if not STATIONS_INPUT_PATH.exists():
        raise FileNotFoundError(f"Cannot build HK scale stations before base stations exist: {STATIONS_INPUT_PATH}")

    base_stations = pd.read_csv(STATIONS_INPUT_PATH)
    rows: list[dict[str, Any]] = []
    for index in range(HK_SCALE_HUB_COUNT):
        source_station = base_stations.iloc[index % len(base_stations)]
        rows.append(
            {
                "station_id": f"HKCS{index + 1:03d}",
                "station_name": f"HK Scale {source_station['location']} Charging Hub {index + 1:03d}",
                "location": source_station["location"],
                "stop_id": source_station["stop_id"],
                "lat": source_station["lat"],
                "lon": source_station["lon"],
                "fast_chargers": 12,
                "slow_chargers": 8,
                "fast_power": 120,
                "slow_power": 40,
                "source": (
                    "Synthetic Hong Kong scale charging infrastructure for full electric fleet scenario; "
                    "80 hubs, 960 fast chargers, 640 slow chargers"
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info("Generated HK scale stations at %s", output_path)


def load_input_data(paths: SchedulerPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all processed input tables required by the JSQ scheduler."""

    ensure_scenario_inputs(paths)
    input_paths = [
        paths.trips_input_path,
        paths.vehicles_input_path,
        paths.stations_input_path,
        paths.prices_input_path,
        paths.energy_input_path,
    ]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}. Run `make data` and `make train` first.")

    logger.info("Reading trips from %s", paths.trips_input_path)
    trips = pd.read_csv(paths.trips_input_path, dtype={"trip_id": str, "route_id": str})
    vehicles = pd.read_csv(paths.vehicles_input_path, dtype={"bus_id": str, "assigned_route": str})
    stations = pd.read_csv(paths.stations_input_path, dtype={"station_id": str})
    prices = pd.read_csv(paths.prices_input_path)
    energy = pd.read_csv(paths.energy_input_path, dtype={"trip_id": str, "route_id": str})

    validate_input_tables(trips, vehicles, stations, prices, energy)
    trips = merge_trip_energy(trips, energy)
    return trips, vehicles, stations, prices


def validate_input_tables(
    trips: pd.DataFrame,
    vehicles: pd.DataFrame,
    stations: pd.DataFrame,
    prices: pd.DataFrame,
    energy: pd.DataFrame,
) -> None:
    """Validate table schemas and basic numeric ranges."""

    required_columns = {
        "trips": {"trip_id", "route_id", "start_time", "end_time", "origin", "destination", "distance_km"},
        "vehicles": {"bus_id", "battery_capacity", "initial_soc", "max_soc", "min_soc", "assigned_route"},
        "stations": {"station_id", "fast_chargers", "slow_chargers", "fast_power", "slow_power"},
        "prices": {"start_time", "end_time", "price_per_kwh", "currency"},
        "energy": {"trip_id", "predicted_energy_kwh"},
    }
    tables = {
        "trips": trips,
        "vehicles": vehicles,
        "stations": stations,
        "prices": prices,
        "energy": energy,
    }
    for name, columns in required_columns.items():
        missing_columns = sorted(columns - set(tables[name].columns))
        if missing_columns:
            raise ValueError(f"{name} table is missing required columns: {missing_columns}")
        if tables[name].empty:
            raise ValueError(f"{name} table is empty.")

    numeric_checks = [
        (vehicles, "battery_capacity", 0.0),
        (stations, "fast_power", 0.0),
        (stations, "slow_power", 0.0),
        (prices, "price_per_kwh", 0.0),
        (energy, "predicted_energy_kwh", 0.0),
    ]
    for data, column, lower_bound in numeric_checks:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not bool((data[column] > lower_bound).all()):
            raise ValueError(f"{column} must be greater than {lower_bound}.")


def merge_trip_energy(trips: pd.DataFrame, energy: pd.DataFrame) -> pd.DataFrame:
    """Attach predicted trip energy and derived time fields to the trip table."""

    merged = trips.merge(
        energy[["trip_id", "predicted_energy_kwh"]],
        on="trip_id",
        how="left",
        validate="one_to_one",
    )
    missing_energy = int(merged["predicted_energy_kwh"].isna().sum())
    if missing_energy:
        raise ValueError(f"Missing predicted energy for {missing_energy} trips.")

    merged["start_min"] = merged["start_time"].map(parse_time_to_minutes)
    merged["end_min"] = merged["end_time"].map(parse_time_to_minutes)
    merged.loc[merged["end_min"] < merged["start_min"], "end_min"] += MINUTES_PER_DAY
    merged["predicted_energy_kwh"] = pd.to_numeric(merged["predicted_energy_kwh"], errors="raise")
    merged["distance_km"] = pd.to_numeric(merged["distance_km"], errors="coerce").fillna(0.0)
    return merged.sort_values(["start_min", "end_min", "trip_id"]).reset_index(drop=True)


def build_vehicle_states(vehicles: pd.DataFrame) -> list[VehicleState]:
    """Convert vehicle rows to mutable simulation state objects."""

    states: list[VehicleState] = []
    for row in vehicles.itertuples(index=False):
        states.append(
            VehicleState(
                bus_id=str(row.bus_id),
                battery_capacity_kwh=float(row.battery_capacity),
                current_soc=float(row.initial_soc),
                max_soc=float(row.max_soc),
                min_soc=float(row.min_soc),
                assigned_route=str(row.assigned_route),
            )
        )
    return states


def build_charger_queues(stations: pd.DataFrame) -> list[ChargerQueue]:
    """Build one FCFS queue for each fast and slow charger in the station table."""

    queues: list[ChargerQueue] = []
    for row in stations.itertuples(index=False):
        station_id = str(row.station_id)
        for index in range(int(row.fast_chargers)):
            queues.append(
                ChargerQueue(
                    station_id=station_id,
                    charger_id=f"{station_id}-F{index + 1:02d}",
                    charger_type="fast",
                    power_kw=float(row.fast_power),
                )
            )
        for index in range(int(row.slow_chargers)):
            queues.append(
                ChargerQueue(
                    station_id=station_id,
                    charger_id=f"{station_id}-S{index + 1:02d}",
                    charger_type="slow",
                    power_kw=float(row.slow_power),
                )
            )
    if not queues:
        raise ValueError("stations table does not define any charger queues.")
    return queues


def build_price_periods(prices: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert price rows into minute ranges for charging-cost integration."""

    periods: list[dict[str, Any]] = []
    for row in prices.itertuples(index=False):
        start_min = parse_time_to_minutes(str(row.start_time))
        end_min = parse_time_to_minutes(str(row.end_time))
        periods.append(
            {
                "start_min": start_min,
                "end_min": end_min,
                "price_per_kwh": float(row.price_per_kwh),
                "currency": str(row.currency),
            }
        )
    return periods


def get_price_at_minute(service_minute: float, price_periods: list[dict[str, Any]]) -> float:
    """Return the electricity price active at an absolute service minute."""

    day_minute = service_minute % MINUTES_PER_DAY
    for period in price_periods:
        if period["start_min"] <= day_minute < period["end_min"]:
            return float(period["price_per_kwh"])
    return float(price_periods[-1]["price_per_kwh"])


def calculate_charging_cost(
    start_min: float,
    end_min: float,
    power_kw: float,
    price_periods: list[dict[str, Any]],
) -> float:
    """Calculate time-of-use charging cost for a constant-power charging interval."""

    if end_min <= start_min:
        return 0.0

    total_cost = 0.0
    current_min = start_min
    boundaries = sorted({period["start_min"] for period in price_periods} | {period["end_min"] for period in price_periods})
    while current_min < end_min:
        day_start = int(current_min // MINUTES_PER_DAY) * MINUTES_PER_DAY
        next_boundaries = [day_start + boundary for boundary in boundaries if day_start + boundary > current_min]
        next_boundary = min(next_boundaries) if next_boundaries else day_start + MINUTES_PER_DAY
        segment_end = min(end_min, next_boundary)
        energy_kwh = power_kw * (segment_end - current_min) / 60.0
        total_cost += energy_kwh * get_price_at_minute(current_min, price_periods)
        current_min = segment_end
    return total_cost


def select_vehicle(
    vehicles: list[VehicleState],
    route_id: str,
    start_min: float,
    required_energy_kwh: float,
) -> VehicleState | None:
    """Select an available vehicle that can complete the fixed trip."""

    candidates = []
    for vehicle in vehicles:
        energy_after_trip = vehicle.current_soc * vehicle.battery_capacity_kwh - required_energy_kwh
        if vehicle.available_time_min <= start_min and energy_after_trip >= vehicle.min_soc * vehicle.battery_capacity_kwh:
            route_bonus = 0 if vehicle.assigned_route == route_id else 1
            candidates.append((route_bonus, vehicle.available_time_min, -vehicle.current_soc, vehicle.bus_id, vehicle))
    if not candidates:
        return None
    return min(candidates)[-1]


def select_jsq_queue(
    queues: list[ChargerQueue],
    arrival_min: float,
    charge_energy_kwh: float,
    config: JSQConfig,
) -> tuple[ChargerQueue, float, float, float]:
    """Choose the charger queue with the shortest estimated completion time."""

    best: tuple[float, float, str, ChargerQueue, float, float] | None = None
    for queue in queues:
        start_min = max(arrival_min + config.charger_travel_time_min, queue.available_time_min)
        duration_min = charge_energy_kwh / queue.power_kw * 60.0
        finish_min = start_min + duration_min
        candidate = (finish_min, start_min, queue.charger_id, queue, duration_min, queue.available_time_min)
        if best is None or candidate[:3] < best[:3]:
            best = candidate

    if best is None:
        raise ValueError("No charger queues are available for JSQ selection.")
    finish_min, start_min, _, queue, duration_min, queue_available_min = best
    wait_time_min = max(0.0, start_min - arrival_min - config.charger_travel_time_min)
    queue.available_time_min = finish_min
    return queue, start_min, wait_time_min, duration_min


def build_schedule_row(
    trip: pd.Series,
    vehicle: VehicleState | None,
    status: str,
    miss_reason: str = "",
) -> dict[str, Any]:
    """Create a normalized schedule row with empty charging fields."""

    return {
        "trip_id": trip["trip_id"],
        "route_id": trip["route_id"],
        "bus_id": "" if vehicle is None else vehicle.bus_id,
        "status": status,
        "miss_reason": miss_reason,
        "origin": trip["origin"],
        "destination": trip["destination"],
        "trip_start_time": trip["start_time"],
        "trip_end_time": trip["end_time"],
        "trip_start_min": float(trip["start_min"]),
        "trip_end_min": float(trip["end_min"]),
        "distance_km": float(trip["distance_km"]),
        "predicted_energy_kwh": float(trip["predicted_energy_kwh"]),
        "soc_before_trip": None,
        "soc_after_trip": None,
        "charge_requested": False,
        "station_id": "",
        "charger_id": "",
        "charger_type": "",
        "charging_power_kw": 0.0,
        "charge_start_time": "",
        "charge_end_time": "",
        "charge_start_min": None,
        "charge_end_min": None,
        "wait_time_min": 0.0,
        "charge_duration_min": 0.0,
        "charged_energy_kwh": 0.0,
        "charging_cost": 0.0,
        "currency": "",
        "soc_after_charge": None,
    }


def schedule_charge_if_needed(
    row: dict[str, Any],
    vehicle: VehicleState,
    trip_end_min: float,
    queues: list[ChargerQueue],
    price_periods: list[dict[str, Any]],
    config: JSQConfig,
) -> None:
    """Apply JSQ charging after a trip when SOC falls below the trigger threshold."""

    if vehicle.current_soc >= config.charge_trigger_soc:
        return

    target_soc = min(config.target_soc, vehicle.max_soc)
    charge_energy_kwh = max(0.0, (target_soc - vehicle.current_soc) * vehicle.battery_capacity_kwh)
    if charge_energy_kwh <= 0:
        return

    queue, start_min, wait_time_min, duration_min = select_jsq_queue(queues, trip_end_min, charge_energy_kwh, config)
    end_min = start_min + duration_min
    charging_cost = calculate_charging_cost(start_min, end_min, queue.power_kw, price_periods)

    vehicle.current_soc = target_soc
    vehicle.available_time_min = end_min
    row.update(
        {
            "charge_requested": True,
            "station_id": queue.station_id,
            "charger_id": queue.charger_id,
            "charger_type": queue.charger_type,
            "charging_power_kw": queue.power_kw,
            "charge_start_time": format_minutes(start_min),
            "charge_end_time": format_minutes(end_min),
            "charge_start_min": start_min,
            "charge_end_min": end_min,
            "wait_time_min": wait_time_min,
            "charge_duration_min": duration_min,
            "charged_energy_kwh": charge_energy_kwh,
            "charging_cost": charging_cost,
            "currency": price_periods[0]["currency"],
            "soc_after_charge": vehicle.current_soc,
        }
    )


def run_jsq_scheduler(
    config: JSQConfig | None = None,
    scenario_name: str = "small_demo",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run the JSQ baseline scheduler.

    Vehicles run fixed trips in chronological order. When a completed trip leaves
    a vehicle below the charge trigger, it joins the charger queue with the
    shortest estimated completion time. Each charger queue follows FCFS.
    """

    resolved_config = config or JSQConfig()
    paths = build_scheduler_paths(scenario_name)
    start_time = time.perf_counter()
    logger.info("Running JSQ scenario: %s (%s)", paths.scenario_name, paths.description)
    trips, vehicles, stations, prices = load_input_data(paths)
    vehicle_states = build_vehicle_states(vehicles)
    charger_queues = build_charger_queues(stations)
    price_periods = build_price_periods(prices)

    rows: list[dict[str, Any]] = []
    for _, trip in trips.iterrows():
        required_energy_kwh = float(trip["predicted_energy_kwh"])
        vehicle = select_vehicle(vehicle_states, str(trip["route_id"]), float(trip["start_min"]), required_energy_kwh)
        if vehicle is None:
            rows.append(build_schedule_row(trip, None, "unserved", "no_available_vehicle"))
            continue

        row = build_schedule_row(trip, vehicle, "completed")
        soc_before_trip = vehicle.current_soc
        vehicle.current_soc -= required_energy_kwh / vehicle.battery_capacity_kwh
        vehicle.available_time_min = float(trip["end_min"])
        row["soc_before_trip"] = soc_before_trip
        row["soc_after_trip"] = vehicle.current_soc
        row["soc_after_charge"] = vehicle.current_soc

        schedule_charge_if_needed(
            row=row,
            vehicle=vehicle,
            trip_end_min=float(trip["end_min"]),
            queues=charger_queues,
            price_periods=price_periods,
            config=resolved_config,
        )
        rows.append(row)

    schedule = pd.DataFrame(rows)
    metrics = calculate_metrics(schedule, vehicle_states, start_time, resolved_config, paths)
    save_outputs(schedule, metrics, paths)
    logger.info(
        "JSQ scheduler finished: completed=%s, unserved=%s, cost=%.2f, runtime=%.2fs",
        metrics["completed_trip_count"],
        metrics["unserved_trip_count"],
        metrics["total_charging_cost"],
        metrics["runtime_seconds"],
    )
    return schedule, metrics


def calculate_metrics(
    schedule: pd.DataFrame,
    vehicles: list[VehicleState],
    start_time: float,
    config: JSQConfig,
    paths: SchedulerPaths,
) -> dict[str, Any]:
    """Calculate aggregate JSQ schedule metrics."""

    completed = schedule[schedule["status"] == "completed"]
    charges = schedule[schedule["charge_requested"]]
    trip_count = int(len(schedule))
    completed_count = int(len(completed))
    unserved_count = trip_count - completed_count
    final_soc_values = [vehicle.current_soc for vehicle in vehicles]
    return {
        "algorithm": "JSQ",
        "scenario_name": paths.scenario_name,
        "description": "Join the Shortest Queue baseline with FCFS charger queues",
        "scenario_description": paths.description,
        "trip_count": trip_count,
        "completed_trip_count": completed_count,
        "unserved_trip_count": unserved_count,
        "completion_rate": float(completed_count / trip_count) if trip_count else 0.0,
        "vehicle_count": int(len(vehicles)),
        "charging_event_count": int(len(charges)),
        "total_predicted_energy_kwh": float(completed["predicted_energy_kwh"].sum()),
        "total_charged_energy_kwh": float(charges["charged_energy_kwh"].sum()),
        "total_charging_cost": float(charges["charging_cost"].sum()),
        "average_wait_time_min": float(charges["wait_time_min"].mean()) if not charges.empty else 0.0,
        "max_wait_time_min": float(charges["wait_time_min"].max()) if not charges.empty else 0.0,
        "average_charge_duration_min": float(charges["charge_duration_min"].mean()) if not charges.empty else 0.0,
        "min_final_soc": float(min(final_soc_values)) if final_soc_values else 0.0,
        "max_final_soc": float(max(final_soc_values)) if final_soc_values else 0.0,
        "runtime_seconds": float(time.perf_counter() - start_time),
        "input_paths": {
            "trips": str(paths.trips_input_path.relative_to(PROJECT_ROOT)),
            "vehicles": str(paths.vehicles_input_path.relative_to(PROJECT_ROOT)),
            "stations": str(paths.stations_input_path.relative_to(PROJECT_ROOT)),
            "prices": str(paths.prices_input_path.relative_to(PROJECT_ROOT)),
            "energy_predictions": str(paths.energy_input_path.relative_to(PROJECT_ROOT)),
        },
        "schedule_output_path": str(paths.schedule_output_path.relative_to(PROJECT_ROOT)),
        "metrics_output_path": str(paths.metrics_output_path.relative_to(PROJECT_ROOT)),
        "config": asdict(config),
    }


def save_outputs(schedule: pd.DataFrame, metrics: dict[str, Any], paths: SchedulerPaths) -> None:
    """Persist schedule and metrics artifacts."""

    paths.schedule_output_path.parent.mkdir(parents=True, exist_ok=True)
    paths.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(paths.schedule_output_path, index=False)
    with paths.metrics_output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    logger.info("Saved JSQ schedule to %s", paths.schedule_output_path)
    logger.info("Saved JSQ metrics to %s", paths.metrics_output_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for JSQ scenario runs."""

    parser = argparse.ArgumentParser(description="Run JSQ baseline charging scheduler.")
    parser.add_argument(
        "--scenario",
        choices=["small_demo", "hk_scale", "default"],
        default="small_demo",
        help="Experiment scenario to run. small_demo is the default quick scenario.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint used by `python -m src.baseline_jsq` and `make jsq`."""

    args = parse_args()
    run_jsq_scheduler(scenario_name=args.scenario)


if __name__ == "__main__":
    main()
