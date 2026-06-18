#!/usr/bin/env python3
"""Run a genetic-algorithm charging scheduler."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.baseline_jsq import (
    PROJECT_ROOT,
    ChargerQueue,
    SchedulerPaths,
    VehicleState,
    build_charger_queues,
    build_price_periods,
    build_scheduler_paths,
    build_vehicle_states,
    calculate_charging_cost,
    format_minutes,
    get_price_at_minute,
    load_input_data,
    run_jsq_scheduler,
)


GA_SCHEDULE_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "schedules" / "ga_schedule.csv"
GA_METRICS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "ga_metrics.json"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GAConfig:
    """Configuration for the compact policy-level genetic search."""

    scenario_name: str = "default"
    population_size: int = 10
    generations: int = 6
    elite_count: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.25
    random_seed: int = 42
    charge_trigger_min: float = 0.22
    charge_trigger_max: float = 0.45
    target_soc_min: float = 0.55
    target_soc_max: float = 0.90
    cost_weight_max: float = 2.5
    wait_weight_max: float = 1.0
    slow_charger_penalty_min: float = -20.0
    slow_charger_penalty_max: float = 40.0
    unserved_trip_penalty: float = 100_000.0
    wait_time_penalty: float = 0.5
    low_soc_penalty: float = 10_000.0


@dataclass(frozen=True)
class TripRecord:
    """A compact immutable trip record used by fast repeated GA simulations."""

    trip_id: str
    route_id: str
    origin: str
    destination: str
    start_time: str
    end_time: str
    start_min: float
    end_min: float
    distance_km: float
    predicted_energy_kwh: float


@dataclass(frozen=True)
class Chromosome:
    """Charging policy parameters optimized by the genetic algorithm."""

    trigger_soc: float
    offpeak_target_soc: float
    peak_target_soc: float
    cost_weight: float
    wait_weight: float
    slow_charger_penalty: float

    def normalized_key(self) -> tuple[float, ...]:
        """Return a rounded key for memoizing fitness evaluations."""

        return (
            round(self.trigger_soc, 4),
            round(self.offpeak_target_soc, 4),
            round(self.peak_target_soc, 4),
            round(self.cost_weight, 4),
            round(self.wait_weight, 4),
            round(self.slow_charger_penalty, 4),
        )


@dataclass
class SimulationResult:
    """Aggregate result from simulating one charging policy."""

    trip_count: int
    completed_trip_count: int
    unserved_trip_count: int
    charging_event_count: int
    total_predicted_energy_kwh: float
    total_charged_energy_kwh: float
    total_charging_cost: float
    total_wait_time_min: float
    max_wait_time_min: float
    total_charge_duration_min: float
    min_final_soc: float
    max_final_soc: float
    fitness_score: float
    schedule: pd.DataFrame | None = None


def clamp(value: float, lower_bound: float, upper_bound: float) -> float:
    """Clamp a floating-point value to an inclusive interval."""

    return max(lower_bound, min(upper_bound, value))


def trips_to_records(trips: pd.DataFrame) -> list[TripRecord]:
    """Convert the merged trip table into lightweight records for repeated runs."""

    records: list[TripRecord] = []
    for row in trips.itertuples(index=False):
        records.append(
            TripRecord(
                trip_id=str(row.trip_id),
                route_id=str(row.route_id),
                origin=str(row.origin),
                destination=str(row.destination),
                start_time=str(row.start_time),
                end_time=str(row.end_time),
                start_min=float(row.start_min),
                end_min=float(row.end_min),
                distance_km=float(row.distance_km),
                predicted_energy_kwh=float(row.predicted_energy_kwh),
            )
        )
    return records


def clone_vehicle_states(vehicles: list[VehicleState]) -> list[VehicleState]:
    """Return a fresh mutable vehicle-state copy for one simulation."""

    return [
        VehicleState(
            bus_id=vehicle.bus_id,
            battery_capacity_kwh=vehicle.battery_capacity_kwh,
            current_soc=vehicle.current_soc,
            max_soc=vehicle.max_soc,
            min_soc=vehicle.min_soc,
            assigned_route=vehicle.assigned_route,
            available_time_min=0.0,
        )
        for vehicle in vehicles
    ]


def clone_charger_queues(queues: list[ChargerQueue]) -> list[ChargerQueue]:
    """Return a fresh mutable charger-queue copy for one simulation."""

    return [
        ChargerQueue(
            station_id=queue.station_id,
            charger_id=queue.charger_id,
            charger_type=queue.charger_type,
            power_kw=queue.power_kw,
            available_time_min=0.0,
        )
        for queue in queues
    ]


def select_vehicle(
    vehicles: list[VehicleState],
    route_id: str,
    start_min: float,
    required_energy_kwh: float,
) -> VehicleState | None:
    """Select an available vehicle using the same tie-breakers as JSQ."""

    best: tuple[int, float, float, str, VehicleState] | None = None
    for vehicle in vehicles:
        energy_after_trip = vehicle.current_soc * vehicle.battery_capacity_kwh - required_energy_kwh
        minimum_energy = vehicle.min_soc * vehicle.battery_capacity_kwh
        if vehicle.available_time_min <= start_min and energy_after_trip >= minimum_energy:
            route_bonus = 0 if vehicle.assigned_route == route_id else 1
            candidate = (route_bonus, vehicle.available_time_min, -vehicle.current_soc, vehicle.bus_id, vehicle)
            if best is None or candidate[:4] < best[:4]:
                best = candidate
    return None if best is None else best[-1]


def is_peak_minute(service_minute: float, price_periods: list[dict[str, Any]]) -> bool:
    """Return True when the active price is above the cheapest available tariff."""

    active_price = get_price_at_minute(service_minute, price_periods)
    minimum_price = min(float(period["price_per_kwh"]) for period in price_periods)
    return active_price > minimum_price


def choose_target_soc(
    chromosome: Chromosome,
    vehicle: VehicleState,
    service_minute: float,
    price_periods: list[dict[str, Any]],
) -> float:
    """Choose the target charge level for the current tariff period."""

    configured_target = chromosome.peak_target_soc if is_peak_minute(service_minute, price_periods) else chromosome.offpeak_target_soc
    target_soc = max(configured_target, chromosome.trigger_soc + 0.03)
    return min(vehicle.max_soc, target_soc)


def select_ga_queue(
    queues: list[ChargerQueue],
    arrival_min: float,
    charge_energy_kwh: float,
    chromosome: Chromosome,
    price_periods: list[dict[str, Any]],
) -> tuple[ChargerQueue, float, float, float, float]:
    """Select the charger queue minimizing the GA policy score."""

    best: tuple[float, float, str, ChargerQueue, float, float, float] | None = None
    for queue in queues:
        start_min = max(arrival_min, queue.available_time_min)
        duration_min = charge_energy_kwh / queue.power_kw * 60.0
        end_min = start_min + duration_min
        wait_time_min = max(0.0, start_min - arrival_min)
        charging_cost = calculate_charging_cost(start_min, end_min, queue.power_kw, price_periods)
        charger_penalty = chromosome.slow_charger_penalty if queue.charger_type == "slow" else 0.0
        score = (
            end_min
            + chromosome.wait_weight * wait_time_min
            + chromosome.cost_weight * charging_cost
            + charger_penalty
        )
        candidate = (score, start_min, queue.charger_id, queue, duration_min, wait_time_min, charging_cost)
        if best is None or candidate[:3] < best[:3]:
            best = candidate

    if best is None:
        raise ValueError("No charger queues are available for GA selection.")

    _, start_min, _, queue, duration_min, wait_time_min, charging_cost = best
    queue.available_time_min = start_min + duration_min
    return queue, start_min, wait_time_min, duration_min, charging_cost


def build_schedule_row(trip: TripRecord, vehicle: VehicleState | None, status: str, miss_reason: str = "") -> dict[str, Any]:
    """Create one schedule output row with the JSQ-compatible schema."""

    return {
        "trip_id": trip.trip_id,
        "route_id": trip.route_id,
        "bus_id": "" if vehicle is None else vehicle.bus_id,
        "status": status,
        "miss_reason": miss_reason,
        "origin": trip.origin,
        "destination": trip.destination,
        "trip_start_time": trip.start_time,
        "trip_end_time": trip.end_time,
        "trip_start_min": trip.start_min,
        "trip_end_min": trip.end_min,
        "distance_km": trip.distance_km,
        "predicted_energy_kwh": trip.predicted_energy_kwh,
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


def simulate_chromosome(
    chromosome: Chromosome,
    trips: list[TripRecord],
    initial_vehicles: list[VehicleState],
    charger_templates: list[ChargerQueue],
    price_periods: list[dict[str, Any]],
    config: GAConfig,
    collect_schedule: bool = False,
) -> SimulationResult:
    """Run a full-day schedule simulation for one chromosome."""

    vehicles = clone_vehicle_states(initial_vehicles)
    queues = clone_charger_queues(charger_templates)
    rows: list[dict[str, Any]] = []
    completed_count = 0
    unserved_count = 0
    charging_event_count = 0
    total_predicted_energy_kwh = 0.0
    total_charged_energy_kwh = 0.0
    total_charging_cost = 0.0
    total_wait_time_min = 0.0
    max_wait_time_min = 0.0
    total_charge_duration_min = 0.0

    for trip in trips:
        required_energy_kwh = trip.predicted_energy_kwh
        vehicle = select_vehicle(vehicles, trip.route_id, trip.start_min, required_energy_kwh)
        if vehicle is None:
            unserved_count += 1
            if collect_schedule:
                rows.append(build_schedule_row(trip, None, "unserved", "no_available_vehicle"))
            continue

        completed_count += 1
        total_predicted_energy_kwh += required_energy_kwh
        row = build_schedule_row(trip, vehicle, "completed") if collect_schedule else None
        soc_before_trip = vehicle.current_soc
        vehicle.current_soc -= required_energy_kwh / vehicle.battery_capacity_kwh
        vehicle.available_time_min = trip.end_min
        if row is not None:
            row["soc_before_trip"] = soc_before_trip
            row["soc_after_trip"] = vehicle.current_soc
            row["soc_after_charge"] = vehicle.current_soc

        if vehicle.current_soc < chromosome.trigger_soc:
            target_soc = choose_target_soc(chromosome, vehicle, trip.end_min, price_periods)
            charge_energy_kwh = max(0.0, (target_soc - vehicle.current_soc) * vehicle.battery_capacity_kwh)
            if charge_energy_kwh > 0:
                queue, start_min, wait_time_min, duration_min, charging_cost = select_ga_queue(
                    queues,
                    trip.end_min,
                    charge_energy_kwh,
                    chromosome,
                    price_periods,
                )
                end_min = start_min + duration_min
                vehicle.current_soc = target_soc
                vehicle.available_time_min = end_min
                charging_event_count += 1
                total_charged_energy_kwh += charge_energy_kwh
                total_charging_cost += charging_cost
                total_wait_time_min += wait_time_min
                max_wait_time_min = max(max_wait_time_min, wait_time_min)
                total_charge_duration_min += duration_min
                if row is not None:
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

        if row is not None:
            rows.append(row)

    final_soc_values = [vehicle.current_soc for vehicle in vehicles]
    min_final_soc = float(min(final_soc_values)) if final_soc_values else 0.0
    max_final_soc = float(max(final_soc_values)) if final_soc_values else 0.0
    low_soc_shortfall = max(0.0, chromosome.trigger_soc - min_final_soc)
    fitness_score = (
        unserved_count * config.unserved_trip_penalty
        + total_charging_cost
        + total_wait_time_min * config.wait_time_penalty
        + low_soc_shortfall * config.low_soc_penalty
    )
    schedule = pd.DataFrame(rows) if collect_schedule else None
    return SimulationResult(
        trip_count=len(trips),
        completed_trip_count=completed_count,
        unserved_trip_count=unserved_count,
        charging_event_count=charging_event_count,
        total_predicted_energy_kwh=total_predicted_energy_kwh,
        total_charged_energy_kwh=total_charged_energy_kwh,
        total_charging_cost=total_charging_cost,
        total_wait_time_min=total_wait_time_min,
        max_wait_time_min=max_wait_time_min,
        total_charge_duration_min=total_charge_duration_min,
        min_final_soc=min_final_soc,
        max_final_soc=max_final_soc,
        fitness_score=fitness_score,
        schedule=schedule,
    )


def repair_chromosome(chromosome: Chromosome, config: GAConfig) -> Chromosome:
    """Normalize a chromosome into valid SOC and weight ranges."""

    trigger_soc = clamp(chromosome.trigger_soc, config.charge_trigger_min, config.charge_trigger_max)
    minimum_target = max(config.target_soc_min, trigger_soc + 0.03)
    offpeak_target_soc = clamp(chromosome.offpeak_target_soc, minimum_target, config.target_soc_max)
    peak_target_soc = clamp(chromosome.peak_target_soc, minimum_target, offpeak_target_soc)
    return Chromosome(
        trigger_soc=trigger_soc,
        offpeak_target_soc=offpeak_target_soc,
        peak_target_soc=peak_target_soc,
        cost_weight=clamp(chromosome.cost_weight, 0.0, config.cost_weight_max),
        wait_weight=clamp(chromosome.wait_weight, 0.0, config.wait_weight_max),
        slow_charger_penalty=clamp(
            chromosome.slow_charger_penalty,
            config.slow_charger_penalty_min,
            config.slow_charger_penalty_max,
        ),
    )


def random_chromosome(rng: random.Random, config: GAConfig) -> Chromosome:
    """Create one random valid charging-policy chromosome."""

    trigger_soc = rng.uniform(config.charge_trigger_min, config.charge_trigger_max)
    offpeak_target_soc = rng.uniform(max(config.target_soc_min, trigger_soc + 0.03), config.target_soc_max)
    peak_target_soc = rng.uniform(max(config.target_soc_min, trigger_soc + 0.03), offpeak_target_soc)
    return Chromosome(
        trigger_soc=trigger_soc,
        offpeak_target_soc=offpeak_target_soc,
        peak_target_soc=peak_target_soc,
        cost_weight=rng.uniform(0.0, config.cost_weight_max),
        wait_weight=rng.uniform(0.0, config.wait_weight_max),
        slow_charger_penalty=rng.uniform(config.slow_charger_penalty_min, config.slow_charger_penalty_max),
    )


def seed_population(rng: random.Random, config: GAConfig) -> list[Chromosome]:
    """Create an initial population containing JSQ-equivalent and heuristic seeds."""

    seeds = [
        Chromosome(0.30, 0.90, 0.90, 0.0, 0.0, 0.0),
        Chromosome(0.28, 0.88, 0.72, 1.0, 0.25, 5.0),
        Chromosome(0.35, 0.90, 0.78, 0.5, 0.75, 15.0),
        Chromosome(0.25, 0.82, 0.65, 2.0, 0.10, -5.0),
    ]
    population = [repair_chromosome(chromosome, config) for chromosome in seeds[: config.population_size]]
    while len(population) < config.population_size:
        population.append(random_chromosome(rng, config))
    return population


def crossover(parent_a: Chromosome, parent_b: Chromosome, rng: random.Random, config: GAConfig) -> Chromosome:
    """Blend two parent policies into one child policy."""

    values_a = parent_a.normalized_key()
    values_b = parent_b.normalized_key()
    child_values = []
    for value_a, value_b in zip(values_a, values_b):
        if rng.random() < 0.5:
            child_values.append(value_a)
        else:
            blend = rng.random()
            child_values.append(value_a * blend + value_b * (1.0 - blend))
    return repair_chromosome(Chromosome(*child_values), config)


def mutate(chromosome: Chromosome, rng: random.Random, config: GAConfig) -> Chromosome:
    """Apply bounded Gaussian mutation to a policy chromosome."""

    values = list(chromosome.normalized_key())
    mutation_scales = [0.035, 0.045, 0.045, 0.35, 0.15, 8.0]
    for index, scale in enumerate(mutation_scales):
        if rng.random() < config.mutation_rate:
            values[index] += rng.gauss(0.0, scale)
    return repair_chromosome(Chromosome(*values), config)


def tournament_select(
    ranked_population: list[tuple[float, Chromosome]],
    rng: random.Random,
    config: GAConfig,
) -> Chromosome:
    """Select a parent using tournament selection over ranked fitness results."""

    sample_size = min(config.tournament_size, len(ranked_population))
    contenders = rng.sample(ranked_population, sample_size)
    return min(contenders, key=lambda item: item[0])[1]


def run_genetic_search(
    trips: list[TripRecord],
    initial_vehicles: list[VehicleState],
    charger_templates: list[ChargerQueue],
    price_periods: list[dict[str, Any]],
    config: GAConfig,
) -> tuple[Chromosome, SimulationResult, list[dict[str, Any]]]:
    """Optimize the compact charging policy with a genetic algorithm."""

    rng = random.Random(config.random_seed)
    population = seed_population(rng, config)
    fitness_cache: dict[tuple[float, ...], SimulationResult] = {}
    history: list[dict[str, Any]] = []

    def evaluate(chromosome: Chromosome) -> SimulationResult:
        key = chromosome.normalized_key()
        if key not in fitness_cache:
            fitness_cache[key] = simulate_chromosome(
                chromosome,
                trips,
                initial_vehicles,
                charger_templates,
                price_periods,
                config,
            )
        return fitness_cache[key]

    for generation in range(config.generations):
        ranked_population = sorted(
            ((evaluate(chromosome).fitness_score, chromosome) for chromosome in population),
            key=lambda item: item[0],
        )
        best_score, best_chromosome = ranked_population[0]
        best_result = evaluate(best_chromosome)
        history.append(
            {
                "generation": generation + 1,
                "best_fitness_score": float(best_score),
                "best_completed_trip_count": int(best_result.completed_trip_count),
                "best_unserved_trip_count": int(best_result.unserved_trip_count),
                "best_total_charging_cost": float(best_result.total_charging_cost),
                "best_chromosome": asdict(best_chromosome),
            }
        )
        logger.info(
            "GA generation %s/%s: completed=%s, unserved=%s, cost=%.2f, fitness=%.2f",
            generation + 1,
            config.generations,
            best_result.completed_trip_count,
            best_result.unserved_trip_count,
            best_result.total_charging_cost,
            best_score,
        )

        next_population = [chromosome for _, chromosome in ranked_population[: config.elite_count]]
        while len(next_population) < config.population_size:
            parent_a = tournament_select(ranked_population, rng, config)
            parent_b = tournament_select(ranked_population, rng, config)
            child = crossover(parent_a, parent_b, rng, config)
            next_population.append(mutate(child, rng, config))
        population = next_population

    ranked_population = sorted(
        ((evaluate(chromosome).fitness_score, chromosome) for chromosome in population),
        key=lambda item: item[0],
    )
    best_chromosome = ranked_population[0][1]
    best_result = evaluate(best_chromosome)
    return best_chromosome, best_result, history


def charge_requested_mask(schedule: pd.DataFrame) -> pd.Series:
    """Return a robust boolean mask for schedule rows with charging events."""

    if "charge_requested" not in schedule.columns:
        return pd.Series(False, index=schedule.index)
    values = schedule["charge_requested"]
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def summarize_baseline(path: Path) -> dict[str, Any]:
    """Read the JSQ schedule artifact and summarize comparable metrics."""

    if not path.exists():
        return {"available": False, "schedule_path": str(path.relative_to(PROJECT_ROOT))}

    schedule = pd.read_csv(path)
    completed = schedule[schedule["status"] == "completed"]
    charges = schedule[charge_requested_mask(schedule)]
    trip_count = int(len(schedule))
    completed_count = int(len(completed))
    total_cost = float(charges["charging_cost"].sum()) if "charging_cost" in charges else 0.0
    return {
        "available": True,
        "schedule_path": str(path.relative_to(PROJECT_ROOT)),
        "trip_count": trip_count,
        "completed_trip_count": completed_count,
        "unserved_trip_count": trip_count - completed_count,
        "completion_rate": float(completed_count / trip_count) if trip_count else 0.0,
        "charging_event_count": int(len(charges)),
        "total_charging_cost": total_cost,
        "average_wait_time_min": float(charges["wait_time_min"].mean()) if not charges.empty else 0.0,
        "max_wait_time_min": float(charges["wait_time_min"].max()) if not charges.empty else 0.0,
    }


def ensure_baseline_summary(paths: SchedulerPaths) -> dict[str, Any]:
    """Load the same-scenario JSQ baseline, generating it when missing."""

    baseline_path = paths.schedule_output_path
    if not baseline_path.exists():
        logger.info(
            "JSQ baseline schedule not found at %s; generating scenario %s before GA comparison.",
            baseline_path,
            paths.scenario_name,
        )
        run_jsq_scheduler(scenario_name=paths.scenario_name)
    return summarize_baseline(baseline_path)


def build_metrics(
    result: SimulationResult,
    chromosome: Chromosome,
    baseline: dict[str, Any],
    history: list[dict[str, Any]],
    paths: SchedulerPaths,
    config: GAConfig,
    start_time: float,
) -> dict[str, Any]:
    """Build the JSON metrics artifact for the selected GA schedule."""

    average_wait_time_min = result.total_wait_time_min / result.charging_event_count if result.charging_event_count else 0.0
    average_charge_duration_min = (
        result.total_charge_duration_min / result.charging_event_count if result.charging_event_count else 0.0
    )
    metrics: dict[str, Any] = {
        "algorithm": "GA",
        "description": "Genetic algorithm over charging-policy parameters with JSQ schedule as baseline reference",
        "trip_count": int(result.trip_count),
        "completed_trip_count": int(result.completed_trip_count),
        "unserved_trip_count": int(result.unserved_trip_count),
        "completion_rate": float(result.completed_trip_count / result.trip_count) if result.trip_count else 0.0,
        "vehicle_count": int(len(build_vehicle_states(pd.read_csv(paths.vehicles_input_path)))),
        "charging_event_count": int(result.charging_event_count),
        "total_predicted_energy_kwh": float(result.total_predicted_energy_kwh),
        "total_charged_energy_kwh": float(result.total_charged_energy_kwh),
        "total_charging_cost": float(result.total_charging_cost),
        "average_wait_time_min": float(average_wait_time_min),
        "max_wait_time_min": float(result.max_wait_time_min),
        "average_charge_duration_min": float(average_charge_duration_min),
        "min_final_soc": float(result.min_final_soc),
        "max_final_soc": float(result.max_final_soc),
        "fitness_score": float(result.fitness_score),
        "runtime_seconds": float(time.perf_counter() - start_time),
        "input_paths": {
            "trips": str(paths.trips_input_path.relative_to(PROJECT_ROOT)),
            "vehicles": str(paths.vehicles_input_path.relative_to(PROJECT_ROOT)),
            "stations": str(paths.stations_input_path.relative_to(PROJECT_ROOT)),
            "prices": str(paths.prices_input_path.relative_to(PROJECT_ROOT)),
            "energy_predictions": str(paths.energy_input_path.relative_to(PROJECT_ROOT)),
            "baseline_schedule": str(paths.schedule_output_path.relative_to(PROJECT_ROOT)),
        },
        "schedule_output_path": str(GA_SCHEDULE_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "metrics_output_path": str(GA_METRICS_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "best_chromosome": asdict(chromosome),
        "config": asdict(config),
        "ga_history": history,
        "baseline_comparison": baseline,
    }

    if baseline.get("available"):
        metrics["baseline_comparison"].update(
            {
                "completed_trip_delta": int(result.completed_trip_count - baseline["completed_trip_count"]),
                "unserved_trip_delta": int(result.unserved_trip_count - baseline["unserved_trip_count"]),
                "charging_cost_delta": float(result.total_charging_cost - baseline["total_charging_cost"]),
                "charging_cost_delta_pct": (
                    float((result.total_charging_cost - baseline["total_charging_cost"]) / baseline["total_charging_cost"])
                    if baseline["total_charging_cost"]
                    else 0.0
                ),
            }
        )
    return metrics


def save_outputs(schedule: pd.DataFrame, metrics: dict[str, Any]) -> None:
    """Persist GA schedule and metrics artifacts."""

    GA_SCHEDULE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GA_METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(GA_SCHEDULE_OUTPUT_PATH, index=False)
    with GA_METRICS_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    logger.info("Saved GA schedule to %s", GA_SCHEDULE_OUTPUT_PATH)
    logger.info("Saved GA metrics to %s", GA_METRICS_OUTPUT_PATH)


def run_ga_optimizer(config: GAConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the GA optimizer and write schedule/metrics artifacts."""

    resolved_config = config or GAConfig()
    start_time = time.perf_counter()
    paths = build_scheduler_paths(resolved_config.scenario_name)
    trips_df, vehicles_df, stations_df, prices_df = load_input_data(paths)
    trips = trips_to_records(trips_df)
    initial_vehicles = build_vehicle_states(vehicles_df)
    charger_templates = build_charger_queues(stations_df)
    price_periods = build_price_periods(prices_df)
    baseline = ensure_baseline_summary(paths)

    best_chromosome, _, history = run_genetic_search(
        trips,
        initial_vehicles,
        charger_templates,
        price_periods,
        resolved_config,
    )
    final_result = simulate_chromosome(
        best_chromosome,
        trips,
        initial_vehicles,
        charger_templates,
        price_periods,
        resolved_config,
        collect_schedule=True,
    )
    if final_result.schedule is None:
        raise RuntimeError("GA final simulation did not produce a schedule.")

    metrics = build_metrics(final_result, best_chromosome, baseline, history, paths, resolved_config, start_time)
    save_outputs(final_result.schedule, metrics)
    logger.info(
        "GA optimizer finished: completed=%s, unserved=%s, cost=%.2f, runtime=%.2fs",
        metrics["completed_trip_count"],
        metrics["unserved_trip_count"],
        metrics["total_charging_cost"],
        metrics["runtime_seconds"],
    )
    return final_result.schedule, metrics


def main() -> None:
    """CLI entrypoint used by `python -m src.ga_optimizer` and `make ga`."""

    run_ga_optimizer()


if __name__ == "__main__":
    main()
