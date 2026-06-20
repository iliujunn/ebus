#!/usr/bin/env python3
"""Generate presentation-ready figures for scenario comparison results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "metrics" / "penetration_scenario_summary.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
ENERGY_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "energy_predictions.csv"
PATH_CANDIDATES_PATH = PROJECT_ROOT / "data" / "processed" / "path_candidates.csv"
JSQ_FULL_SCHEDULE_PATH = PROJECT_ROOT / "outputs" / "schedules" / "jsq_schedule_full.csv"
GA_FULL_SCHEDULE_PATH = PROJECT_ROOT / "outputs" / "schedules" / "ga_schedule_full.csv"
GA_FULL_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "ga_metrics_full.json"

SCENARIO_ORDER = ["current", "planned", "full"]
ALGORITHM_ORDER = ["JSQ", "GA"]
SCENARIO_LABELS = {
    "current": "Current\n150 buses",
    "planned": "Planned\n750 buses",
    "full": "Full\n5,870 buses",
}
ALGORITHM_COLORS = {
    "JSQ": "#4C78A8",
    "GA": "#F58518",
}
MODEL_COLORS = {
    "Fixed kWh/km": "#4C78A8",
    "Neural network": "#F58518",
}
PATH_STRATEGY_COLORS = {
    "Shortest distance": "#4C78A8",
    "Lowest energy": "#54A24B",
}
REQUIRED_COLUMNS = {
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
}


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON artifact with stable UTF-8 formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def compact_duration_label(minutes: float) -> str:
    """Format service minutes as HH:MM for schedule axes."""

    rounded = int(round(minutes))
    hours, minute = divmod(rounded, 60)
    return f"{hours:02d}:{minute:02d}"


def load_summary(path: Path) -> pd.DataFrame:
    """Load and validate the cross-scenario summary CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Scenario summary not found: {path}. Run `make scenario-summary` first.")

    data = pd.read_csv(path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(f"Scenario summary is missing required columns: {missing_columns}")

    data = data[data["scenario"].isin(SCENARIO_ORDER) & data["algorithm"].isin(ALGORITHM_ORDER)].copy()
    if data.empty:
        raise ValueError("Scenario summary does not contain current/planned/full JSQ/GA rows.")

    expected_pairs = {(scenario, algorithm) for scenario in SCENARIO_ORDER for algorithm in ALGORITHM_ORDER}
    actual_pairs = set(zip(data["scenario"], data["algorithm"], strict=True))
    missing_pairs = sorted(expected_pairs - actual_pairs)
    if missing_pairs:
        raise ValueError(f"Scenario summary is missing scenario/algorithm rows: {missing_pairs}")

    for column in REQUIRED_COLUMNS - {"scenario", "algorithm"}:
        data[column] = pd.to_numeric(data[column], errors="raise")

    return data


def pivot_metric(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return a scenario-by-algorithm table in stable display order."""

    pivoted = data.pivot(index="scenario", columns="algorithm", values=metric)
    return pivoted.reindex(index=SCENARIO_ORDER, columns=ALGORITHM_ORDER)


def metric_value(data: pd.DataFrame, scenario: str, algorithm: str, metric: str) -> float:
    """Return one numeric metric value from the summary table."""

    row = data[(data["scenario"] == scenario) & (data["algorithm"] == algorithm)]
    if row.empty:
        raise ValueError(f"Missing row for {scenario}/{algorithm}")
    return float(row.iloc[0][metric])


def format_compact_number(value: float) -> str:
    """Format large chart labels compactly."""

    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 10_000:
        return f"{value / 1_000:.0f}K"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def format_label(value: float, label_kind: str) -> str:
    """Format a bar label for one metric type."""

    if label_kind == "currency":
        return f"${format_compact_number(value)}"
    if label_kind == "minutes":
        return f"{value:.2f}"
    if label_kind == "percent":
        return f"{value * 100:.1f}%"
    return format_compact_number(value)


def format_money(value: float) -> str:
    """Format a currency value for annotations."""

    return f"HKD {format_compact_number(value)}"


def style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Apply shared chart styling."""

    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E8EB", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD1D8")
    ax.spines["bottom"].set_color("#CBD1D8")
    ax.tick_params(axis="both", colors="#2F3A45")


def format_axis_tick(value: float, label_kind: str) -> str:
    """Format y-axis ticks without cluttering the chart."""

    if label_kind == "currency":
        return f"${format_compact_number(value)}"
    if label_kind == "percent":
        return f"{value * 100:.0f}%"
    if label_kind == "minutes":
        return f"{value:g}"
    return format_compact_number(value)


def annotate_grouped_bars(ax: plt.Axes, label_kind: str) -> None:
    """Add value labels above grouped bars."""

    _, y_max = ax.get_ylim()
    label_offset = y_max * 0.018 if y_max else 0.2
    for container in ax.containers:
        for bar in container:
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + label_offset,
                format_label(height, label_kind),
                ha="center",
                va="bottom",
                fontsize=9,
                color="#23313F",
            )


def add_improvement_notes(ax: plt.Axes, values: pd.DataFrame, metric: str, lower_is_better: bool) -> None:
    """Annotate GA-vs-JSQ percentage changes where JSQ has a non-zero baseline."""

    _, y_max = ax.get_ylim()
    y_position = y_max * 0.91
    for index, scenario in enumerate(SCENARIO_ORDER):
        jsq = float(values.loc[scenario, "JSQ"])
        ga = float(values.loc[scenario, "GA"])
        if jsq == 0:
            continue
        delta_pct = (ga - jsq) / jsq
        if abs(delta_pct) < 0.001:
            continue
        direction = "lower" if (delta_pct < 0) == lower_is_better else "higher"
        ax.text(
            index,
            y_position,
            f"GA {abs(delta_pct) * 100:.1f}% {direction}",
            ha="center",
            va="center",
            fontsize=9,
            color="#1D4E89" if direction == "lower" else "#8A4B08",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D8DEE6"},
        )


def ga_improvement(data: pd.DataFrame, scenario: str, metric: str) -> tuple[float, float, float, float]:
    """Return JSQ, GA, absolute improvement, and percentage improvement."""

    jsq = metric_value(data, scenario, "JSQ", metric)
    ga = metric_value(data, scenario, "GA", metric)
    improvement = jsq - ga
    improvement_pct = improvement / jsq if jsq else 0.0
    return jsq, ga, improvement, improvement_pct


def plot_grouped_bar(
    data: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    label_kind: str = "number",
    lower_is_better: bool | None = None,
    y_limit: tuple[float, float] | None = None,
) -> None:
    """Create one grouped JSQ-vs-GA bar chart."""

    values = pivot_metric(data, metric)
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=150)
    x_positions = range(len(SCENARIO_ORDER))
    bar_width = 0.34

    for offset_index, algorithm in enumerate(ALGORITHM_ORDER):
        offset = (offset_index - 0.5) * bar_width
        ax.bar(
            [x + offset for x in x_positions],
            values[algorithm].to_numpy(),
            width=bar_width,
            label=algorithm,
            color=ALGORITHM_COLORS[algorithm],
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER])
    style_axis(ax, title, ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_axis_tick(value, label_kind)))
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    if y_limit is not None:
        ax.set_ylim(*y_limit)
    else:
        max_value = float(values.max().max())
        ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1.0)

    annotate_grouped_bars(ax, label_kind)
    if lower_is_better is not None:
        add_improvement_notes(ax, values, metric, lower_is_better)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cost_savings(data: pd.DataFrame, output_path: Path) -> None:
    """Plot the money saved by GA instead of another raw cost chart."""

    scenarios = ["planned", "full"]
    savings = [ga_improvement(data, scenario, "total_charging_cost") for scenario in scenarios]
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=150)
    values = [item[2] for item in savings]
    y_positions = range(len(scenarios))

    ax.barh(y_positions, values, color="#2E7D32", height=0.46)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([SCENARIO_LABELS[scenario].replace("\n", " / ") for scenario in scenarios])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_money(value)))
    style_axis(ax, "GA Charging Cost Savings vs JSQ", "Cost saved by GA")
    ax.set_ylabel("")
    ax.set_xlim(0, max(values) * 1.45)

    for index, (scenario, (jsq, ga, saving, saving_pct)) in enumerate(zip(scenarios, savings, strict=True)):
        ax.text(
            saving + max(values) * 0.03,
            index,
            f"{format_money(saving)} saved\n{saving_pct * 100:.1f}% lower\nJSQ {format_money(jsq)} -> GA {format_money(ga)}",
            va="center",
            ha="left",
            fontsize=10,
            color="#1B3D1F",
        )

    ax.text(
        0,
        -0.58,
        "Current scenario has no charging demand in the generated schedules, so cost savings are naturally zero.",
        ha="left",
        va="center",
        fontsize=9,
        color="#5B6570",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_full_wait_reduction(data: pd.DataFrame, output_path: Path) -> None:
    """Plot full-scale wait-time reduction using a dumbbell chart."""

    metrics = [
        ("average_wait_time_min", "Average wait"),
        ("max_wait_time_min", "Maximum wait"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=150)
    y_positions = range(len(metrics))

    for index, (metric, label) in enumerate(metrics):
        jsq, ga, reduction, reduction_pct = ga_improvement(data, "full", metric)
        ax.plot([ga, jsq], [index, index], color="#9BA7B4", linewidth=4, solid_capstyle="round")
        ax.scatter(jsq, index, s=120, color=ALGORITHM_COLORS["JSQ"], zorder=3, label="JSQ" if index == 0 else "")
        ax.scatter(ga, index, s=120, color=ALGORITHM_COLORS["GA"], zorder=3, label="GA" if index == 0 else "")
        ax.text(jsq + 2.0, index, f"{jsq:.1f} min", va="center", ha="left", fontsize=10, color="#23313F")
        if ga < 6:
            ax.text(ga, index - 0.13, f"{ga:.1f} min", va="top", ha="center", fontsize=10, color="#23313F")
        else:
            ax.text(max(ga - 1.5, 0), index, f"{ga:.1f} min", va="center", ha="right", fontsize=10, color="#23313F")
        ax.text(
            (jsq + ga) / 2,
            index + 0.23,
            f"{reduction:.1f} min lower ({reduction_pct * 100:.1f}%)",
            va="bottom",
            ha="center",
            fontsize=10,
            color="#1D4E89",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D8DEE6"},
        )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([label for _, label in metrics])
    ax.set_ylim(-0.45, len(metrics) - 0.35)
    ax.set_xlim(0, max(metric_value(data, "full", "JSQ", "max_wait_time_min") * 1.14, 10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    style_axis(ax, "Full-Scale Charging Wait Reduction", "")
    ax.set_xlabel("Wait time (minutes)")
    ax.set_ylabel("")
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_efficiency(data: pd.DataFrame, output_path: Path) -> None:
    """Plot per-1,000-trip charging cost and energy to make scenarios comparable."""

    normalized = data.copy()
    normalized["cost_per_1000_trips"] = normalized["total_charging_cost"] / normalized["trip_count"] * 1000
    normalized["energy_mwh_per_1000_trips"] = normalized["total_charged_energy_kwh"] / normalized["trip_count"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), dpi=150)
    configs = [
        ("cost_per_1000_trips", "Charging Cost per 1,000 Trips", "Cost per 1,000 trips", "currency"),
        ("energy_mwh_per_1000_trips", "Charged Energy per 1,000 Trips", "MWh per 1,000 trips", "number"),
    ]

    for ax, (metric, title, ylabel, label_kind) in zip(axes, configs, strict=True):
        values = pivot_metric(normalized, metric)
        x_positions = range(len(SCENARIO_ORDER))
        bar_width = 0.34
        for offset_index, algorithm in enumerate(ALGORITHM_ORDER):
            offset = (offset_index - 0.5) * bar_width
            ax.bar(
                [x + offset for x in x_positions],
                values[algorithm].to_numpy(),
                width=bar_width,
                label=algorithm,
                color=ALGORITHM_COLORS[algorithm],
            )
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([SCENARIO_LABELS[scenario].replace("\n", "\n") for scenario in SCENARIO_ORDER], fontsize=9)
        style_axis(ax, title, ylabel)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_axis_tick(value, label_kind)))
        max_value = float(values.max().max())
        ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1.0)
        annotate_grouped_bars(ax, label_kind)

    axes[0].legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(1.08, -0.16))
    fig.suptitle("Scale-Normalized Charging Efficiency", fontsize=15, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_report_dashboard(data: pd.DataFrame, output_path: Path) -> None:
    """Create a one-page result dashboard for slides."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    ax_scale, ax_cost, ax_wait, ax_emissions = axes.ravel()

    jsq_rows = data[data["algorithm"] == "JSQ"].set_index("scenario").reindex(SCENARIO_ORDER)
    x_positions = range(len(SCENARIO_ORDER))
    ax_scale.bar(x_positions, jsq_rows["trip_count"].to_numpy() / 1000, color="#52796F", width=0.42)
    ax_scale.set_xticks(list(x_positions))
    ax_scale.set_xticklabels([SCENARIO_LABELS[scenario].replace("\n", "\n") for scenario in SCENARIO_ORDER], fontsize=9)
    style_axis(ax_scale, "Scenario Scale", "Trips (thousand)")
    for index, scenario in enumerate(SCENARIO_ORDER):
        trips = metric_value(data, scenario, "JSQ", "trip_count")
        vehicles = metric_value(data, scenario, "JSQ", "vehicle_count")
        ax_scale.text(index, trips / 1000 + 2.0, f"{trips:,.0f} trips\n{vehicles:,.0f} buses", ha="center", fontsize=9)

    savings = [ga_improvement(data, scenario, "total_charging_cost") for scenario in ["planned", "full"]]
    ax_cost.bar([0, 1], [item[3] * 100 for item in savings], color="#2E7D32", width=0.48)
    ax_cost.set_xticks([0, 1])
    ax_cost.set_xticklabels(["Planned", "Full"])
    style_axis(ax_cost, "GA Cost Reduction", "Reduction vs JSQ")
    ax_cost.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax_cost.set_ylim(0, 60)
    for index, (_, _, saving, saving_pct) in enumerate(savings):
        ax_cost.text(index, saving_pct * 100 + 2.2, f"{saving_pct * 100:.1f}%\n{format_money(saving)}", ha="center", fontsize=9)

    wait_metrics = [ga_improvement(data, "full", "average_wait_time_min"), ga_improvement(data, "full", "max_wait_time_min")]
    ax_wait.bar([0, 1], [item[3] * 100 for item in wait_metrics], color="#1D4E89", width=0.48)
    ax_wait.set_xticks([0, 1])
    ax_wait.set_xticklabels(["Average", "Maximum"])
    style_axis(ax_wait, "Full-Scale Wait Reduction", "Reduction vs JSQ")
    ax_wait.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax_wait.set_ylim(0, 85)
    for index, (_, _, reduction, reduction_pct) in enumerate(wait_metrics):
        ax_wait.text(index, reduction_pct * 100 + 2.5, f"{reduction_pct * 100:.1f}%\n{reduction:.1f} min", ha="center", fontsize=9)

    emissions = [ga_improvement(data, scenario, "total_charging_co2_kg") for scenario in ["planned", "full"]]
    ax_emissions.bar([0, 1], [item[2] / 1000 for item in emissions], color="#6A4C93", width=0.48)
    ax_emissions.set_xticks([0, 1])
    ax_emissions.set_xticklabels(["Planned", "Full"])
    style_axis(ax_emissions, "Charging CO2 Reduction", "Tonnes CO2 lower")
    for index, (_, _, reduction, reduction_pct) in enumerate(emissions):
        ax_emissions.text(index, reduction / 1000 + 7, f"{reduction / 1000:.0f} t\n{reduction_pct * 100:.1f}%", ha="center", fontsize=9)

    fig.suptitle("Three-Stage Electrification Scheduling Results", fontsize=17, weight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Calculate standard regression metrics for one prediction series."""

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def build_energy_prediction_analysis(metrics_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Compare fixed kWh/km baseline against the trained neural model."""

    if not ENERGY_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Energy predictions not found: {ENERGY_PREDICTIONS_PATH}. Run `make train` first.")

    data = pd.read_csv(ENERGY_PREDICTIONS_PATH)
    required = {"distance_km", "energy_kwh", "predicted_energy_kwh"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"energy_predictions.csv is missing required columns: {missing}")

    data = data.copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="raise")

    fixed_kwh_per_km = float(data["energy_kwh"].sum() / data["distance_km"].sum())
    data["fixed_energy_kwh"] = data["distance_km"] * fixed_kwh_per_km
    data["fixed_error_kwh"] = data["fixed_energy_kwh"] - data["energy_kwh"]
    data["nn_error_kwh"] = data["predicted_energy_kwh"] - data["energy_kwh"]

    metrics = {
        "input_path": str(ENERGY_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
        "sample_count": int(len(data)),
        "fixed_kwh_per_km": fixed_kwh_per_km,
        "models": {
            "Fixed kWh/km": regression_metrics(data["energy_kwh"], data["fixed_energy_kwh"]),
            "Neural network": regression_metrics(data["energy_kwh"], data["predicted_energy_kwh"]),
        },
    }
    fixed = metrics["models"]["Fixed kWh/km"]
    neural = metrics["models"]["Neural network"]
    metrics["headline"] = {
        "mae_delta_kwh": neural["mae"] - fixed["mae"],
        "mae_reduction_pct": (fixed["mae"] - neural["mae"]) / fixed["mae"] if fixed["mae"] else 0.0,
        "rmse_delta_kwh": neural["rmse"] - fixed["rmse"],
        "rmse_reduction_pct": (fixed["rmse"] - neural["rmse"]) / fixed["rmse"] if fixed["rmse"] else 0.0,
        "r2_delta": neural["r2"] - fixed["r2"],
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "energy_prediction_comparison.json", metrics)
    return data, metrics


def plot_energy_metric_comparison(metrics: dict, output_path: Path) -> None:
    """Plot MAE/RMSE/R2 for fixed-distance and neural energy models."""

    metric_names = ["mae", "rmse", "r2"]
    labels = ["MAE (kWh)", "RMSE (kWh)", "R2"]
    model_names = list(MODEL_COLORS)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), dpi=150)
    for ax, metric_name, label in zip(axes, metric_names, labels, strict=True):
        values = [metrics["models"][model][metric_name] for model in model_names]
        ax.bar(model_names, values, color=[MODEL_COLORS[model] for model in model_names], width=0.5)
        style_axis(ax, label, label)
        ax.set_xlabel("")
        max_value = max(values)
        min_value = min(values)
        if metric_name == "r2":
            ax.set_ylim(max(-0.05, min_value - 0.12), min(1.05, max_value + 0.12))
        else:
            ax.set_ylim(0, max_value * 1.25)
        for index, value in enumerate(values):
            ax.text(index, value + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03, f"{value:.3f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", labelrotation=20)

    fig.suptitle("Energy Prediction Model Comparison", fontsize=15, weight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(data: pd.DataFrame, output_path: Path, sample_size: int = 5000) -> None:
    """Plot actual energy against neural-network predictions."""

    sample = data.sample(n=min(sample_size, len(data)), random_state=42)
    actual = sample["energy_kwh"]
    predicted = sample["predicted_energy_kwh"]
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    ax.scatter(actual, predicted, s=8, alpha=0.28, color="#F58518", edgecolors="none")
    ax.plot([low, high], [low, high], color="#23313F", linestyle="--", linewidth=1.2, label="Perfect prediction")
    style_axis(ax, "Actual vs Neural Network Predicted Energy", "Predicted energy (kWh)")
    ax.set_xlabel("Actual energy (kWh)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_error_distribution(data: pd.DataFrame, output_path: Path) -> None:
    """Plot prediction error distributions for the fixed and neural models."""

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=150)
    ax.hist(data["fixed_error_kwh"], bins=80, alpha=0.55, color=MODEL_COLORS["Fixed kWh/km"], label="Fixed kWh/km")
    ax.hist(data["nn_error_kwh"], bins=80, alpha=0.55, color=MODEL_COLORS["Neural network"], label="Neural network")
    ax.axvline(0, color="#23313F", linewidth=1.0)
    style_axis(ax, "Energy Prediction Error Distribution", "Trip count")
    ax.set_xlabel("Prediction error (predicted - actual, kWh)")
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_green_path_analysis(metrics_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Aggregate shortest-distance and lowest-energy path strategies."""

    if not PATH_CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"Path candidates not found: {PATH_CANDIDATES_PATH}. Run `make data-derived` first.")

    data = pd.read_csv(PATH_CANDIDATES_PATH)
    required = {"trip_id", "path_id", "path_type", "distance_km", "energy_kwh", "carbon_kgco2"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"path_candidates.csv is missing required columns: {missing}")

    data = data.copy()
    for column in ["distance_km", "energy_kwh", "carbon_kgco2"]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    shortest = data.loc[data.groupby("trip_id")["distance_km"].idxmin()].copy()
    lowest_energy = data.loc[data.groupby("trip_id")["energy_kwh"].idxmin()].copy()
    strategies = [("Shortest distance", shortest), ("Lowest energy", lowest_energy)]
    rows = []
    for strategy, frame in strategies:
        rows.append(
            {
                "strategy": strategy,
                "trip_count": int(frame["trip_id"].nunique()),
                "total_distance_km": float(frame["distance_km"].sum()),
                "total_energy_kwh": float(frame["energy_kwh"].sum()),
                "total_carbon_kgco2": float(frame["carbon_kgco2"].sum()),
                "average_distance_km": float(frame["distance_km"].mean()),
                "average_energy_kwh": float(frame["energy_kwh"].mean()),
                "average_carbon_kgco2": float(frame["carbon_kgco2"].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    type_summary = (
        data.groupby("path_type", as_index=False)
        .agg(
            trip_count=("trip_id", "nunique"),
            total_distance_km=("distance_km", "sum"),
            total_energy_kwh=("energy_kwh", "sum"),
            total_carbon_kgco2=("carbon_kgco2", "sum"),
            average_distance_km=("distance_km", "mean"),
            average_energy_kwh=("energy_kwh", "mean"),
            average_carbon_kgco2=("carbon_kgco2", "mean"),
        )
        .sort_values("total_energy_kwh")
        .reset_index(drop=True)
    )
    selected_pairs = shortest[["trip_id", "path_id"]].merge(
        lowest_energy[["trip_id", "path_id"]],
        on="trip_id",
        suffixes=("_shortest", "_lowest_energy"),
    )
    same_path_count = int((selected_pairs["path_id_shortest"] == selected_pairs["path_id_lowest_energy"]).sum())
    shortest_row = summary[summary["strategy"] == "Shortest distance"].iloc[0]
    green_row = summary[summary["strategy"] == "Lowest energy"].iloc[0]
    metrics = {
        "input_path": str(PATH_CANDIDATES_PATH.relative_to(PROJECT_ROOT)),
        "candidate_count": int(len(data)),
        "trip_count": int(data["trip_id"].nunique()),
        "strategies": rows,
        "path_type_strategies": type_summary.to_dict(orient="records"),
        "headline": {
            "same_shortest_and_lowest_energy_path_count": same_path_count,
            "same_shortest_and_lowest_energy_path_pct": same_path_count / int(data["trip_id"].nunique()),
            "distance_delta_km": float(green_row["total_distance_km"] - shortest_row["total_distance_km"]),
            "energy_delta_kwh": float(green_row["total_energy_kwh"] - shortest_row["total_energy_kwh"]),
            "energy_reduction_pct": float(
                (shortest_row["total_energy_kwh"] - green_row["total_energy_kwh"])
                / shortest_row["total_energy_kwh"]
            ),
            "carbon_delta_kgco2": float(green_row["total_carbon_kgco2"] - shortest_row["total_carbon_kgco2"]),
            "carbon_reduction_pct": float(
                (shortest_row["total_carbon_kgco2"] - green_row["total_carbon_kgco2"])
                / shortest_row["total_carbon_kgco2"]
            ),
        },
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_dir / "green_path_summary.csv", index=False)
    type_summary.to_csv(metrics_dir / "green_path_type_summary.csv", index=False)
    write_json(metrics_dir / "green_path_summary.json", metrics)
    return summary, metrics


def plot_green_path_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot distance, energy, and carbon totals for green path strategies."""

    metrics = [
        ("total_distance_km", "Distance", "km", "#4C78A8"),
        ("total_energy_kwh", "Energy", "kWh", "#54A24B"),
        ("total_carbon_kgco2", "Carbon", "kg CO2", "#6A4C93"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), dpi=150)
    for ax, (metric, title, ylabel, color) in zip(axes, metrics, strict=True):
        values = summary.set_index("strategy").loc[list(PATH_STRATEGY_COLORS), metric]
        ax.bar(values.index, values.to_numpy(), color=[PATH_STRATEGY_COLORS[name] for name in values.index], width=0.52)
        style_axis(ax, title, ylabel)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_compact_number(value)))
        ax.set_ylim(0, float(values.max()) * 1.16)
        for index, value in enumerate(values):
            ax.text(index, value + float(values.max()) * 0.025, format_compact_number(float(value)), ha="center", fontsize=9)
        ax.tick_params(axis="x", labelrotation=18)
        ax.patches[1].set_color(color)

    fig.suptitle("Green Path Planning Strategy Comparison", fontsize=15, weight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_path_type_comparison(metrics: dict, output_path: Path) -> None:
    """Plot aggregate distance, energy, and carbon by candidate path type."""

    summary = pd.DataFrame(metrics["path_type_strategies"])
    label_map = {
        "shorter_with_more_slope": "Shorter\nmore slope",
        "balanced_low_congestion": "Balanced\nlow congestion",
        "longer_but_flatter": "Longer\nflatter",
    }
    order = ["shorter_with_more_slope", "balanced_low_congestion", "longer_but_flatter"]
    summary = summary.set_index("path_type").reindex(order).reset_index()
    x_labels = [label_map.get(value, value) for value in summary["path_type"]]
    metrics_to_plot = [
        ("total_distance_km", "Total Distance", "km", "#4C78A8"),
        ("total_energy_kwh", "Total Energy", "kWh", "#54A24B"),
        ("total_carbon_kgco2", "Total Carbon", "kg CO2", "#6A4C93"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), dpi=150)
    for ax, (metric, title, ylabel, color) in zip(axes, metrics_to_plot, strict=True):
        values = summary[metric].to_numpy()
        ax.bar(range(len(values)), values, color=color, width=0.56)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(x_labels, fontsize=9)
        style_axis(ax, title, ylabel)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_compact_number(value)))
        ax.set_ylim(0, float(max(values)) * 1.16)
        for index, value in enumerate(values):
            ax.text(index, value + float(max(values)) * 0.025, format_compact_number(float(value)), ha="center", fontsize=9)

    fig.suptitle("Candidate Path Type Impact", fontsize=15, weight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_schedule(path: Path) -> pd.DataFrame:
    """Load a scheduler output CSV with numeric schedule fields."""

    if not path.exists():
        raise FileNotFoundError(f"Schedule not found: {path}")
    data = pd.read_csv(path)
    numeric_columns = [
        "trip_start_min",
        "trip_end_min",
        "charge_start_min",
        "charge_end_min",
        "soc_before_trip",
        "soc_after_trip",
        "soc_after_charge",
        "charging_cost",
        "wait_time_min",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["charge_requested"] = data["charge_requested"].astype(str).str.lower().isin(["true", "1", "yes"])
    return data


def plot_charging_gantt(schedule: pd.DataFrame, title: str, output_path: Path, max_events: int = 28) -> None:
    """Plot a compact charging Gantt chart for a scheduler output."""

    charging = schedule[schedule["charge_requested"]].dropna(subset=["charge_start_min", "charge_end_min"]).copy()
    if charging.empty:
        fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=150)
        ax.text(0.5, 0.5, "No charging events in this scenario", ha="center", va="center", fontsize=13)
        ax.axis("off")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    charging = charging.sort_values(["charge_start_min", "charge_end_min"]).head(max_events)
    labels = [f"{index + 1:02d}" for index in range(len(charging))]

    fig_height = max(5.2, min(9.0, 0.28 * len(charging) + 2.2))
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=150)
    colors = charging["charger_type"].map({"fast": "#F58518", "slow": "#4C78A8"}).fillna("#9BA7B4")
    y_positions = np.arange(len(charging))
    ax.barh(
        y_positions,
        charging["charge_end_min"] - charging["charge_start_min"],
        left=charging["charge_start_min"],
        height=0.66,
        color=colors,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    style_axis(ax, title, "Event index")
    ax.set_xlabel("Service time")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: compact_duration_label(value)))
    for y, row in zip(y_positions, charging.itertuples(index=False), strict=True):
        label = f"{row.bus_id} / {row.charger_id}"
        ax.text(
            float(row.charge_start_min) + 2.0,
            y,
            label,
            va="center",
            ha="left",
            fontsize=7,
            color="white" if row.charger_type == "fast" else "#23313F",
            clip_on=True,
        )
    fast_patch = plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#F58518", markersize=10, label="Fast")
    slow_patch = plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#4C78A8", markersize=10, label="Slow")
    ax.legend(handles=[fast_patch, slow_patch], frameon=False, ncols=2, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_charging_load_curve(
    jsq_schedule: pd.DataFrame,
    ga_schedule: pd.DataFrame,
    output_path: Path,
    interval_min: int = 15,
) -> None:
    """Plot aggregate active charging power over the service day."""

    time_grid = np.arange(0, 24 * 60 + interval_min, interval_min)

    def active_power(schedule: pd.DataFrame) -> np.ndarray:
        charging = schedule[schedule["charge_requested"]].dropna(subset=["charge_start_min", "charge_end_min"])
        load = np.zeros(len(time_grid), dtype=float)
        for row in charging.itertuples(index=False):
            start = float(row.charge_start_min)
            end = float(row.charge_end_min)
            power = float(row.charging_power_kw)
            mask = (time_grid >= start) & (time_grid < end)
            load[mask] += power
        return load

    jsq_load = active_power(jsq_schedule)
    ga_load = active_power(ga_schedule)

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
    ax.plot(time_grid, jsq_load / 1000, color=ALGORITHM_COLORS["JSQ"], linewidth=2, label="JSQ")
    ax.plot(time_grid, ga_load / 1000, color=ALGORITHM_COLORS["GA"], linewidth=2, label="GA")
    ax.fill_between(time_grid, 0, jsq_load / 1000, color=ALGORITHM_COLORS["JSQ"], alpha=0.12)
    ax.fill_between(time_grid, 0, ga_load / 1000, color=ALGORITHM_COLORS["GA"], alpha=0.12)
    style_axis(ax, "Full-Scale Active Charging Load", "Active charging power (MW)")
    ax.set_xlabel("Service time")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: compact_duration_label(value)))
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.set_xlim(0, 24 * 60)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_soc_curves(schedule: pd.DataFrame, title: str, output_path: Path, max_buses: int = 8) -> None:
    """Plot SOC trajectories for the busiest buses in a schedule."""

    completed = schedule[schedule["status"].astype(str).str.lower() == "completed"].copy()
    if completed.empty:
        raise ValueError("Schedule contains no completed trips for SOC plotting.")

    bus_ids = completed["bus_id"].value_counts().head(max_buses).index.tolist()
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    cmap = plt.get_cmap("tab10")

    for index, bus_id in enumerate(bus_ids):
        bus_rows = completed[completed["bus_id"] == bus_id].sort_values(["trip_start_min", "trip_end_min"])
        times: list[float] = []
        soc_values: list[float] = []
        for row in bus_rows.itertuples(index=False):
            if pd.notna(row.trip_start_min) and pd.notna(row.soc_before_trip):
                times.append(float(row.trip_start_min))
                soc_values.append(float(row.soc_before_trip))
            if pd.notna(row.trip_end_min) and pd.notna(row.soc_after_trip):
                times.append(float(row.trip_end_min))
                soc_values.append(float(row.soc_after_trip))
            if getattr(row, "charge_requested") and pd.notna(row.charge_end_min) and pd.notna(row.soc_after_charge):
                times.append(float(row.charge_end_min))
                soc_values.append(float(row.soc_after_charge))
        if times:
            ax.plot(times, soc_values, linewidth=1.6, alpha=0.9, color=cmap(index % 10), label=bus_id)

    ax.axhline(0.2, color="#B23A48", linestyle="--", linewidth=1.0, label="Safety SOC 20%")
    style_axis(ax, title, "SOC")
    ax.set_xlabel("Service time")
    ax.set_ylim(0.15, 0.95)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: compact_duration_label(value)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax.legend(frameon=False, ncols=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ga_convergence(metrics_path: Path, output_path: Path) -> None:
    """Plot GA convergence using saved generation history."""

    if not metrics_path.exists():
        raise FileNotFoundError(f"GA metrics not found: {metrics_path}. Run GA first.")
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    history = metrics.get("ga_history", [])
    if not history:
        raise ValueError(f"GA metrics contain no ga_history: {metrics_path}")

    data = pd.DataFrame(history)
    fig, ax_cost = plt.subplots(figsize=(9.5, 5.6), dpi=150)
    ax_cost.plot(
        data["generation"],
        data["best_total_charging_cost"],
        color="#F58518",
        marker="o",
        linewidth=2,
        label="Best charging cost",
    )
    style_axis(ax_cost, "GA Convergence on Full Scenario", "Best charging cost")
    ax_cost.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${format_compact_number(value)}"))
    ax_cost.set_xlabel("Generation")

    ax_fitness = ax_cost.twinx()
    ax_fitness.plot(
        data["generation"],
        data["best_fitness_score"],
        color="#4C78A8",
        marker="s",
        linewidth=1.8,
        linestyle="--",
        label="Best fitness",
    )
    ax_fitness.set_ylabel("Best fitness score")
    ax_fitness.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_compact_number(value)))
    ax_fitness.spines["top"].set_visible(False)
    ax_fitness.spines["right"].set_color("#CBD1D8")

    handles_1, labels_1 = ax_cost.get_legend_handles_labels()
    handles_2, labels_2 = ax_fitness.get_legend_handles_labels()
    ax_cost.legend(handles_1 + handles_2, labels_1 + labels_2, frameon=False, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_figures(summary_csv: Path, output_dir: Path) -> list[Path]:
    """Generate all analysis and scenario comparison figures."""

    data = load_summary(summary_csv)
    metrics_dir = DEFAULT_METRICS_DIR
    energy_data, energy_metrics = build_energy_prediction_analysis(metrics_dir)
    path_summary, path_metrics = build_green_path_analysis(metrics_dir)
    jsq_full_schedule = load_schedule(JSQ_FULL_SCHEDULE_PATH)
    ga_full_schedule = load_schedule(GA_FULL_SCHEDULE_PATH)

    output_paths = [
        output_dir / "scenario_cost_savings.png",
        output_dir / "scenario_full_wait_reduction.png",
        output_dir / "scenario_normalized_efficiency.png",
        output_dir / "scenario_report_dashboard.png",
        output_dir / "energy_model_metric_comparison.png",
        output_dir / "energy_actual_vs_predicted.png",
        output_dir / "energy_prediction_error_distribution.png",
        output_dir / "green_path_strategy_comparison.png",
        output_dir / "green_path_type_comparison.png",
        output_dir / "jsq_charging_gantt_full.png",
        output_dir / "ga_charging_gantt_full.png",
        output_dir / "charging_load_curve_full.png",
        output_dir / "jsq_soc_curves_full.png",
        output_dir / "ga_soc_curves_full.png",
        output_dir / "ga_convergence_full.png",
    ]
    plot_cost_savings(data, output_paths[0])
    plot_full_wait_reduction(data, output_paths[1])
    plot_normalized_efficiency(data, output_paths[2])
    plot_report_dashboard(data, output_paths[3])
    plot_energy_metric_comparison(energy_metrics, output_paths[4])
    plot_actual_vs_predicted(energy_data, output_paths[5])
    plot_prediction_error_distribution(energy_data, output_paths[6])
    plot_green_path_comparison(path_summary, output_paths[7])
    plot_path_type_comparison(path_metrics, output_paths[8])
    plot_charging_gantt(jsq_full_schedule, "JSQ Full-Scale Charging Gantt", output_paths[9])
    plot_charging_gantt(ga_full_schedule, "GA Full-Scale Charging Gantt", output_paths[10])
    plot_charging_load_curve(jsq_full_schedule, ga_full_schedule, output_paths[11])
    plot_soc_curves(jsq_full_schedule, "JSQ Full-Scale SOC Curves", output_paths[12])
    plot_soc_curves(ga_full_schedule, "GA Full-Scale SOC Curves", output_paths[13])
    plot_ga_convergence(GA_FULL_METRICS_PATH, output_paths[14])
    return output_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate scenario comparison figures.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    output_paths = generate_figures(args.summary_csv, args.output_dir)
    for path in output_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
