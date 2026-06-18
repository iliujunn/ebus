#!/usr/bin/env python3
"""Download Hong Kong public transport GTFS data and normalize trip records."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


DEFAULT_URL = "https://static.data.gov.hk/td/pt-headway-sc/gtfs.zip"


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_time(value: str) -> int:
    hour, minute, second = (int(part) for part in value.split(":"))
    return hour * 3600 + minute * 60 + second


def format_time(seconds: int) -> str:
    hour = seconds // 3600
    minute = seconds % 3600 // 60
    second = seconds % 60
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def clean_stop_name(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.split("|")[0]
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def stop_row_distance_km(rows: list[dict[str, str]], stops: dict[str, dict[str, str]]) -> float:
    total = 0.0
    last_stop: dict[str, str] | None = None
    for row in rows:
        stop = stops.get(row["stop_id"], {})
        if not all(stop.get(k) for k in ("stop_lat", "stop_lon")):
            last_stop = None
            continue
        if last_stop:
            total += haversine_km(
                float(last_stop["stop_lat"]),
                float(last_stop["stop_lon"]),
                float(stop["stop_lat"]),
                float(stop["stop_lon"]),
            )
        last_stop = stop
    return total


def normalize_trips(gtfs_dir: Path, output_path: Path, max_routes: int, max_trips: int) -> int:
    routes = read_csv(gtfs_dir / "routes.txt")
    trips = read_csv(gtfs_dir / "trips.txt")
    stop_times = read_csv(gtfs_dir / "stop_times.txt")
    stops = {row["stop_id"]: row for row in read_csv(gtfs_dir / "stops.txt")}

    # GTFS route_type=3 is bus. If the feed omits route_type, keep the route.
    bus_routes = [
        row["route_id"]
        for row in routes
        if row.get("route_type", "3") == "3"
    ][:max_routes]
    bus_route_set = set(bus_routes)
    trip_route = {
        row["trip_id"]: row["route_id"]
        for row in trips
        if row.get("route_id") in bus_route_set
    }

    by_trip: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in stop_times:
        trip_id = row.get("trip_id", "")
        if trip_id in trip_route:
            by_trip[trip_id].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trip_id",
        "route_id",
        "start_time",
        "end_time",
        "origin",
        "destination",
        "distance_km",
        "candidate_path",
        "source",
    ]
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for trip_id, rows in by_trip.items():
            rows.sort(key=lambda row: int(row.get("stop_sequence") or 0))
            first = rows[0]
            last = rows[-1]
            start_time = parse_time(first.get("departure_time") or first["arrival_time"])
            end_time = parse_time(last.get("arrival_time") or last["departure_time"])
            origin = clean_stop_name(stops.get(first["stop_id"], {}).get("stop_name", first["stop_id"]))
            destination = clean_stop_name(stops.get(last["stop_id"], {}).get("stop_name", last["stop_id"]))

            distance_km = ""
            if first.get("shape_dist_traveled") and last.get("shape_dist_traveled"):
                distance_km = f"{(float(last['shape_dist_traveled']) - float(first['shape_dist_traveled'])) / 1000:.2f}"
            if not distance_km or float(distance_km) <= 0:
                path_km = stop_row_distance_km(rows, stops)
                if path_km > 0:
                    distance_km = f"{path_km:.2f}"
            if not distance_km or float(distance_km) <= 0:
                first_stop = stops.get(first["stop_id"], {})
                last_stop = stops.get(last["stop_id"], {})
                if all(first_stop.get(k) for k in ("stop_lat", "stop_lon")) and all(
                    last_stop.get(k) for k in ("stop_lat", "stop_lon")
                ):
                    km = haversine_km(
                        float(first_stop["stop_lat"]),
                        float(first_stop["stop_lon"]),
                        float(last_stop["stop_lat"]),
                        float(last_stop["stop_lon"]),
                    )
                    distance_km = f"{km:.2f}"
            if not distance_km or float(distance_km) <= 0:
                duration_hours = max(1 / 60, (end_time - start_time) / 3600)
                distance_km = f"{max(0.5, duration_hours * 12):.2f}"

            writer.writerow(
                {
                    "trip_id": trip_id,
                    "route_id": trip_route[trip_id],
                    "start_time": format_time(start_time),
                    "end_time": format_time(end_time),
                    "origin": origin,
                    "destination": destination,
                    "distance_km": distance_km,
                    "candidate_path": trip_route[trip_id],
                    "source": "DATA.GOV.HK pt-headway GTFS",
                }
            )
            count += 1
            if count >= max_trips:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-dir", default="data/raw/hk_gtfs")
    parser.add_argument("--output", default="data/processed/trips_hk_gtfs.csv")
    parser.add_argument("--max-routes", type=int, default=3)
    parser.add_argument("--max-trips", type=int, default=1000)
    parser.add_argument("--refresh", action="store_true", help="Download GTFS even when a local zip already exists.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    zip_path = raw_dir / "gtfs.zip"
    extract_dir = raw_dir / "extracted"

    if args.refresh or not zip_path.exists():
        print(f"Downloading {args.url}")
        download(args.url, zip_path)
    else:
        print(f"Using existing {zip_path}")

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    count = normalize_trips(extract_dir, Path(args.output), args.max_routes, args.max_trips)
    print(f"Wrote {count} normalized trips to {args.output}")


if __name__ == "__main__":
    main()
