"""Generate an entirely synthetic Aktau route for the Kazakhstan G-TON demo.

The road shape is an illustrative OpenStreetMap/OSRM planning reference, not a
recorded vehicle journey. Neither the positions nor their timestamps come from
the test video. The full route extends beyond the short video's duration.
"""

import argparse
import bisect
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "data" / "gps" / "road_test_gps.csv"
DEMO_NOTICE = "DEMO GPS DATA - NOT REAL VEHICLE COORDINATES"
EARTH_RADIUS_METERS = 6_371_000.0
SPEED_METERS_PER_SECOND = 8.0
# Illustrative Aktau road geometry (latitude, longitude), based on a public
# OpenStreetMap / OSRM route. No live routing service is needed to run this demo.
# Planning reference: https://www.openstreetmap.org/#map=16/43.6482/51.1687
AKTAU_ROUTE = (
    (43.652926, 51.162915),
    (43.649855, 51.165927),
    (43.649711, 51.166078),
    (43.649893, 51.166428),
    (43.649940, 51.166975),
    (43.649755, 51.167423),
    (43.649644, 51.167533),
    (43.648944, 51.168223),
    (43.647967, 51.169187),
    (43.647296, 51.169848),
    (43.646411, 51.170738),
    (43.646127, 51.171024),
    (43.645267, 51.171888),
    (43.644895, 51.172264),
    (43.644605, 51.172556),
    (43.644036, 51.173158),
    (43.643419, 51.173774),
    (43.643144, 51.174062),
    (43.643434, 51.174572),
)


def distance_meters(first, second):
    lat_a, lat_b = math.radians(first[0]), math.radians(second[0])
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(second[1] - first[1])
    value = (math.sin(delta_lat / 2) ** 2
             + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2)
    return 2 * EARTH_RADIUS_METERS * math.asin(math.sqrt(min(1.0, max(0.0, value))))


def generate_points():
    # DEMO coordinates ONLY. These are invented and must never be presented
    # as real GPS measured from the video or a real vehicle journey.
    cumulative = [0.0]
    for first, second in zip(AKTAU_ROUTE, AKTAU_ROUTE[1:]):
        cumulative.append(cumulative[-1] + distance_meters(first, second))
    duration_seconds = math.ceil(cumulative[-1] / SPEED_METERS_PER_SECOND)
    points = []
    for second in range(duration_seconds + 1):
        distance = min(second * SPEED_METERS_PER_SECOND, cumulative[-1])
        segment = min(bisect.bisect_right(cumulative, distance) - 1, len(AKTAU_ROUTE) - 2)
        fraction = (distance - cumulative[segment]) / (cumulative[segment + 1] - cumulative[segment])
        first, last = AKTAU_ROUTE[segment], AKTAU_ROUTE[segment + 1]
        latitude = first[0] + fraction * (last[0] - first[0])
        longitude = first[1] + fraction * (last[1] - first[1])
        points.append({
            "timestamp_seconds": second,
            "latitude": latitude,
            "longitude": longitude,
        })
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite-demo", action="store_true",
                        help="Explicitly replace the existing demo GPS CSV.")
    args = parser.parse_args()
    print(DEMO_NOTICE, flush=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Preserve an existing GPS file rather than overwriting possible real data.
    if OUTPUT_PATH.exists() and not args.overwrite_demo:
        raise FileExistsError(f"GPS file already exists; use --overwrite-demo to replace demo data: {OUTPUT_PATH}")
    if OUTPUT_PATH.is_symlink():
        raise ValueError("Refusing to replace a linked GPS file")
    points = generate_points()
    with OUTPUT_PATH.open("w" if args.overwrite_demo else "x", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["timestamp_seconds", "latitude", "longitude"])
        writer.writeheader()
        writer.writerows(points)
    length = sum(distance_meters(first, second) for first, second in zip(AKTAU_ROUTE, AKTAU_ROUTE[1:]))
    print(f"Aktau, Kazakhstan - synthetic road route: {length / 1000:.2f} km")
    print(f"Generated {len(points)} demo points from 0 to {points[-1]['timestamp_seconds']} seconds")
    print("Coordinates and timestamps are synthetic; they were not recorded from the test video.")
    print(f"GPS CSV: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
