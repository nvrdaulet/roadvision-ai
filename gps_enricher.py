"""Enrich defect candidates with DEMO GPS, then group nearby candidates.

This demo uses synthetic vehicle positions as candidate positions, not measured
defect locations. Pairwise Haversine links <= 12 m form connected components:
merging is transitive, so endpoints in a group can be more than 12 m apart.
Merged coordinates come from the earliest timestamp; image-center fields come
from the highest-confidence member. frames_seen sums observations, including
overlaps, and detection_count counts original physical defect candidates.
"""

import bisect
import csv
import json
import math
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFECTS_PATH = ROOT / "runs" / "video_v3" / "defects.csv"
GPS_PATH = ROOT / "data" / "gps" / "road_test_gps.csv"
OUTPUT_DIR = ROOT / "runs" / "gps"
DEMO_NOTICE = "DEMO GPS DATA - NOT REAL VEHICLE COORDINATES"
EARTH_RADIUS_METERS = 6_371_000.0
MERGE_DISTANCE_METERS = 12.0
REQUIRED_FIELDS = (
    "defect_id", "track_ids", "first_frame", "last_frame", "frames_seen",
    "max_confidence", "timestamp_seconds", "center_x", "center_y",
)


def read_csv(path, required):
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fields = reader.fieldnames or []
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        return fields, list(reader)


def finite_number(value, field):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite: {value}")
    return result


def load_gps(path):
    _, rows = read_csv(path, ("timestamp_seconds", "latitude", "longitude"))
    points = []
    for row in rows:
        timestamp = finite_number(row["timestamp_seconds"], "timestamp_seconds")
        latitude = finite_number(row["latitude"], "latitude")
        longitude = finite_number(row["longitude"], "longitude")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("GPS latitude/longitude is outside valid bounds")
        points.append((timestamp, latitude, longitude))
    points.sort()
    if not points:
        raise ValueError("GPS CSV contains no points")
    if len({point[0] for point in points}) != len(points):
        raise ValueError("GPS timestamps must be unique")
    return points


def interpolate_gps(timestamp, points, timestamps):
    index = bisect.bisect_left(timestamps, timestamp)
    if index < len(points) and timestamps[index] == timestamp:
        return points[index][1], points[index][2]
    if index == 0 or index == len(points):
        raise ValueError(
            f"Defect timestamp {timestamp} is outside GPS coverage "
            f"[{timestamps[0]}, {timestamps[-1]}]; extrapolation is disabled"
        )
    before, after = points[index - 1], points[index]
    fraction = (timestamp - before[0]) / (after[0] - before[0])
    return (
        before[1] + fraction * (after[1] - before[1]),
        before[2] + fraction * (after[2] - before[2]),
    )


def haversine_distance(latitude_a, longitude_a, latitude_b, longitude_b):
    lat_a, lat_b = math.radians(latitude_a), math.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(longitude_b - longitude_a)
    value = (math.sin(delta_lat / 2) ** 2
             + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2)
    return 2 * EARTH_RADIUS_METERS * math.asin(math.sqrt(max(0.0, min(1.0, value))))


def enrich_defects(rows, points):
    timestamps = [point[0] for point in points]
    enriched = []
    defect_ids = set()
    for original in rows:
        row = dict(original)
        if row["defect_id"] in defect_ids:
            raise ValueError(f"Duplicate defect_id: {row['defect_id']}")
        defect_ids.add(row["defect_id"])
        ids = json.loads(row["track_ids"])
        if not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids):
            raise ValueError("track_ids must be a nonempty JSON list of integer IDs")
        for field in ("first_frame", "last_frame", "frames_seen"):
            row[field] = int(row[field])
        if (row["first_frame"] < 1 or row["last_frame"] < row["first_frame"]
                or row["frames_seen"] < 1):
            raise ValueError(f"Invalid frame counts for defect {row['defect_id']}")
        for field in ("max_confidence", "timestamp_seconds", "center_x", "center_y"):
            row[field] = finite_number(row[field], field)
        if not 0 <= row["max_confidence"] <= 1:
            raise ValueError("max_confidence must be between 0 and 1")
        row["latitude"], row["longitude"] = interpolate_gps(row["timestamp_seconds"], points, timestamps)
        enriched.append(row)
    return enriched


def merge_gps_duplicates(rows):
    """Group connected <=12 m pairs, using only original interpolated points."""
    ordered = sorted(rows, key=lambda row: (row["timestamp_seconds"], str(row["defect_id"])))
    parents = list(range(len(ordered)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, first in enumerate(ordered):
        for other_index in range(index + 1, len(ordered)):
            other = ordered[other_index]
            distance = haversine_distance(
                first["latitude"], first["longitude"], other["latitude"], other["longitude"]
            )
            if distance <= MERGE_DISTANCE_METERS:
                parents[find(other_index)] = find(index)
    groups = {}
    for index, row in enumerate(ordered):
        groups.setdefault(find(index), []).append(row)

    merged = []
    for members in groups.values():
        earliest = members[0]
        best = min(members, key=lambda row: (-row["max_confidence"],
                                             row["timestamp_seconds"], str(row["defect_id"])))
        row = dict(best)
        row.update({
            "defect_id": len(merged) + 1,
            "track_ids": json.dumps(sorted({
                track_id for member in members for track_id in json.loads(member["track_ids"])
            })),
            "first_frame": min(member["first_frame"] for member in members),
            "last_frame": max(member["last_frame"] for member in members),
            "frames_seen": sum(member["frames_seen"] for member in members),
            "max_confidence": best["max_confidence"],
            "timestamp_seconds": earliest["timestamp_seconds"],
            "latitude": earliest["latitude"],
            "longitude": earliest["longitude"],
            "source_defect_ids": json.dumps([member["defect_id"] for member in members]),
            "detection_count": len(members),
        })
        merged.append(row)
    return merged


def recreate_outputs():
    """Validate and recreate only runs/gps; reject links and junctions."""
    target = OUTPUT_DIR
    for path in (ROOT / "runs", target):
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"Refusing linked output directory: {path}")
    if target.resolve() != ROOT / "runs" / "gps":
        raise ValueError("Unexpected resolved output directory")
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"Output location is not a directory: {target}")
        for directory, subdirs, files in os.walk(target, followlinks=False):
            for name in subdirs + files:
                path = Path(directory) / name
                if (path.is_symlink() or path.is_junction()
                        or not path.resolve().is_relative_to(target)):
                    raise ValueError(f"Refusing linked or external output entry: {path}")
        shutil.rmtree(target)
    target.mkdir(parents=True)


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print(DEMO_NOTICE, flush=True)
    fields, rows = read_csv(DEFECTS_PATH, REQUIRED_FIELDS)
    points = load_gps(GPS_PATH)
    enriched = enrich_defects(rows, points)
    merged = merge_gps_duplicates(enriched)
    # Validate inputs and calculate results before replacing generated outputs.
    recreate_outputs()
    enriched_fields = fields + [field for field in ("latitude", "longitude") if field not in fields]
    write_csv(OUTPUT_DIR / "defects_with_gps.csv", enriched_fields, enriched)
    merged_fields = enriched_fields + ["source_defect_ids", "detection_count"]
    write_csv(OUTPUT_DIR / "unique_defects.csv", merged_fields, merged)
    print(f"Physical defect candidates before GPS merge: {len(enriched)}")
    print(f"Unique defects after GPS merge: {len(merged)}")
    print(f"Duplicates merged: {len(enriched) - len(merged)}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(DEMO_NOTICE)


if __name__ == "__main__":
    main()
