"""Filter road detections before ByteTrack and export confirmed pothole events.

Frame numbers start at 1; timestamps identify the best-confidence snapshot.
Raw counts mean YOLO detections after its standard postprocessing, before our
road-region and duplicate filters. An event is a track, not a GPS-deduplicated
physical pothole. The annotated output video does not include audio.
"""

import csv
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "runs" / "video_v2"
FIELDS = (
    "event_id", "track_id", "first_frame", "last_frame", "frames_seen",
    "max_confidence", "timestamp_seconds",
)


def box_iou(first, second):
    """Intersection over union for two (x1, y1, x2, y2) boxes."""
    intersection = (
        max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
        * max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    )
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def filter_indices(boxes, confidences, frame_height):
    """Keep road-region boxes, then greedily suppress overlaps >= 0.45."""
    candidates = [
        index for index, box in enumerate(boxes)
        if (box[1] + box[3]) / 2 > 0.30 * frame_height
        and box[2] > box[0] and box[3] > box[1]
    ]
    candidates.sort(key=lambda index: (-confidences[index], index))
    kept = []
    for index in candidates:
        if all(box_iou(boxes[index], boxes[other]) < 0.45 for other in kept):
            kept.append(index)
    return kept


def recreate_outputs():
    """Recreate only runs/video_v2, refusing symlinks and directory junctions."""
    target = OUTPUT_DIR
    for path in (ROOT / "runs", target):
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"Refusing linked output directory: {path}")
    if target.resolve() != ROOT / "runs" / "video_v2":
        raise ValueError("Unexpected resolved output directory")
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"Output location is not a directory: {target}")
        # Validate every existing path before recursive deletion on Windows.
        for directory, subdirs, files in os.walk(target, followlinks=False):
            for name in subdirs + files:
                path = Path(directory) / name
                if (path.is_symlink() or path.is_junction()
                        or not path.resolve().is_relative_to(target)):
                    raise ValueError(f"Refusing linked or external output entry: {path}")
        shutil.rmtree(target)
    (target / "detections").mkdir(parents=True)


@dataclass
class EventTrack:
    track_id: int
    first_frame: int
    last_frame: int = 0
    frames_seen: int = 0
    max_confidence: float = 0.0
    best_frame: int = 0
    best_snapshot: bytes = b""
    best_box: tuple = ()
    event_id: int | None = None

    @property
    def confirmed(self):
        return self.frames_seen >= 5 and self.max_confidence >= 0.35


def draw_box(image, coordinates, confidence, track_id, confirmed=False, event_id=None):
    import cv2

    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, math.floor(coordinates[0])))
    y1 = max(0, min(height - 1, math.floor(coordinates[1])))
    x2 = max(x1, min(width - 1, math.ceil(coordinates[2])))
    y2 = max(y1, min(height - 1, math.ceil(coordinates[3])))
    label = f"Pothole {confidence:.2f} ID: {track_id if track_id is not None else 'pending'}"
    if confirmed:
        label += f" CONFIRMED event {event_id}"
    color = (0, 200, 0) if confirmed else (0, 165, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.65, width / 1100)
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 2)
    text_x = max(0, min(x1, width - text_width - 4))
    text_y = max(text_height + 4, y1 - 6)
    cv2.rectangle(image, (text_x, text_y - text_height - 4),
                  (text_x + text_width + 4, text_y + baseline), color, -1)
    cv2.putText(image, label, (text_x + 2, text_y), font, scale,
                (0, 0, 0), 2, cv2.LINE_AA)


def main():
    import cv2
    import numpy as np
    from ultralytics import YOLO

    source = ROOT / "data" / "videos" / "road_test.mp4"
    weights = ROOT / "models" / "Samdutse-pothole-yolov8" / "best.pt"
    for path in (source, weights):
        if not path.is_file():
            raise FileNotFoundError(path)

    model = YOLO(str(weights))
    pothole_ids = sorted(
        class_id for class_id, name in model.names.items()
        if name.strip().lower() in {"pothole", "potholes"}
    )
    if not pothole_ids:
        raise ValueError(f"No pothole class in model: {model.names}")
    raw_count = filtered_count = 0
    frame_detections = []

    def filter_before_tracking(predictor):
        nonlocal raw_count, filtered_count, frame_detections
        frame_detections = []
        for index, result in enumerate(predictor.results):
            boxes = result.boxes
            if boxes is None:
                continue
            raw_count += len(boxes)
            keep = filter_indices(
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), result.orig_shape[0]
            )
            filtered_count += len(keep)
            predictor.results[index] = result[keep]
            # Preserve original detector boxes before tracking replaces them
            # with Kalman-filtered coordinates or omits tentative tracks.
            frame_detections = [
                (tuple(boxes.xyxy[item].tolist()), float(boxes.conf[item].item()))
                for item in keep
            ]

    # Register before the first model.track() so both filters run before
    # Ultralytics' ByteTrack callback, including on frames with zero boxes.
    model.add_callback("on_predict_postprocess_end", filter_before_tracking)
    capture = cv2.VideoCapture(str(source))
    writer = None
    tracks = {}
    total_frames = confirmed_events = 0
    video_path = OUTPUT_DIR / "road_test_annotated.mp4"
    csv_path = OUTPUT_DIR / "events.csv"

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video FPS: {fps}")
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        success, frame = capture.read()
        if not success:
            raise RuntimeError("Video contains no decodable frames")
        height, width = frame.shape[:2]
        recreate_outputs()
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot write output video: {video_path}")
        print(f"Processing {expected_frames} frames at {fps:.3f} FPS", flush=True)

        while success:
            total_frames += 1
            model.track(
                source=frame, persist=True, tracker="bytetrack.yaml",
                conf=0.30, imgsz=960, classes=pothole_ids, save=False, verbose=False,
            )
            annotated = frame.copy()
            tracker = model.predictor.trackers[0]
            assigned_indices = set()
            # ByteTrack's public Results omit the first observation of most
            # tentative tracks. Include those actual observations, but never
            # count Kalman predictions on frames without matched detections.
            for tracked in tracker.tracked_stracks:
                if tracked.frame_id != tracker.frame_id:
                    continue
                detection_index = int(tracked.idx)
                if detection_index in assigned_indices:
                    raise RuntimeError("ByteTrack assigned one detection to multiple track IDs")
                assigned_indices.add(detection_index)
                coords, confidence = frame_detections[detection_index]
                track_id = int(tracked.track_id)
                record = tracks.setdefault(track_id, EventTrack(track_id, total_frames))
                if record.last_frame != total_frames:
                    record.frames_seen += 1
                record.last_frame = total_frames
                if confidence > record.max_confidence:
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if not ok:
                        raise RuntimeError(f"Cannot encode snapshot for track {track_id}")
                    record.best_snapshot = encoded.tobytes()
                    record.best_box = coords
                    record.best_frame = total_frames
                    record.max_confidence = confidence
                if record.confirmed and record.event_id is None:
                    confirmed_events += 1
                    record.event_id = confirmed_events
                draw_box(annotated, coords, confidence, track_id,
                         confirmed=record.confirmed, event_id=record.event_id)
            for index, (coords, confidence) in enumerate(frame_detections):
                if index not in assigned_indices:
                    draw_box(annotated, coords, confidence, track_id=None)
            writer.write(annotated)
            if total_frames % 100 == 0:
                print(f"Processed {total_frames}/{expected_frames} frames | "
                      f"raw: {raw_count} | filtered: {filtered_count} | "
                      f"confirmed pothole events: {confirmed_events}", flush=True)
            success, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if expected_frames > 0 and total_frames != expected_frames:
        print(f"Notice: input reports {expected_frames} frames, but OpenCV decoded "
              f"{total_frames}. The output contains the decoded frames.")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        csv_writer.writeheader()
        for record in sorted(
            (track for track in tracks.values() if track.confirmed),
            key=lambda track: track.event_id,
        ):
            snapshot = cv2.imdecode(np.frombuffer(record.best_snapshot, dtype=np.uint8), cv2.IMREAD_COLOR)
            if snapshot is None:
                raise RuntimeError(f"Cannot decode snapshot for track {record.track_id}")
            draw_box(snapshot, record.best_box, record.max_confidence, record.track_id,
                     confirmed=True, event_id=record.event_id)
            snapshot_path = OUTPUT_DIR / "detections" / (
                f"event_{record.event_id:04d}_track_{record.track_id}.jpg"
            )
            if not cv2.imwrite(str(snapshot_path), snapshot):
                raise RuntimeError(f"Cannot save snapshot: {snapshot_path}")
            csv_writer.writerow({
                "event_id": record.event_id, "track_id": record.track_id,
                "first_frame": record.first_frame, "last_frame": record.last_frame,
                "frames_seen": record.frames_seen, "max_confidence": record.max_confidence,
                "timestamp_seconds": (record.best_frame - 1) / fps,
            })

    print(f"Total frames: {total_frames}")
    print(f"Raw model detections: {raw_count}")
    print(f"Detections after duplicate filtering (road region only): {filtered_count}")
    print(f"Confirmed pothole events: {confirmed_events}")
    print(f"Ignored temporary tracks (<5 frames): {sum(t.frames_seen < 5 for t in tracks.values())}")
    print(f"Tracks rejected for confidence (<0.35): "
          f"{sum(t.frames_seen >= 5 and not t.confirmed for t in tracks.values())}")
    print(f"Output video: {video_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
