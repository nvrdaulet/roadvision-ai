"""Track potholes, then merge confirmed fragments into physical defect candidates.

Frames are one-based. Timestamps and pixel centers refer to the best-confidence
snapshot. frames_seen sums original track observations, including overlaps.
Merge edges are tested on original confirmed tracks; connected components
merge transitively. These image-space candidates are not GPS-deduplicated.
The final video shows merged candidate IDs retrospectively and has no audio.
"""

import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "runs" / "video_v3"
FIELDS = (
    "defect_id", "track_ids", "first_frame", "last_frame", "frames_seen",
    "max_confidence", "timestamp_seconds", "center_x", "center_y",
)


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(first, second):
    intersection = (
        max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
        * max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    )
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def filter_indices(boxes, confidences, frame_height):
    """Apply the same road-center and IoU >= 0.45 filters as v2."""
    candidates = [
        index for index, box in enumerate(boxes)
        if (box[1] + box[3]) / 2 > 0.30 * frame_height and box_area(box) > 0
    ]
    candidates.sort(key=lambda index: (-confidences[index], index))
    kept = []
    for index in candidates:
        if all(box_iou(boxes[index], boxes[other]) < 0.45 for other in kept):
            kept.append(index)
    return kept


def recreate_outputs():
    """Recreate only runs/video_v3 after validating all deletion paths."""
    target = OUTPUT_DIR
    for path in (ROOT / "runs", target):
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"Refusing linked output directory: {path}")
    if target.resolve() != ROOT / "runs" / "video_v3":
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
    (target / "detections").mkdir(parents=True)


@dataclass
class TrackRecord:
    track_id: int
    first_frame: int
    last_frame: int = 0
    frames_seen: int = 0
    max_confidence: float = 0.0
    best_bbox: tuple = ()
    best_bbox_center_x: float = 0.0
    best_bbox_center_y: float = 0.0
    best_frame: int = 0
    best_snapshot: bytes = b""

    @property
    def confirmed(self):
        return self.frames_seen >= 5 and self.max_confidence >= 0.35


@dataclass
class DefectCandidate:
    defect_id: int
    track_ids: list[int]
    first_frame: int
    last_frame: int
    frames_seen: int
    best_track: TrackRecord


def tracks_match(first, second, fps, frame_diagonal):
    """Check the time, center-distance and symmetric scale requirements."""
    gap_frames = max(0, max(first.first_frame, second.first_frame)
                     - min(first.last_frame, second.last_frame))
    if gap_frames / fps > 1.0:
        return False
    distance = math.hypot(
        first.best_bbox_center_x - second.best_bbox_center_x,
        first.best_bbox_center_y - second.best_bbox_center_y,
    )
    if distance / frame_diagonal > 0.15:
        return False
    first_area, second_area = box_area(first.best_bbox), box_area(second.best_bbox)
    if min(first_area, second_area) <= 0:
        return False
    # Use larger/smaller for order-independent matching. This also keeps the
    # inverse ratio >= 0.35 (1 / 2.85 is approximately 0.35088).
    area_ratio = max(first_area, second_area) / min(first_area, second_area)
    return 0.35 <= area_ratio <= 2.85


def merge_tracks(tracks, fps, width, height):
    """Build connected components from compatible original confirmed tracks."""
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Merging requires valid FPS and frame dimensions")
    records = sorted(
        (track for track in tracks if track.confirmed),
        key=lambda track: (track.first_frame, track.track_id),
    )
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    diagonal = math.hypot(width, height)
    for index, first in enumerate(records):
        for other_index in range(index + 1, len(records)):
            if tracks_match(first, records[other_index], fps, diagonal):
                parents[find(other_index)] = find(index)

    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)
    candidates = []
    for members in groups.values():
        # Confidence ties choose the earlier snapshot, then lower track ID.
        best = min(members, key=lambda track: (-track.max_confidence,
                                               track.best_frame, track.track_id))
        candidates.append(DefectCandidate(
            defect_id=len(candidates) + 1,
            track_ids=sorted(track.track_id for track in members),
            first_frame=min(track.first_frame for track in members),
            last_frame=max(track.last_frame for track in members),
            frames_seen=sum(track.frames_seen for track in members),
            best_track=best,
        ))
    return candidates


def draw_box(image, bbox, confidence, track_id, defect_id=None):
    import cv2

    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, math.floor(bbox[0])))
    y1 = max(0, min(height - 1, math.floor(bbox[1])))
    x2 = max(x1, min(width - 1, math.ceil(bbox[2])))
    y2 = max(y1, min(height - 1, math.ceil(bbox[3])))
    color = (0, 200, 0) if defect_id is not None else (0, 165, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    lines = [f"Pothole {confidence:.2f} Track: {track_id if track_id is not None else 'pending'}"]
    if defect_id is not None:
        lines.append(f"Physical defect candidate {defect_id}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.65, width / 1000)
    sizes = [cv2.getTextSize(line, font, scale, 2) for line in lines]
    line_height = max(size[0][1] + size[1] + 6 for size in sizes)
    text_width = max(size[0][0] for size in sizes)
    text_x = max(0, min(x1, width - text_width - 4))
    top = max(0, y1 - line_height * len(lines))
    cv2.rectangle(image, (text_x, top),
                  (text_x + text_width + 4, top + line_height * len(lines)), color, -1)
    for index, line in enumerate(lines):
        y = top + index * line_height + sizes[index][0][1] + 3
        cv2.putText(image, line, (text_x + 2, y), font, scale, (0, 0, 0), 2, cv2.LINE_AA)


def render_video(source, video_path, observations, track_to_defect, fps, width, height):
    """Second pass draws the final merged IDs without repeating inference."""
    import cv2

    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot reopen video: {source}")
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot write output video: {video_path}")
        for frame_number, detections in enumerate(observations, start=1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended early during rendering at frame {frame_number}")
            for bbox, confidence, track_id in detections:
                draw_box(frame, bbox, confidence, track_id, track_to_defect.get(track_id))
            writer.write(frame)
            if frame_number % 500 == 0:
                print(f"Rendered {frame_number}/{len(observations)} frames", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def save_candidates(candidates, fps):
    import cv2
    import numpy as np

    csv_path = OUTPUT_DIR / "defects.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        for candidate in candidates:
            best = candidate.best_track
            snapshot = cv2.imdecode(np.frombuffer(best.best_snapshot, dtype=np.uint8), cv2.IMREAD_COLOR)
            if snapshot is None:
                raise RuntimeError(f"Cannot decode snapshot for track {best.track_id}")
            draw_box(snapshot, best.best_bbox, best.max_confidence,
                     best.track_id, candidate.defect_id)
            path = OUTPUT_DIR / "detections" / f"defect_{candidate.defect_id:04d}.jpg"
            if not cv2.imwrite(str(path), snapshot):
                raise RuntimeError(f"Cannot save snapshot: {path}")
            writer.writerow({
                "defect_id": candidate.defect_id,
                "track_ids": json.dumps(candidate.track_ids),
                "first_frame": candidate.first_frame,
                "last_frame": candidate.last_frame,
                "frames_seen": candidate.frames_seen,
                "max_confidence": best.max_confidence,
                "timestamp_seconds": (best.best_frame - 1) / fps,
                "center_x": best.best_bbox_center_x,
                "center_y": best.best_bbox_center_y,
            })


def main():
    import cv2
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
            # Raw counts are after normal YOLO postprocessing, before our filters.
            raw_count += len(boxes)
            keep = filter_indices(
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), result.orig_shape[0]
            )
            filtered_count += len(keep)
            predictor.results[index] = result[keep]
            frame_detections = [
                (tuple(boxes.xyxy[item].tolist()), float(boxes.conf[item].item()))
                for item in keep
            ]

    # Register first so the filters execute before Ultralytics' tracker callback.
    model.add_callback("on_predict_postprocess_end", filter_before_tracking)
    capture = cv2.VideoCapture(str(source))
    tracks = {}
    observations = []
    total_frames = 0
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
        print(f"Tracking {expected_frames} reported frames at {fps:.3f} FPS", flush=True)
        while success:
            total_frames += 1
            model.track(
                source=frame, persist=True, tracker="bytetrack.yaml",
                conf=0.30, imgsz=960, classes=pothole_ids, save=False, verbose=False,
            )
            tracker = model.predictor.trackers[0]
            assignments = {}
            # Include real tentative observations from frame one of each ID,
            # and exclude lost-track predictions with no observation this frame.
            for tracked in tracker.tracked_stracks:
                if tracked.frame_id != tracker.frame_id:
                    continue
                index = int(tracked.idx)
                if not 0 <= index < len(frame_detections) or index in assignments:
                    raise RuntimeError("Invalid or duplicate ByteTrack detection index")
                assignments[index] = int(tracked.track_id)
            frame_observations = []
            encoded_snapshot = None
            for index, (bbox, confidence) in enumerate(frame_detections):
                track_id = assignments.get(index)
                frame_observations.append((bbox, confidence, track_id))
                if track_id is None:
                    continue
                record = tracks.setdefault(track_id, TrackRecord(track_id, total_frames))
                if record.last_frame != total_frames:
                    record.frames_seen += 1
                record.last_frame = total_frames
                if confidence > record.max_confidence:
                    if encoded_snapshot is None:
                        ok, encoded = cv2.imencode(".jpg", frame)
                        if not ok:
                            raise RuntimeError(f"Cannot encode frame {total_frames}")
                        encoded_snapshot = encoded.tobytes()
                    record.max_confidence = confidence
                    record.best_bbox = bbox
                    record.best_bbox_center_x = (bbox[0] + bbox[2]) / 2
                    record.best_bbox_center_y = (bbox[1] + bbox[3]) / 2
                    record.best_frame = total_frames
                    record.best_snapshot = encoded_snapshot
            observations.append(frame_observations)
            if total_frames % 100 == 0:
                print(f"Processed {total_frames}/{expected_frames} frames | raw: {raw_count} | "
                      f"filtered: {filtered_count} | confirmed tracks: "
                      f"{sum(record.confirmed for record in tracks.values())}", flush=True)
            success, frame = capture.read()
    finally:
        capture.release()

    if expected_frames > 0 and total_frames != expected_frames:
        print(f"Notice: input reports {expected_frames} frames, but OpenCV decoded "
              f"{total_frames}. The output contains the decoded frames.")
    confirmed_count = sum(track.confirmed for track in tracks.values())
    candidates = merge_tracks(tracks.values(), fps, width, height)
    track_to_defect = {
        track_id: candidate.defect_id
        for candidate in candidates for track_id in candidate.track_ids
    }
    video_path = OUTPUT_DIR / "road_test_annotated.mp4"
    print(f"Rendering {len(candidates)} physical defect candidates...", flush=True)
    render_video(source, video_path, observations, track_to_defect, fps, width, height)
    save_candidates(candidates, fps)

    print(f"Total frames: {total_frames}")
    print(f"Raw detections: {raw_count}")
    print(f"Detections after filtering: {filtered_count}")
    print(f"Confirmed tracks before merging: {confirmed_count}")
    print(f"Physical defect candidates after merging: {len(candidates)}")
    print(f"Tracks merged: {confirmed_count - len(candidates)}")
    print(f"Ignored tracks: {len(tracks) - confirmed_count}")
    print(f"Output video: {video_path}")
    print(f"CSV: {OUTPUT_DIR / 'defects.csv'}")


if __name__ == "__main__":
    main()
