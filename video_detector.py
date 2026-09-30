"""Track potholes with ByteTrack and export confirmed track evidence.

Frames are numbered from 1. timestamp_seconds refers to the best-confidence
crop, using (frame_number - 1) / FPS. Tentative boxes are shown in orange;
confirmed boxes are green. Unique potholes are counted by tracker ID, so a
lost object that receives a new ID may be counted again. Output has no audio.
"""

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIELDS = (
    "pothole_id", "track_id", "first_frame", "last_frame", "frames_seen",
    "max_confidence", "timestamp_seconds",
)


@dataclass
class TrackRecord:
    track_id: int
    first_frame: int
    last_frame: int = 0
    frames_seen: int = 0
    max_confidence: float = 0.0
    best_frame: int = 0
    best_crop: bytes = b""
    pothole_id: int | None = None

    @property
    def confirmed(self) -> bool:
        return self.frames_seen >= 3 and self.max_confidence >= 0.35


def prepare_outputs(output_dir: Path) -> Path:
    """Remove only this script's named outputs; never delete directories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.resolve() != ROOT / "runs" / "video":
        raise ValueError("Output directory resolves outside the intended location")
    detections_dir = output_dir / "detections"
    detections_dir.mkdir(exist_ok=True)
    if detections_dir.resolve() != output_dir / "detections":
        raise ValueError("Detections directory resolves outside the intended location")
    generated = [output_dir / "road_test_annotated.mp4", output_dir / "potholes.csv"]
    generated.extend(
        path for path in detections_dir.iterdir()
        if re.fullmatch(r"pothole_\d+_track_\d+\.jpg", path.name)
    )
    for path in generated:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Refusing to replace non-regular output: {path}")
    for path in generated:
        if path.is_file():
            path.unlink()
    return detections_dir


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

    raw_detections = 0

    def count_raw_detections(predictor):
        # Registered before ByteTrack's callback: count detector boxes before
        # association can drop tentative/unmatched detections.
        nonlocal raw_detections
        raw_detections += sum(
            len(result.boxes) for result in predictor.results
            if result.boxes is not None
        )

    model.add_callback("on_predict_postprocess_end", count_raw_detections)
    capture = cv2.VideoCapture(str(source))
    writer = None
    tracks = {}
    frames_processed = 0
    confirmed_count = 0
    untracked_detections = 0
    output_dir = ROOT / "runs" / "video"
    video_path = output_dir / "road_test_annotated.mp4"
    csv_path = output_dir / "potholes.csv"

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video FPS: {fps}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        success, frame = capture.read()
        if not success:
            raise RuntimeError("Video contains no decodable frames")
        height, width = frame.shape[:2]
        detections_dir = prepare_outputs(output_dir)
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create MP4 video: {video_path}")
        print(f"Processing {total_frames} frames at {fps:.3f} FPS", flush=True)

        while success:
            frames_processed += 1
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.25,
                imgsz=960,
                classes=pothole_ids,
                save=False,
                verbose=False,
            )
            annotated = frame.copy()
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    confidence = float(box.conf.item())
                    coords = box.xyxy[0].tolist()
                    x1 = max(0, min(width - 1, math.floor(coords[0])))
                    y1 = max(0, min(height - 1, math.floor(coords[1])))
                    x2 = max(x1 + 1, min(width, math.ceil(coords[2])))
                    y2 = max(y1 + 1, min(height, math.ceil(coords[3])))
                    color = (0, 165, 255)
                    label = f"Pothole {confidence:.2f} ID: pending"
                    if box.id is None:
                        untracked_detections += 1
                    else:
                        track_id = int(box.id.item())
                        record = tracks.setdefault(
                            track_id, TrackRecord(track_id, frames_processed)
                        )
                        if record.last_frame != frames_processed:
                            record.frames_seen += 1
                        record.last_frame = frames_processed
                        if confidence > record.max_confidence:
                            encoded_ok, encoded = cv2.imencode(
                                ".jpg", frame[y1:y2, x1:x2]
                            )
                            if not encoded_ok:
                                raise RuntimeError(f"Cannot encode crop for track {track_id}")
                            record.best_crop = encoded.tobytes()
                            record.max_confidence = confidence
                            record.best_frame = frames_processed
                        if record.confirmed and record.pothole_id is None:
                            confirmed_count += 1
                            record.pothole_id = confirmed_count
                        label = f"Pothole {confidence:.2f} ID: {track_id}"
                        if record.confirmed:
                            color = (0, 200, 0)
                            label += " CONFIRMED"
                    cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), color, 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = min(0.65, width / 850)
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 2)
                    text_x = max(0, min(x1, width - text_width - 4))
                    text_y = max(text_height + 4, y1 - 6)
                    cv2.rectangle(
                        annotated, (text_x, text_y - text_height - 4),
                        (text_x + text_width + 4, text_y + baseline), color, -1,
                    )
                    cv2.putText(annotated, label, (text_x + 2, text_y), font,
                                scale, (0, 0, 0), 2, cv2.LINE_AA)
            writer.write(annotated)
            if frames_processed % 100 == 0:
                print(f"Processed {frames_processed}/{total_frames} frames | "
                      f"raw detections: {raw_detections} | "
                      f"confirmed: {confirmed_count}", flush=True)
            success, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        csv_writer.writeheader()
        for record in sorted(
            (record for record in tracks.values() if record.confirmed),
            key=lambda record: record.pothole_id,
        ):
            crop_path = detections_dir / (
                f"pothole_{record.pothole_id:04d}_track_{record.track_id}.jpg"
            )
            crop_path.write_bytes(record.best_crop)
            csv_writer.writerow({
                "pothole_id": record.pothole_id,
                "track_id": record.track_id,
                "first_frame": record.first_frame,
                "last_frame": record.last_frame,
                "frames_seen": record.frames_seen,
                "max_confidence": record.max_confidence,
                "timestamp_seconds": (record.best_frame - 1) / fps,
            })

    temporary = sum(record.frames_seen < 3 for record in tracks.values())
    low_confidence = sum(
        record.frames_seen >= 3 and not record.confirmed for record in tracks.values()
    )
    print(f"Total frames processed: {frames_processed}")
    print(f"Raw detections: {raw_detections}")
    print(f"Confirmed unique potholes: {confirmed_count}")
    print(f"Ignored temporary detections (track IDs seen in <3 frames): {temporary}")
    print(f"Other unconfirmed tracks (maximum confidence <0.35): {low_confidence}")
    print(f"Returned detections without a track ID: {untracked_detections}")
    print(f"Output video path: {video_path}")
    print(f"CSV path: {csv_path}")


if __name__ == "__main__":
    main()
