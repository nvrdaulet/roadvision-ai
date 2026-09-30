"""Save pothole boxes and confidences for manual visual inspection."""

import csv
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent


def main():
    from ultralytics import YOLO
    from ultralytics.data.utils import IMG_FORMATS

    test_dir = WORKSPACE_ROOT / "data" / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    images = sorted(
        (path for path in test_dir.glob("pothole_*")
         if path.is_file() and path.suffix.lower().lstrip(".") in IMG_FORMATS),
        key=lambda path: path.name.lower(),
    )
    if not images:
        raise ValueError(f"No pothole_* images found in {test_dir}")

    weights = WORKSPACE_ROOT / "models" / "Samdutse-pothole-yolov8" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Existing Samdutse model not found: {weights}")
    model = YOLO(str(weights))
    pothole_ids = sorted(
        class_id for class_id, name in model.names.items()
        if name.strip().lower() in {"pothole", "potholes"}
    )
    if not pothole_ids:
        raise ValueError(f"No pothole class found in model labels: {model.names}")

    output_dir = WORKSPACE_ROOT / "runs" / "manual_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "review.csv"
    # Preserve existing outputs as well as source files.
    for destination in [csv_path, *(output_dir / image.name for image in images)]:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {destination}")

    with csv_path.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["filename", "detection_count", "max_confidence"]
        )
        writer.writeheader()
        for image in images:
            confidences = []
            results = model.predict(
                source=str(image),
                conf=0.25,
                imgsz=960,
                classes=pothole_ids,
                save=False,
                stream=True,
                verbose=False,
            )
            for result in results:
                if result.boxes is not None:
                    confidences.extend(float(box.conf.item()) for box in result.boxes)
                # Also save images with no boxes so every input can be reviewed.
                result.save(
                    filename=str(output_dir / image.name),
                    boxes=True,
                    labels=True,
                    conf=True,
                )
            writer.writerow({
                "filename": image.name,
                "detection_count": len(confidences),
                "max_confidence": max(confidences, default=0.0),
            })
            csv_file.flush()
            print(f"{image.name}: {len(confidences)} detections")

    print(f"Saved {len(images)} images to: {output_dir}")
    print(f"Review CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
