"""Evaluate image-level pothole presence using filename ground truth."""

import csv
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
FIELDS = [
    "filename", "expected_pothole", "pothole_detected",
    "detection_count", "max_confidence", "correct",
]


def expected_pothole(path: Path) -> bool:
    name = path.name.lower()
    if name.startswith("pothole_"):
        return True
    if name.startswith(("crack_", "normal_")):
        return False
    raise ValueError(f"Cannot determine expected label from filename: {path.name}")


def main():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    from ultralytics.data.utils import IMG_FORMATS

    test_dir = WORKSPACE_ROOT / "data" / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    images = sorted(
        (path for path in test_dir.rglob("*")
         if path.is_file() and path.suffix.lower().lstrip(".") in IMG_FORMATS),
        key=lambda path: str(path).lower(),
    )
    # Validate labels before inference so unknown prefixes cannot skew metrics.
    labeled_images = [(path, expected_pothole(path)) for path in images]
    rows = []
    tp = fn = fp = tn = 0

    if images:
        model_dir = WORKSPACE_ROOT / "models" / "Samdutse-pothole-yolov8"
        weights = model_dir / "best.pt"
        if not weights.is_file():
            model_dir.mkdir(parents=True, exist_ok=True)
            weights = Path(hf_hub_download(
                repo_id="Samdutse/pothole-yolov8",
                filename="best.pt",
                local_dir=str(model_dir),
            ))
        model = YOLO(str(weights))
        pothole_ids = {
            class_id for class_id, name in model.names.items()
            if name.strip().lower() in {"pothole", "potholes"}
        }
        if not pothole_ids:
            raise ValueError(f"No pothole class found in model labels: {model.names}")

        for image, expected in labeled_images:
            confidences = []
            results = model.predict(
                source=str(image),
                conf=0.20,
                imgsz=960,
                classes=sorted(pothole_ids),
                save=False,
                stream=True,
                verbose=False,
            )
            for result in results:
                if result.boxes is not None:
                    confidences.extend(float(box.conf.item()) for box in result.boxes)
            detected = bool(confidences)
            if expected and detected:
                tp += 1
            elif expected:
                fn += 1
            elif detected:
                fp += 1
            else:
                tn += 1
            rows.append({
                "filename": image.relative_to(test_dir).as_posix(),
                "expected_pothole": str(expected).lower(),
                "pothole_detected": str(detected).lower(),
                "detection_count": len(confidences),
                "max_confidence": max(confidences, default=0.0),
                "correct": str(expected == detected).lower(),
            })

    output_path = WORKSPACE_ROOT / "runs" / "evaluation" / "pothole_model_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total images: {len(images)}")
    print(f"True Positives: {tp}")
    print(f"False Negatives: {fn}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    for name, numerator, denominator in (
        ("Recall", tp, tp + fn),
        ("Precision", tp, tp + fp),
        ("Accuracy", tp + tn, len(images)),
    ):
        value = f"{100 * numerator / denominator:.2f}%" if denominator else "N/A (zero denominator)"
        print(f"{name}: {value}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
