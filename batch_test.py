"""Evaluate filename-labeled road images with the existing RoadVision model."""

import csv
import json
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
CRACK_CLASSES = {"longitudinal_crack", "transverse_crack", "alligator_crack"}
FIELDS = [
    "filename", "expected_label", "detected_classes", "maximum_confidence",
    "detections_count", "is_correct",
]


def expected_label(filename: str) -> str:
    name = filename.lower()
    for prefix, label in (
        ("pothole_", "pothole"),
        ("crack_", "crack"),
        ("normal_", "no road damage"),
    ):
        if name.startswith(prefix):
            return label
    return "unknown"


def main():
    from app.detector import RoadDamageDetector

    test_dir = WORKSPACE_ROOT / "data" / "test"
    images = sorted(
        (path for path in test_dir.rglob("*")
         if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda path: str(path).lower(),
    )
    output_path = WORKSPACE_ROOT / "runs" / "evaluation" / "results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tested = {"pothole": 0, "crack": 0, "no road damage": 0}
    correct = dict.fromkeys(tested, 0)
    unknown_count = 0
    detector = RoadDamageDetector() if images else None

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        for image in images:
            label = expected_label(image.name)
            classes = set()
            maximum_confidence = 0.0
            detections_count = 0
            # Match app/detector.py's current inference and saving settings.
            results = detector.model.predict(
                source=str(image),
                conf=0.10,
                imgsz=960,
                save=True,
                project=str(detector.output_dir.parent),
                name=detector.output_dir.name,
                exist_ok=True,
                stream=True,
            )
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    classes.add(result.names[int(box.cls.item())])
                    maximum_confidence = max(maximum_confidence, float(box.conf.item()))
                    detections_count += 1

            normalized_classes = {
                name.strip().lower().replace(" ", "_").replace("-", "_")
                for name in classes
            }
            is_correct = (
                (label == "pothole" and "pothole" in normalized_classes)
                or (label == "crack" and bool(CRACK_CLASSES & normalized_classes))
                or (label == "no road damage" and detections_count == 0)
            )
            if label in tested:
                tested[label] += 1
                correct[label] += int(is_correct)
            else:
                # Evaluate every image, but do not invent labels for accuracy.
                unknown_count += 1

            writer.writerow({
                "filename": image.relative_to(test_dir).as_posix(),
                "expected_label": label,
                "detected_classes": json.dumps(sorted(classes)),
                "maximum_confidence": maximum_confidence,
                "detections_count": detections_count,
                "is_correct": is_correct if label != "unknown" else "",
            })
            csv_file.flush()

    labeled_count = sum(tested.values())
    percentage = 100 * sum(correct.values()) / labeled_count if labeled_count else 0.0
    print(f"Total images: {len(images)}")
    print(f"Pothole images tested: {tested['pothole']}")
    print(f"Potholes correctly detected: {correct['pothole']}")
    print(f"Crack images tested: {tested['crack']}")
    print(f"Cracks correctly detected: {correct['crack']}")
    print(f"Normal images tested: {tested['no road damage']}")
    print(f"Normal images correctly classified: {correct['no road damage']}")
    print(f"Overall correct percentage: {percentage:.2f}%")
    if unknown_count:
        print(f"Images with unknown prefixes (excluded from accuracy): {unknown_count}")
    if not images:
        print(f"No .jpg, .jpeg, or .png images found in {test_dir}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
