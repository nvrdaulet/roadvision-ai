"""Compare image-level pothole classification across confidence thresholds.

CSV metrics are fractions from 0 to 1; the console displays percentages.
Undefined ratios are reported as zero. F1 ties favor the lower threshold.
"""

import csv
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
FIELDS = (
    "confidence_threshold", "true_positives", "false_negatives",
    "false_positives", "true_negatives", "recall", "precision",
    "accuracy", "f1_score",
)


def expected_pothole(path: Path) -> bool:
    name = path.name.lower()
    if name.startswith("pothole_"):
        return True
    if name.startswith(("crack_", "normal_")):
        return False
    raise ValueError(f"Cannot determine expected label from filename: {path.name}")


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main():
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
    if not images:
        raise ValueError(f"No supported images found in {test_dir}")
    labeled_images = [(image, expected_pothole(image)) for image in images]
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

    rows = []
    for threshold in sorted(THRESHOLDS):
        print(f"Evaluating {len(images)} images at confidence {threshold:.2f}...", flush=True)
        tp = fn = fp = tn = 0
        for image, expected in labeled_images:
            detected = False
            results = model.predict(
                source=str(image),
                conf=threshold,
                imgsz=960,
                classes=pothole_ids,
                save=False,
                stream=True,
                verbose=False,
            )
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    detected = True
            if expected and detected:
                tp += 1
            elif expected:
                fn += 1
            elif detected:
                fp += 1
            else:
                tn += 1
        rows.append({
            "confidence_threshold": threshold,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "recall": ratio(tp, tp + fn),
            "precision": ratio(tp, tp + fp),
            "accuracy": ratio(tp + tn, len(images)),
            "f1_score": ratio(2 * tp, 2 * tp + fp + fn),
        })

    output_path = WORKSPACE_ROOT / "runs" / "evaluation" / "threshold_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal images: {len(images)}")
    print("TP=True Positives, FN=False Negatives, FP=False Positives, TN=True Negatives")
    print(f"{'Conf':>6} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} "
          f"{'Recall':>9} {'Precision':>10} {'Accuracy':>9} {'F1':>9}")
    for row in rows:
        print(
            f"{row['confidence_threshold']:>6.2f} "
            f"{row['true_positives']:>4} {row['false_negatives']:>4} "
            f"{row['false_positives']:>4} {row['true_negatives']:>4} "
            f"{row['recall']:>9.2%} {row['precision']:>10.2%} "
            f"{row['accuracy']:>9.2%} {row['f1_score']:>9.2%}"
        )
    best = max(rows, key=lambda row: row['f1_score'])
    print(f"\nBEST THRESHOLD BY F1: {best['confidence_threshold']:.2f} "
          f"(F1: {best['f1_score']:.2%}; ties favor the lower threshold)")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
