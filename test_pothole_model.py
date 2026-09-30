"""Test the Samdutse pothole model on pothole-labeled images."""

from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent


def main():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    test_dir = WORKSPACE_ROOT / "data" / "test"
    images = sorted(
        (path for path in test_dir.glob("pothole_*")
         if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda path: path.name.lower(),
    )
    detected_images = 0
    output_dir = WORKSPACE_ROOT / "runs" / "pothole_test"

    if images:
        # Keep these weights separate from the existing RDD2022 best.pt.
        model_dir = WORKSPACE_ROOT / "models" / "Samdutse-pothole-yolov8"
        model_dir.mkdir(parents=True, exist_ok=True)
        weights = model_dir / "best.pt"
        if not weights.is_file():
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

        for image in images:
            confidences = []
            results = model.predict(
                source=str(image),
                conf=0.20,
                imgsz=960,
                classes=sorted(pothole_ids),
                save=True,
                project=str(output_dir.parent),
                name=output_dir.name,
                exist_ok=True,
                stream=True,
                verbose=False,
            )
            for result in results:
                if result.boxes is not None:
                    confidences.extend(float(box.conf.item()) for box in result.boxes)
            if confidences:
                detected_images += 1
            values = ", ".join(f"{value:.4f}" for value in confidences)
            print(
                f"{image.name}: potholes detected={len(confidences)}, "
                f"confidence values=[{values}]"
            )
    else:
        print(f"No matching pothole images found in {test_dir}")

    percentage = 100 * detected_images / len(images) if images else 0.0
    print(f"Total pothole images: {len(images)}")
    print(f"Images where at least one pothole was detected: {detected_images}")
    print(f"Detection percentage: {percentage:.2f}%")
    if images:
        print(f"Annotated images saved to: {output_dir}")


if __name__ == "__main__":
    main()
