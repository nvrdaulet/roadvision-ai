"""Road damage detection for local images and videos."""

from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class RoadDamageDetector:
    """Download the weights as needed and load YOLO once per instance."""

    def __init__(self):
        models_dir = WORKSPACE_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        weights_path = models_dir / "best.pt"
        if not weights_path.is_file():
            weights_path = Path(hf_hub_download(
                repo_id="dronefreak/rdd2022-yolov8s",
                filename="best.pt",
                local_dir=str(models_dir),
            ))

        self.model = YOLO(str(weights_path))
        self.output_dir = WORKSPACE_ROOT / "runs" / "detect" / "roadvision"

    def detect(self, source: str | Path) -> Path:
        """Save annotations, print every detection, and return the output folder.

        Relative input paths are resolved from the workspace root. Bounding
        boxes use pixel coordinates in (x1, y1, x2, y2) order.
        """
        source = Path(source).expanduser()
        if not source.is_absolute():
            source = WORKSPACE_ROOT / source
        if not source.is_file():
            raise FileNotFoundError(f"Image or video not found: {source}")

        results = self.model.predict(
            source=str(source),
            conf=0.10,
            imgsz=960,
            save=True,
            project=str(self.output_dir.parent),
            name=self.output_dir.name,
            exist_ok=True,
            stream=True,
        )
        total_detections = 0
        # Exhaust the stream so all video frames are saved and writers close.
        for frame_number, result in enumerate(results, start=1):
            if result.boxes is None:
                continue
            for box in result.boxes:
                total_detections += 1
                class_name = result.names[int(box.cls.item())]
                confidence = float(box.conf.item())
                bbox = box.xyxy[0].tolist()
                print(
                    f"Frame {frame_number}: class={class_name}, "
                    f"confidence={confidence:.4f}, bbox={bbox}"
                )

        if total_detections == 0:
            print("No road damage detected")
        else:
            print(f"Total number of detections: {total_detections}")

        print(f"Annotated results saved to: {self.output_dir}")
        return self.output_dir
