"""Run RoadVision AI on an image or video."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Detect road damage in an image or video.")
    parser.add_argument("source", help="Path to an image or video file")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.is_absolute():
        source = Path(__file__).resolve().parent / source
    if not source.is_file():
        parser.error(f"Image or video not found: {source}")

    from app.detector import RoadDamageDetector

    detector = RoadDamageDetector()
    detector.detect(source)


if __name__ == "__main__":
    main()
