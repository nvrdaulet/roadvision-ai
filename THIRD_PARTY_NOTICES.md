# Third-party notices

RoadVision AI integrates third-party software, pretrained models, and mapping data. The RoadVision AI team did **not** train the pothole detector from scratch and does not claim ownership of third-party weights or datasets. Licensing the project's original source code does not replace upstream terms.

The upstream model and Ultralytics licensing pages below were reviewed on 2026-10-01. Consult the upstream sources for subsequent changes and the terms applicable to the versions you use.

## Ultralytics YOLO

**Project:** Ultralytics YOLO  
**Website:** [https://ultralytics.com](https://ultralytics.com)  
**License:** AGPL-3.0 for open-source usage, with separate Enterprise licensing available.

RoadVision AI uses Ultralytics YOLOv8 inference and its ByteTrack tracking integration. The detector architecture and underlying framework are third-party work.

Sources: [Official licensing information](https://www.ultralytics.com/license), [Ultralytics source and license](https://github.com/ultralytics/ultralytics).

## Pothole detection model

**Model:** Samdutse/pothole-yolov8  
**Source:** [https://huggingface.co/Samdutse/pothole-yolov8](https://huggingface.co/Samdutse/pothole-yolov8)  
**Description:** YOLOv8s fine-tuned for pothole detection.

RoadVision AI uses the published `best.pt` weights for inference. The upstream model card identifies the Smartathon pothole dataset on Roboflow Universe as the fine-tuning dataset. The model publisher's training work and dataset provenance must not be attributed to the RoadVision AI team.

**Important:** The Hugging Face model page does not clearly state a separate model license. Do not claim ownership of these pretrained weights. **Users should verify the upstream model license before commercial deployment.** This notice does not establish permission to redistribute or commercially use the weights or their training data.

## Earlier road-damage model

The earlier `main.py` and `app/detector.py` integration uses [dronefreak/rdd2022-yolov8s](https://huggingface.co/dronefreak/rdd2022-yolov8s), a separate pretrained road-damage detector. Its model page identifies an AGPL-3.0 license and references the RDD2022 dataset. Consult the model card and its linked dataset notices before using that integration. RoadVision AI does not claim ownership of that model or dataset.

## Tracking, dashboard, and processing libraries

RoadVision AI acknowledges these projects and their contributors:

- [ByteTrack](https://github.com/FoundationVision/ByteTrack): multi-object tracking, accessed through Ultralytics' implementation.
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub): access to published pretrained weights.
- [OpenCV](https://opencv.org): video decoding, image processing, and annotation output.
- [Streamlit](https://streamlit.io), [pandas](https://pandas.pydata.org), [Folium](https://python-visualization.github.io/folium/), and [streamlit-folium](https://github.com/randyzwitch/streamlit-folium): dashboard and map components.
- [SQLite](https://sqlite.org): local maintenance request storage.

These projects retain their respective licenses and notices. The dependency manifests also include supporting packages; consult each distributed package's own license files when redistributing an installation. This document is not a replacement for those notices.

## Maps, route reference, and demo evidence

Map data is credited to **© OpenStreetMap contributors**. See the [OpenStreetMap copyright and license page](https://www.openstreetmap.org/copyright). The synthetic Aktau route uses OpenStreetMap/OSRM road geometry as a planning reference; [OSRM](https://project-osrm.org) is acknowledged for routing tooling.

**DEMO GPS DATA — NOT REAL VEHICLE COORDINATES.** Demo coordinates and timestamps are synthetic and were not recorded from the test video. Snapshot detections come from real model inference, but this does not establish ownership or licensing rights over the underlying images, videos, or pretrained model.

The [RoadVision AI source-code license](LICENSE) does not grant new permissions for third-party media, datasets, models, or map data. Retain applicable attribution and verify the relevant upstream permissions before reuse.
