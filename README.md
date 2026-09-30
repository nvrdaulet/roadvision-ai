# RoadVision AI

AI-powered road defect monitoring and maintenance workflow system.

**Demo:** [https://roadly.streamlit.app](https://roadly.streamlit.app)

RoadVision AI connects road video analysis to a map and a maintenance workflow for reviewing defects, assigning teams, and tracking repair status.

**The RoadVision AI team did NOT train the pothole detector from scratch.** The project integrates the pretrained [Samdutse/pothole-yolov8](https://huggingface.co/Samdutse/pothole-yolov8) detector, based on **Ultralytics YOLOv8**. Its upstream model card describes a YOLOv8s model fine-tuned for pothole detection.

RoadVision AI's implemented contribution includes **video processing, multi-frame tracking, defect event merging, GPS enrichment, geographic deduplication, interactive mapping, maintenance workflow, and the Streamlit dashboard**. The project does not claim ownership of third-party pretrained weights or datasets.

## Features

- Process video frame by frame with the pretrained pothole detector and persistent ByteTrack IDs.
- Filter detections by road region, suppress overlapping boxes, and confirm tracks across multiple frames.
- Merge fragmented tracks into physical defect candidates and save their best-confidence snapshots.
- Attach GPS positions by timestamp interpolation and merge nearby candidates using Haversine distance.
- Explore an interactive map, evidence snapshots, confidence metrics, and defect records.
- Filter by AI confidence, observation count, status, and priority.
- Assign Road Team A, Road Team B, or Emergency Crew to maintenance requests.
- Track `NEW`, `ASSIGNED`, `IN_PROGRESS`, and `FIXED` statuses with editable `LOW`, `MEDIUM`, or `HIGH` priority.
- Save maintenance decisions in SQLite, with a session-only fallback when storage is unavailable.
- Run the dashboard from the bundled demo dataset without video files or model weights.

**AI confidence is a model score, not road damage severity.** Maintenance priority is a separate, editable workflow field. Initial priority is `HIGH` for confidence ≥ 0.70 with at least three observations, `MEDIUM` for confidence ≥ 0.55 otherwise, and `LOW` for the remaining records.

## Architecture

**Camera → AI Detection → Tracking → GPS → Deduplication → Map → Maintenance Request → Repair**

| Stage | Current implementation |
| --- | --- |
| Camera | The intended source is road footage from buses or service vehicles. The current pipeline reads a local test video. |
| AI Detection | `video_detector_v3.py` loads Samdutse's pretrained YOLOv8 pothole model once and processes the video frames. |
| Tracking | Persistent ByteTrack IDs, multi-frame confirmation, and post-tracking event merging produce physical defect candidates. |
| GPS | `generate_demo_gps.py` creates a synthetic Aktau route. `gps_enricher.py` matches candidate timestamps to GPS points and interpolates between them. |
| Deduplication | GPS candidates linked within 12 meters are grouped with Haversine distance. Original track and source defect IDs are preserved. |
| Map | `dashboard/app.py` displays prepared records and snapshots with Streamlit, pandas, Folium, and streamlit-folium. |
| Maintenance Request | The dashboard stores status, priority, team, and timestamps in `data/roadvision.db` when SQLite is available. |
| Repair | Users can mark a request `FIXED`. This is workflow state, not independent verification that a physical repair occurred. |

The current video pipeline uses `conf=0.30`, `imgsz=960`, and `bytetrack.yaml`. It ignores boxes whose centers are in the upper 30% of the frame and suppresses duplicate boxes with IoU ≥ 0.45 before tracking. A track must appear in at least five frames and reach maximum confidence ≥ 0.35 to be confirmed.

Confirmed tracks can merge when their frame ranges overlap or the gap is at most one second, their best-box centers are within 0.15 of the frame diagonal, and their box areas have similar scale. The output is called **physical defect candidates**; it is not a verified count of distinct real-world potholes.

The dashboard reads prepared outputs. Opening the dashboard does not run AI inference, collect camera footage, or acquire live GPS.

## Demo

Open the [RoadVision AI public demo](https://roadly.streamlit.app).

> **DEMO GPS DATA — NOT REAL VEHICLE COORDINATES**

The public demo uses **synthetic GPS coordinates in Aktau, Kazakhstan**. These coordinates and timestamps were not recorded from the test video and must not be presented as measured vehicle or pothole locations. **The AI detections shown in the snapshots come from real model inference.**

The bundled dataset contains 10 demo defect records and 10 corresponding snapshots. The dashboard chooses its data sources as a pair:

| Source | Defect records | Snapshots |
| --- | --- | --- |
| Local processing, preferred when both paths exist | `runs/gps/unique_defects.csv` | `runs/video_v3/detections/` |
| Public fallback | `demo_data/unique_defects.csv` | `demo_data/snapshots/` |

When using the fallback, the dashboard displays **“Public demo dataset loaded.”** Local snapshots are selected using the first source defect ID; public snapshots are named by demo defect ID (`defect_0001.jpg` through `defect_0010.jpg`).

Try selecting a marker, reviewing its snapshot, changing a maintenance request's team or status, and filtering the map. Maintenance actions are local demo records; they are not sent to a municipal road service.

## Installation

Use a Python virtual environment from the repository root. The processing scripts require Python 3.12 or newer; the current development environment uses Python 3.14.

For the dashboard, install only its dependencies. In Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r dashboard/requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate` instead. The dashboard can use `demo_data/` immediately; YOLO weights and test videos are not needed for this path.

To run the optional local detection pipeline, install the full processing environment:

```bash
python -m pip install -r requirements.txt
```

Provide a video you have permission to use at `data/videos/road_test.mp4`. Download the pretrained weights into the location expected by the video scripts:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Samdutse/pothole-yolov8', filename='best.pt', local_dir='models/Samdutse-pothole-yolov8')"
```

The download requires internet access. Review the model licensing notes below before using or redistributing the weights. `video_detector_v3.py` expects this weights file to exist; it does not download it itself.

To generate local demo outputs:

```bash
python video_detector_v3.py
python generate_demo_gps.py
python gps_enricher.py
```

`video_detector_v3.py` recreates `runs/video_v3/`, and `gps_enricher.py` recreates `runs/gps/`. Preserve any outputs you need before rerunning them. The GPS generator refuses to replace an existing GPS CSV unless you explicitly run `python generate_demo_gps.py --overwrite-demo` to replace demo data.

## Running the dashboard

With the virtual environment active, run from the repository root:

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit. For Streamlit deployment, use `dashboard/app.py` as the entrypoint and retain `dashboard/requirements.txt` and the `demo_data/` directory in the repository.

SQLite stores workflow fields separately from detection records in `data/roadvision.db`. If SQLite cannot be read or written, the app keeps edits in the current browser session and displays a notice. Session-only edits disappear when that session ends; hosted SQLite files may also reset when the app restarts.

## Project structure

```text
roadvision-ai/
├── dashboard/
│   ├── app.py                       # Map, filters, evidence, and workflow UI
│   ├── maintenance.py               # SQLite maintenance records
│   ├── session_maintenance.py       # Fallback when SQLite is unavailable
│   └── requirements.txt             # Dashboard dependencies
├── demo_data/
│   ├── unique_defects.csv           # Bundled public demo records
│   └── snapshots/                   # 10 snapshots named by demo defect ID
├── app/detector.py                  # Earlier general road-damage detector
├── main.py                          # Earlier image/video CLI
├── video_detector.py                # Initial tracking pipeline
├── video_detector_v2.py             # Road-region and duplicate filtering
├── video_detector_v3.py             # Tracking and defect event merging
├── generate_demo_gps.py             # Synthetic Aktau GPS generation
├── gps_enricher.py                  # GPS interpolation and deduplication
├── batch_test.py                    # General road-damage image evaluation
├── test_pothole_model.py            # Pothole image inference
├── evaluate_pothole_model.py        # Image-level pothole evaluation
├── threshold_sweep.py               # Confidence-threshold comparison
├── inspect_pothole_detections.py    # Manual inspection outputs
├── models/                          # Local downloaded weights
├── data/                            # Local inputs, GPS CSV, and SQLite DB
├── runs/                            # Generated annotations and CSV outputs
├── requirements.txt                 # Full processing dependencies
├── README.md
├── THIRD_PARTY_NOTICES.md
└── LICENSE
```

`models/`, local input files, and generated `runs/` outputs are not required to view the bundled demo and may be absent in a fresh checkout.

The earlier `main.py` / `app/detector.py` path uses the separate pretrained [dronefreak/rdd2022-yolov8s](https://huggingface.co/dronefreak/rdd2022-yolov8s) road-damage model. The current pothole video pipeline described above uses **Samdutse/pothole-yolov8**.

## Current limitations

- GPS is synthetic in the demo. Vehicle positions stand in for candidate locations; the system does not calculate surveyed defect coordinates from camera geometry.
- Tracking and merging use heuristics. Fragmented tracks can remain separate, and nearby defects can be merged incorrectly. Pairwise merges are transitive, so endpoints of a GPS group can be more than 12 meters apart.
- `detection_count` counts merged physical defect candidates, not video frames or confirmed observations from independent vehicle passes.
- AI detections can be wrong or missed. Confidence does not measure damage severity, and the demo does not establish accuracy on Kazakhstan roads.
- Existing filename-based evaluation measures image-level pothole presence, not a bounding-box mAP benchmark or validated road inventory.
- The current pipeline processes local files, and its annotated video has no audio. Live fleet ingestion and real GPS telemetry are not implemented.
- The maintenance workflow has no authentication, role-based permissions, external dispatch, or durable shared hosted database. Treat public workflow edits as demonstration data.

## Third-party components

| Component | Role |
| --- | --- |
| [Samdutse/pothole-yolov8](https://huggingface.co/Samdutse/pothole-yolov8) | Pretrained pothole detector weights |
| [Ultralytics YOLO](https://ultralytics.com) | YOLOv8 inference and tracking integration |
| [ByteTrack](https://github.com/FoundationVision/ByteTrack) | Multi-object tracking algorithm used through Ultralytics |
| [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub) | Model distribution and downloads |
| [OpenCV](https://opencv.org) | Video and image processing |
| [Streamlit](https://streamlit.io), [pandas](https://pandas.pydata.org), [Folium](https://python-visualization.github.io/folium/), [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) | Dashboard, records, and interactive map |
| [SQLite](https://sqlite.org) | Local maintenance storage |
| [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) and [OSRM](https://project-osrm.org) | Map data and the planning reference for the synthetic route |

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and licensing notes.

## Citation and acknowledgements

When referencing this integration, cite **RoadVision AI — AI-powered road defect monitoring and maintenance workflow system**, with the [source repository](https://github.com/nvrdaulet/roadvision-ai) and [public demo](https://roadly.streamlit.app). Identify the repository revision used when reporting reproducible results.

Please also acknowledge the [Samdutse model publisher](https://huggingface.co/Samdutse/pothole-yolov8), [Ultralytics](https://github.com/ultralytics/ultralytics), and the authors of [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://github.com/FoundationVision/ByteTrack). Use the upstream projects' citation instructions when citing their work.

The Samdutse model card credits the Smartathon pothole dataset on Roboflow Universe for its fine-tuning. That training was upstream work, not training performed by the RoadVision AI team. Credit also belongs to the maintainers of the dashboard libraries, mapping tools, and OpenStreetMap contributors. RoadVision AI does not claim ownership of their software, models, datasets, or map data.

## License notes

RoadVision AI's original source code is licensed under the **GNU Affero General Public License, version 3 (AGPL-3.0)**. See the complete, unmodified text in [LICENSE](LICENSE), obtained from the [GNU license source](https://www.gnu.org/licenses/agpl-3.0.txt).

Ultralytics provides AGPL-3.0 for open-source usage, with separate Enterprise licensing available. Refer to the [official Ultralytics licensing page](https://www.ultralytics.com/license) for its terms. AGPL-3.0 includes corresponding-source obligations for modified software offered over a network; review the full license when distributing or hosting the project.

The Hugging Face page for **Samdutse/pothole-yolov8 does not clearly state a separate model license**. RoadVision AI does not claim ownership of these pretrained weights. **Users should verify the upstream model license before commercial deployment.**

The source-code license does not claim ownership of, or grant new rights to, third-party weights, datasets, videos, images, or map data. Their respective upstream terms still apply; check those terms before reuse or redistribution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
