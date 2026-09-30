"""RoadVision AI dashboard. Run: python -m streamlit run dashboard/app.py."""

import base64
import hashlib
import html
import json
import math
import sqlite3
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

try:
    from dashboard.maintenance import (
        PRIORITIES, STATUSES, TEAMS,
    )
    from dashboard.session_maintenance import SESSION_ONLY_KEY, load_workflow, save_workflow
except ModuleNotFoundError:
    # Streamlit also supports launching from inside the dashboard directory.
    from maintenance import PRIORITIES, STATUSES, TEAMS
    from session_maintenance import SESSION_ONLY_KEY, load_workflow, save_workflow


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CSV_PATH = ROOT / "runs" / "gps" / "unique_defects.csv"
LOCAL_SNAPSHOT_DIR = ROOT / "runs" / "video_v3" / "detections"
# Select a complete source pair: local snapshots use source IDs, whereas the
# committed public snapshots are named by the demo CSV's own defect IDs.
USING_PUBLIC_DEMO = not (LOCAL_CSV_PATH.is_file() and LOCAL_SNAPSHOT_DIR.is_dir())
CSV_PATH = ROOT / "demo_data" / "unique_defects.csv" if USING_PUBLIC_DEMO else LOCAL_CSV_PATH
SNAPSHOT_DIR = ROOT / "demo_data" / "snapshots" if USING_PUBLIC_DEMO else LOCAL_SNAPSHOT_DIR
DATABASE_PATH = ROOT / "data" / "roadvision.db"
DEMO_BANNER = "DEMO GPS DATA — NOT REAL VEHICLE COORDINATES"
TABLE_COLUMNS = [
    "defect_id", "latitude", "longitude", "max_confidence",
    "detection_count", "timestamp_seconds", "status",
]
REQUIRED_COLUMNS = set(TABLE_COLUMNS[:-1]) | {"source_defect_ids"}


def parse_source_ids(value):
    """Preserve source order: the first original ID selects the snapshot."""
    try:
        values = json.loads(value) if isinstance(value, str) else value
        if not isinstance(values, list) or not values:
            return []
        result = []
        for item in values:
            if isinstance(item, bool) or not str(item).isdigit() or int(item) < 1:
                return []
            result.append(int(item))
        return result
    except (ValueError, TypeError):
        return []


def load_data(path):
    """Read validated data; never fill in missing coordinates or records."""
    data = pd.read_csv(path, dtype={"defect_id": "string", "source_defect_ids": "string"})
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")
    numeric = ["latitude", "longitude", "max_confidence", "detection_count", "timestamp_seconds"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = data[numeric].notna().all(axis=1)
    for column in numeric:
        valid &= data[column].map(lambda value: pd.notna(value) and math.isfinite(value))
    valid &= data["latitude"].between(-90, 90) & data["longitude"].between(-180, 180)
    valid &= data["max_confidence"].between(0, 1)
    valid &= (data["detection_count"] >= 1) & (data["detection_count"] % 1 == 0)
    valid &= data["timestamp_seconds"] >= 0
    valid &= data["defect_id"].notna() & data["defect_id"].str.strip().ne("")
    invalid_count = int((~valid).sum())
    data = data.loc[valid].copy()
    duplicate_count = int(data["defect_id"].duplicated().sum())
    data = data.drop_duplicates("defect_id", keep="first")
    data["detection_count"] = data["detection_count"].astype(int)
    data["source_ids"] = data["source_defect_ids"].apply(parse_source_ids)
    return data, invalid_count, duplicate_count


def confidence_style(confidence):
    if confidence >= 0.70:
        return "green", "High"
    if confidence >= 0.50:
        return "blue", "Medium"
    return "gray", "Low"


def popup_html(row):
    sources = row["source_ids"]
    snapshot_html = '<p style="color:#64748b">Snapshot unavailable</p>'
    if USING_PUBLIC_DEMO:
        demo_id = str(row["defect_id"])
        snapshot_id = int(demo_id) if demo_id.isdigit() and int(demo_id) > 0 else None
    else:
        snapshot_id = sources[0] if sources else None
    if snapshot_id is not None:
        snapshot_path = SNAPSHOT_DIR / f"defect_{snapshot_id:04d}.jpg"
        try:
            image_bytes = snapshot_path.read_bytes()
            if image_bytes:
                encoded = base64.b64encode(image_bytes).decode("ascii")
                snapshot_html = (
                    f'<img src="data:image/jpeg;base64,{encoded}" '
                    f'alt="Snapshot for defect {html.escape(str(row["defect_id"]))}" '
                    'style="width:100%;border-radius:8px;margin-top:10px" '
                    'onerror="this.outerHTML=\'<p>Snapshot unavailable</p>\'">'
                )
        except OSError:
            pass
    details = [
        ("Latitude", f"{row['latitude']:.7f}"),
        ("Longitude", f"{row['longitude']:.7f}"),
        ("Max AI confidence", f"{row['max_confidence']:.1%}"),
        ("Timestamp", f"{row['timestamp_seconds']:.2f} seconds"),
        ("Merged observations", str(row["detection_count"])),
        ("Source defect IDs", ", ".join(map(str, sources)) if sources else "Unavailable"),
        ("Priority", str(row["priority"])),
        ("Status", str(row["status"])),
        ("Responsible team", str(row["assigned_to"]) or "Unassigned"),
    ]
    rows = "".join(
        f'<tr><td style="padding:4px 12px 4px 0;color:#64748b">{html.escape(label)}</td>'
        f'<td style="padding:4px 0;font-weight:600">{html.escape(value)}</td></tr>'
        for label, value in details
    )
    return (
        '<div style="width:330px;max-width:100%;font:13px/1.45 Arial,sans-serif;color:#172b4d">'
        '<div style="color:#92400e;font-size:10px;font-weight:700">DEMO GPS · NOT REAL COORDINATES</div>'
        f'<h3 style="margin:8px 0">Defect {html.escape(str(row["defect_id"]))}</h3>'
        f'<table>{rows}</table>{snapshot_html}</div>'
    )


def build_map(data, center):
    defect_map = folium.Map(
        location=center, zoom_start=16, tiles="OpenStreetMap", control_scale=True,
    )
    for _, row in data.iterrows():
        color, label = confidence_style(row["max_confidence"])
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            tooltip=html.escape(f"Defect {row['defect_id']} · AI confidence {row['max_confidence']:.1%} ({label})"),
            popup=folium.Popup(popup_html(row), max_width=370),
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(defect_map)
    return defect_map


def maintenance_editor(data):
    st.subheader("Create maintenance request")
    if data.empty:
        st.info("No defects match the filters. Adjust the filters to create or update a request.")
        return
    st.caption("Select a visible defect, update its maintenance details, then save.")
    defect_ids = data["defect_id"].tolist()
    if st.session_state.get("maintenance_defect") not in defect_ids:
        st.session_state["maintenance_defect"] = defect_ids[0]
    # Keep the selector outside the form so switching defects loads saved values.
    selected_id = st.selectbox("Defect", defect_ids, key="maintenance_defect",
                               format_func=lambda value: f"Defect {value}")
    record = data.loc[data["defect_id"] == selected_id].iloc[0]
    revision = f"{selected_id}-{record['updated_at']}"
    teams = ("", *TEAMS)
    with st.form(f"maintenance-form-{revision}", enter_to_submit=False):
        assigned_to = st.selectbox(
            "Responsible team", teams, index=teams.index(record["assigned_to"]),
            format_func=lambda team: team or "Unassigned", key=f"team-{revision}",
        )
        priority = st.selectbox("Priority", PRIORITIES,
                                index=PRIORITIES.index(record["priority"]),
                                key=f"priority-{revision}")
        status = st.selectbox("Status", STATUSES,
                              index=STATUSES.index(record["status"]), key=f"status-{revision}")
        st.caption("Priority is the maintenance team's working priority. AI confidence remains unchanged.")
        submitted = st.form_submit_button("Save maintenance request", type="primary",
                                           icon=":material/save:", key="save_maintenance")
    if submitted:
        try:
            save_workflow(selected_id, status, priority, assigned_to, DATABASE_PATH, st.session_state)
        except (OSError, sqlite3.Error, ValueError) as error:
            st.error(f"Unable to save maintenance request: {error}")
        else:
            destination = "saved for this session only" if st.session_state.get(SESSION_ONLY_KEY) else "saved"
            st.session_state["maintenance_notice"] = f"Maintenance request for defect {selected_id} {destination}."
            st.rerun()
    st.caption(f"Created: {record['created_at']} · Updated: {record['updated_at']} (UTC)")


st.set_page_config(page_title="RoadVision AI", page_icon=":material/add_road:", layout="wide")
st.warning(DEMO_BANNER, icon=":material/warning:")
st.title("RoadVision AI")
st.markdown("AI-powered road defect monitoring")
st.caption("Kazakhstan G-TON demo · Aktau · Road-maintenance workflow")
st.badge("Demo mode", icon=":material/science:", color="orange")
st.caption("Detection is real AI inference. GPS coordinates are synthetic for demonstration.")
with st.container(border=True):
    st.subheader("System workflow")
    st.markdown("Camera → AI Detection → GPS → Deduplication → Maintenance Request → Repair")
with st.expander("How it works"):
    st.markdown(
        "1. Cameras on buses and service vehicles capture the road.\n"
        "2. AI detects road defects in the captured images.\n"
        "3. GPS attaches coordinates to each observation. This demo uses a synthetic Aktau route, not GPS recorded from the video.\n"
        "4. Repeated observations are merged into defect candidates.\n"
        "5. The road service receives a maintenance request, assigns a team, and tracks progress through repair."
    )
if notice := st.session_state.pop("maintenance_notice", None):
    st.success(notice)

with st.sidebar:
    st.header("Filters")
    st.caption("Filters apply to metrics, map, tables and the maintenance selector.")
    selected_statuses = st.multiselect("Status", STATUSES, default=list(STATUSES), key="filter_status")
    selected_priorities = st.multiselect("Priority", PRIORITIES, default=list(PRIORITIES), key="filter_priority")
    minimum_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05,
                                   format="%.2f", key="filter_confidence",
                                   help="Model confidence, not maintenance priority.")
    minimum_observations = st.number_input(
        "Minimum observation count", min_value=1, value=1, step=1, key="filter_observations",
        help="Number of source defect candidates merged into each GPS group.",
    )
    st.divider()
    st.caption("Demo GPS route · Maintenance uses local SQLite when available.")
    st.caption("On a hosted demo, saved maintenance changes may reset when the app restarts.")

try:
    data, invalid_count, duplicate_count = load_data(CSV_PATH)
except FileNotFoundError:
    st.info("No defect data is available yet. Expected local processing data in runs/gps/unique_defects.csv "
            "or the public dataset in demo_data/unique_defects.csv.")
    st.stop()
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Unable to load defect data: {error}")
    st.stop()
if USING_PUBLIC_DEMO:
    st.caption("Public demo dataset loaded.")
if invalid_count:
    st.warning(f"Skipped {invalid_count} row(s) with missing or invalid values.")
if duplicate_count:
    st.warning(f"Skipped {duplicate_count} duplicate defect ID(s).")

try:
    metadata = pd.DataFrame(
        load_workflow(data.to_dict("records"), DATABASE_PATH, st.session_state),
        columns=["defect_id", "status", "priority", "assigned_to", "created_at", "updated_at"],
    )
    metadata["defect_id"] = metadata["defect_id"].astype("string")
    # Saved workflow values (SQLite or session fallback) own user edits.
    data = data.drop(columns=["status", "priority", "assigned_to", "created_at", "updated_at"], errors="ignore")
    data = data.merge(metadata, on="defect_id", how="left", validate="one_to_one")
except (OSError, sqlite3.Error, ValueError) as error:
    st.error(f"Unable to load maintenance records: {error}")
    st.stop()
if st.session_state.get(SESSION_ONLY_KEY):
    st.info("SQLite storage is unavailable. Maintenance changes are kept for this browser session only "
            "and will be lost when the session ends.")

filtered = data.loc[
    (data["max_confidence"] >= minimum_confidence)
    & (data["detection_count"] >= minimum_observations)
    & data["status"].isin(selected_statuses)
    & data["priority"].isin(selected_priorities)
].copy()
st.caption(f"Showing {len(filtered)} of {len(data)} defects · DEMO GPS")
with st.container(horizontal=True):
    st.metric("Unique road defects", len(filtered), border=True)
    st.metric("Total merged observations", int(filtered["detection_count"].sum()),
              border=True, help="Sum of detection_count for the visible GPS groups.")
    average = f"{filtered['max_confidence'].mean():.1%}" if not filtered.empty else "—"
    highest = f"{filtered['max_confidence'].max():.1%}" if not filtered.empty else "—"
    st.metric("Average AI confidence", average, border=True)
    st.metric("Highest AI confidence", highest, border=True)
with st.container(horizontal=True):
    st.metric("New", int(filtered["status"].eq("NEW").sum()), border=True)
    st.metric("In progress", int(filtered["status"].eq("IN_PROGRESS").sum()), border=True)
    st.metric("Fixed", int(filtered["status"].eq("FIXED").sum()), border=True)

st.subheader("Defect map")
st.markdown("**AI CONFIDENCE** · :green[High ≥ 0.70] · :blue[Medium ≥ 0.50 and < 0.70] · :gray[Low < 0.50]")
st.caption("Marker colors show AI confidence, not maintenance priority or road damage severity. Select a marker for details and a snapshot.")
if data.empty:
    st.info("No valid defect records are available. No map coordinates have been generated.")
else:
    center = [float(data["latitude"].mean()), float(data["longitude"].mean())]
    if filtered.empty:
        st.info("No defects match these filters. Adjust a filter to show more results.")
    # Refresh popup metadata immediately after a workflow edit.
    map_revision = hashlib.sha256(filtered.drop(columns="source_ids").to_json().encode()).hexdigest()[:16]
    # This responsive option belongs to streamlit-folium, not native Streamlit.
    st_folium(build_map(filtered, center), height=510, use_container_width=True,
              returned_objects=[], key=f"defects-{USING_PUBLIC_DEMO}-{map_revision}")

st.subheader("Defect records")
st.dataframe(
    filtered[TABLE_COLUMNS], hide_index=True, width="stretch",
    column_config={
        "latitude": st.column_config.NumberColumn("latitude", format="%.7f"),
        "longitude": st.column_config.NumberColumn("longitude", format="%.7f"),
        "max_confidence": st.column_config.NumberColumn("max_confidence", format="%.3f"),
        "timestamp_seconds": st.column_config.NumberColumn("timestamp_seconds", format="%.2f"),
    },
)

maintenance_editor(filtered)

st.subheader("Maintenance requests")
st.caption("Priority can be edited independently of the model's confidence.")
requests = filtered[[
    "defect_id", "priority", "status", "assigned_to", "latitude", "longitude",
    "max_confidence", "detection_count",
]].rename(columns={"max_confidence": "AI confidence", "detection_count": "observations"})
st.dataframe(
    requests, hide_index=True, width="stretch",
    column_config={
        "latitude": st.column_config.NumberColumn("latitude", format="%.7f"),
        "longitude": st.column_config.NumberColumn("longitude", format="%.7f"),
        "AI confidence": st.column_config.NumberColumn("AI confidence", format="percent"),
    },
)
