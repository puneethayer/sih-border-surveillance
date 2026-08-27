"""
Detection Dashboard — v6: 2-camera integration
=================================================
Adds a REAL camera source on top of v1-v5's dummy simulation:

  - RealCameraSource opens two physical cameras via cv2.VideoCapture
    (device index 0 and 1 by default — configurable in the sidebar) and
    returns actual frames. Detections stay DUMMY (random boxes) since
    no real model is wired up yet — swap `get_detections()` inside
    RealCameraSource for your model call when it's ready.
  - A sidebar toggle switches between "Dummy simulation" (no hardware
    needed, always works) and "Real cameras" (needs webcams attached).
  - Capture objects are cached in session_state so they're opened ONCE,
    not re-opened every fragment refresh (which would be slow/broken).
  - If a real camera fails to open or a read fails, that feed shows a
    "NO SIGNAL" placeholder instead of crashing the app.
  - A "Release cameras" button frees the hardware cleanly.

Interface is unchanged: both sources implement
    get_frame(camera) -> np.ndarray
    get_detections(camera, frame) -> list[dict]
so everything else in the app (layout, alerts, zone, analytics) is
identical regardless of which source is active.
"""

import random
import uuid
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime

import numpy as np
import cv2
import streamlit as st
import pandas as pd

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Detection Dashboard", layout="wide", page_icon="🎥")
st.title("🎥 Detection Dashboard")
st.caption("Video: real webcams (or simulated). Detections: dummy until a model is wired up.")

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
FRAME_W, FRAME_H = 640, 480

# The 3 real capabilities this dashboard is built to receive tomorrow:
#   1. Human detection + tracking   -> category "person", has a track_id
#   2. Vehicle detection + classify -> category "vehicle", has a vehicle "type"
#   3. Face detection                -> category "face", nested inside a person box
PERSON_TRACK_LABEL = "person"
VEHICLE_TYPES = ["car", "truck", "bus", "motorcycle"]
FACE_LABEL = "face"

CATEGORY_COLORS = {  # BGR
    "person": (60, 200, 255),
    "vehicle": (255, 150, 60),
    "face": (120, 255, 120),
}
CAMERA = "Camera 1"
ZONE = (200, 120, 480, 380)
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


# ----------------------------------------------------------------------
# Detection source interface — same contract for dummy and real
# ----------------------------------------------------------------------
class DetectionSource(ABC):
    @abstractmethod
    def get_frame(self, camera: str) -> np.ndarray: ...
    @abstractmethod
    def get_detections(self, camera: str, frame: np.ndarray) -> list: ...


def _draw_zone(frame):
    zx1, zy1, zx2, zy2 = ZONE
    overlay = frame.copy()
    cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 90, 200), -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 140, 255), 2)
    cv2.putText(frame, "ZONE", (zx1 + 6, zy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
    return frame


def _in_zone(x, y):
    zx1, zy1, zx2, zy2 = ZONE
    return zx1 <= x <= zx2 and zy1 <= y <= zy2


# ----------------------------------------------------------------------
# DUMMY source — no hardware needed, always works
# ----------------------------------------------------------------------
class DummyDetectionSource(DetectionSource):
    def __init__(self):
        self.objects = self._init_objects()
        self._next_track_id = 1

    def _init_objects(self, n=5):
        objs = []
        for _ in range(n):
            is_person = random.random() < 0.6
            objs.append({
                "category": "person" if is_person else "vehicle",
                "vehicle_type": None if is_person else random.choice(VEHICLE_TYPES),
                "track_id": self._new_track_id() if is_person else None,
                "x": random.uniform(50, FRAME_W - 50), "y": random.uniform(50, FRAME_H - 50),
                "vx": random.uniform(-3, 3), "vy": random.uniform(-3, 3),
                "size": random.randint(40, 90),
            })
        return objs

    def _new_track_id(self):
        tid = getattr(self, "_next_track_id", 1)
        self._next_track_id = tid + 1
        return f"T{tid:03d}"

    def get_frame(self, camera):
        frame = np.full((FRAME_H, FRAME_W, 3), 25, dtype=np.uint8)
        for gx in range(0, FRAME_W, 40):
            cv2.line(frame, (gx, 0), (gx, FRAME_H), (40, 40, 40), 1)
        for gy in range(0, FRAME_H, 40):
            cv2.line(frame, (0, gy), (FRAME_W, gy), (40, 40, 40), 1)
        frame = _draw_zone(frame)
        for obj in self.objects:
            obj["x"] += obj["vx"]; obj["y"] += obj["vy"]
            if obj["x"] < 30 or obj["x"] > FRAME_W - 30: obj["vx"] *= -1
            if obj["y"] < 30 or obj["y"] > FRAME_H - 30: obj["vy"] *= -1
        cv2.putText(frame, f"SIMULATED — frame {st.session_state.frame_count}",
                    (10, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return frame

    def get_detections(self, camera, frame):
        """
        Returns detections shaped like the 3 target capabilities:
          - person:  {category:"person", track_id, box, confidence, severity, in_zone}
                     may include a nested "face" sub-detection
          - vehicle: {category:"vehicle", vehicle_type, box, confidence, severity, in_zone}
        This is the exact shape a real tracker + classifier + face model
        should hand back tomorrow — only this method's body changes.
        """
        detections = []
        for obj in self.objects:
            if random.random() < 0.35:
                half = obj["size"] // 2
                x1, y1 = int(obj["x"] - half), int(obj["y"] - half)
                x2, y2 = int(obj["x"] + half), int(obj["y"] + half)
                confidence = round(random.uniform(0.5, 0.99), 2)
                severity = "high" if confidence >= 0.85 else "medium" if confidence >= 0.7 else "low"
                in_zone = _in_zone(obj["x"], obj["y"])

                if obj["category"] == "person":
                    det = {
                        "category": "person", "label": PERSON_TRACK_LABEL,
                        "track_id": obj["track_id"], "box": (x1, y1, x2, y2),
                        "confidence": confidence, "severity": severity, "in_zone": in_zone,
                    }
                    # simulate a face detection nested inside the person box, ~50% of the time
                    if random.random() < 0.5:
                        fw, fh = (x2 - x1) // 3, (y2 - y1) // 3
                        fx1, fy1 = x1 + (x2 - x1) // 3, y1 + 5
                        det["face_box"] = (fx1, fy1, fx1 + fw, fy1 + fh)
                    detections.append(det)
                else:
                    detections.append({
                        "category": "vehicle", "label": obj["vehicle_type"],
                        "vehicle_type": obj["vehicle_type"], "box": (x1, y1, x2, y2),
                        "confidence": confidence, "severity": severity, "in_zone": in_zone,
                    })
        return detections


# ----------------------------------------------------------------------
# REAL source — 2 physical cameras via OpenCV
# ----------------------------------------------------------------------
class RealCameraSource(DetectionSource):
    """
    Opens the real webcam by device index. Detections are still random
    placeholders (no model yet) — replace `get_detections()`'s body with
    your actual inference call when ready; `get_frame()` needs no changes.
    """

    def __init__(self, device_index: int):
        self.device_index = device_index
        self.cap = None

    def _get_cap(self):
        if self.cap is None:
            # cv2.CAP_DSHOW is the faster/more reliable backend on Windows;
            # harmless to pass on other OSes but you can drop it if it errors.
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        return self.cap

    def get_frame(self, camera):
        cap = self._get_cap()
        ok, frame = (False, None)
        if cap is not None and cap.isOpened():
            ok, frame = cap.read()

        if not ok or frame is None:
            frame = np.full((FRAME_H, FRAME_W, 3), 15, dtype=np.uint8)
            cv2.putText(frame, "NO SIGNAL", (FRAME_W // 2 - 90, FRAME_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2)
            cv2.putText(frame, f"device {self.device_index} not available",
                        (20, FRAME_H // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        else:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            frame = _draw_zone(frame)
            cv2.putText(frame, f"LIVE — frame {st.session_state.frame_count}",
                        (10, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        return frame

    def get_detections(self, camera, frame):
        # Placeholder only — replace this with real model calls, e.g.:
        #   people = tracker(frame)     -> category "person" + track_id
        #   vehicles = classifier(frame) -> category "vehicle" + vehicle_type
        #   faces = face_model(frame)    -> nested "face_box" per person
        # Keep the same dict shape as DummyDetectionSource.get_detections()
        # so nothing else in the app needs to change.
        detections = []
        if random.random() < 0.25:
            x1 = random.randint(0, FRAME_W - 120)
            y1 = random.randint(0, FRAME_H - 120)
            x2, y2 = x1 + random.randint(60, 120), y1 + random.randint(60, 120)
            confidence = round(random.uniform(0.5, 0.99), 2)
            severity = "high" if confidence >= 0.85 else "medium" if confidence >= 0.7 else "low"
            in_zone = _in_zone((x1 + x2) / 2, (y1 + y2) / 2)
            if random.random() < 0.6:
                det = {"category": "person", "label": PERSON_TRACK_LABEL, "track_id": f"T{random.randint(1,99):03d}",
                       "box": (x1, y1, x2, y2), "confidence": confidence, "severity": severity, "in_zone": in_zone}
                if random.random() < 0.5:
                    fw, fh = (x2 - x1) // 3, (y2 - y1) // 3
                    fx1, fy1 = x1 + (x2 - x1) // 3, y1 + 5
                    det["face_box"] = (fx1, fy1, fx1 + fw, fy1 + fh)
                detections.append(det)
            else:
                vt = random.choice(VEHICLE_TYPES)
                detections.append({"category": "vehicle", "label": vt, "vehicle_type": vt,
                                    "box": (x1, y1, x2, y2), "confidence": confidence,
                                    "severity": severity, "in_zone": in_zone})
        return detections

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = CATEGORY_COLORS.get(det["category"], (0, 255, 0))
        thickness = 3 if det["severity"] == "high" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        tag = "⚠ " if det["in_zone"] else ""
        if det["category"] == "person":
            caption = f"{tag}person {det['track_id']} {det['confidence']:.2f}"
        elif det["category"] == "vehicle":
            caption = f"{tag}{det['vehicle_type']} {det['confidence']:.2f}"
        else:
            caption = f"{tag}{det['label']} {det['confidence']:.2f}"
        cv2.putText(frame, caption, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # nested face detection, drawn inside the person box
        if det.get("face_box"):
            fx1, fy1, fx2, fy2 = det["face_box"]
            fcolor = CATEGORY_COLORS["face"]
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), fcolor, 1)
            cv2.putText(frame, "face", (fx1, max(fy1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, fcolor, 1)
    return frame


def apply_preprocessing(frame, mode: str):
    """Preprocessing hook — runs on every frame before detection/display,
    real or dummy. Useful groundwork: a real model often wants the same
    normalization/filtering step applied consistently, so this is the one
    place to add it later (e.g. denoising, contrast correction)."""
    if mode == "Grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == "Night vision":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boosted = cv2.convertScaleAbs(gray, alpha=1.8, beta=30)
        tinted = np.zeros_like(frame)
        tinted[:, :, 1] = boosted  # green channel only
        return tinted
    return frame


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
defaults = {
    "alerts": [], "frame_count": 0, "total_detections": 0, "last_snapshot": None,
    "source_mode": "Real webcam", "device_index": 0,
    "preprocessing": "None",
    "heatmap": np.zeros((FRAME_H, FRAME_W), dtype=np.float32),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "dummy_source" not in st.session_state:
    st.session_state.dummy_source = DummyDetectionSource()
if "real_source" not in st.session_state:
    st.session_state.real_source = RealCameraSource(st.session_state.device_index)

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Controls")
source_mode = st.sidebar.radio("Video source", ["Real webcam", "Dummy simulation"],
                                index=["Real webcam", "Dummy simulation"].index(st.session_state.source_mode))
st.session_state.source_mode = source_mode

if source_mode == "Real webcam":
    new_idx = st.sidebar.number_input("Device index", 0, 10, st.session_state.device_index)
    if new_idx != st.session_state.device_index:
        st.session_state.real_source.release()
        st.session_state.device_index = new_idx
        st.session_state.real_source = RealCameraSource(new_idx)
    source: DetectionSource = st.session_state.real_source
    if st.sidebar.button("🔌 Release webcam"):
        st.session_state.real_source.release()
else:
    source = st.session_state.dummy_source

st.sidebar.markdown("---")
run = st.sidebar.checkbox("▶ Run live loop", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (sec)", 0.1, 2.0, 0.3, 0.1)
max_alerts = st.sidebar.number_input("Max alerts kept in table", 10, 500, 100)

st.sidebar.markdown("---")
search = st.sidebar.text_input("Search alerts (label/track ID contains)")
category_filter = st.sidebar.multiselect(
    "Filter by capability", ["person", "vehicle"], default=["person", "vehicle"]
)
vehicle_type_filter = st.sidebar.multiselect("Vehicle types", VEHICLE_TYPES, default=VEHICLE_TYPES)
face_only = st.sidebar.checkbox("Only show detections with a face", value=False)
severity_filter = st.sidebar.multiselect("Filter by severity", ["high", "medium", "low"],
                                          default=["high", "medium", "low"])
zone_only = st.sidebar.checkbox("Show zone alerts only", value=False)
unacked_only = st.sidebar.checkbox("Unacknowledged only", value=False)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Detections are still dummy/random on every source, real or simulated. "
    "Replace RealCameraSource.get_detections() with a real model call when ready."
)

st.sidebar.markdown("---")
st.session_state.preprocessing = st.sidebar.selectbox(
    "Frame preprocessing", ["None", "Grayscale", "Night vision"],
    index=["None", "Grayscale", "Night vision"].index(st.session_state.preprocessing)
)

st.sidebar.markdown("---")
st.sidebar.subheader("Settings")
import json as _json
settings_blob = {
    "source_mode": st.session_state.source_mode,
    "device_index": st.session_state.device_index,
    "preprocessing": st.session_state.preprocessing,
}
st.sidebar.download_button(
    "💾 Export settings", _json.dumps(settings_blob, indent=2), "dashboard_settings.json",
    "application/json", use_container_width=True
)
uploaded_settings = st.sidebar.file_uploader("📂 Import settings", type="json", label_visibility="collapsed")
if uploaded_settings is not None:
    try:
        loaded = _json.load(uploaded_settings)
        st.session_state.source_mode = loaded.get("source_mode", "Real webcam")
        st.session_state.device_index = loaded.get("device_index", 0)
        st.session_state.preprocessing = loaded.get("preprocessing", "None")
        st.sidebar.success("Settings loaded — adjust above if needed.")
    except Exception as e:
        st.sidebar.error(f"Couldn't read settings file: {e}")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_live, tab_analytics, tab_heatmap = st.tabs(["🟢 Live Monitor", "📊 Analytics", "🔥 Heatmap"])

with tab_live:
    stat1, stat2, stat3, stat4, stat5 = st.columns(5)
    stat1.metric("Frame", st.session_state.frame_count)
    stat2.metric("Total detections", st.session_state.total_detections)
    stat3.metric("Alerts in view", len(st.session_state.alerts))
    stat4.metric("In-zone alerts", sum(1 for a in st.session_state.alerts if a.get("in_zone")))
    stat5.metric("Source", "🔴 Live" if source_mode == "Real webcam" else "🎲 Simulated")
    st.markdown("---")

    col_video, col_alerts = st.columns([2, 1])

    @st.fragment(run_every=refresh_interval if run else None)
    def live_panel():
        frame = source.get_frame(CAMERA)
        frame = apply_preprocessing(frame, st.session_state.preprocessing)
        detections = source.get_detections(CAMERA, frame)
        frame = draw_detections(frame, detections)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if 0 <= cy < FRAME_H and 0 <= cx < FRAME_W:
                cv2.circle(st.session_state.heatmap, (cx, cy), 25, 1.0, -1)

        for det in detections:
            display_label = (
                f"person {det['track_id']}" if det["category"] == "person" else det["vehicle_type"]
            )
            st.session_state.alerts.append({
                "id": str(uuid.uuid4())[:8], "time": datetime.now().strftime("%H:%M:%S"),
                "camera": CAMERA, "category": det["category"], "label": display_label,
                "track_id": det.get("track_id"), "vehicle_type": det.get("vehicle_type"),
                "has_face": bool(det.get("face_box")),
                "confidence": det["confidence"],
                "severity": det["severity"], "in_zone": det["in_zone"], "acknowledged": False,
                "escalated": False, "notes": "",
                "box": det["box"],
            })
            st.session_state.total_detections += 1
            if det["severity"] == "high" and det["in_zone"]:
                st.toast(f"⚠ High-severity {display_label} detected in zone!", icon="🚨")

        st.session_state.alerts = st.session_state.alerts[-max_alerts:]
        st.session_state.frame_count += 1

        with col_video:
            st.subheader("Live Feed")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", use_container_width=True)

            snap_col1, snap_col2 = st.columns([1, 3])
            if snap_col1.button("📸 Snapshot"):
                ok, buf = cv2.imencode(".png", frame)
                if ok:
                    st.session_state.last_snapshot = buf.tobytes()
            if st.session_state.last_snapshot is not None:
                snap_col2.download_button(
                    "Download last snapshot", st.session_state.last_snapshot,
                    file_name=f"snapshot_{datetime.now().strftime('%H%M%S')}.png",
                    mime="image/png", key=f"dl_{st.session_state.frame_count}"
                )

        with col_alerts:
            st.subheader("Alerts")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear all", use_container_width=True, key=f"clear_{st.session_state.frame_count}"):
                    st.session_state.alerts = []
            with c2:
                if st.session_state.alerts:
                    csv = pd.DataFrame(st.session_state.alerts).to_csv(index=False)
                    st.download_button("Export CSV", csv, "alerts.csv", "text/csv",
                                        use_container_width=True, key=f"csv_{st.session_state.frame_count}")

            visible = [
                a for a in st.session_state.alerts
                if a["category"] in category_filter and a["severity"] in severity_filter
                and (a["category"] != "vehicle" or a["vehicle_type"] in vehicle_type_filter)
                and (not face_only or a["has_face"])
                and (not zone_only or a["in_zone"])
                and (not unacked_only or not a["acknowledged"])
                and (search.lower() in a["label"].lower() or
                     (a.get("track_id") and search.lower() in a["track_id"].lower())
                     if search else True)
            ]
            visible.sort(key=lambda a: a["time"], reverse=True)
            visible.sort(key=lambda a: (a["acknowledged"], SEVERITY_RANK[a["severity"]]))

            if visible:
                df = pd.DataFrame(visible)
                display_cols = ["time", "category", "label", "has_face", "confidence", "severity",
                                 "in_zone", "acknowledged", "escalated", "notes"]
                edited = st.data_editor(
                    df[display_cols], use_container_width=True, height=380,
                    disabled=["time", "category", "label", "has_face", "confidence", "severity", "in_zone"],
                    column_config={
                        "acknowledged": st.column_config.CheckboxColumn("Ack?"),
                        "escalated": st.column_config.CheckboxColumn("Escalate?"),
                        "has_face": st.column_config.CheckboxColumn("Face?"),
                        "notes": st.column_config.TextColumn("Notes", width="medium"),
                        "confidence": st.column_config.NumberColumn(format="%.2f"),
                    },
                    hide_index=True, key=f"editor_{st.session_state.frame_count}",
                )
                # write acknowledged/escalated/notes edits back into the real alert log
                id_lookup = {a["id"]: a for a in st.session_state.alerts}
                for i, row in edited.iterrows():
                    alert_id = df.iloc[i]["id"] if "id" in df.columns else None
                    if alert_id in id_lookup:
                        id_lookup[alert_id]["acknowledged"] = bool(row["acknowledged"])
                        id_lookup[alert_id]["escalated"] = bool(row["escalated"])
                        id_lookup[alert_id]["notes"] = row["notes"]
            else:
                st.info("No alerts match the current filters.")

    live_panel()
    if not run:
        st.info("Loop paused. Check '▶ Run live loop' in the sidebar to resume.")

with tab_analytics:
    st.subheader("Detection Analytics")
    if st.session_state.alerts:
        df = pd.DataFrame(st.session_state.alerts)
        st.markdown("**Detections per minute**")
        df["minute"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.floor("min").dt.strftime("%H:%M")
        st.line_chart(df.groupby("minute").size())

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**By capability**")
            st.bar_chart(df["category"].value_counts())
        with col_b:
            st.markdown("**By severity**")
            st.bar_chart(df["severity"].value_counts())

        col_c, col_d = st.columns(2)
        with col_c:
            vdf = df[df["category"] == "vehicle"]
            st.markdown("**Vehicle types**")
            if not vdf.empty:
                st.bar_chart(vdf["vehicle_type"].value_counts())
            else:
                st.caption("No vehicle detections yet.")
        with col_d:
            pdf = df[df["category"] == "person"]
            st.markdown("**Face detection rate (of people)**")
            if not pdf.empty:
                st.bar_chart(pdf["has_face"].value_counts())
            else:
                st.caption("No person detections yet.")

    else:
        st.info("No detections logged yet — let the live loop run for a bit.")

with tab_heatmap:
    st.subheader("Where detections cluster in the frame")
    st.caption("Builds up as detections come in — shows hotspots across the whole session, "
               "independent of which camera is currently selected.")
    if st.session_state.heatmap.max() > 0:
        norm = cv2.normalize(st.session_state.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        st.image(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), use_container_width=True)
        if st.button("Reset heatmap"):
            st.session_state.heatmap = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
            st.rerun()
    else:
        st.info("No detections yet — let the live loop run for a bit to build up the heatmap.")
