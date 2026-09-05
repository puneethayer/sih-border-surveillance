from pathlib import Path
from datetime import datetime
import sqlite3

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates


# ============================================================
# IBVAP AI BORDER SURVEILLANCE COMMAND CENTER
# ============================================================

st.set_page_config(
    page_title="IBVAP | AI Border Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "cctv.mp4"
MODEL_PATH = ROOT / "yolo26n.pt"
OUTPUT_PATH = ROOT / "intrusion_result.mp4"
DB_PATH = ROOT / "database" / "ibvap.db"
SNAPSHOT_DIR = ROOT / "logs" / "snapshots"


# ============================================================
# SESSION STATE
# ============================================================

for key, default in {
    "camera": None,
    "fence_points": [],
    "last_click": None,
    "fence_version": 0,
    "detection_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# SMALL CSS ONLY - NO CUSTOM HTML
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(0,180,255,.07), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(0,255,170,.04), transparent 24%),
            #070b12;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetric"] {
        background: #0e1722;
        border: 1px solid rgba(130,160,190,.16);
        border-radius: 14px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #8ba0b4;
    }

    [data-testid="stMetricValue"] {
        color: #edf7ff;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE
# ============================================================

def get_events():
    """Read all database events once per Streamlit rerun."""
    if not DB_PATH.exists():
        return []

    connection = None

    try:
        connection = sqlite3.connect(str(DB_PATH))
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                id,
                timestamp,
                camera_id,
                object_type,
                track_id,
                event_type,
                confidence,
                snapshot_path,
                plate_number
            FROM events
            ORDER BY timestamp DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]

    except sqlite3.Error:
        return []

    finally:
        if connection is not None:
            connection.close()


def format_timestamp(value):
    if not value:
        return "-"

    try:
        return datetime.fromisoformat(str(value)).strftime(
            "%d %b %Y • %H:%M:%S"
        )
    except Exception:
        return str(value)


def resolve_snapshot(value):
    """
    Resolve absolute Windows paths and project-relative paths.
    """
    if not value:
        return None

    raw = str(value)
    path = Path(raw)

    if path.is_absolute() and path.exists():
        return path

    candidates = [
        ROOT / path,
        ROOT / raw.replace("\\", "/"),
        SNAPSHOT_DIR / path.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


# ============================================================
# VIDEO
# ============================================================

@st.cache_data(show_spinner=False)
def get_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    success, frame = cap.read()
    cap.release()

    if not success:
        return None

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ============================================================
# FENCE DRAWING
# ============================================================

def draw_fence(frame, points):
    image = frame.copy()

    if len(points) >= 3:
        polygon = np.asarray(points, dtype=np.int32)
        overlay = image.copy()

        cv2.fillPoly(
            overlay,
            [polygon],
            (255, 90, 0),
        )

        image = cv2.addWeighted(
            overlay,
            0.18,
            image,
            0.82,
            0,
        )

    if len(points) >= 2:
        for index in range(len(points) - 1):
            cv2.line(
                image,
                points[index],
                points[index + 1],
                (255, 170, 30),
                4,
            )

    if len(points) >= 3:
        cv2.line(
            image,
            points[-1],
            points[0],
            (255, 170, 30),
            4,
        )

    for index, (x, y) in enumerate(points, start=1):
        cv2.circle(
            image,
            (x, y),
            9,
            (255, 170, 30),
            -1,
        )

        cv2.circle(
            image,
            (x, y),
            11,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            image,
            f"P{index}",
            (x + 12, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"P{index}",
            (x + 12, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 170, 30),
            1,
            cv2.LINE_AA,
        )

    return image


# ============================================================
# LOAD DATA
# ============================================================

all_events = get_events()

intrusion_events = [
    event
    for event in all_events
    if str(event.get("event_type", "")).lower() == "intrusion"
]

detection_events = [
    event
    for event in all_events
    if str(event.get("event_type", "")).lower() == "detection"
]

camera_count = len(
    {
        str(event.get("camera_id"))
        for event in all_events
        if event.get("camera_id")
    }
)


# ============================================================
# HEADER
# ============================================================

title_left, title_right = st.columns([4, 1])

with title_left:
    st.title("🛡️ IBVAP COMMAND CENTER")
    st.caption(
        "Intelligent Border Virtual AI Protection • "
        "YOLO Vision • Virtual Fence • SQLite Evidence"
    )

with title_right:
    st.success("● SYSTEM ONLINE")
    st.caption("AI surveillance active")


# ============================================================
# DATABASE STATUS
# ============================================================

if DB_PATH.exists():

    if all_events:
        latest = format_timestamp(
            all_events[0].get("timestamp")
        )
        st.success(
            f"🟢 Database connected • Latest event: {latest}"
        )
    else:
        st.info(
            "🟢 Database connected • No events recorded yet"
        )

else:

    st.warning(
        f"Database not found yet: {DB_PATH}"
    )


# ============================================================
# TOP METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "SYSTEM STATUS",
        "ONLINE",
    )

with m2:
    st.metric(
        "🚨 INTRUSIONS",
        len(intrusion_events),
        delta="THREAT" if intrusion_events else "SECURE",
        delta_color="inverse" if intrusion_events else "normal",
    )

with m3:
    st.metric(
        "AI EVENTS",
        len(all_events),
    )

with m4:
    st.metric(
        "CAMERAS",
        camera_count if camera_count else 1,
    )


# ============================================================
# MAIN TABS
# ============================================================

surveillance_tab, events_tab, analytics_tab = st.tabs(
    [
        "📡 SURVEILLANCE",
        "🚨 EVENT INTELLIGENCE",
        "📊 ANALYTICS",
    ]
)


# ============================================================
# SURVEILLANCE
# ============================================================

with surveillance_tab:

    st.subheader("📹 Surveillance Network")

    cam1, cam2, cam3 = st.columns(3)

    with cam1:
        with st.container(border=True):
            st.markdown("### 📹 CCTV-01")
            st.caption("Primary perimeter camera")
            st.success("● READY")

            if st.button(
                "OPEN CCTV-01",
                key="open_cctv1",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.camera = "CAM_01"
                st.session_state.fence_points = []
                st.session_state.last_click = None
                st.session_state.detection_result = None
                st.session_state.fence_version += 1
                st.rerun()

    with cam2:
        with st.container(border=True):
            st.markdown("### 📹 CCTV-02")
            st.caption("Vehicle monitoring node")
            st.info("● STANDBY")
            st.button(
                "CCTV-02 UNAVAILABLE",
                key="cam2_unavailable",
                disabled=True,
                use_container_width=True,
            )

    with cam3:
        with st.container(border=True):
            st.markdown("### 📹 CCTV-03")
            st.caption("Additional surveillance node")
            st.info("● STANDBY")
            st.button(
                "CCTV-03 UNAVAILABLE",
                key="cam3_unavailable",
                disabled=True,
                use_container_width=True,
            )


    # --------------------------------------------------------
    # CAMERA WORKFLOW
    # --------------------------------------------------------

    if st.session_state.camera is None:

        st.info(
            "Select CCTV-01 to define a restricted virtual zone "
            "and run the real YOLO intrusion detector."
        )

    else:

        if not VIDEO_PATH.exists():
            st.error(
                f"CCTV video not found:\n{VIDEO_PATH}"
            )
            st.stop()

        frame = get_first_frame(VIDEO_PATH)

        if frame is None:
            st.error(
                "The CCTV video exists but its first frame could not be read."
            )
            st.stop()

        st.divider()

        setup_left, setup_right = st.columns([2, 3])

        with setup_left:
            st.subheader("🎥 CCTV-01")
            st.success("● CAMERA ACTIVE")

        with setup_right:
            st.info(
                "Click 3–4 points on the image to create the "
                "restricted virtual fence."
            )


        # ----------------------------------------------------
        # FENCE CONTROLS
        # ----------------------------------------------------

        undo_col, reset_col, close_col, count_col = st.columns(
            [1, 1, 1, 2]
        )

        with undo_col:
            if st.button(
                "↩️ UNDO",
                use_container_width=True,
                disabled=not st.session_state.fence_points,
            ):
                st.session_state.fence_points.pop()
                st.session_state.last_click = None
                st.session_state.detection_result = None
                st.session_state.fence_version += 1
                st.rerun()

        with reset_col:
            if st.button(
                "🔄 RESET",
                use_container_width=True,
            ):
                st.session_state.fence_points = []
                st.session_state.last_click = None
                st.session_state.detection_result = None
                st.session_state.fence_version += 1
                st.rerun()

        with close_col:
            if st.button(
                "✕ CLOSE",
                use_container_width=True,
            ):
                st.session_state.camera = None
                st.session_state.fence_points = []
                st.session_state.last_click = None
                st.session_state.detection_result = None
                st.session_state.fence_version += 1
                st.rerun()

        with count_col:
            st.info(
                f"📍 Fence points: "
                f"{len(st.session_state.fence_points)} / 4"
            )


        # ----------------------------------------------------
        # INTERACTIVE FENCE IMAGE
        # ----------------------------------------------------

        preview = draw_fence(
            frame,
            st.session_state.fence_points,
        )

        clicked = streamlit_image_coordinates(
            Image.fromarray(preview),
            width=848,
            key=f"ibvap_fence_{st.session_state.fence_version}",
            cursor="crosshair",
        )


        if clicked is not None:

            x = int(clicked.get("x", 0))
            y = int(clicked.get("y", 0))

            current_click = (x, y)

            if current_click != st.session_state.last_click:

                st.session_state.last_click = current_click

                if len(st.session_state.fence_points) < 4:

                    x = max(
                        0,
                        min(
                            x,
                            frame.shape[1] - 1,
                        ),
                    )

                    y = max(
                        0,
                        min(
                            y,
                            frame.shape[0] - 1,
                        ),
                    )

                    st.session_state.fence_points.append(
                        (x, y)
                    )

                    st.rerun()


        # ----------------------------------------------------
        # FENCE STATUS
        # ----------------------------------------------------

        point_count = len(
            st.session_state.fence_points
        )

        if point_count == 0:
            st.info(
                "👆 Click Point 1 on the restricted area's first corner."
            )

        elif point_count == 1:
            st.warning(
                "Point 1 placed • Select Point 2."
            )

        elif point_count == 2:
            st.warning(
                "2 points placed • Select Point 3."
            )

        elif point_count == 3:
            st.success(
                "✅ Fence is valid. Add Point 4 for a cleaner quadrilateral."
            )

        else:
            st.success(
                "🟢 FOUR-POINT RESTRICTED ZONE READY"
            )


        # ----------------------------------------------------
        # COORDINATES
        # ----------------------------------------------------

        if st.session_state.fence_points:

            point_columns = st.columns(4)

            for index, point in enumerate(
                st.session_state.fence_points
            ):

                with point_columns[index]:
                    st.metric(
                        f"POINT {index + 1}",
                        f"{point[0]}, {point[1]}",
                    )


        # ----------------------------------------------------
        # RUN DETECTION
        # ----------------------------------------------------

        st.divider()

        ready = len(
            st.session_state.fence_points
        ) >= 3

        run_button = st.button(
            "🚨 RUN AI INTRUSION DETECTION",
            key="run_intrusion",
            type="primary",
            use_container_width=True,
            disabled=not ready,
        )

        if run_button:

            try:
                from intrusion import run_detection as detector_run
            except Exception as error:
                st.error(
                    "Could not import run_detection from intrusion.py"
                )
                st.exception(error)
                st.stop()


            if not MODEL_PATH.exists():
                st.error(
                    f"YOLO model not found:\n{MODEL_PATH}"
                )
                st.stop()


            if OUTPUT_PATH.exists():
                try:
                    OUTPUT_PATH.unlink()
                except OSError:
                    pass


            st.subheader("🤖 AI Analysis")

            progress = st.progress(
                0,
                text="Initializing YOLO vision engine...",
            )

            try:

                result = detector_run(
                    video_path=str(VIDEO_PATH),
                    fence_points=list(
                        st.session_state.fence_points
                    ),
                    output_path=str(OUTPUT_PATH),
                    camera_id="CAM_01",
                )

                progress.progress(
                    100,
                    text="AI analysis completed.",
                )

                st.session_state.detection_result = result

                st.rerun()

            except Exception as error:

                progress.empty()

                st.error(
                    "❌ Detection failed."
                )

                st.exception(error)

                st.stop()


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        if st.session_state.detection_result is not None:

            result = (
                st.session_state.detection_result
            )

            run_events = result.get(
                "events",
                []
            )

            frames_processed = result.get(
                "frames_processed",
                0
            )

            intrusion_count = result.get(
                "intrusion_events",
                len(run_events)
            )

            st.divider()

            st.subheader(
                "🎯 Mission Results"
            )

            r1, r2, r3, r4 = st.columns(4)

            with r1:
                st.metric(
                    "FRAMES ANALYSED",
                    frames_processed
                )

            with r2:
                st.metric(
                    "🚨 INTRUSIONS",
                    intrusion_count
                )

            with r3:
                st.metric(
                    "CAMERA",
                    "CAM_01"
                )

            with r4:
                st.metric(
                    "AI STATUS",
                    "THREAT"
                    if intrusion_count
                    else "SECURE"
                )


            if intrusion_count:

                st.error(
                    f"🚨 {intrusion_count} INTRUSION EVENT(S) DETECTED"
                )

            else:

                st.success(
                    "🟢 PERIMETER SECURE — NO INTRUSION DETECTED"
                )


            # ------------------------------------------------
            # PROCESSED VIDEO
            # ------------------------------------------------

            if OUTPUT_PATH.exists():

                st.subheader(
                    "🎥 Processed Surveillance Feed"
                )

                st.video(
                    str(OUTPUT_PATH)
                )


            # ------------------------------------------------
            # CURRENT RUN EVENTS
            # ------------------------------------------------

            if run_events:

                st.subheader(
                    "🚨 Intrusions From This Run"
                )

                run_rows = []

                for event in run_events:

                    confidence = event.get(
                        "confidence"
                    )

                    snapshot_value = (
                        event.get("snapshot_path")
                        or event.get("snapshot")
                        or "-"
                    )

                    run_rows.append(
                        {
                            "TIME": event.get(
                                "time",
                                "-"
                            ),
                            "OBJECT": str(
                                event.get(
                                    "category",
                                    "-"
                                )
                            ).upper(),
                            "TRACK ID": event.get(
                                "track_id",
                                "-"
                            ),
                            "CONFIDENCE": (
                                f"{float(confidence) * 100:.1f}%"
                                if confidence is not None
                                else "-"
                            ),
                            "SNAPSHOT": str(
                                snapshot_value
                            ),
                            "DATABASE": event.get(
                                "database_status",
                                "SUCCESS"
                            ),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(run_rows),
                    use_container_width=True,
                    hide_index=True,
                )


                # --------------------------------------------
                # CURRENT RUN SNAPSHOTS
                # --------------------------------------------

                st.subheader(
                    "📸 Intrusion Evidence"
                )

                current_snapshots = []

                for event in run_events:

                    snapshot = resolve_snapshot(
                        event.get(
                            "snapshot_path"
                        )
                        or event.get("snapshot")
                    )

                    if snapshot is not None:
                        current_snapshots.append(
                            (
                                event,
                                snapshot
                            )
                        )

                if current_snapshots:

                    snapshot_columns = st.columns(
                        min(3, len(current_snapshots))
                    )

                    for index, (event, image) in enumerate(
                        current_snapshots
                    ):

                        with snapshot_columns[
                            index % len(snapshot_columns)
                        ]:

                            st.image(
                                str(image),
                                caption=(
                                    f"🚨 {event.get('category', 'object')} "
                                    f"• Track {event.get('track_id', '-')}"
                                ),
                                use_container_width=True,
                            )

                else:

                    st.info(
                        "Intrusion events were recorded, but no snapshot "
                        "file could be resolved on disk."
                    )


            # ------------------------------------------------
            # DATABASE CONFIRMATION
            # ------------------------------------------------

            st.success(
                "💾 SQLite database has been updated with the intrusion event(s)."
            )

            refreshed_events = get_events()

            new_intrusions = [
                event
                for event in refreshed_events
                if str(
                    event.get("event_type", "")
                ).lower() == "intrusion"
            ]

            if new_intrusions:

                st.subheader(
                    "🗄️ Latest Database Event"
                )

                latest_db = new_intrusions[0]

                latest_confidence = latest_db.get(
                    "confidence"
                )

                latest_confidence_text = (

                    f"{float(latest_confidence) * 100:.1f}%"

                    if latest_confidence is not None

                    else "-"

                )


                evidence_left, evidence_right = st.columns(
                    [1.2, 1]
                )

                with evidence_left:

                    st.info(
                        f"""
                        Camera: {latest_db.get("camera_id", "-")}

                        Object: {latest_db.get("object_type", "-")}

                        Track ID: {latest_db.get("track_id", "-")}

                        Confidence: {latest_confidence_text}

                        Time: {format_timestamp(latest_db.get("timestamp"))}
                        """
                    )

                with evidence_right:

                    latest_image = resolve_snapshot(
                        latest_db.get(
                            "snapshot_path"
                        )
                    )

                    if latest_image:

                        st.image(
                            str(latest_image),
                            caption="Latest stored evidence",
                            use_container_width=True,
                        )


# ============================================================
# EVENT INTELLIGENCE
# ============================================================

with events_tab:

    st.subheader(
        "🚨 Event Intelligence"
    )

    current_events = get_events()

    if not current_events:

        st.info(
            "No database events have been recorded yet."
        )

    else:

        filter1, filter2, filter3 = st.columns(3)

        with filter1:

            event_filter = st.selectbox(
                "Event Type",
                [
                    "All",
                    "Intrusion",
                    "Detection",
                ],
                key="events_type_filter",
            )

        with filter2:

            camera_values = sorted(
                {
                    str(
                        event.get(
                            "camera_id"
                        )
                    )
                    for event in current_events
                    if event.get("camera_id")
                }
            )

            camera_filter = st.selectbox(
                "Camera",
                ["All"] + camera_values,
                key="events_camera_filter",
            )

        with filter3:

            object_values = sorted(
                {
                    str(
                        event.get(
                            "object_type"
                        )
                    )
                    for event in current_events
                    if event.get("object_type")
                }
            )

            object_filter = st.selectbox(
                "Object Type",
                ["All"] + object_values,
                key="events_object_filter",
            )


        filtered_events = current_events


        if event_filter != "All":

            filtered_events = [

                event
                for event in filtered_events
                if str(
                    event.get(
                        "event_type",
                        ""
                    )
                ).lower()
                == event_filter.lower()

            ]


        if camera_filter != "All":

            filtered_events = [

                event
                for event in filtered_events
                if str(
                    event.get(
                        "camera_id",
                        ""
                    )
                )
                == camera_filter

            ]


        if object_filter != "All":

            filtered_events = [

                event
                for event in filtered_events
                if str(
                    event.get(
                        "object_type",
                        ""
                    )
                )
                == object_filter

            ]


        rows = []

        for event in filtered_events[:100]:

            confidence = event.get(
                "confidence"
            )

            rows.append(
                {
                    "TIME": format_timestamp(
                        event.get(
                            "timestamp"
                        )
                    ),
                    "CAMERA": event.get(
                        "camera_id",
                        "-"
                    ),
                    "OBJECT": event.get(
                        "object_type",
                        "-"
                    ),
                    "TRACK ID": event.get(
                        "track_id",
                        "-"
                    ),
                    "EVENT": (
                        "🚨 INTRUSION"
                        if str(
                            event.get(
                                "event_type",
                                ""
                            )
                        ).lower()
                        == "intrusion"
                        else "DETECTION"
                    ),
                    "CONFIDENCE": (
                        f"{float(confidence) * 100:.1f}%"
                        if confidence is not None
                        else "-"
                    ),
                    "PLATE": event.get(
                        "plate_number"
                    ) or "-",
                }
            )


        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=500,
        )


        csv_df = pd.DataFrame(rows)

        if not csv_df.empty:

            st.download_button(
                "⬇️ DOWNLOAD FILTERED CSV",
                data=csv_df.to_csv(index=False).encode(
                    "utf-8"
                ),
                file_name="ibvap_event_log.csv",
                mime="text/csv",
                use_container_width=True,
            )


        st.divider()

        filtered_intrusions = [

            event
            for event in filtered_events
            if str(
                event.get(
                    "event_type",
                    ""
                )
            ).lower() == "intrusion"

        ]


        if filtered_intrusions:

            st.subheader(
                "📸 Intrusion Evidence Gallery"
            )

            gallery_items = []

            for event in filtered_intrusions[:9]:

                image = resolve_snapshot(
                    event.get(
                        "snapshot_path"
                    )
                )

                if image:
                    gallery_items.append(
                        (
                            event,
                            image
                        )
                    )


            if gallery_items:

                gallery_columns = st.columns(
                    min(3, len(gallery_items))
                )

                for index, (event, image) in enumerate(
                    gallery_items
                ):

                    with gallery_columns[
                        index % len(gallery_columns)
                    ]:

                        st.image(
                            str(image),
                            use_container_width=True
                        )

                        st.caption(
                            f"{event.get('object_type', 'object')} "
                            f"• Track {event.get('track_id', '-')}"
                        )


# ============================================================
# ANALYTICS
# ============================================================

with analytics_tab:

    st.subheader(
        "📊 Surveillance Analytics"
    )

    analytics_events = get_events()

    if not analytics_events:

        st.info(
            "Analytics will appear after the first detection run."
        )

    else:

        analytics_left, analytics_right = st.columns(2)


        with analytics_left:

            event_series = pd.Series(
                [
                    str(
                        event.get(
                            "event_type",
                            "unknown"
                        )
                    ).title()
                    for event in analytics_events
                ]
            ).value_counts()

            st.markdown(
                "#### Events by Type"
            )

            st.bar_chart(
                event_series
            )


        with analytics_right:

            object_series = pd.Series(
                [
                    str(
                        event.get(
                            "object_type",
                            "unknown"
                        )
                    ).title()
                    for event in analytics_events
                ]
            ).value_counts()

            st.markdown(
                "#### Objects Detected"
            )

            st.bar_chart(
                object_series
            )


        st.divider()


        camera_series = pd.Series(
            [
                str(
                    event.get(
                        "camera_id",
                        "unknown"
                    )
                )
                for event in analytics_events
            ]
        ).value_counts()


        st.markdown(
            "#### Events by Camera"
        )

        st.bar_chart(
            camera_series
        )


        st.divider()


        st.subheader(
            "🚨 Intrusion Summary"
        )

        intrusion_summary = [

            event
            for event in analytics_events
            if str(
                event.get(
                    "event_type",
                    ""
                )
            ).lower() == "intrusion"

        ]


        if intrusion_summary:

            summary_rows = []

            for event in intrusion_summary:

                confidence = event.get(
                    "confidence"
                )

                summary_rows.append(
                    {
                        "TIME": format_timestamp(
                            event.get(
                                "timestamp"
                            )
                        ),
                        "CAMERA": event.get(
                            "camera_id",
                            "-"
                        ),
                        "OBJECT": event.get(
                            "object_type",
                            "-"
                        ),
                        "TRACK ID": event.get(
                            "track_id",
                            "-"
                        ),
                        "CONFIDENCE": (
                            f"{float(confidence) * 100:.1f}%"
                            if confidence is not None
                            else "-"
                        ),
                    }
                )


            st.dataframe(
                pd.DataFrame(
                    summary_rows[:50]
                ),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.success(
                "No intrusion events in the database."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ IBVAP • Intelligent Border Virtual AI Protection • "
    "YOLO Vision Engine • Virtual Perimeter • SQLite Evidence System"
)
