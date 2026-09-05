import os
import sqlite3
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IBVAP Detection Dashboard",
    page_icon="🚨",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "database" / "ibvap.db"

LOGS_DIR = PROJECT_ROOT / "logs"
SNAPSHOTS_DIR = LOGS_DIR / "snapshots"


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_connection():
    """Connect to the project's SQLite database."""
    return sqlite3.connect(str(DB_PATH))


def get_all_events():
    """Get all events from SQLite."""
    if not DB_PATH.exists():
        return []

    conn = get_connection()
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
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

    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return []

    finally:
        conn.close()


def get_intrusion_events():
    """Get only intrusion events."""
    if not DB_PATH.exists():
        return []

    conn = get_connection()
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
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
            WHERE event_type = 'intrusion'
            ORDER BY timestamp DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]

    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return []

    finally:
        conn.close()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

VEHICLE_TYPES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "vehicle"
}


def get_category(object_type):
    """Convert database object type into dashboard category."""

    if not object_type:
        return "unknown"

    object_type = str(object_type).lower()

    if object_type == "person":
        return "person"

    if object_type in VEHICLE_TYPES:
        return "vehicle"

    return object_type


def get_severity(event):
    """Calculate dashboard severity."""

    if event.get("event_type") == "intrusion":
        return "high"

    confidence = event.get("confidence")

    if confidence is None:
        return "low"

    if confidence >= 0.85:
        return "high"

    if confidence >= 0.70:
        return "medium"

    return "low"


def get_label(event):
    """Create readable object label."""

    object_type = str(event.get("object_type") or "unknown")
    track_id = event.get("track_id")

    if object_type.lower() == "person":
        if track_id is not None:
            return f"person {track_id}"
        return "person"

    return object_type


def resolve_snapshot_path(snapshot_path):
    """
    Resolve snapshot path stored in database.

    Handles both:
        C:\\Users\\...\\logs\\snapshots\\image.jpg

    and:
        logs/snapshots/image.jpg
    """

    if not snapshot_path:
        return None

    path = Path(str(snapshot_path))

    # Absolute path
    if path.is_absolute() and path.exists():
        return path

    # Path relative to project root
    candidate = PROJECT_ROOT / path

    if candidate.exists():
        return candidate

    # Try converting Windows backslashes
    cleaned = str(snapshot_path).replace("\\", "/")
    candidate = PROJECT_ROOT / cleaned

    if candidate.exists():
        return candidate

    # Try only filename inside logs/snapshots
    filename = Path(cleaned).name
    candidate = SNAPSHOTS_DIR / filename

    if candidate.exists():
        return candidate

    return None


def format_timestamp(timestamp):
    """Make timestamp easier to read."""

    if not timestamp:
        return "-"

    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(timestamp)


# ============================================================
# LOAD DATABASE
# ============================================================

if not DB_PATH.exists():

    st.error(
        f"""
        ❌ Database not found.

        Expected database:

        `{DB_PATH}`

        Make sure your project looks like:

        CCTV-YOLO/
        ├── app.py
        ├── intrusion.py
        └── database/
            └── ibvap.db
        """
    )

    st.stop()


all_events = get_all_events()
intrusion_events = get_intrusion_events()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎛️ Controls")

st.sidebar.success("🟢 Database Connected")

st.sidebar.caption(
    f"SQLite database:\n{DB_PATH}"
)

st.sidebar.markdown("---")


# Auto refresh
auto_refresh = st.sidebar.checkbox(
    "🔄 Auto refresh",
    value=True
)

refresh_seconds = st.sidebar.slider(
    "Refresh interval (seconds)",
    1,
    10,
    3
)


st.sidebar.markdown("---")


# Event filter
event_filter = st.sidebar.selectbox(
    "Event type",
    [
        "All events",
        "Intrusion only",
        "Detection only"
    ]
)


# Camera filter
cameras = sorted(
    list(
        set(
            str(e.get("camera_id"))
            for e in all_events
            if e.get("camera_id") is not None
        )
    )
)

camera_filter = st.sidebar.selectbox(
    "Camera",
    ["All cameras"] + cameras
)


# Object filter
object_types = sorted(
    list(
        set(
            str(e.get("object_type"))
            for e in all_events
            if e.get("object_type") is not None
        )
    )
)

object_filter = st.sidebar.selectbox(
    "Object type",
    ["All objects"] + object_types
)


st.sidebar.markdown("---")


if st.sidebar.button("🔄 Refresh now", use_container_width=True):
    st.rerun()


# ============================================================
# FILTER EVENTS
# ============================================================

filtered_events = all_events.copy()


if event_filter == "Intrusion only":
    filtered_events = [
        e for e in filtered_events
        if e.get("event_type") == "intrusion"
    ]

elif event_filter == "Detection only":
    filtered_events = [
        e for e in filtered_events
        if e.get("event_type") == "detection"
    ]


if camera_filter != "All cameras":
    filtered_events = [
        e for e in filtered_events
        if str(e.get("camera_id")) == camera_filter
    ]


if object_filter != "All objects":
    filtered_events = [
        e for e in filtered_events
        if str(e.get("object_type")) == object_filter
    ]


# ============================================================
# HEADER
# ============================================================

st.title("🚨 IBVAP Detection Dashboard")

st.caption(
    "Real intrusion events from YOLO detection → SQLite database → dashboard"
)


# ============================================================
# STATUS
# ============================================================

if all_events:

    latest_event = all_events[0]

    st.success(
        f"🟢 Database online • "
        f"Latest event: {format_timestamp(latest_event.get('timestamp'))}"
    )

else:

    st.warning(
        "⚠️ Database connected, but no events have been recorded yet."
    )


# ============================================================
# TOP METRICS
# ============================================================

total_events = len(all_events)

total_intrusions = len(intrusion_events)

total_detections = sum(
    1
    for e in all_events
    if e.get("event_type") == "detection"
)

unique_cameras = len(
    set(
        str(e.get("camera_id"))
        for e in all_events
        if e.get("camera_id") is not None
    )
)


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Events",
        total_events
    )


with col2:
    st.metric(
        "🚨 Intrusions",
        total_intrusions
    )


with col3:
    st.metric(
        "Detections",
        total_detections
    )


with col4:
    st.metric(
        "Cameras",
        unique_cameras
    )


st.markdown("---")


# ============================================================
# TABS
# ============================================================

tab_live, tab_intrusions, tab_analytics, tab_snapshot = st.tabs(
    [
        "🟢 Live Events",
        "🚨 Intrusions",
        "📊 Analytics",
        "📸 Snapshots"
    ]
)


# ============================================================
# LIVE EVENTS TAB
# ============================================================

with tab_live:

    st.subheader("Live Database Events")

    if filtered_events:

        table_data = []

        for event in filtered_events:

            confidence = event.get("confidence")

            if confidence is not None:
                confidence_display = f"{confidence * 100:.1f}%"
            else:
                confidence_display = "-"

            table_data.append(
                {
                    "Time": format_timestamp(
                        event.get("timestamp")
                    ),

                    "Camera": event.get(
                        "camera_id"
                    ),

                    "Object": event.get(
                        "object_type"
                    ),

                    "Track ID": event.get(
                        "track_id"
                    ),

                    "Event": (
                        "🚨 INTRUSION"
                        if event.get("event_type") == "intrusion"
                        else "Detection"
                    ),

                    "Confidence": confidence_display,

                    "Severity": get_severity(event),

                    "Plate": event.get(
                        "plate_number"
                    ) or "-"
                }
            )

        df = pd.DataFrame(table_data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=500
        )

    else:

        st.info(
            "No events match the current filters."
        )


# ============================================================
# INTRUSION TAB
# ============================================================

with tab_intrusions:

    st.subheader("🚨 Real Intrusion Alerts")

    if intrusion_events:

        for index, event in enumerate(intrusion_events[:20]):

            confidence = event.get("confidence")

            if confidence is not None:
                confidence_percent = confidence * 100
                confidence_text = f"{confidence_percent:.1f}%"
            else:
                confidence_text = "N/A"

            timestamp = format_timestamp(
                event.get("timestamp")
            )

            camera = event.get(
                "camera_id",
                "Unknown"
            )

            object_type = event.get(
                "object_type",
                "Unknown"
            )

            track_id = event.get(
                "track_id",
                "-"
            )

            snapshot_path = resolve_snapshot_path(
                event.get("snapshot_path")
            )

            with st.container(border=True):

                left, middle, right = st.columns(
                    [2, 2, 1]
                )

                with left:

                    st.markdown(
                        "### 🚨 INTRUSION DETECTED"
                    )

                    st.write(
                        f"**Camera:** {camera}"
                    )

                    st.write(
                        f"**Object:** {object_type}"
                    )

                    st.write(
                        f"**Track ID:** {track_id}"
                    )

                with middle:

                    st.write(
                        f"**Time:** {timestamp}"
                    )

                    st.write(
                        f"**Confidence:** {confidence_text}"
                    )

                    st.write(
                        "**Severity:** 🔴 HIGH"
                    )

                with right:

                    if snapshot_path:

                        st.image(
                            str(snapshot_path),
                            caption="Intrusion snapshot",
                            width=180
                        )

                    else:

                        st.caption(
                            "No snapshot available"
                        )

    else:

        st.info(
            "No intrusion events found in the database."
        )


# ============================================================
# ANALYTICS TAB
# ============================================================

with tab_analytics:

    st.subheader("📊 Detection Analytics")

    if all_events:

        analytics_df = pd.DataFrame(
            all_events
        )

        # ----------------------------------------------------
        # Event type
        # ----------------------------------------------------

        st.markdown("### Events by Type")

        event_counts = (
            analytics_df["event_type"]
            .value_counts()
        )

        st.bar_chart(
            event_counts
        )

        # ----------------------------------------------------
        # Object types
        # ----------------------------------------------------

        st.markdown("### Objects Detected")

        object_counts = (
            analytics_df["object_type"]
            .value_counts()
        )

        st.bar_chart(
            object_counts
        )

        # ----------------------------------------------------
        # Camera activity
        # ----------------------------------------------------

        st.markdown("### Events by Camera")

        camera_counts = (
            analytics_df["camera_id"]
            .value_counts()
        )

        st.bar_chart(
            camera_counts
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        st.markdown("### Confidence")

        confidence_df = analytics_df[
            analytics_df["confidence"].notna()
        ].copy()

        if not confidence_df.empty:

            confidence_df["confidence_percent"] = (
                confidence_df["confidence"] * 100
            )

            st.line_chart(
                confidence_df[
                    "confidence_percent"
                ].reset_index(drop=True)
            )

        else:

            st.info(
                "No confidence data available."
            )

    else:

        st.info(
            "No database events available for analytics."
        )


# ============================================================
# SNAPSHOTS TAB
# ============================================================

with tab_snapshot:

    st.subheader("📸 Intrusion Snapshots")

    snapshot_events = [
        e for e in intrusion_events
        if resolve_snapshot_path(
            e.get("snapshot_path")
        )
    ]

    if snapshot_events:

        cols = st.columns(3)

        for index, event in enumerate(
            snapshot_events[:30]
        ):

            snapshot_path = resolve_snapshot_path(
                event.get("snapshot_path")
            )

            if not snapshot_path:
                continue

            with cols[index % 3]:

                st.image(
                    str(snapshot_path),
                    use_container_width=True
                )

                st.caption(
                    f"🚨 {event.get('object_type')} | "
                    f"Track {event.get('track_id')} | "
                    f"{format_timestamp(event.get('timestamp'))}"
                )

    else:

        st.info(
            "No intrusion snapshots found."
        )


# ============================================================
# DATABASE INFORMATION
# ============================================================

st.markdown("---")

with st.expander("🗄️ Database Information"):

    st.write(
        f"**Database:** `{DB_PATH}`"
    )

    st.write(
        f"**Total rows:** {len(all_events)}"
    )

    st.write(
        f"**Intrusion rows:** {len(intrusion_events)}"
    )

    if DB_PATH.exists():

        size_kb = DB_PATH.stat().st_size / 1024

        st.write(
            f"**Database size:** {size_kb:.2f} KB"
        )


# ============================================================
# AUTO REFRESH
# ============================================================

if auto_refresh:

    st.markdown(
        f"""
        <meta http-equiv="refresh"
              content="{refresh_seconds}">
        """,
        unsafe_allow_html=True
    )

    st.caption(
        f"🔄 Dashboard automatically refreshes "
        f"every {refresh_seconds} seconds."
    )