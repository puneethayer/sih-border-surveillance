import sqlite3
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates


# ============================================================
# IBVAP COMMAND CENTER
# ============================================================

st.set_page_config(
    page_title="IBVAP | Border AI Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

VIDEO = ROOT / "cctv.mp4"
MODEL = ROOT / "yolo26n.pt"
OUTPUT = ROOT / "intrusion_result.mp4"

DB = ROOT / "database" / "ibvap.db"
SNAPSHOTS = ROOT / "logs" / "snapshots"


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(0, 180, 255, 0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(0, 255, 170, 0.06),
            transparent 25%
        ),
        #070b12;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* ================= HEADER ================= */

.hero {
    padding: 26px 30px;
    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            rgba(18,31,45,.98),
            rgba(7,13,22,.98)
        );

    border: 1px solid rgba(80,160,200,.22);

    box-shadow:
        0 20px 60px rgba(0,0,0,.30);

    margin-bottom: 20px;
}

.brand {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: .5px;
}

.brand span {
    color: #39d9ff;
}

.subtitle {
    color: #8499ad;
    font-size: 13px;
    margin-top: 6px;
}

.online {
    text-align: right;
    color: #42e8a5;
    font-size: 14px;
    font-weight: 800;
}

.online small {
    display: block;
    color: #70869a;
    font-weight: 500;
    margin-top: 5px;
}


/* ================= METRIC CARDS ================= */

.card {
    background:
        linear-gradient(
            145deg,
            #101a25,
            #0b1119
        );

    border: 1px solid rgba(120,160,190,.15);

    border-radius: 16px;

    padding: 18px;

    min-height: 105px;
}

.card-title {
    color: #7f94a8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}

.card-value {
    color: #edf7ff;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.blue {
    color: #39d9ff;
}

.green {
    color: #42e8a5;
}

.red {
    color: #ff5f67;
}


/* ================= CAMERA ================= */

.camera {
    border: 1px solid rgba(57,217,255,.25);

    border-radius: 16px;

    padding: 17px;

    background:
        linear-gradient(
            145deg,
            #0e1924,
            #091019
        );

    margin-bottom: 10px;
}

.camera-title {
    font-size: 16px;
    font-weight: 800;
}

.camera-status {
    color: #42e8a5;
    font-size: 12px;
    font-weight: 700;
}


/* ================= INSTRUCTIONS ================= */

.instruction {
    border-left: 3px solid #39d9ff;

    padding: 13px 16px;

    background: rgba(57,217,255,.045);

    border-radius: 8px;

    color: #a8bdce;

    font-size: 13px;

    margin: 10px 0 16px;
}


/* ================= ALERTS ================= */

.alert-box {
    border: 1px solid rgba(255,80,90,.45);

    border-radius: 16px;

    padding: 20px;

    background:
        linear-gradient(
            135deg,
            rgba(80,15,25,.60),
            rgba(35,10,15,.55)
        );

    box-shadow:
        0 0 35px rgba(255,60,70,.08);

    margin: 15px 0;
}

.secure-box {
    border: 1px solid rgba(66,232,165,.28);

    border-radius: 16px;

    padding: 20px;

    background:
        rgba(12,42,34,.30);

    margin: 15px 0;
}


/* ================= BUTTONS ================= */

div.stButton > button {
    border-radius: 10px;

    min-height: 43px;

    font-weight: 700;

    border: 1px solid rgba(120,160,190,.15);
}

button[kind="primary"] {
    background:
        linear-gradient(
            90deg,
            #087ea3,
            #13a9c9
        );

    border: none;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;

    color: #52677a;

    font-size: 11px;

    margin-top: 40px;

    line-height: 1.7;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "camera" not in st.session_state:
    st.session_state.camera = None

if "points" not in st.session_state:
    st.session_state.points = []

if "last_click" not in st.session_state:
    st.session_state.last_click = None

if "fence_version" not in st.session_state:
    st.session_state.fence_version = 0

if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# DATABASE
# ============================================================

def get_events():

    if not DB.exists():
        return []

    conn = None

    try:

        conn = sqlite3.connect(str(DB))

        conn.row_factory = sqlite3.Row

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

    except sqlite3.Error:
        return []

    finally:

        if conn:
            conn.close()


def format_time(value):

    if not value:
        return "-"

    try:

        return datetime.fromisoformat(
            str(value)
        ).strftime(
            "%d %b %Y • %H:%M:%S"
        )

    except Exception:

        return str(value)


# ============================================================
# SNAPSHOT PATH
# ============================================================

def find_snapshot(value):

    if not value:
        return None

    path = Path(str(value))

    if path.is_absolute() and path.exists():
        return path

    candidates = [

        ROOT / path,

        ROOT / str(value).replace("\\", "/"),

        SNAPSHOTS / path.name

    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    return None


# ============================================================
# FIRST VIDEO FRAME
# ============================================================

@st.cache_data
def get_first_frame(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        return None

    success, frame = cap.read()

    cap.release()

    if not success:
        return None

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# DRAW FENCE
# ============================================================

def draw_fence(frame, points):

    image = frame.copy()

    # Restricted area
    if len(points) >= 3:

        polygon = np.array(
            points,
            dtype=np.int32
        )

        overlay = image.copy()

        cv2.fillPoly(
            overlay,
            [polygon],
            (255, 90, 0)
        )

        image = cv2.addWeighted(
            overlay,
            0.18,
            image,
            0.82,
            0
        )

    # Lines
    if len(points) >= 2:

        for i in range(
            len(points) - 1
        ):

            cv2.line(
                image,
                points[i],
                points[i + 1],
                (255, 170, 30),
                4
            )

    # Close polygon
    if len(points) >= 3:

        cv2.line(
            image,
            points[-1],
            points[0],
            (255, 170, 30),
            4
        )

    # Points
    for i, (x, y) in enumerate(points):

        cv2.circle(
            image,
            (x, y),
            10,
            (255, 170, 30),
            -1
        )

        cv2.circle(
            image,
            (x, y),
            12,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"P{i + 1}",
            (x + 13, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            f"P{i + 1}",
            (x + 13, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 170, 30),
            1,
            cv2.LINE_AA
        )

    return image


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="brand">
            🛡️ <span>IBVAP</span> COMMAND CENTER
        </div>

        <div class="subtitle">
            Intelligent Border Virtual AI Protection
            • AI-Powered Intrusion Surveillance
        </div>

        <div class="online">
            ● SYSTEM ONLINE
            <small>
                YOLO Vision • Virtual Fence • SQLite
            </small>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATABASE STATUS
# ============================================================

events = get_events()

intrusions = [
    event
    for event in events
    if event.get("event_type") == "intrusion"
]

detections = [
    event
    for event in events
    if event.get("event_type") == "detection"
]


# ============================================================
# TOP METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)


with m1:

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                SYSTEM STATUS
            </div>

            <div class="card-value green">
                ONLINE
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m2:

    threat_class = (
        "red"
        if intrusions
        else "green"
    )

    st.markdown(
        f"""
        <div class="card">

            <div class="card-title">
                ACTIVE THREATS
            </div>

            <div class="card-value {threat_class}">
                {len(intrusions)}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m3:

    st.markdown(
        f"""
        <div class="card">

            <div class="card-title">
                AI EVENTS
            </div>

            <div class="card-value blue">
                {len(events)}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m4:

    db_status = (
        "CONNECTED"
        if DB.exists()
        else "WAITING"
    )

    db_class = (
        "green"
        if DB.exists()
        else "red"
    )

    st.markdown(
        f"""
        <div class="card">

            <div class="card-title">
                DATABASE
            </div>

            <div class="card-value {db_class}">
                {db_status}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SURVEILLANCE NETWORK
# ============================================================

st.markdown(
    """
    <div style="margin-top:25px">

        <h2>📡 Surveillance Network</h2>

        <p style="color:#8196aa;font-size:13px">
            Select a camera to begin AI-assisted perimeter inspection.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


c1, c2, c3 = st.columns(3)


with c1:

    st.markdown(
        """
        <div class="camera">

            <div class="camera-title">
                📹 CCTV-01
            </div>

            <div class="camera-status">
                ● READY
            </div>

            <div style="color:#71869a;font-size:12px;margin-top:5px">
                Primary perimeter camera
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "OPEN CCTV-01",
        key="open_cctv1",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.camera = "CAM_01"

        st.session_state.points = []

        st.session_state.last_click = None

        st.session_state.result = None

        st.session_state.fence_version += 1

        st.rerun()


with c2:

    st.markdown(
        """
        <div class="camera">

            <div class="camera-title">
                📹 CCTV-02
            </div>

            <div style="color:#65798a;font-size:12px;font-weight:700">
                ● STANDBY
            </div>

            <div style="color:#71869a;font-size:12px;margin-top:5px">
                Vehicle monitoring node
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        """
        <div class="camera">

            <div class="camera-title">
                📹 CCTV-03
            </div>

            <div style="color:#65798a;font-size:12px;font-weight:700">
                ● STANDBY
            </div>

            <div style="color:#71869a;font-size:12px;margin-top:5px">
                Additional surveillance node
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CAMERA NOT SELECTED
# ============================================================

if st.session_state.camera is None:

    st.markdown(
        """
        <div class="secure-box">

            <h3 style="color:#42e8a5;margin-top:0">
                🟢 PERIMETER STATUS: READY
            </h3>

            <span style="color:#8196aa">
                Select CCTV-01 to configure a virtual
                restricted zone and start AI detection.
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer">
            🛡️ IBVAP • INTELLIGENT BORDER VIRTUAL AI PROTECTION
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# CCTV SETUP
# ============================================================

if not VIDEO.exists():

    st.error(
        f"CCTV video not found:\n{VIDEO}"
    )

    st.stop()


frame = get_first_frame(VIDEO)


if frame is None:

    st.error(
        "Unable to read cctv.mp4."
    )

    st.stop()


st.markdown(
    """
    <div style="margin-top:25px">

        <h2>
            🎥 CCTV-01
            <span style="color:#42e8a5;font-size:14px">
                ● LIVE
            </span>
        </h2>

        <p style="color:#8196aa;font-size:13px">
            Virtual perimeter configuration
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INSTRUCTIONS
# ============================================================

st.markdown(
    """
    <div class="instruction">

        <b style="color:#39d9ff">
            VIRTUAL FENCE SETUP
        </b>
        <br><br>

        ① Click the corners of the restricted area
        &nbsp;&nbsp; • &nbsp;&nbsp;

        ② Select 3–4 points
        &nbsp;&nbsp; • &nbsp;&nbsp;

        ③ Use UNDO if needed
        &nbsp;&nbsp; • &nbsp;&nbsp;

        ④ Run AI Detection

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FENCE CONTROLS
# ============================================================

b1, b2, b3, b4 = st.columns(
    [1, 1, 1, 2]
)


with b1:

    if st.button(
        "↩️ UNDO",
        use_container_width=True,
        disabled=not st.session_state.points
    ):

        st.session_state.points.pop()

        st.session_state.last_click = None

        st.session_state.result = None

        st.session_state.fence_version += 1

        st.rerun()


with b2:

    if st.button(
        "🔄 RESET",
        use_container_width=True
    ):

        st.session_state.points = []

        st.session_state.last_click = None

        st.session_state.result = None

        st.session_state.fence_version += 1

        st.rerun()


with b3:

    if st.button(
        "✕ CLOSE",
        use_container_width=True
    ):

        st.session_state.camera = None

        st.session_state.points = []

        st.session_state.last_click = None

        st.session_state.result = None

        st.session_state.fence_version += 1

        st.rerun()


with b4:

    st.info(
        f"📍 FENCE POINTS: "
        f"{len(st.session_state.points)} / 4"
    )


# ============================================================
# INTERACTIVE IMAGE
# ============================================================

shown = draw_fence(
    frame,
    st.session_state.points
)


clicked = streamlit_image_coordinates(
    Image.fromarray(shown),
    width=848,
    key=f"ibvap_fence_{st.session_state.fence_version}",
    cursor="crosshair"
)


# ============================================================
# CLICK HANDLER
# ============================================================

if clicked is not None:

    x = int(
        clicked.get(
            "x",
            0
        )
    )

    y = int(
        clicked.get(
            "y",
            0
        )
    )

    current_click = (x, y)

    if current_click != st.session_state.last_click:

        st.session_state.last_click = current_click

        if len(st.session_state.points) < 4:

            x = max(
                0,
                min(
                    x,
                    frame.shape[1] - 1
                )
            )

            y = max(
                0,
                min(
                    y,
                    frame.shape[0] - 1
                )
            )

            st.session_state.points.append(
                (x, y)
            )

            st.rerun()


# ============================================================
# FENCE STATUS
# ============================================================

point_count = len(
    st.session_state.points
)


if point_count == 0:

    st.info(
        "👆 Waiting for Point 1..."
    )


elif point_count == 1:

    st.warning(
        "Point 1 placed • Select Point 2"
    )


elif point_count == 2:

    st.warning(
        "2 points placed • Select Point 3"
    )


elif point_count == 3:

    st.success(
        "✅ 3 points placed • Add Point 4 for a clean quadrilateral"
    )


else:

    st.success(
        "🟢 RESTRICTED ZONE LOCKED • 4-point fence ready"
    )


# ============================================================
# POINT COORDINATES
# ============================================================

if st.session_state.points:

    cols = st.columns(4)

    for i, point in enumerate(
        st.session_state.points
    ):

        with cols[i]:

            st.metric(
                f"POINT {i + 1}",
                f"{point[0]}, {point[1]}"
            )


# ============================================================
# RUN AI DETECTION
# ============================================================

st.write("")


run_button = st.button(
    "🚨  RUN AI INTRUSION DETECTION",
    use_container_width=True,
    type="primary",
    disabled=len(st.session_state.points) < 3
)


if run_button:

    try:

        from intrusion import run_detection

    except Exception as error:

        st.error(
            "Could not import intrusion.py"
        )

        st.exception(error)

        st.stop()


    if not MODEL.exists():

        st.error(
            f"YOLO model not found:\n{MODEL}"
        )

        st.stop()


    # Remove old output
    if OUTPUT.exists():

        try:
            OUTPUT.unlink()
        except Exception:
            pass


    st.markdown(
        """
        <div style="margin-top:25px">

            <h2>🤖 AI Analysis</h2>

            <p style="color:#8196aa">
                YOLO is analysing CCTV-01 against
                the selected virtual perimeter.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    progress = st.progress(0)

    status = st.empty()

    status.info(
        "🔄 Initializing YOLO vision engine..."
    )


    try:

        result = run_detection(

            video_path=str(VIDEO),

            fence_points=st.session_state.points,

            output_path=str(OUTPUT),

            camera_id="CAM_01"

        )


        progress.progress(100)

        status.success(
            "✅ AI analysis completed successfully."
        )

        st.session_state.result = result

        st.rerun()


    except Exception as error:

        progress.empty()

        status.error(
            "❌ Detection failed."
        )

        st.exception(error)

        st.stop()


# ============================================================
# RESULTS
# ============================================================

if st.session_state.result is not None:

    result = st.session_state.result

    result_events = result.get(
        "events",
        []
    )

    frames = result.get(
        "frames_processed",
        0
    )

    intrusion_count = result.get(
        "intrusion_events",
        len(result_events)
    )


    st.markdown("---")


    st.markdown(
        """
        <div style="margin-top:10px">

            <h2>🎯 Mission Results</h2>

            <p style="color:#8196aa">
                AI analysis output from CCTV-01.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    r1, r2, r3, r4 = st.columns(4)


    with r1:

        st.metric(
            "FRAMES ANALYSED",
            frames
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


    # ========================================================
    # BIG STATUS
    # ========================================================

    if intrusion_count:

        st.markdown(
            f"""
            <div class="alert-box">

                <h2 style="color:#ff6670;margin-top:0">
                    🚨 INTRUSION DETECTED
                </h2>

                <p style="color:#d8aeb2">
                    <b>{intrusion_count}</b>
                    intrusion event(s) detected by CCTV-01.
                    Evidence has been stored in SQLite.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="secure-box">

                <h2 style="color:#42e8a5;margin-top:0">
                    🟢 PERIMETER SECURE
                </h2>

                <p style="color:#91b8aa">
                    No intrusion event was detected
                    for the selected virtual zone.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROCESSED VIDEO
    # ========================================================

    if OUTPUT.exists():

        st.markdown(
            "### 🎥 Processed Surveillance Feed"
        )

        st.video(
            OUTPUT.read_bytes()
        )


    # ========================================================
    # CURRENT RUN EVENTS
    # ========================================================

    if result_events:

        st.markdown(
            "### 🚨 Detected Threats"
        )


        rows = []


        for event in result_events:

            confidence = event.get(
                "confidence"
            )


            rows.append({

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

                "CENTROID": str(
                    event.get(
                        "centroid",
                        "-"
                    )
                )

            })


        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# EVIDENCE
# ============================================================

all_events = get_events()

db_intrusions = [

    event
    for event in all_events
    if event.get("event_type") == "intrusion"

]


st.markdown("---")


st.markdown(
    """
    <div>

        <h2>🗄️ Evidence & Event Intelligence</h2>

        <p style="color:#8196aa">
            Recorded intrusion events from the IBVAP SQLite database.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LATEST THREAT
# ============================================================

if db_intrusions:

    latest = db_intrusions[0]

    confidence = latest.get(
        "confidence"
    )

    confidence_text = (

        f"{float(confidence) * 100:.1f}%"

        if confidence is not None

        else "N/A"

    )


    left, right = st.columns(
        [1.4, 1]
    )


    with left:

        st.markdown(
            f"""
            <div class="alert-box">

                <h3 style="color:#ff6670;margin-top:0">
                    🚨 LATEST THREAT
                </h3>

                <b>Camera:</b>
                {latest.get("camera_id", "-")}
                <br><br>

                <b>Object:</b>
                {latest.get("object_type", "-")}
                <br><br>

                <b>Track ID:</b>
                {latest.get("track_id", "-")}
                <br><br>

                <b>Confidence:</b>
                {confidence_text}
                <br><br>

                <b>Time:</b>
                {format_time(latest.get("timestamp"))}

            </div>
            """,
            unsafe_allow_html=True
        )


    with right:

        image = find_snapshot(
            latest.get(
                "snapshot_path"
            )
        )


        if image:

            st.image(
                str(image),
                caption="Captured intrusion evidence",
                use_container_width=True
            )

        else:

            st.info(
                "No snapshot available."
            )


# ============================================================
# EVENT LOG
# ============================================================

if all_events:

    st.markdown(
        "### 📋 Event Log"
    )


    table = []


    for event in all_events[:25]:

        confidence = event.get(
            "confidence"
        )


        table.append({

            "TIME": format_time(
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

            "TRACK": event.get(
                "track_id",
                "-"
            ),

            "EVENT": (

                "🚨 INTRUSION"

                if event.get(
                    "event_type"
                ) == "intrusion"

                else "DETECTION"

            ),

            "CONFIDENCE": (

                f"{float(confidence) * 100:.1f}%"

                if confidence is not None

                else "-"

            ),

            "PLATE": event.get(
                "plate_number"
            ) or "-"

        })


    st.dataframe(
        pd.DataFrame(table),
        use_container_width=True,
        hide_index=True,
        height=430
    )


# ============================================================
# SNAPSHOT GALLERY
# ============================================================

valid_snapshots = []


for event in db_intrusions[:9]:

    image = find_snapshot(
        event.get(
            "snapshot_path"
        )
    )


    if image:

        valid_snapshots.append(
            (
                event,
                image
            )
        )


if valid_snapshots:

    st.markdown(
        "### 📸 Evidence Gallery"
    )


    gallery = st.columns(3)


    for i, (event, image) in enumerate(
        valid_snapshots
    ):

        with gallery[i % 3]:

            st.image(
                str(image),
                use_container_width=True
            )

            st.caption(
                f"🚨 {event.get('object_type', 'object')} "
                f"• Track {event.get('track_id', '-')} "
                f"• {format_time(event.get('timestamp'))}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🛡️ <b>IBVAP</b> • INTELLIGENT BORDER VIRTUAL AI PROTECTION
        <br>

        YOLO Vision Engine • Virtual Perimeter • SQLite Evidence System

    </div>
    """,
    unsafe_allow_html=True
)