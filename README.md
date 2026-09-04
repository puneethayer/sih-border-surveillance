# 🛡️ IBVAP — Intelligent Border Virtual AI Protection

### AI-Powered CCTV Intrusion Detection & Virtual Perimeter Surveillance System

IBVAP (**Intelligent Border Virtual AI Protection**) is an AI-assisted CCTV surveillance prototype designed for **border security, Border Out Posts (BOPs), check posts, border roads, restricted areas, and other sensitive locations**.

The system uses **YOLO-based object detection and tracking** together with a user-defined **virtual fence/perimeter** to identify when detected objects enter a restricted area.

When an intrusion is detected, IBVAP records the event, captures an evidence snapshot, stores the event in SQLite, maintains a CSV log, and displays the results through a browser-based surveillance dashboard.

---

# 📌 Problem Statement

Border security locations require continuous monitoring of CCTV feeds. Manual monitoring of multiple cameras for long periods is:

* Fatigue-prone
* Difficult to scale
* Time-consuming
* Dependent on constant human attention
* Vulnerable to missed intrusion events

IBVAP provides an **AI-assisted first layer of surveillance** that automatically analyzes CCTV footage and identifies potential intrusions into predefined restricted zones.

The system is intended to assist human security personnel rather than completely replace them.

---

# 🎯 Objectives

The main objectives of IBVAP are:

* Automate the first level of CCTV monitoring.
* Detect relevant objects in surveillance footage.
* Track detected objects across video frames.
* Define restricted areas using a virtual fence.
* Detect when an object enters the restricted zone.
* Generate intrusion events.
* Capture visual evidence of detected intrusions.
* Store intrusion information in a database.
* Maintain a CSV event log.
* Display surveillance results through a web-based dashboard.
* Provide event history and analytics for monitoring.

---

# 🧠 Core Concept

The fundamental workflow of IBVAP is:

```text
                 CCTV / VIDEO INPUT
                         │
                         ▼
              ┌─────────────────────┐
              │   OpenCV Processing │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   YOLO Detection    │
              │     + Tracking      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Virtual Fence     │
              │   / Restricted Zone │
              └──────────┬──────────┘
                         │
                         ▼
                 Object in Zone?
                    /          \
                  YES           NO
                   │             │
                   ▼             ▼
            INTRUSION EVENT    Continue
                   │
          ┌────────┼───────────┐
          ▼        ▼           ▼
       Snapshot  SQLite       CSV
          │        │           │
          └────────┼───────────┘
                   │
                   ▼
            Streamlit Dashboard
                   │
          ┌────────┼──────────┐
          ▼        ▼          ▼
      Live View  Event Log  Analytics
```

---

# 🚀 Features Implemented

## 1. AI-Based Object Detection

IBVAP uses the **Ultralytics YOLO** framework for object detection.

The current detection configuration focuses on relevant COCO classes:

| Class ID | Object |
| -------: | ------ |
|        0 | Person |
|        2 | Car    |
|        5 | Bus    |
|        7 | Truck  |

The detection engine loads the YOLO model and processes the CCTV video frame-by-frame.

---

## 2. Object Tracking

The system uses YOLO tracking with:

```text
ByteTrack
```

Tracking allows objects to maintain a **Track ID** across frames.

This makes it possible to determine whether the same detected object is entering or remaining within the restricted area instead of treating every frame as a completely new detection.

---

## 3. Virtual Fence / Restricted Zone

The operator can define a restricted area directly on the CCTV frame.

The virtual fence is represented as a polygon using selected `(x, y)` points.

At least **three points** are required to create a valid fence.

The interface displays the selected points and visually draws the virtual perimeter over the CCTV image.

---

## 4. Intrusion Detection

The system checks tracked objects against the virtual fence.

When an object satisfies the intrusion condition, IBVAP:

* Identifies the object
* Records its Track ID
* Records its category
* Records confidence
* Records its position/centroid
* Generates an intrusion event
* Captures a snapshot
* Stores the event in SQLite
* Adds the event to the CSV log

The detection engine also maintains active intrusion tracking to avoid treating every frame as a separate active intrusion.

---

# 📸 Evidence Capture

When an intrusion occurs, IBVAP captures a snapshot of the relevant frame.

The snapshot is stored inside:

```text
logs/
└── snapshots/
```

The snapshot path is associated with the corresponding database event so that the dashboard can retrieve and display the evidence.

The dashboard can display the captured intrusion snapshot with the caption:

```text
Captured intrusion snapshot
```

---

# 🗄️ Database

IBVAP uses:

```text
SQLite
```

for local event storage.

The database is located at:

```text
database/ibvap.db
```

The event system stores information including:

* Event ID
* Timestamp
* Camera ID
* Object type
* Track ID
* Event type
* Confidence
* Snapshot path
* Plate number field where applicable

The Streamlit application retrieves events from the `events` table and displays them in the dashboard.

---

# 📄 CSV Event Logging

In addition to SQLite, IBVAP maintains a CSV event log.

The CSV contains fields including:

```text
Date
Time
Object ID
Category
Event
Confidence
Centroid X
Centroid Y
Snapshot
```

This provides a simple portable record of intrusion events for inspection and reporting.

---

# 🎥 Processed CCTV Output

The detection engine processes the input CCTV video and creates an annotated output video.

The output can contain:

* YOLO detections
* Object tracking
* Virtual fence
* Intrusion indication
* Active intruder count
* Frame number
* Intrusion event count
* IBVAP identification

The processed video is saved as:

```text
intrusion_result.mp4
```

The Streamlit dashboard can then display the processed CCTV video directly in the browser.

---

# 🖥️ Command Center Dashboard

IBVAP includes a browser-based **Streamlit Command Center**.

The dashboard is designed as the main operator interface.

The current interface includes:

### 📹 Surveillance

Used for:

* CCTV selection
* Video viewing
* Virtual fence setup
* Starting AI detection
* Viewing processed CCTV footage

The current implementation includes a primary camera:

```text
CCTV-01
CAM_01
```

---

### 🚨 Event Log

Displays previously recorded detection/intrusion events from the SQLite database.

Event information can include:

* Time
* Camera
* Object
* Track ID
* Confidence
* Event type
* Snapshot/evidence

---

### 📊 Analytics

The dashboard provides surveillance analytics based on recorded events.

Current analytics include:

* Events by type
* Objects detected
* Events by camera
* Intrusion summary

The dashboard generates charts from the stored event data.

---

# 📊 Dashboard Metrics

The Command Center displays high-level system metrics such as:

```text
SYSTEM STATUS
ACTIVE INTRUSIONS
AI EVENTS
CAMERAS
```

The dashboard also displays database connection status and the latest recorded event.

---

# 🎨 User Interface

The frontend is built using **Streamlit** with custom CSS styling.

The interface includes:

* IBVAP Command Center branding
* System status indicator
* Surveillance dashboard
* Event log
* Analytics
* CCTV cards
* Intrusion alerts
* Evidence snapshots
* Processed CCTV playback
* Metrics and charts

## The current design uses a dark surveillance/command-center visual style.

# 🧰 Technology Stack

## AI / Computer Vision

* **Python**
* **Ultralytics YOLO**
* **OpenCV**
* **ByteTrack**
* **NumPy**

## Frontend / Dashboard

* **Streamlit**
* **Pandas**
* **Pillow**
* **streamlit-image-coordinates**
* Custom CSS

The current frontend uses `streamlit-image-coordinates` for browser-based coordinate selection rather than `streamlit-drawable-canvas`.

## Database

* **SQLite**

## Logging

* SQLite event database
* CSV event log
* Evidence snapshots

---

# 📁 Current Project Components

The project contains/uses components conceptually organized around:

```text
IBVAP/
│
├── app.py / Streamlit application
│
├── intrusion.py
│       └── AI detection + tracking + intrusion engine
│
├── database/
│       └── ibvap.db
│
├── logs/
│   ├── intrusion_log.csv
│   └── snapshots/
│
├── cctv.mp4
│
├── yolo26n.pt
│
├── intrusion_result.mp4
│
└── README.md
```

> File names may differ depending on the current GitHub version. The repository's actual structure should be treated as the source of truth.

---

# 🔄 Current End-to-End Workflow

The current prototype is designed around this workflow:

### Step 1 — Select CCTV

The operator selects the CCTV feed.

### Step 2 — View CCTV

The system obtains the CCTV/video frame.

### Step 3 — Define Virtual Fence

The operator clicks points on the CCTV frame to define the restricted area.

### Step 4 — Start AI Detection

The application calls the detection engine with:

```text
Video Path
Fence Points
Output Path
Camera ID
```

The frontend passes the selected fence points directly to the detection function.

### Step 5 — YOLO Processing

The detection engine:

```text
Loads YOLO
      ↓
Opens CCTV video
      ↓
Processes frames
      ↓
Detects objects
      ↓
Tracks objects with ByteTrack
```

### Step 6 — Virtual Fence Analysis

Detected objects are evaluated against the user-defined virtual perimeter.

### Step 7 — Intrusion Event

If the intrusion condition is satisfied:

```text
Intrusion Detected
       ↓
Snapshot Captured
       ↓
SQLite Event
       ↓
CSV Log
       ↓
Event returned to frontend
```

### Step 8 — Results

The dashboard displays:

* Processed video
* Number of frames processed
* Number of intrusion events
* Detected objects
* Track IDs
* Confidence
* Centroid
* Snapshot information
* Database events

---

# 🧩 Current Architecture

```text
┌─────────────────────┐
│    CCTV / Video     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│       OpenCV        │
│   Video Processing  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     YOLO Model      │
│ Detection + Tracking│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    ByteTrack        │
│   Object Tracking   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Virtual Fence     │
│ Intrusion Analysis  │
└──────────┬──────────┘
           │
           ▼
      ┌────┴────┐
      │         │
      ▼         ▼
  Snapshot   Event Data
      │         │
      │     ┌───┴────┐
      │     │ SQLite │
      │     └───┬────┘
      │         │
      ▼         ▼
   Evidence   CSV Log
      │         │
      └────┬────┘
           │
           ▼
┌─────────────────────┐
│  Streamlit Command  │
│       Center        │
└─────────────────────┘
```

---

# 👥 Team Development

IBVAP is intended to be maintained as a shared GitHub project.

Recommended workflow:

```text
main
 │
 ├── frontend
 ├── backend
 ├── ai-detection
 └── database
```

Team members should work on separate branches and merge completed changes into `main` through Pull Requests.

Before committing changes:

```bash
git pull
```

After completing a feature:

```bash
git add .
git commit -m "Describe your change"
git push
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd IBVAP
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

If `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

The main project dependencies include the Python computer-vision, AI, dashboard, and data-processing packages used by the application.

## 4. Verify project files

Ensure the required model and test video are available:

```text
yolo26n.pt
cctv.mp4
```

The detection engine expects the YOLO model and CCTV input in the project structure used by the application.

## 5. Run the application

```bash
streamlit run app.py
```

The exact entry-point filename should match the current repository version.

---

# 🧪 Prototype Testing

A basic demonstration can be performed using:

1. Start the Streamlit application.
2. Open CCTV-01.
3. Display the CCTV frame.
4. Select at least three points to create the virtual fence.
5. Start AI detection.
6. Allow YOLO to process the CCTV video.
7. Observe detected objects and tracking.
8. Check whether objects enter the restricted zone.
9. Verify the intrusion alert.
10. Verify the generated snapshot.
11. Verify the SQLite event.
12. Verify the CSV log.
13. Verify the processed CCTV video.
14. Check the Event Log and Analytics tabs.

---

# 📌 Current Implementation Status

| Component                      | Status                    |
| ------------------------------ | ------------------------- |
| CCTV/video input               | ✅ Implemented             |
| YOLO object detection          | ✅ Implemented             |
| Object tracking                | ✅ Implemented             |
| ByteTrack                      | ✅ Implemented             |
| Virtual fence                  | ✅ Implemented             |
| Intrusion detection            | ✅ Implemented             |
| Intrusion snapshots            | ✅ Implemented             |
| SQLite event storage           | ✅ Implemented             |
| CSV logging                    | ✅ Implemented             |
| Processed video                | ✅ Implemented             |
| Streamlit dashboard            | ✅ Implemented             |
| Surveillance view              | ✅ Implemented             |
| Event log                      | ✅ Implemented             |
| Analytics                      | ✅ Implemented             |
| Evidence display               | ✅ Implemented             |
| Multi-camera architecture      | 🟡 Prototype / expandable |
| Dedicated REST backend API     | 🔴 Planned                |
| Real-time CCTV/RTSP deployment | 🔴 Planned                |
| Advanced identity recognition  | 🔴 Future scope           |
| Production deployment          | 🔴 Future scope           |

---

# 🚧 Current Development Priorities

The next major stage of IBVAP development is **system integration and backend separation**.

The priority is to move toward:

```text
CCTV
  ↓
AI Detection Engine
  ↓
Backend API
  ↓
Database
  ↓
Frontend Dashboard
```

The current prototype already demonstrates the detection → event → database → dashboard workflow within the Streamlit application.

The next development stage should make this architecture more modular and scalable.

---

# 🔮 Future Enhancements

Potential future improvements include:

### Real-Time CCTV

Replace or supplement test video with:

* RTSP camera streams
* IP cameras
* Live CCTV feeds

### Multiple Cameras

Support multiple simultaneous cameras:

```text
CAM_01
CAM_02
CAM_03
CAM_04
...
```

### Central Backend

Introduce a dedicated backend API for:

* Event creation
* Event retrieval
* Camera management
* User management
* Database operations
* Alert management

### Real-Time Alerts

Possible future notification channels:

* Dashboard alerts
* Email
* SMS
* Mobile notifications
* Control-room alarms

### Advanced Recognition

Future versions could incorporate:

* Face recognition
* License plate recognition
* Authorized-person identification
* Whitelist / blacklist management

### Improved Tracking

Potential improvements include:

* More robust multi-object tracking
* Re-identification
* Better duplicate-event handling
* Intrusion cooldown periods
* Persistent object tracking

### Deployment

The prototype can eventually be adapted for:

* Dedicated surveillance servers
* Edge devices
* GPU-based inference
* Cloud infrastructure
* Centralized command centers

---

# ⚠️ Limitations

IBVAP is currently a **prototype / proof-of-concept**.

Important limitations include:

* Performance depends on hardware.
* Detection accuracy depends on the trained/model configuration and video conditions.
* Poor lighting, occlusion, camera angle, and image quality can affect detection.
* A predefined virtual fence is required for intrusion analysis.
* The current prototype primarily demonstrates video-file based surveillance.
* Production-scale multi-camera deployment requires further optimization.
* AI detection should be treated as an assistance mechanism and not as an infallible security decision system.

---

# 🔐 Security Considerations

A production deployment should additionally address:

* Authentication
* Authorization
* Secure API communication
* Database access control
* Secure storage of evidence
* Encryption
* Audit logging
* User roles
* Secure camera credentials
* Protection of surveillance footage

These are important future requirements beyond the current academic prototype.

---

# 📚 Project Significance

IBVAP demonstrates how AI and computer vision can be applied to surveillance environments where continuous human monitoring is difficult.

The project combines:

```text
Artificial Intelligence
        +
Computer Vision
        +
Object Tracking
        +
Geofencing / Virtual Perimeter
        +
Database Systems
        +
Web-Based Visualization
```

to create an integrated AI-assisted surveillance prototype.

---

# 🎓 Academic Scope

IBVAP demonstrates concepts from:

* Artificial Intelligence
* Machine Learning
* Computer Vision
* Object Detection
* Object Tracking
* Image Processing
* Database Management
* Python Programming
* Web Application Development
* Software Engineering
* Data Visualization

---

# 🏁 Final Project Goal

The long-term goal of IBVAP is to provide an intelligent surveillance pipeline capable of:

```text
DETECT
  ↓
TRACK
  ↓
ANALYZE
  ↓
IDENTIFY INTRUSION
  ↓
CAPTURE EVIDENCE
  ↓
STORE EVENT
  ↓
ALERT OPERATOR
  ↓
DISPLAY & ANALYZE
```

The system is designed as an **AI-assisted surveillance layer** that helps security personnel monitor restricted areas more efficiently.

---

## 🛡️ IBVAP

**Intelligent Border Virtual AI Protection**

> AI-Powered Intrusion Surveillance • YOLO Vision • Virtual Perimeter • SQLite Evidence System

**Project Type:** Academic Prototype
**Domain:** AI / Computer Vision / Border Security / Surveillance
**Primary Language:** Python
**Interface:** Streamlit
**Detection:** YOLO
**Tracking:** ByteTrack
**Database:** SQLite
