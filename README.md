# Border Security CCTV Surveillance System (Prototype)

## Problem Statement

Border Out Posts (BOPs), check posts, and border roads require continuous
human monitoring of CCTV feeds to detect unauthorized intrusions. Manual
monitoring is fatigue-prone and does not scale across multiple cameras or
long shifts. This project explores an AI-assisted approach to automatically
detect and log intrusion events from CCTV footage.

## Motivation

Automating even the first layer of surveillance — detecting when a person
enters a restricted area — can reduce the manual monitoring burden and
create a reliable, timestamped audit trail of events for security review.

## Current Status: Prototype

**This is an early-stage prototype, not a deployment-ready system.** It has
been tested only on a single pre-recorded video file, not live CCTV/RTSP
feeds, and has not been evaluated for real-world border deployment
conditions (night vision, weather, multiple cameras, network reliability,
etc.).

## System Architecture (Current)

```
CCTV video file (.mp4)
        ↓
YOLO object detection (person class only)
        ↓
ByteTrack object tracking (assigns persistent IDs)
        ↓
Custom polygon zone check (point-in-polygon test)
        ↓
Intrusion event detection (new entry only, no duplicate alerts)
        ↓
Event logging (CSV) + Snapshot capture + Annotated output video
```

## Current Features

- ✅ Person detection using YOLO (`yolo26n.pt`)
- ✅ Multi-person tracking with persistent IDs (ByteTrack)
- ✅ **Custom polygon-shaped restricted zone**, selected interactively by
  mouse click on the first video frame (not a fixed rectangle)
- ✅ Intrusion detection: triggers only on zone *entry*, not every frame a
  person remains inside (avoids duplicate/spam alerts)
- ✅ CSV event logging (date, time, person ID, event type, snapshot path)
- ✅ Automatic snapshot capture on each new intrusion event
- ✅ Annotated output video (green box = normal, red box = intruder,
  live status banner, live event counter)
- ✅ GPU-accelerated inference (tested on NVIDIA RTX 4050, 6GB VRAM)

## Not Yet Implemented

- ❌ Real-time CCTV / RTSP camera input (currently video-file only)
- ❌ Multiple simultaneous camera support
- ❌ ANPR / automatic number plate recognition
- ❌ Face recognition
- ❌ Automated alert/notification system (SMS, email, sound alarm)
- ❌ Web dashboard / live monitoring UI
- ❌ Persistent database (currently flat CSV file)
- ❌ Authentication / access control
- ❌ Deployment packaging (Docker, service installer, etc.)

## Technologies Used

| Component | Technology |
|---|---|
| Object detection | YOLO (Ultralytics, `yolo26n.pt`) |
| Object tracking | ByteTrack |
| Video/image processing | OpenCV |
| Deep learning backend | PyTorch (CUDA-enabled) |
| Language | Python 3.12 |

## Hardware Requirements

- **Tested on:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM), CUDA 12.4
- A CUDA-capable NVIDIA GPU is strongly recommended for real-time
  performance. CPU-only inference is possible but significantly slower.
- Minimum ~6GB VRAM recommended for the current model/settings.

## Installation

### Prerequisites

- Python 3.12 (tested version)
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA driver supporting CUDA 12.4 or compatible

### Setup steps

1. Clone this repository:
```
   git clone <your-repo-url>
   cd CCTV-YOLO
```

2. Create and activate a virtual environment:
```
   python -m venv .venv
   .venv\Scripts\activate
```
   *(Windows PowerShell/CMD command shown; adjust for other OS.)*

3. Install PyTorch with CUDA support first (this project was tested with
   CUDA 12.4 — adjust the CUDA version in the command below to match your
   own GPU/driver if different, using the official selector at
   https://pytorch.org/get-started/locally/):
```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

4. Install remaining dependencies:
```
   pip install -r requirements.txt
```

5. Download the YOLO model weights (`yolo26n.pt`) — **not included in this
   repository** due to file size. Place it in the project root directory.
   *(Add the exact download source/link here once confirmed.)*

6. Place your input video as `cctv.mp4` in the project root, or edit the
   `VIDEO_PATH` variable in `intrusion.py` to point to your own file.

### Running the project

```
python intrusion.py
```

- A window will open showing the first frame of the video.
- **Left-click** to place points and draw your restricted zone polygon
  (minimum 3 points).
- Press **Enter** to confirm the zone, **R** to reset points, **Q** to
  cancel.
- The system will then process the video, showing live detection and
  intrusion status. Press **Q** at any time to stop early.

## How Intrusion Detection Works

1. The first frame of the video is used to let the user draw a custom
   polygon marking the restricted zone.
2. Each subsequent frame is passed through YOLO with ByteTrack, detecting
   and tracking people (class 0 only).
3. For each tracked person, the center point of their bounding box is
   checked against the polygon using a point-in-polygon test.
4. If a person's center enters the polygon and they were **not** already
   inside it in the previous frame, this is logged as a new intrusion
   event — this prevents the same person from generating repeated alerts
   every single frame they remain in the zone.
5. Each new event is logged to `logs/intrusion_log.csv` (date, time,
   person ID, event, snapshot path) and a snapshot image is saved to
   `logs/snapshots/`.

### Known limitation

If the tracker loses a person's ID (e.g. due to occlusion) and reassigns a
new ID when they reappear, this will currently be logged as a new
intrusion event, even if it is the same person who never left the zone.
This is a limitation of ID-based re-identification and is a planned area
for improvement.

## Project Structure

```
CCTV-YOLO/
├── intrusion.py          # Main intrusion detection script
├── requirements.txt      # Python dependencies
├── README.md
├── .gitignore
└── logs/                 # Generated at runtime (not committed)
    ├── intrusion_log.csv
    └── snapshots/
```

*(`.venv/`, `cctv.mp4`, `intrusion_result.mp4`, `yolo26n.pt`, and `runs/`
are excluded from version control — see `.gitignore`.)*

## Current Limitations

- Works only on pre-recorded video files, not live camera feeds
- Single-camera only
- No automated alerting beyond terminal output and CSV logging
- No authentication or access control on the system itself
- Not tested under real-world border conditions (weather, lighting,
  night vision, network interruptions)
- Person re-identification across occlusion/ID-switches is not handled
- This is a functional prototype for demonstration purposes and has
  **not** been validated for real-world border security deployment

## Future Scope

Planned development, in order:

1. Reliable entry/exit detection improvements, reduced false positives
2. Real-time camera support (webcam, then RTSP/IP cameras)
3. Multi-camera support
4. Vehicle detection + ANPR (automatic number plate recognition) with OCR
5. Face recognition (subject to legal/operational review)
6. Automated alert system (sound, notifications, severity levels)
7. Web-based live monitoring dashboard
8. Database backend, proper logging/error handling, configuration files
9. Deployment packaging and documentation

## Team Contribution

*(To be filled in)*

| Name | Contribution |
|---|---|
|   |   |
|   |   |

## Disclaimer

This project is a student/prototype-stage system built for demonstration
and learning purposes. It is not certified or validated for operational
border security use.