# 🛡️ IBVAP — Intelligent Border Video Analytics Platform

### AI-Based Intelligent Video Analytics Platform for Border Surveillance Using Existing CCTV Infrastructure

IBVAP (**Intelligent Border Video Analytics Platform**) is an AI-powered video surveillance prototype designed to transform existing CCTV infrastructure into an intelligent surveillance system.

The platform uses **Artificial Intelligence, Computer Vision, Object Detection, Tracking, and Video Analytics** to analyze CCTV video streams and identify potential security threats in real time.

IBVAP is designed to work with **existing CCTV infrastructure**, reducing the need for expensive dedicated surveillance hardware.

---

## 📌 Problem Statement

Border security forces deploy CCTV cameras at:

* Border Out Posts (BOPs)
* Check posts
* Border roads
* Restricted areas
* Strategic installations

Conventional CCTV systems primarily provide video recording and live monitoring, requiring security personnel to continuously observe multiple camera feeds.

Advanced surveillance capabilities such as:

* Human detection and tracking
* Vehicle detection and classification
* Face detection
* Automatic Number Plate Recognition (ANPR)
* Intrusion detection
* Suspicious activity detection
* Night-time movement detection
* Real-time alerts

often require specialized hardware or proprietary systems.

This makes large-scale deployment expensive and difficult, particularly in remote border locations.

---

# 💡 Proposed Solution

IBVAP provides a **software-based AI surveillance platform** that enhances existing CCTV infrastructure using Artificial Intelligence and Computer Vision.

The system receives video from a CCTV camera and detects objects such as **Humans and Vehicles**.

A user can draw a **virtual intrusion/restricted area** on the CCTV video.

The core threat-detection rule is:

> **If a detected Human or Vehicle enters the drawn intrusion area, the system identifies the event as a potential THREAT and generates an alert.**

Objects outside the intrusion area are treated as normal detections and do not generate an intrusion threat alert.

---

# 🚀 Key Features

## 👤 Human Detection

The system detects humans appearing in the CCTV video feed.

Human detections are displayed simply as:

```text
Human
```

The system does **not assign individual IDs** such as:

```text
Person 1
Person 2
Person 3
```

The focus is on detecting human presence and determining whether the human enters a restricted area.

---

## 🚗 Vehicle Detection and Classification

The system detects vehicles appearing in the CCTV footage.

Depending on the AI model, supported vehicle classes may include:

* Car
* Motorcycle
* Bus
* Truck
* Other supported vehicle classes

Vehicles are also checked against the defined intrusion area.

If a vehicle enters the restricted area, it is classified as a potential threat.

Example:

```text
Vehicle detected
Vehicle entered intrusion area
⚠️ THREAT: Vehicle Intrusion
```

---

# 🚧 Virtual Intrusion Area

IBVAP allows the user to **draw a restricted/intrusion area directly on the CCTV video**.

This area represents a location that should not be entered by unauthorized humans or vehicles.

### Example

```text
┌─────────────────────────────────────┐
│                                     │
│         CCTV VIDEO                  │
│                                     │
│      ┌──────────────────────┐       │
│      │   RESTRICTED AREA    │       │
│      │                      │       │
│      │   👤 Human           │       │
│      │                      │       │
│      │   🚗 Vehicle         │       │
│      └──────────────────────┘       │
│                                     │
└─────────────────────────────────────┘
```

If a detected object enters this area, the system generates a threat alert.

---

# ⚠️ Threat Detection Logic

The main threat-detection logic of IBVAP is based on the object's position relative to the user-defined intrusion area.

```text
              CCTV VIDEO
                   │
                   ▼
            AI Object Detection
                   │
           ┌───────┴───────┐
           ▼               ▼
        👤 Human        🚗 Vehicle
           │               │
           └───────┬───────┘
                   ▼
          Check Object Position
                   │
                   ▼
       Is it inside the drawn
          intrusion area?
             /          \
           YES           NO
            │             │
            ▼             ▼
       ⚠️ THREAT       Normal
         ALERT         Detection
            │
            ▼
      Event Logging
            │
            ▼
      Dashboard Alert
```

---

# 🚨 Threat Examples

### Human enters intrusion area

```text
👤 Human detected
📍 Intrusion area entered

⚠️ THREAT: Human entered restricted area
```

### Vehicle enters intrusion area

```text
🚗 Vehicle detected
📍 Intrusion area entered

⚠️ THREAT: Vehicle entered restricted area
```

### Human outside intrusion area

```text
👤 Human detected

Status: Normal
```

### Vehicle outside intrusion area

```text
🚗 Vehicle detected

Status: Normal
```

---

# 🎯 Core System Rule

IBVAP follows a simple rule:

```text
IF Human enters intrusion area
        ↓
     THREAT

IF Vehicle enters intrusion area
        ↓
     THREAT

IF Human stays outside intrusion area
        ↓
     NORMAL

IF Vehicle stays outside intrusion area
        ↓
     NORMAL
```

This makes the system focused on **location-based security threats** rather than simply detecting objects.

---

# 🧠 Human Tracking

The system can track detected humans across consecutive video frames to understand their movement.

However, IBVAP does **not display individual person IDs**.

Instead of:

```text
Person 1
Person 2
Person 3
```

the system represents detections as:

```text
Human
```

The important information is whether the detected human is inside or outside the restricted area.

---

# 🚗 Vehicle Tracking

Detected vehicles can also be tracked across consecutive frames.

The system checks their movement relative to the intrusion area.

If a vehicle crosses into the restricted zone:

```text
Vehicle
     ↓
Enters Intrusion Area
     ↓
⚠️ THREAT
     ↓
Alert + Event Log
```

---

# 🚨 Real-Time Alert Generation

When a Human or Vehicle enters the intrusion area, the system generates a real-time alert.

Example:

```text
┌──────────────────────────────────────┐
│          🚨 SECURITY ALERT           │
├──────────────────────────────────────┤
│ ⚠️ Human entered intrusion area      │
│ Time: 10:32:15                       │
└──────────────────────────────────────┘
```

or:

```text
┌──────────────────────────────────────┐
│          🚨 SECURITY ALERT           │
├──────────────────────────────────────┤
│ ⚠️ Vehicle entered intrusion area    │
│ Time: 10:35:42                       │
└──────────────────────────────────────┘
```

---

# 📝 Event Logging

Threat events can be stored in an SQLite database.

The system can record:

| Field     | Description                 |
| --------- | --------------------------- |
| Date      | Date of event               |
| Time      | Time of event               |
| Detection | Human / Vehicle             |
| Event     | Intrusion                   |
| Zone      | User-defined intrusion area |
| Status    | Threat                      |

Example:

| Time     | Detection | Event     | Status    |
| -------- | --------- | --------- | --------- |
| 10:32:15 | Human     | Intrusion | ⚠️ Threat |
| 10:35:42 | Vehicle   | Intrusion | ⚠️ Threat |

---

# 🖥️ Dashboard

IBVAP uses a **Streamlit-based dashboard** for centralized surveillance monitoring.

The dashboard can display:

* 📹 Live CCTV feed
* 👤 Human detections
* 🚗 Vehicle detections
* 🚧 Drawn intrusion area
* ⚠️ Threat alerts
* 📝 Event logs
* 📊 Detection information
* 🌙 Night monitoring status

### Dashboard Concept

```text
┌───────────────────────────────────────────────────────────┐
│              🛡️ IBVAP SURVEILLANCE DASHBOARD              │
├────────────────────────────────┬──────────────────────────┤
│                                │                          │
│        📹 CCTV VIDEO           │      🚨 ALERTS           │
│                                │                          │
│    ┌──────────────────────┐    │ ⚠️ Human - Threat       │
│    │                      │    │ ⚠️ Vehicle - Threat     │
│    │   👤 Human           │    │                          │
│    │                      │    │                          │
│    │   🚧 Restricted      │    │                          │
│    │      Area            │    │                          │
│    │                      │    │                          │
│    └──────────────────────┘    │                          │
│                                │                          │
├────────────────────────────────┴──────────────────────────┤
│                     📝 EVENT LOGS                          │
├──────────┬────────────┬──────────────┬─────────────────────┤
│ Time     │ Detection  │ Event        │ Status              │
├──────────┼────────────┼──────────────┼─────────────────────┤
│ 10:32:15 │ Human      │ Intrusion    │ ⚠️ Threat            │
│ 10:35:42 │ Vehicle    │ Intrusion    │ ⚠️ Threat            │
└──────────┴────────────┴──────────────┴─────────────────────┘
```

---

# 🏗️ System Architecture

```text
                 EXISTING CCTV CAMERA
                         │
                         ▼
                ┌──────────────────┐
                │   IP VIDEO FEED  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │      OpenCV      │
                │  Frame Processing│
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    YOLO MODEL    │
                │  Object Detection│
                └────────┬─────────┘
                         │
                ┌────────┴─────────┐
                │                  │
                ▼                  ▼
             👤 Human           🚗 Vehicle
             Detection          Detection
                │                  │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ Drawn Intrusion  │
                │      Area        │
                └────────┬─────────┘
                         │
                         ▼
                Object inside area?
                    /          \
                  YES           NO
                   │             │
                   ▼             ▼
             ⚠️ THREAT        NORMAL
                ALERT        DETECTION
                   │
                   ▼
             ┌──────────────┐
             │ Event Logger │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │    SQLite    │
             │   Database   │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │  Streamlit   │
             │  Dashboard   │
             └──────────────┘
```

---

# 🔄 Working Process

### Step 1 — CCTV Input

The system receives a live video stream from an existing CCTV camera.

### Step 2 — Frame Capture

OpenCV captures frames from the CCTV stream.

### Step 3 — AI Object Detection

The YOLO-based model detects objects such as humans and vehicles.

### Step 4 — Draw Intrusion Area

The user defines a restricted area by drawing a virtual boundary on the video.

### Step 5 — Position Analysis

The system checks the position of every detected Human and Vehicle.

### Step 6 — Intrusion Check

The system determines whether the detected object has entered the defined intrusion area.

### Step 7 — Threat Identification

If the detected object is inside the intrusion area:

```text
Human → Threat
Vehicle → Threat
```

### Step 8 — Alert Generation

A real-time threat alert is displayed on the dashboard.

### Step 9 — Event Logging

The threat event is stored in the database for future reference.

---

# 🧠 Technologies Used

| Technology             | Purpose                             |
| ---------------------- | ----------------------------------- |
| **Python**             | Main programming language           |
| **OpenCV**             | Video and image processing          |
| **YOLO / Ultralytics** | AI object detection                 |
| **Streamlit**          | Web-based dashboard                 |
| **SQLite**             | Event and alert storage             |
| **EasyOCR**            | Number plate/text recognition       |
| **NumPy**              | Numerical and image-data processing |

---

# 📁 Project Structure

```text
IBVAP/
│
├── app.py
│
├── models/
│   └── yolov_model.pt
│
├── detection/
│   ├── human_detection.py
│   ├── vehicle_detection.py
│   └── tracking.py
│
├── intrusion/
│   └── virtual_fence.py
│
├── alerts/
│   └── alert_manager.py
│
├── database/
│   └── database.py
│
├── anpr/
│   └── number_plate.py
│
├── utils/
│   └── video_utils.py
│
├── requirements.txt
│
└── README.md
```

> The exact project structure may change during development.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Enter the project directory:

```bash
cd IBVAP
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
opencv-python
ultralytics
streamlit
numpy
easyocr
```

SQLite is included with standard Python installations.

---

# ▶️ Running the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your web browser.

---

# 🎯 Project Objectives

IBVAP aims to:

1. Enhance existing CCTV infrastructure using AI.
2. Automatically detect humans and vehicles.
3. Allow users to define a virtual intrusion area.
4. Detect when a human enters the intrusion area.
5. Detect when a vehicle enters the intrusion area.
6. Classify such intrusion events as potential threats.
7. Generate real-time threat alerts.
8. Store security events for future analysis.
9. Reduce dependence on expensive dedicated surveillance hardware.
10. Improve situational awareness and response time.

---

# 🌙 Night-Time Movement Detection

The system can be extended to monitor movement during night-time or low-light conditions.

If a human or vehicle is detected entering a restricted area during night-time, the system can generate a potential threat alert.

Example:

```text
🌙 NIGHT MONITORING

👤 Human detected
📍 Restricted area entered

⚠️ THREAT ALERT
```

---

# 🔢 Automatic Number Plate Recognition

IBVAP can use **EasyOCR** along with image-processing techniques for Automatic Number Plate Recognition.

The ANPR module can be used to:

1. Detect a vehicle.
2. Identify the number-plate region.
3. Extract text from the plate.
4. Store the recognized plate information.
5. Associate the event with the detected vehicle.

This feature can be integrated with the intrusion detection system.

For example:

```text
🚗 Vehicle detected
📍 Intrusion area entered
🔢 Number Plate: XXXXXXXX

⚠️ THREAT: Vehicle Intrusion
```

---

# 🔮 Future Scope

Future versions of IBVAP can include:

* Advanced ANPR
* Face detection
* Authorized face-recognition capabilities
* Improved multi-object tracking
* Suspicious activity detection
* Loitering detection
* Abandoned vehicle detection
* Improved night-time detection
* Multi-camera monitoring
* Centralized command dashboard
* Advanced notification systems
* Edge AI deployment
* Integration with authorized command and control systems

---

# 📈 Advantages

### 💰 Cost Effective

Uses existing CCTV infrastructure rather than requiring dedicated smart surveillance cameras.

### 🔧 Software-Based

AI-powered intelligence can be added through software.

### 📡 Scalable

The architecture can be extended to multiple CCTV cameras and locations.

### ⚡ Real-Time

The system can analyze video and generate alerts in real time.

### 🚧 Flexible

The user can define the restricted area according to the surveillance environment.

### 🧠 AI-Assisted

Reduces the need for continuous manual observation of every CCTV feed.

---

# 🌐 Potential Applications

IBVAP can potentially be adapted for monitoring:

* 🛡️ Border Out Posts (BOPs)
* 🚧 Check posts
* 🛣️ Border roads
* 🏭 Industrial facilities
* 🏢 Restricted buildings
* 🔒 High-security zones
* 🏗️ Critical infrastructure
* 🎓 Large campuses
* 🪖 Strategic installations

---

# 📊 Expected Outcome

IBVAP aims to transform conventional CCTV surveillance into an intelligent AI-assisted monitoring system.

```text
Existing CCTV
      ↓
AI Video Analysis
      ↓
Human / Vehicle Detection
      ↓
Draw Intrusion Area
      ↓
Check Object Position
      ↓
┌────────────────────────────┐
│ Is Human/Vehicle inside?   │
└─────────────┬──────────────┘
              │
        ┌─────┴─────┐
        ↓           ↓
       YES          NO
        ↓           ↓
   ⚠️ THREAT     NORMAL
     ALERT       MONITORING
        ↓
   Event Logging
        ↓
   Dashboard Alert
```

---

# 🔐 Privacy & Security

IBVAP is primarily designed as a **security-event detection system**.

The basic intrusion-detection system focuses on:

```text
👤 Human
🚗 Vehicle
📍 Location
🚧 Intrusion Area
⚠️ Threat Event
```

The system does not need to assign individual IDs to humans for basic intrusion detection.

Any future facial-recognition or identity-related functionality should only be implemented with appropriate authorization, privacy safeguards, and compliance with applicable laws and organizational policies.

---

# 👥 Project Information

### Project Name

**IBVAP**

### Full Name

**Intelligent Border Video Analytics Platform**

### Domain

* Artificial Intelligence
* Machine Learning
* Computer Vision
* Video Analytics
* Border Surveillance
* Software Engineering

---

# 🏆 Project Vision

> **Transform existing CCTV infrastructure into an intelligent AI-assisted surveillance network that detects potential human and vehicle intrusions and provides real-time threat alerts.**

---

# 📜 Disclaimer

IBVAP is an **academic/prototype project** developed for research, learning, and demonstration purposes.

Real-world deployment in border-security or other high-security environments would require appropriate testing, authorization, cybersecurity controls, privacy safeguards, reliability validation, and compliance with applicable laws, regulations, and organizational policies.


