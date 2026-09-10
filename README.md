# 🛡️ IBVAP — Intelligent Border Video Analytics Platform

### AI-Based Intelligent Video Analytics Platform for Border Surveillance Using Existing CCTV Infrastructure

**IBVAP (Intelligent Border Video Analytics Platform)** is an AI-powered video surveillance prototype designed to transform existing CCTV infrastructure into an intelligent surveillance network.

The platform uses **Artificial Intelligence, Machine Learning, Computer Vision, Object Detection, Object Tracking, Pose Estimation, and Video Analytics** to analyze CCTV video streams and identify potential security threats and suspicious activities in real time.

The system is designed to work with **existing CCTV infrastructure**, reducing the need for expensive dedicated surveillance hardware.

---

# 📌 Problem Statement

Border security forces deploy CCTV cameras at:

* Border Out Posts (BOPs)
* Check posts
* Border roads
* Strategic locations
* Restricted and sensitive areas

Conventional CCTV systems primarily provide video recording and live monitoring, requiring security personnel to continuously observe multiple camera feeds.

Advanced surveillance functionalities such as:

* Human detection and tracking
* Vehicle detection and classification
* Face detection
* Automatic Number Plate Recognition (ANPR)
* Intrusion detection
* Suspicious activity detection
* Night-time movement detection
* Real-time alert generation

often require specialized hardware or proprietary surveillance solutions.

This makes large-scale deployment **costly, complex, and difficult**, particularly in remote border areas.

---

# 💡 Proposed Solution

IBVAP provides a **software-based AI surveillance platform** capable of enhancing existing CCTV infrastructure.

The platform receives live video streams from standard CCTV cameras and performs real-time analysis using AI and Computer Vision.

The system has two major security layers:

### 1. 🚧 Intrusion Detection

A user can draw a **virtual restricted/intrusion area** on the CCTV video.

If a detected **Human or Vehicle enters the drawn intrusion area**, the system classifies the event as a **potential threat** and generates an alert.

### 2. 🧠 Suspicious Activity Detection

The system also analyzes human movement and posture to identify potentially suspicious behavioral indicators.

Suspicious activity can be detected **anywhere in the CCTV frame**, not only inside the virtual intrusion area.

Possible indicators include:

* Loitering
* Rapid movement
* Erratic movement
* Crouching
* Hands raised
* Other configurable behavioral indicators

These indicators generate alerts for operator attention and are **not treated as proof of criminal activity**.

---

# 🚀 Key Features

## 👤 Human Detection

IBVAP detects humans appearing in the CCTV video feed using AI-based object detection.

Human detections are displayed simply as:

```text
Human
```

The system does **not display individual labels such as**:

```text
Person 1
Person 2
Person 3
```

The focus is on detecting human presence, movement, location, and potential security events.

---

# 🚗 Vehicle Detection and Classification

The system detects vehicles appearing in the CCTV footage and can classify them according to the supported AI model.

Possible vehicle classes include:

* Car
* Motorcycle
* Bus
* Truck
* Other supported vehicle classes

Vehicles are also analyzed against the defined intrusion area.

If a vehicle enters the restricted area, it is treated as a **potential threat**.

Example:

```text
🚗 Vehicle detected

📍 Vehicle entered intrusion area

⚠️ THREAT: Vehicle Intrusion
```

---

# 🚧 Virtual Intrusion Area

IBVAP allows the operator to **draw a restricted area directly on the CCTV video**.

This virtual area represents a location where unauthorized human or vehicle entry should generate an alert.

### Basic Concept

```text
┌─────────────────────────────────────┐
│           CCTV VIDEO                │
│                                     │
│      👤 Human                       │
│                                     │
│              ┌───────────────┐      │
│              │  RESTRICTED   │      │
│              │     AREA      │      │
│              │               │      │
│              │    🚗 Vehicle │      │
│              └───────────────┘      │
│                                     │
└─────────────────────────────────────┘
```

The system continuously checks detected objects against the user-defined area.

---

# ⚠️ Threat Detection Logic

The main intrusion rule is:

```text
Human detected
      ↓
Is Human inside intrusion area?
      ↓
     YES
      ↓
⚠️ POTENTIAL THREAT
```

and:

```text
Vehicle detected
      ↓
Is Vehicle inside intrusion area?
      ↓
     YES
      ↓
⚠️ POTENTIAL THREAT
```

Objects outside the restricted area remain under normal monitoring.

---

## 🎯 Core Threat Rules

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

Therefore, **both humans and vehicles can trigger intrusion threat alerts**.

---

# 🧠 Suspicious Activity Detection

Suspicious Activity Detection is a separate security layer from the virtual-fence intrusion system.

The purpose is to answer:

> **“Is a person behaving unusually enough that the security operator should be alerted?”**

The system analyzes **movement and human posture over time** rather than relying only on a single frame.

---

# 🔍 Suspicious Activity Indicators

## 1. 🕒 Loitering

Loitering refers to a person remaining around the same area for an unusually long period with limited displacement.

Conceptually:

```text
Person detected
      ↓
Moves slightly
      ↓
Does not leave the area
      ↓
Remains for a period of time
      ↓
⚠️ LOITERING
```

Loitering detection is **not restricted to the virtual intrusion area**.

A person can trigger a loitering alert anywhere within the CCTV frame if the configured conditions are satisfied.

---

## 2. 🏃 Rapid Movement

Rapid movement can be identified by analyzing the displacement of a tracked person's position over time.

```text
Frame N
Person → Position A
        ↓
Frame N+1
Person → Position B
        ↓
Large displacement
        ↓
⚠️ RAPID MOVEMENT
```

---

## 3. 🔄 Erratic Movement

Erratic movement refers to irregular or rapidly changing movement patterns.

Example:

```text
Normal:

→ → → → →


Erratic:

→ ↗ ← ↘ → ←
```

The system analyzes movement history and direction changes to identify unusual movement patterns.

---

## 4. 🧎 Crouching

The system can use **human pose estimation** to analyze body posture.

Relevant body keypoints include:

* Shoulders
* Hips
* Knees
* Ankles

These keypoints can be used to identify a low or crouched posture.

```text
Human Pose
     ↓
Shoulder
     ↓
Hip
     ↓
Knee
     ↓
Ankle

Posture Analysis
     ↓
⚠️ CROUCHING INDICATOR
```

---

## 5. 🙌 Hands Raised

Pose estimation can also be used to identify raised-hand posture.

The system compares the position of the wrists with the shoulders.

Conceptually:

```text
     Wrist
       ↑
     Elbow
       ↑
    Shoulder

       ↓

⚠️ HANDS RAISED
```

---

# 🧠 AI Pipeline for Suspicious Activity

The suspicious-activity pipeline can be represented as:

```text
CCTV Video
     ↓
YOLO Detection
     ↓
Human Detection
     ↓
Object Tracking
     ↓
Movement History + Pose History
     ↓
Behavior Analysis
     ↓
┌────────────┬────────────┬─────────────┐
│            │            │             │
Loitering   Rapid       Erratic      Posture
            Movement     Movement        │
                                      ┌───┴────┐
                                      │        │
                                   Crouching  Hands
                                              Raised
     └────────────┬────────────┬──────────────┘
                  ↓
         Temporal Confirmation
                  ↓
       ⚠️ SUSPICIOUS ACTIVITY
                  ↓
          Alert + Snapshot
                  ↓
             Event Log
```

---

# ⏱️ Temporal Confirmation

The system should not generate a suspicious-activity alert based on one unusual frame alone.

Instead, behavioral indicators can be observed across multiple frames.

```text
Frame 1 → Behavior Candidate
Frame 2 → Behavior Candidate
Frame 3 → Behavior Candidate
Frame 4 → Behavior Candidate
       ↓
Enough evidence
       ↓
⚠️ CONFIRMED SUSPICIOUS ACTIVITY
```

This approach helps reduce false alerts caused by temporary or accidental movements.

---

# 📊 Suspicion Score

The system can use a heuristic suspicion score to prioritize events.

Conceptually:

```text
Loitering          → Score +
Rapid Movement     → Score +
Erratic Movement   → Score +
Crouching          → Score +
Hands Raised       → Score +
```

Multiple indicators can increase the overall priority of an event.

The score is a **heuristic confidence/priority indicator**, not a probability that a person is committing a crime.

---

# 🚨 Real-Time Alerts

When a potential security event is detected, IBVAP can generate a real-time alert.

### Intrusion Alert

```text
🚨 SECURITY ALERT

⚠️ Human entered intrusion area

Status: POTENTIAL THREAT
```

### Vehicle Intrusion Alert

```text
🚨 SECURITY ALERT

⚠️ Vehicle entered intrusion area

Status: POTENTIAL THREAT
```

### Suspicious Activity Alert

```text
🚨 SECURITY ALERT

⚠️ Suspicious Activity Detected

Activity: LOITERING
Status: REVIEW REQUIRED
```

---

# 🔊 Audible Alerts

The system can provide an audible alert when a security event is confirmed.

The alert pipeline is:

```text
Security Event
      ↓
Event Confirmation
      ↓
┌─────┼─────┐
↓     ↓     ↓
Video Sound  Log
Alert Alert  Event
```

For Windows-based testing, an audio alert can be implemented using Python's built-in `winsound` module.

---

# 📸 Event Snapshots

When a suspicious activity or intrusion event is confirmed, the system can save a snapshot of the relevant CCTV frame.

```text
CCTV Frame
    ↓
Event Detected
    ↓
Snapshot Generated
    ↓
logs/snapshots/
```

This provides visual evidence for the operator to review the event.

---

# 📝 Event Logging

Security events can be stored for future review.

The system can maintain information such as:

| Field     | Description                       |
| --------- | --------------------------------- |
| Date      | Date of event                     |
| Time      | Time of event                     |
| Detection | Human / Vehicle                   |
| Event     | Intrusion / Suspicious Activity   |
| Activity  | Loitering / Rapid Movement / etc. |
| Location  | Detected location                 |
| Status    | Threat / Suspicious               |
| Snapshot  | Associated event image            |

Example:

| Time     | Detection | Event     | Status        |
| -------- | --------- | --------- | ------------- |
| 10:32:15 | Human     | Intrusion | ⚠️ Threat     |
| 10:35:42 | Vehicle   | Intrusion | ⚠️ Threat     |
| 10:38:10 | Human     | Loitering | ⚠️ Suspicious |
| 10:41:25 | Human     | Crouching | ⚠️ Suspicious |

---

# 🗄️ SQLite Database

SQLite can be used to store security events and alerts.

The architecture is:

```text
Security Event
      ↓
Event Manager
      ↓
SQLite Database
      ↓
Streamlit Dashboard
```

This allows previously generated events to be displayed and reviewed from the dashboard.

---

# 🔢 Automatic Number Plate Recognition (ANPR)

IBVAP can integrate **Automatic Number Plate Recognition** using computer vision and OCR techniques.

The ANPR pipeline can be:

```text
Vehicle Detection
       ↓
Number Plate Detection
       ↓
Plate Region Extraction
       ↓
Image Processing
       ↓
EasyOCR
       ↓
Number Plate Text
       ↓
Event Logging
```

Example:

```text
🚗 Vehicle detected
📍 Intrusion area entered
🔢 Number Plate: XXXXXXXX

⚠️ THREAT: Vehicle Intrusion
```

---

# 🌙 Night-Time Movement Detection

IBVAP can monitor movement during night-time or low-light conditions.

If a human or vehicle is detected entering a restricted area during night-time, the system can generate a potential threat alert.

Example:

```text
🌙 NIGHT MONITORING ACTIVE

👤 Human detected
📍 Intrusion area entered

⚠️ POTENTIAL THREAT
```

---

# 🖥️ Streamlit Dashboard

IBVAP uses **Streamlit** to provide a centralized surveillance dashboard.

The dashboard can include:

* 📹 Live CCTV feed
* 👤 Human detection
* 🚗 Vehicle detection
* 🚧 Drawn intrusion area
* ⚠️ Threat alerts
* 🧠 Suspicious activity alerts
* 📝 Event logs
* 📊 Detection information
* 🌙 Night monitoring status
* 📸 Event snapshots

### Dashboard Concept

```text
┌─────────────────────────────────────────────────────────────┐
│            🛡️ IBVAP SURVEILLANCE DASHBOARD                 │
├────────────────────────────────┬────────────────────────────┤
│                                │                            │
│         📹 CCTV FEED           │       🚨 ALERTS            │
│                                │                            │
│   ┌────────────────────────┐   │ ⚠️ Human - Threat          │
│   │                        │   │ ⚠️ Vehicle - Threat        │
│   │   👤 Human             │   │ ⚠️ Loitering              │
│   │                        │   │                            │
│   │   🚧 Intrusion Area    │   │                            │
│   │                        │   │                            │
│   └────────────────────────┘   │                            │
│                                │                            │
├────────────────────────────────┴────────────────────────────┤
│                       📝 EVENT LOGS                          │
├──────────┬────────────┬────────────────┬─────────────────────┤
│ Time     │ Detection  │ Event          │ Status              │
├──────────┼────────────┼────────────────┼─────────────────────┤
│ 10:32:15 │ Human      │ Intrusion      │ ⚠️ Threat             │
│ 10:35:42 │ Vehicle    │ Intrusion      │ ⚠️ Threat             │
│ 10:38:10 │ Human      │ Loitering      │ ⚠️ Suspicious         │
└──────────┴────────────┴────────────────┴─────────────────────┘
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
                 │                  │
                 │           Vehicle Classification
                 │
                 ▼
          Object Tracking
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
 Intrusion Analysis   Behavior Analysis
        │                  │
        ▼                  ▼
 Drawn Restricted     Movement + Pose
      Area                 History
        │                  │
        ▼                  ▼
   Inside Area?       Suspicious?
     /     \           /      \
   YES      NO       YES       NO
    │        │        │         │
    ▼        ▼        ▼         ▼
⚠️ THREAT  NORMAL  ⚠️ SUSPICIOUS  NORMAL
    │                 ACTIVITY
    │                    │
    └──────────┬─────────┘
               ▼
       Alert Generation
               │
        ┌──────┼───────┐
        ▼      ▼       ▼
      Sound Snapshot  Log
               │       │
               └───┬───┘
                   ▼
              SQLite DB
                   │
                   ▼
            Streamlit Dashboard
```

---

# 🔄 Overall Working Process

### Step 1 — CCTV Input

The system receives a live video stream from an existing CCTV camera.

### Step 2 — Frame Processing

OpenCV captures and processes video frames.

### Step 3 — AI Object Detection

The YOLO-based model detects humans and vehicles.

### Step 4 — Object Tracking

Detected objects can be tracked across consecutive frames to analyze movement.

### Step 5 — Draw Intrusion Area

The operator defines a restricted area by drawing a virtual boundary on the CCTV view.

### Step 6 — Intrusion Analysis

The system checks whether a detected Human or Vehicle enters the restricted area.

### Step 7 — Threat Generation

If a Human or Vehicle enters the defined intrusion area:

```text
⚠️ POTENTIAL THREAT
```

is generated.

### Step 8 — Suspicious Activity Analysis

Human movement and posture can be analyzed independently of the virtual fence.

### Step 9 — Suspicious Activity Detection

Possible indicators include:

```text
Loitering
Rapid Movement
Erratic Movement
Crouching
Hands Raised
```

### Step 10 — Alert Generation

Confirmed events generate alerts on the dashboard.

### Step 11 — Snapshot and Logging

Relevant events can be saved as snapshots and stored in the database.

---

# 🔐 Two-Layer Security Model

IBVAP uses two complementary security mechanisms.

## Layer 1 — Location-Based Intrusion Detection

```text
Human / Vehicle
       ↓
Drawn Intrusion Area
       ↓
Enters Area?
       ↓
YES
       ↓
⚠️ POTENTIAL THREAT
```

## Layer 2 — Behavioral Suspicious Activity Detection

```text
Human
  ↓
Movement + Pose
  ↓
Behavior Analysis
  ↓
Unusual Indicator?
  ↓
YES
  ↓
⚠️ SUSPICIOUS ACTIVITY
```

Both layers can operate independently.

For example:

```text
Human
  ↓
Loitering
  +
Enters Restricted Area
  ↓
⚠️ SUSPICIOUS ACTIVITY
+
⚠️ INTRUSION THREAT
```

---

# 🧠 Technologies Used

| Technology             | Purpose                             |
| ---------------------- | ----------------------------------- |
| **Python**             | Main programming language           |
| **OpenCV**             | Video and image processing          |
| **YOLO / Ultralytics** | Object detection                    |
| **YOLO Pose**          | Human pose estimation               |
| **Object Tracking**    | Movement tracking across frames     |
| **Streamlit**          | Web-based dashboard                 |
| **SQLite**             | Event and alert storage             |
| **EasyOCR**            | Number plate/text recognition       |
| **NumPy**              | Numerical and image-data processing |
| **winsound**           | Windows audible alerts              |

---

# 📁 Project Structure

```text
IBVAP/
│
├── app.py
│
├── models/
│   ├── yolov_model.pt
│   └── yolo11n-pose.pt
│
├── detection/
│   ├── human_detection.py
│   ├── vehicle_detection.py
│   └── tracking.py
│
├── intrusion/
│   └── virtual_fence.py
│
├── suspicious_activity/
│   ├── loitering.py
│   ├── rapid_movement.py
│   ├── erratic_movement.py
│   └── posture_analysis.py
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
├── logs/
│   └── snapshots/
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
3. Classify supported vehicle types.
4. Allow operators to define a virtual intrusion area.
5. Detect human intrusion into restricted areas.
6. Detect vehicle intrusion into restricted areas.
7. Treat unauthorized human and vehicle entry as potential threats.
8. Analyze human behavior for suspicious activity indicators.
9. Generate real-time security alerts.
10. Store security events and snapshots.
11. Reduce dependence on expensive dedicated surveillance hardware.
12. Improve situational awareness and response time.
13. Provide a scalable software-based surveillance platform.

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

# 📈 Advantages

### 💰 Cost Effective

Enhances existing CCTV infrastructure without requiring dedicated smart surveillance cameras for the basic analytics layer.

### 🔧 Software-Based

AI-powered surveillance capabilities are provided primarily through software.

### 📡 Scalable

The architecture can be extended to multiple CCTV cameras and locations.

### ⚡ Real-Time Monitoring

The system can analyze video and generate security alerts in real time.

### 🚧 Flexible Intrusion Detection

Operators can define restricted areas according to the surveillance environment.

### 🧠 AI-Assisted Monitoring

Reduces the need for continuous manual observation of every CCTV feed.

### 🔍 Explainable Security Rules

Intrusion alerts are based on a clear condition:

```text
Object enters restricted area
        ↓
Potential Threat
```

Suspicious activity alerts are based on observable movement and posture indicators.

---

# 🔮 Future Scope

Future versions of IBVAP can include:

* Advanced ANPR
* Improved face detection
* Authorized facial-recognition capabilities
* Improved multi-object tracking
* More advanced suspicious-activity recognition
* Loitering detection improvements
* Abandoned vehicle detection
* Improved night-time detection
* Multi-camera monitoring
* Centralized command dashboard
* Advanced notification mechanisms
* Edge AI deployment
* Cloud-based monitoring
* Integration with authorized command and control systems
* More robust behavior-analysis models
* Improved low-light and adverse-weather detection

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
Object Tracking
      ↓
 ┌───────────────┬────────────────┐
 ↓               ↓                ↓
Intrusion      Movement          Pose
Analysis       Analysis          Analysis
 ↓               ↓                ↓
Restricted     Loitering       Crouching
Area           Rapid Move      Hands Raised
 ↓               ↓                ↓
Human/Vehicle  Erratic Move    Posture
Enters?        Analysis        Analysis
 ↓               ↓                ↓
YES             Suspicious      Suspicious
 ↓               Activity        Activity
⚠️ THREAT           └──────┬────────┘
                           ↓
                  ⚠️ SECURITY ALERT
                           ↓
                  Snapshot + Logging
                           ↓
                    SQLite Database
                           ↓
                  Streamlit Dashboard
```

---

# ⚠️ Important System Interpretation

IBVAP does **not** claim that a detected person is a criminal or that a particular behavior proves malicious intent.

For example:

```text
Loitering       ≠ Criminal
Crouching       ≠ Criminal
Rapid Movement  ≠ Criminal
Hands Raised    ≠ Criminal
```

These are **behavioral security indicators** that can be used to bring an event to the attention of an authorized security operator.

Similarly:

```text
Human enters restricted area
        ↓
Potential Threat
```

and:

```text
Vehicle enters restricted area
        ↓
Potential Threat
```

The system identifies the event for review rather than making a final determination about a person's intent.

---

# 🔐 Privacy & Security

IBVAP is primarily designed as a **security-event detection and monitoring system**.

The basic system focuses on:

```text
👤 Human
🚗 Vehicle
📍 Location
🚧 Intrusion Area
🧠 Behavior
⚠️ Security Event
```

The system does not require assigning individual identity labels to humans for basic intrusion detection.

Any future facial-recognition or identity-related functionality should only be implemented with appropriate authorization, privacy safeguards, cybersecurity controls, and compliance with applicable laws and organizational policies.

---

# 🏆 Project Vision

> **Transform existing CCTV infrastructure into an intelligent AI-assisted surveillance network that detects human and vehicle intrusions, identifies suspicious behavioral indicators, and provides real-time security alerts to authorized operators.**

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
* Object Detection
* Object Tracking
* Border Surveillance
* Software Engineering

---

# 📜 Disclaimer

IBVAP is an **academic/prototype project** developed for research, learning, and demonstration purposes.

Real-world deployment in border-security or other high-security environments would require appropriate testing, authorization, cybersecurity controls, privacy safeguards, reliability validation, human oversight, and compliance with applicable laws, regulations, and organizational policies.



