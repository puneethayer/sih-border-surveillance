from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
VIDEO_PATH = ROOT / "cctv.mp4"
MODEL_PATH = ROOT / "yolo26n.pt"
OUTPUT_PATH = ROOT / "intrusion_result.mp4"
SNAPSHOT_DIR = ROOT / "logs" / "snapshots"
DB_PATH = ROOT / "database" / "ibvap.db"

app = FastAPI(title="IBVAP API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


class DetectionRequest(BaseModel):
    fence_points: list[list[int]] = Field(min_length=3, max_length=4)
    camera_id: str = "CAM_01"
    video: str = "cctv.mp4"


class DetectionState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.status = "idle"
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error: str | None = None
        self.result: dict[str, Any] | None = None
        self.fence_points: list[list[int]] = []
        self.camera_id = "CAM_01"
        self.thread: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            result = self.result or {}
            return {
                "status": self.status,
                "camera_id": self.camera_id,
                "fence_points": self.fence_points,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "frames_processed": result.get("processed_frames", 0),
                "intrusion_count": result.get("intrusion_count", 0),
                "output_video": bool(result.get("output_video")),
            }


state = DetectionState()


def _db_events() -> list[dict[str, Any]]:
    from database.db import get_all_events, init_db

    init_db()
    return get_all_events()


def _run_detection(req: DetectionRequest, video_path: Path) -> None:
    try:
        from intrusion import run_detection

        with state.lock:
            state.status = "running"
            state.started_at = time.time()
            state.finished_at = None
            state.error = None
            state.result = None

        result = run_detection(
            video_path=str(video_path),
            fence_points=[tuple(point) for point in req.fence_points],
            output_path=str(OUTPUT_PATH),
            camera_id=req.camera_id,
        )

        with state.lock:
            state.result = result
            state.status = "completed"
            state.finished_at = time.time()
    except Exception as exc:
        with state.lock:
            state.status = "error"
            state.error = f"{type(exc).__name__}: {exc}"
            state.finished_at = time.time()


def _resolve_video(name: str) -> Path:
    # Deliberately allow only files inside the project root.
    candidate = (ROOT / name).resolve()
    if candidate.parent != ROOT or candidate.name != name:
        raise HTTPException(status_code=400, detail="Invalid video name")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {name}")
    return candidate

VIDEO_FILES = {
    "cctv.mp4": BASE_DIR / "cctv.mp4",
    "intrusion_result.mp4": BASE_DIR / "intrusion_result.mp4",
}


@app.get("/api/video/{filename}")
def get_video(filename: str):
    video_path = VIDEO_FILES.get(filename)

    if video_path is None or not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        video_path,
        media_type="video/mp4"
    )

@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        from database.db import init_db
        init_db()
        db_ok = DB_PATH.exists()
    except Exception:
        db_ok = False

    return {
        "api": "online",
        "ai_model": MODEL_PATH.exists(),
        "database": db_ok,
        "camera_source": VIDEO_PATH.exists(),
        "status": state.status,
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    events = _db_events()
    now = time.time()
    recent = []
    for event in events:
        ts = event.get("timestamp")
        if not ts:
            continue
        try:
            # Stored timestamps are local/ISO strings; the dashboard only needs a recent window.
            import datetime as dt
            parsed = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            if (dt.datetime.now(dt.timezone.utc) - parsed).total_seconds() <= 86400:
                recent.append(event)
        except Exception:
            pass

    confidences = [float(e["confidence"]) for e in events if e.get("confidence") is not None]
    return {
        "active_camera": 1,
        "active_intrusions": int((state.result or {}).get("intrusion_count", 0)),
        "ai_events": len(events),
        "events_24h": len(recent),
        "ai_confidence": round((sum(confidences) / len(confidences)) * 100, 1) if confidences else 0,
        "detection_status": state.status,
    }


@app.get("/api/events")
def events(
    limit: int = Query(50, ge=1, le=500),
    event_type: str | None = None,
    camera_id: str | None = None,
    object_type: str | None = None,
) -> list[dict[str, Any]]:
    rows = _db_events()
    if event_type and event_type.lower() != "all":
        rows = [r for r in rows if str(r.get("event_type", "")).lower() == event_type.lower()]
    if camera_id and camera_id.lower() != "all":
        rows = [r for r in rows if str(r.get("camera_id", "")).lower() == camera_id.lower()]
    if object_type and object_type.lower() != "all":
        rows = [r for r in rows if str(r.get("object_type", "")).lower() == object_type.lower()]
    return rows[:limit]


@app.get("/api/events/latest")
def latest_event() -> dict[str, Any] | None:
    rows = _db_events()
    return rows[0] if rows else None


@app.get("/api/detection/status")
def detection_status() -> dict[str, Any]:
    return state.snapshot()


@app.post("/api/detection/start")
def start_detection(req: DetectionRequest) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail="YOLO model yolo26n.pt is missing")
    if len(req.fence_points) < 3:
        raise HTTPException(status_code=400, detail="At least 3 fence points are required")

    video_path = _resolve_video(req.video)
    with state.lock:
        if state.status == "running":
            raise HTTPException(status_code=409, detail="Detection is already running")
        state.fence_points = req.fence_points
        state.camera_id = req.camera_id
        state.status = "starting"
        state.error = None
        state.result = None

        worker = threading.Thread(target=_run_detection, args=(req, video_path), daemon=True)
        state.thread = worker
        worker.start()

    return {"ok": True, "message": "AI detection started", **state.snapshot()}


@app.post("/api/detection/reset")
def reset_detection() -> dict[str, Any]:
    with state.lock:
        if state.status == "running":
            raise HTTPException(status_code=409, detail="Cannot reset while detection is running")
        state.status = "idle"
        state.started_at = None
        state.finished_at = None
        state.error = None
        state.result = None
        state.fence_points = []
    return state.snapshot()


@app.get("/api/camera/frame")
def first_frame() -> StreamingResponse:
    if not VIDEO_PATH.exists():
        raise HTTPException(status_code=404, detail="cctv.mp4 not found")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(status_code=500, detail="Could not read CCTV frame")
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode CCTV frame")
    return StreamingResponse(iter([encoded.tobytes()]), media_type="image/jpeg")


def mjpeg_stream() -> Any:
    # Before detection completes, show the real CCTV source. Once the processed
    # file exists, stream its processed frames so the operator sees the AI overlays.
    while True:
        source = OUTPUT_PATH if state.status == "completed" and OUTPUT_PATH.exists() else VIDEO_PATH
        if not source.exists():
            time.sleep(0.5)
            continue

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            time.sleep(0.5)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        delay = max(0.01, 1.0 / min(fps, 30.0))

        while True:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                if state.status == "completed":
                    return
                break
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n" + encoded.tobytes() + b"\r\n"
                )
            time.sleep(delay)


@app.get("/api/stream")
def stream() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
    )


@app.get("/api/output-video")
def output_video() -> FileResponse:
    if not OUTPUT_PATH.exists():
        raise HTTPException(status_code=404, detail="No processed output video yet")
    return FileResponse(OUTPUT_PATH, media_type="video/mp4", filename="intrusion_result.mp4")


@app.get("/api/evidence/{filename}")
def evidence(filename: str) -> FileResponse:
    path = (SNAPSHOT_DIR / filename).resolve()
    if path.parent != SNAPSHOT_DIR.resolve() or not path.exists():
        raise HTTPException(status_code=404, detail="Evidence not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/evidence")
def evidence_list(limit: int = Query(50, ge=1, le=200)) -> list[dict[str, Any]]:
    rows = _db_events()
    output = []
    for row in rows[:limit]:
        snapshot = row.get("snapshot_path")
        if snapshot:
            output.append({**row, "evidence_url": f"/api/evidence/{Path(snapshot).name}"})
    return output
