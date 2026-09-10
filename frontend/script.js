(() => {
  "use strict";

  const root = document.documentElement;
  const API = "";
  const $ = (id) => document.getElementById(id);

  const themeToggle = $("themeToggle");
  const themeIcon = $("themeIcon");
  const menuBtn = $("menuBtn");
  const mobileMenu = $("mobileMenu");
  const siteHeader = document.querySelector(".site-header");
  const toast = $("toast");
  const clock = $("clock");

  const liveFeed = $("liveFeed");
  const cctvPanel = $("cctvPanel");
  const fenceCanvas = $("fenceCanvas");
  const fenceHint = $("fenceHint");
  const fenceCoordinates = $("fenceCoordinates");
  const fenceState = $("fenceState");
  const feedState = $("feedState");
  const feedMeta = $("feedMeta");
  const runDetection = $("runDetection");
  const resetFence = $("resetFence");
  const undoFence = $("undoFence");
  const eventList = $("eventList");

  let fencePoints = [];
  let lastDetectionStatus = "idle";

  // ------------------------------------------------------------
  // THEME
  // ------------------------------------------------------------
  const savedTheme = localStorage.getItem("ibvap-theme");
  const systemTheme = window.matchMedia("(prefers-color-scheme: light)").matches ? "day" : "night";
  root.dataset.theme = savedTheme === "day" || savedTheme === "night" ? savedTheme : systemTheme;

  function updateThemeIcon() {
    if (!themeIcon) return;
    const isDay = root.dataset.theme === "day";
    themeIcon.textContent = isDay ? "☀" : "☾";
    themeToggle?.setAttribute("aria-label", isDay ? "Switch to night mode" : "Switch to day mode");
  }

  updateThemeIcon();

  themeToggle?.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "day" ? "night" : "day";
    localStorage.setItem("ibvap-theme", root.dataset.theme);
    updateThemeIcon();
    showToast(root.dataset.theme === "day" ? "Day mode enabled" : "Night mode enabled");
  });

  // ------------------------------------------------------------
  // CLOCK
  // ------------------------------------------------------------
  function updateClock() {
    if (!clock) return;
    const time = new Intl.DateTimeFormat("en-IN", {
      timeZone: "Asia/Kolkata",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false
    }).format(new Date());
    clock.textContent = `${time} IST`;
  }
  updateClock();
  setInterval(updateClock, 1000);

  // ------------------------------------------------------------
  // MOBILE NAVIGATION
  // ------------------------------------------------------------
  menuBtn?.addEventListener("click", () => mobileMenu?.classList.toggle("open"));
  mobileMenu?.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => mobileMenu.classList.remove("open"));
  });

  // ------------------------------------------------------------
  // HEADER
  // ------------------------------------------------------------
  let headerTicking = false;
  function updateHeader() {
    if (!siteHeader) return;
    siteHeader.classList.toggle("scrolled", window.scrollY > 90);
    headerTicking = false;
  }
  window.addEventListener("scroll", () => {
    if (headerTicking) return;
    headerTicking = true;
    requestAnimationFrame(updateHeader);
  }, { passive: true });
  updateHeader();

  // ------------------------------------------------------------
  // SCROLL REVEAL — repeats when scrolling up/down
  // ------------------------------------------------------------
  const revealItems = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
        } else {
          entry.target.classList.remove("visible");
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -50px 0px" });
    revealItems.forEach((el) => observer.observe(el));
  } else {
    revealItems.forEach((el) => el.classList.add("visible"));
  }

  // ------------------------------------------------------------
  // FENCE CANVAS
  // ------------------------------------------------------------
  function resizeFenceCanvas() {
    if (!fenceCanvas || !cctvPanel) return;
    const rect = cctvPanel.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    fenceCanvas.width = Math.max(1, Math.round(rect.width * dpr));
    fenceCanvas.height = Math.max(1, Math.round(rect.height * dpr));
    fenceCanvas.style.width = `${rect.width}px`;
    fenceCanvas.style.height = `${rect.height}px`;
    const ctx = fenceCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawFence();
  }

  function imageCoordinatesFromClient(clientX, clientY) {
    if (!liveFeed || !cctvPanel) return null;
    const rect = liveFeed.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const naturalWidth = liveFeed.naturalWidth || 1;
    const naturalHeight = liveFeed.naturalHeight || 1;
    const x = Math.round(((clientX - rect.left) / rect.width) * naturalWidth);
    const y = Math.round(((clientY - rect.top) / rect.height) * naturalHeight);
    return {
      x: Math.max(0, Math.min(naturalWidth - 1, x)),
      y: Math.max(0, Math.min(naturalHeight - 1, y))
    };
  }

  function canvasPoint(point) {
    if (!liveFeed || !fenceCanvas) return null;
    const imgRect = liveFeed.getBoundingClientRect();
    const canvasRect = fenceCanvas.getBoundingClientRect();
    const naturalWidth = liveFeed.naturalWidth || 1;
    const naturalHeight = liveFeed.naturalHeight || 1;
    return {
      x: (point.x / naturalWidth) * imgRect.width + (imgRect.left - canvasRect.left),
      y: (point.y / naturalHeight) * imgRect.height + (imgRect.top - canvasRect.top)
    };
  }

  function drawFence() {
    if (!fenceCanvas) return;
    const ctx = fenceCanvas.getContext("2d");
    const rect = fenceCanvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!fencePoints.length) return;

    const points = fencePoints.map(canvasPoint).filter(Boolean);
    if (points.length >= 3) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.closePath();
      ctx.fillStyle = "rgba(239, 118, 93, 0.14)";
      ctx.fill();
      ctx.strokeStyle = "rgba(239, 118, 93, 0.95)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    points.forEach((p, index) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#ef765d";
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.font = "800 10px ui-monospace, monospace";
      ctx.fillText(String(index + 1), p.x + 9, p.y - 8);
    });
  }

  fenceCanvas?.addEventListener("click", (event) => {
    if (lastDetectionStatus === "running" || lastDetectionStatus === "starting") return;
    if (fencePoints.length >= 4) {
      showToast("Maximum 4 fence points reached");
      return;
    }
    const point = imageCoordinatesFromClient(event.clientX, event.clientY);
    if (!point) return;
    fencePoints.push(point);
    updateFenceUI();
  });

  undoFence?.addEventListener("click", () => {
    fencePoints.pop();
    updateFenceUI();
  });

  resetFence?.addEventListener("click", async () => {
    fencePoints = [];
    updateFenceUI();
    try {
      await fetch(`${API}/api/detection/reset`, { method: "POST" });
    } catch (_) {}
    showToast("Restricted zone reset");
  });

  window.addEventListener("resize", resizeFenceCanvas);
  liveFeed?.addEventListener("load", resizeFenceCanvas);
  setTimeout(resizeFenceCanvas, 250);

  function updateFenceUI() {
    drawFence();
    const count = fencePoints.length;
    fenceState.textContent = count >= 3 ? "ARMED" : `${count}/3 POINTS`;
    fenceHint.textContent = count < 3
      ? `CLICK ${3 - count} MORE POINT${3 - count === 1 ? "" : "S"} ON THE FRAME`
      : count === 3
        ? "VALID ZONE — ADD A FOURTH POINT OR RUN DETECTION"
        : "FOUR-POINT RESTRICTED ZONE READY";
    fenceCoordinates.textContent = fencePoints.map((p, i) => `P${i + 1} ${p.x},${p.y}`).join("  •  ");
    if (runDetection) runDetection.disabled = count < 3 || lastDetectionStatus === "running" || lastDetectionStatus === "starting";
    if (undoFence) undoFence.disabled = count === 0;
  }

  // ------------------------------------------------------------
  // API DATA
  // ------------------------------------------------------------
  async function api(path, options = {}) {
    const response = await fetch(`${API}${path}`, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options
    });
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        message = data.detail || message;
      } catch (_) {}
      throw new Error(message);
    }
    return response.json();
  }

  async function refreshHealth() {
    try {
      const data = await api("/api/health");
      $("healthAI").textContent = data.ai_model ? "ONLINE" : "ERROR";
      $("healthDB").textContent = data.database ? "ONLINE" : "ERROR";
      $("healthCamera").textContent = data.camera_source ? "CONNECTED" : "ERROR";
      feedMeta.textContent = data.camera_source ? "VIDEO SOURCE: CCTV / CAM_01" : "VIDEO SOURCE: UNAVAILABLE";
    } catch (_) {
      $("healthAI").textContent = "OFFLINE";
      $("healthDB").textContent = "OFFLINE";
      $("healthCamera").textContent = "OFFLINE";
    }
  }

  async function refreshStats() {
    try {
      const data = await api("/api/stats");
      $("statCamera").textContent = String(data.active_camera).padStart(2, "0");
      $("statIntrusions").textContent = String(data.active_intrusions).padStart(2, "0");
      $("statEvents").textContent = String(data.events_24h).padStart(2, "0");
      $("statConfidence").textContent = data.ai_confidence ? `${data.ai_confidence}%` : "--";
    } catch (_) {}
  }

  function formatTime(timestamp) {
    if (!timestamp) return "--:--:--";
    const date = new Date(timestamp.includes("T") ? timestamp : timestamp.replace(" ", "T"));
    if (Number.isNaN(date.getTime())) return String(timestamp).slice(-8);
    return new Intl.DateTimeFormat("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(date);
  }

  function renderEvents(events) {
    const limited = events.slice(0, 6);
    $("eventCount").textContent = String(limited.length).padStart(2, "0");
    if (!limited.length) {
      eventList.innerHTML = `<div class="event event-empty"><time>--:--:--</time><div class="event-indicator"></div><div><b>NO EVENTS RECORDED</b><span>Run AI detection to populate the live event stream.</span><small>CAM_01 • STANDBY</small></div></div>`;
      return;
    }
    eventList.innerHTML = limited.map((event) => {
      const intrusion = String(event.event_type || "").toLowerCase() === "intrusion";
      const objectName = String(event.object_type || "object");
      const confidence = event.confidence == null ? "--" : `${(Number(event.confidence) * 100).toFixed(1)}%`;
      return `<div class="event">
        <time>${formatTime(event.timestamp)}</time>
        <div class="event-indicator ${intrusion ? "danger" : ""}"></div>
        <div><b>${intrusion ? "PERIMETER INTRUSION" : `${objectName.toUpperCase()} DETECTED`}</b><span>${objectName} / Track #${event.track_id ?? "-"} / ${confidence}</span><small>${event.camera_id || "CAM_01"} • ${intrusion ? "ACTIVE" : "LOGGED"}</small></div>
      </div>`;
    }).join("");
  }

  async function refreshEvents() {
    try {
      const data = await api("/api/events?limit=20");
      renderEvents(data);
    } catch (_) {}
  }

  async function refreshDetectionStatus() {
    try {
      const data = await api("/api/detection/status");
      lastDetectionStatus = data.status;
      feedState.textContent = data.status === "running" || data.status === "starting"
        ? "AI PROCESSING"
        : data.status === "completed"
          ? "AI OUTPUT"
          : data.status === "error"
            ? "ERROR"
            : "READY";
      if (data.status === "error") showToast(data.error || "Detection failed");
      updateFenceUI();
      if (data.status === "completed") {
        liveFeed.src = `/api/stream?completed=${Date.now()}`;
        refreshEvents();
        refreshStats();
      }
    } catch (_) {}
  }

  runDetection?.addEventListener("click", async () => {
    if (fencePoints.length < 3) {
      showToast("Define at least 3 fence points first");
      return;
    }
    try {
      runDetection.disabled = true;
      feedState.textContent = "STARTING";
      await api("/api/detection/start", {
        method: "POST",
        body: JSON.stringify({ fence_points: fencePoints, camera_id: "CAM_01", video: "cctv.mp4" })
      });
      liveFeed.src = `/api/stream?run=${Date.now()}`;
      showToast("AI intrusion detection started");
      await refreshDetectionStatus();
    } catch (error) {
      showToast(error.message || "Could not start detection");
      await refreshDetectionStatus();
    }
  });

  // ------------------------------------------------------------
  // VIDEO CAROUSEL
  // ------------------------------------------------------------

  const videoTrack = $("videoTrack");
  const videoSlides = videoTrack
    ? [...videoTrack.querySelectorAll(".video-slide")]
    : [];

  const videoDots = document.querySelectorAll(".carousel-dot");
  const videoPrev = $("videoPrev");
  const videoNext = $("videoNext");
  const videoCounter = $("videoCounter");
  const cameraTitle = $("cameraTitle");

  let currentVideo = 0;

  function updateVideoCarousel(index, smooth = true) {

    if (!videoSlides.length || !videoTrack) return;

    currentVideo =
      (index + videoSlides.length) % videoSlides.length;

    const slide = videoSlides[currentVideo];

    videoTrack.scrollTo({
      left: slide.offsetLeft,
      behavior: smooth ? "smooth" : "auto"
    });

    videoSlides.forEach((item, i) => {
      item.classList.toggle("active", i === currentVideo);
    });

    videoDots.forEach((dot, i) => {
      dot.classList.toggle("active", i === currentVideo);
    });

    if (videoCounter) {
      videoCounter.textContent =
        `${String(currentVideo + 1).padStart(2, "0")} / ${String(videoSlides.length).padStart(2, "0")}`;
    }

    if (cameraTitle) {
      cameraTitle.textContent =
        slide.dataset.camera || `CAM_${String(currentVideo + 1).padStart(2, "0")}`;
    }

    videoSlides.forEach((item, i) => {
      const video = item.querySelector("video");

      if (!video) return;

      if (i === currentVideo) {
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  }

  videoPrev?.addEventListener("click", () => {
    updateVideoCarousel(currentVideo - 1);
  });

  videoNext?.addEventListener("click", () => {
    updateVideoCarousel(currentVideo + 1);
  });

  videoDots.forEach((dot) => {
    dot.addEventListener("click", () => {
      updateVideoCarousel(Number(dot.dataset.slide));
    });
  });

  /* Detect manual horizontal scrolling */
  let carouselScrollTimer;

  videoTrack?.addEventListener("scroll", () => {

    clearTimeout(carouselScrollTimer);

    carouselScrollTimer = setTimeout(() => {

      const index = Math.round(
        videoTrack.scrollLeft / videoTrack.clientWidth
      );

      if (index !== currentVideo) {
        updateVideoCarousel(index, false);
      }

    }, 80);

  });

  window.addEventListener("resize", () => {
    updateVideoCarousel(currentVideo, false);
  });

    updateVideoCarousel(0, false);
    
  // ------------------------------------------------------------
  // STARTUP + POLLING
  // ------------------------------------------------------------
  refreshHealth();
  refreshStats();
  refreshEvents();
  refreshDetectionStatus();
  updateFenceUI();

  setInterval(() => {
    refreshHealth();
    refreshStats();
    refreshEvents();
    refreshDetectionStatus();
  }, 2500);

  function showToast(message) {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(window.__ibvapToast);
    window.__ibvapToast = setTimeout(() => toast.classList.remove("show"), 2200);
  }
})();
