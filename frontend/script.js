(() => {
  "use strict";
  const root = document.documentElement;
  const themeToggle = document.getElementById("themeToggle");
  const themeIcon = document.getElementById("themeIcon");
  const menuBtn = document.getElementById("menuBtn");
  const mobileMenu = document.getElementById("mobileMenu");
  const siteHeader = document.querySelector(".site-header");
  const toast = document.getElementById("toast");
  const clock = document.getElementById("clock");

  /* =========================================================
     THEME
     ========================================================= */
  // Load saved theme, otherwise use the user's system preference.
  const savedTheme = localStorage.getItem("ibvap-theme");
  const systemTheme =
    window.matchMedia("(prefers-color-scheme: light)").matches
      ? "day"
      : "night";
  root.dataset.theme =
    savedTheme === "day" || savedTheme === "night"
      ? savedTheme
      : systemTheme;

  // Update the sun/moon icon.
  function updateThemeIcon() {
    if (!themeIcon) return;
    const isDay = root.dataset.theme === "day";
    themeIcon.textContent = isDay ? "☀" : "☾";
    if (themeToggle) {
      themeToggle.setAttribute(
        "aria-label",
        isDay ? "Switch to night mode" : "Switch to day mode"
      );
      themeToggle.title =
        isDay ? "Switch to night mode" : "Switch to day mode";
    }
  }
  updateThemeIcon();


  // Theme button.
  // IMPORTANT:
  // There is only ONE click handler here.
  // The previous version had two handlers, causing the theme
  // to switch twice and appear as if it wasn't working.
  themeToggle?.addEventListener("click", () => {
    if (root.dataset.theme === "day") {
      root.dataset.theme = "night";
    } else {
      root.dataset.theme = "day";
    }

    // Remember the user's choice.
    localStorage.setItem(
      "ibvap-theme",
      root.dataset.theme
    );
    updateThemeIcon();
    showToast(
      root.dataset.theme === "day"
        ? "Day mode enabled"
        : "Night mode enabled"
    );
  });

  /* =========================================================
     IST CLOCK
     ========================================================= */
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

  // Start clock immediately.
  updateClock();
  // Update every second.
  setInterval(updateClock, 1000);


  /* =========================================================
     MOBILE NAVIGATION
     ========================================================= */
  menuBtn?.addEventListener("click", () => {
    if (!mobileMenu) return;
    mobileMenu.classList.toggle("open");
  });

  // Close mobile menu when a link is clicked.
  mobileMenu?.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      mobileMenu.classList.remove("open");
    });
  });

  /* =========================================================
   SHRINK HEADER ON SCROLL
   ========================================================= */
  let headerTicking = false;

  function updateHeader() {
      if (!siteHeader) return;
      siteHeader.classList.toggle(
          "scrolled",
          window.scrollY > 90
      );
      headerTicking = false;
  }

  window.addEventListener(
      "scroll",
      () => {
          if (headerTicking) return;
          headerTicking = true;
          requestAnimationFrame(updateHeader);
      },
      { passive: true }
  );
  updateHeader();

  /* =========================================================
    SCROLL REVEAL ANIMATIONS
    ========================================================= */
  const revealItems = document.querySelectorAll(".reveal");

  if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(
          (entries) => {
              entries.forEach((entry) => {
                  if (entry.isIntersecting) {
                      entry.target.classList.add("visible");
                  } else {
                      entry.target.classList.remove("visible");
                  }
              });
          },
          {
              threshold: 0.12,
              rootMargin: "0px 0px -50px 0px"
          }
      );

      revealItems.forEach((element) => {
          observer.observe(element);
      });

  } else {
      revealItems.forEach((element) => {
          element.classList.add("visible");
      });
  }


  /* =========================================================
     PLAYBACK BUTTON
     ========================================================= */
  const playDemo =
    document.getElementById("playDemo");

  playDemo?.addEventListener("click", () => {
    showToast(
      "Playback interface ready for backend integration"
    );
  });


  /* =========================================================
     TOAST NOTIFICATION
     ========================================================= */
  function showToast(message) {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("show");

    // Clear any previous timer.
    clearTimeout(window.__ibvapToast);

    // Hide toast after 1.9 seconds.
    window.__ibvapToast = setTimeout(() => {
      toast.classList.remove("show");
    }, 1900);
  }
})();