/* Shared app shell: tab switching + a single /api/state poller that fans out
   the latest state to any registered listener (live view, 3D view, experiment). */

const App = (() => {
  const listeners = [];
  let latestState = { detections: [], fps: 0, backend: "", synthetic: true };

  function onState(fn) {
    listeners.push(fn);
  }

  function getState() {
    return latestState;
  }

  // ---- tabs ----
  function initTabs() {
    const tabs = document.querySelectorAll(".tab");
    const panels = document.querySelectorAll(".panel");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        tabs.forEach((t) => t.classList.remove("active"));
        panels.forEach((p) => p.classList.remove("active"));
        tab.classList.add("active");
        const panel = document.getElementById("tab-" + tab.dataset.tab);
        if (panel) panel.classList.add("active");
        document.dispatchEvent(new CustomEvent("tabchange", { detail: tab.dataset.tab }));
      });
    });
  }

  // ---- status pills ----
  function updateStatus(state) {
    const backend = document.getElementById("pill-backend");
    const fps = document.getElementById("pill-fps");
    const tags = document.getElementById("pill-tags");
    if (backend) {
      backend.textContent = "backend: " + (state.backend || "–");
      backend.classList.toggle("synthetic", !!state.synthetic);
      backend.classList.toggle("live", !state.synthetic);
    }
    if (fps) fps.textContent = "FPS: " + (state.fps ?? "–");
    if (tags) tags.textContent = "tags: " + (state.detections ? state.detections.length : 0);
  }

  // ---- poller ----
  async function poll() {
    try {
      const res = await fetch("/api/state", { cache: "no-store" });
      if (res.ok) {
        latestState = await res.json();
        updateStatus(latestState);
        listeners.forEach((fn) => {
          try { fn(latestState); } catch (e) { console.error(e); }
        });
      }
    } catch (e) {
      // backend not ready / network hiccup – ignore, retry next tick
    } finally {
      setTimeout(poll, 120);
    }
  }

  // ---- toast helper (shared) ----
  let toastEl = null;
  let toastTimer = null;
  function toast(msg, color) {
    if (!toastEl) {
      toastEl = document.createElement("div");
      toastEl.className = "toast";
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = msg;
    toastEl.style.borderColor = color || "var(--accent-2)";
    toastEl.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove("show"), 1800);
  }

  // Attach MJPEG streams only after the document has loaded, so a never-ending
  // multipart stream does not block the page 'load' event (also helps tooling).
  function attachStreams() {
    document.querySelectorAll("img.mjpeg[data-src]").forEach((img) => {
      img.src = img.dataset.src;
    });
  }

  function init() {
    initTabs();
    poll();
    if (document.readyState === "complete") attachStreams();
    else window.addEventListener("load", attachStreams);
  }

  document.addEventListener("DOMContentLoaded", init);

  return { onState, getState, toast };
})();
