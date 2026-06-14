/* Settings tab: build the camera-settings form from the backend spec, apply
   changes live (debounced) and persist them. Values are remembered server-side
   in web/camera_settings.json, so a reload / restart restores them. */

(() => {
  const form = document.getElementById("settings-form");
  const resetBtn = document.getElementById("settings-reset");
  const statusEl = document.getElementById("settings-status");

  let spec = {};
  let values = {};
  let applyTimer = null;
  let loaded = false;

  function setStatus(msg) {
    statusEl.textContent = msg;
  }

  async function fetchSettings() {
    const res = await fetch("/api/settings", { cache: "no-store" });
    const data = await res.json();
    spec = data.spec;
    values = data.settings;
    if (!loaded) {
      build();
      loaded = true;
    }
    syncInputs();
  }

  function build() {
    form.innerHTML = "";
    for (const key of Object.keys(spec)) {
      const meta = spec[key];
      const field = document.createElement("div");
      field.className = "field";
      field.dataset.key = key;

      if (meta.type === "bool") {
        field.innerHTML = `
          <label class="switch">
            <input type="checkbox" data-key="${key}" />
            <span class="track"></span>
            <span>${meta.label}</span>
          </label>`;
        field.querySelector("input").addEventListener("change", (e) => {
          values[key] = e.target.checked;
          syncDisabled();
          queueApply();
        });
      } else {
        const nullable = !!meta.nullable;
        field.innerHTML = `
          <div class="head">
            <label>${meta.label}</label>
            ${nullable ? `<label class="switch"><input type="checkbox" data-enable="${key}" /><span class="track"></span><span class="hint">啟用</span></label>` : ""}
          </div>
          <div class="row">
            <input type="range" data-key="${key}" min="${meta.min}" max="${meta.max}" step="${meta.step}" />
            <span class="val" data-val="${key}"></span>
          </div>`;
        const range = field.querySelector(`input[type=range]`);
        range.addEventListener("input", (e) => {
          values[key] = Number(e.target.value);
          field.querySelector(`[data-val="${key}"]`).textContent = e.target.value;
          queueApply();
        });
        if (nullable) {
          field.querySelector(`input[data-enable="${key}"]`).addEventListener("change", (e) => {
            if (e.target.checked) {
              const def = values[key] == null ? meta.min : values[key];
              values[key] = Number(range.value || def);
            } else {
              values[key] = null;
            }
            syncDisabled();
            queueApply();
          });
        }
      }
      form.appendChild(field);
    }
  }

  function syncInputs() {
    for (const key of Object.keys(spec)) {
      const meta = spec[key];
      const v = values[key];
      if (meta.type === "bool") {
        const cb = form.querySelector(`input[data-key="${key}"]`);
        if (cb) cb.checked = !!v;
      } else {
        const range = form.querySelector(`input[type=range][data-key="${key}"]`);
        const valEl = form.querySelector(`[data-val="${key}"]`);
        if (meta.nullable) {
          const enable = form.querySelector(`input[data-enable="${key}"]`);
          if (enable) enable.checked = v != null;
        }
        if (range) range.value = v == null ? meta.min : v;
        if (valEl) valEl.textContent = v == null ? "—" : v;
      }
    }
    syncDisabled();
  }

  function syncDisabled() {
    for (const key of Object.keys(spec)) {
      const meta = spec[key];
      const field = form.querySelector(`.field[data-key="${key}"]`);
      if (!field) continue;
      let disabled = false;
      if (meta.disabled_when && values[meta.disabled_when]) disabled = true;
      if (meta.nullable && values[key] == null) {
        const range = field.querySelector("input[type=range]");
        const valEl = field.querySelector(`[data-val="${key}"]`);
        if (range) range.disabled = true;
        if (valEl) valEl.textContent = "—";
      } else {
        const range = field.querySelector("input[type=range]");
        if (range) range.disabled = false;
      }
      field.classList.toggle("disabled", disabled);
    }
  }

  function queueApply() {
    clearTimeout(applyTimer);
    setStatus("套用中…");
    applyTimer = setTimeout(apply, 150);
  }

  async function apply() {
    try {
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      const data = await res.json();
      values = data.settings;
      setStatus("已儲存 ✓");
      App.toast("鏡頭設定已套用並儲存");
    } catch (e) {
      setStatus("套用失敗");
    }
  }

  async function reset() {
    // Reset = send defaults by clearing the stored file via a fresh default set.
    const defaults = {
      auto_exposure: false, exposure: 900, brightness: 78,
      contrast: 24, gain: 12, saturation: null,
    };
    values = defaults;
    syncInputs();
    await apply();
  }

  resetBtn.addEventListener("click", reset);

  // Load when the settings tab is first shown (and once on startup).
  document.addEventListener("tabchange", (e) => {
    if (e.detail === "settings") fetchSettings();
  });
  fetchSettings();
})();
