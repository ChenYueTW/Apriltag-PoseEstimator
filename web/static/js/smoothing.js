/* Temporal smoothing controls (on the merged Live tab). Builds from the backend
   spec, live-applies and persists via /api/smoothing. */

(() => {
  const form = document.getElementById("smoothing-form");
  const statusEl = document.getElementById("smoothing-status");

  let spec = {};
  let config = {};
  let loaded = false;
  let timer = null;

  async function fetchConfig() {
    const res = await fetch("/api/smoothing", { cache: "no-store" });
    const data = await res.json();
    spec = data.spec;
    config = data.config;
    if (!loaded) { build(); loaded = true; }
    sync();
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
            <span class="track"></span><span>${meta.label}</span>
          </label>`;
        field.querySelector("input").addEventListener("change", (e) => {
          config[key] = e.target.checked; sync(); queue();
        });
      } else if (meta.type === "number") {
        field.innerHTML = `
          <div class="head"><label>${meta.label}</label></div>
          <div class="row">
            <input type="range" data-key="${key}" min="${meta.min}" max="${meta.max}" step="${meta.step}" />
            <span class="val" data-val="${key}"></span>
          </div>`;
        field.querySelector("input").addEventListener("input", (e) => {
          config[key] = Number(e.target.value);
          field.querySelector(`[data-val="${key}"]`).textContent = e.target.value;
          queue();
        });
      } else if (meta.type === "choice") {
        field.innerHTML = `
          <label>${meta.label}</label>
          <select data-key="${key}">
            ${meta.options.map((o) => `<option value="${o}">${o}</option>`).join("")}
          </select>`;
        field.querySelector("select").addEventListener("change", (e) => {
          config[key] = e.target.value; queue();
        });
      }
      form.appendChild(field);
    }
  }

  function sync() {
    for (const key of Object.keys(spec)) {
      const meta = spec[key];
      const v = config[key];
      if (meta.type === "bool") {
        const cb = form.querySelector(`input[data-key="${key}"]`);
        if (cb) cb.checked = !!v;
      } else if (meta.type === "number") {
        const r = form.querySelector(`input[data-key="${key}"]`);
        const ve = form.querySelector(`[data-val="${key}"]`);
        if (r) r.value = v;
        if (ve) ve.textContent = v;
      } else if (meta.type === "choice") {
        const s = form.querySelector(`select[data-key="${key}"]`);
        if (s) s.value = v;
      }
    }
    // Grey out dependent fields when smoothing is disabled.
    for (const key of Object.keys(spec)) {
      const meta = spec[key];
      const field = form.querySelector(`.field[data-key="${key}"]`);
      if (!field) continue;
      const off = meta.disabled_when_off && !config[meta.disabled_when_off];
      field.classList.toggle("disabled", !!off);
    }
  }

  function queue() {
    clearTimeout(timer);
    statusEl.textContent = "套用中…";
    timer = setTimeout(apply, 150);
  }

  async function apply() {
    try {
      const res = await fetch("/api/smoothing", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const data = await res.json();
      config = data.config;
      sync();
      statusEl.textContent = "已儲存 ✓";
      App.toast("時間平滑設定已套用");
    } catch (e) {
      statusEl.textContent = "套用失敗";
    }
  }

  document.addEventListener("tabchange", (e) => {
    if (e.detail === "live") fetchConfig();
  });
  fetchConfig();
})();
