/* Experiment tab: pair a ground-truth coordinate with the live novel + IPPE
   estimates, build records, and export them to CSV. */

(() => {
  const liveBody = document.getElementById("exp-live-body");
  const tagSelect = document.getElementById("exp-tag");
  const inX = document.getElementById("exp-x");
  const inY = document.getElementById("exp-y");
  const inZ = document.getElementById("exp-z");
  const createBtn = document.getElementById("exp-create");
  const msg = document.getElementById("exp-msg");
  const recordsBody = document.getElementById("exp-records");
  const countEl = document.getElementById("exp-count");
  const exportBtn = document.getElementById("exp-export");
  const clearBtn = document.getElementById("exp-clear");

  function fmt3(v) {
    if (!v) return "—";
    return `(${v[0].toFixed(3)}, ${v[1].toFixed(3)}, ${v[2].toFixed(3)})`;
  }
  function fmtAng(v) {
    if (!v) return "—";
    return `(${v[0].toFixed(1)}, ${v[1].toFixed(1)}, ${v[2].toFixed(1)})`;
  }
  function fmtErr(v) {
    return v == null ? "—" : v.toFixed(4);
  }
  function fmt1(v) {
    return v == null ? "—" : v.toFixed(2);
  }

  // ---- live current estimates + tag dropdown ----
  App.onState((state) => {
    const dets = state.detections || [];
    if (!dets.length) {
      liveBody.innerHTML = '<tr class="empty"><td colspan="3">尚未偵測到 tag</td></tr>';
    } else {
      liveBody.innerHTML = dets
        .map((d) => `
          <tr>
            <td><span class="tag-id">${d.id}</span></td>
            <td class="mono">${fmt3(d.novel_pose)}</td>
            <td class="mono">${fmt3(d.ippe_pose)}</td>
          </tr>`)
        .join("");
    }

    // Keep the tag dropdown in sync with detected ids, preserving the selection.
    const ids = dets.map((d) => String(d.id));
    const current = tagSelect.value;
    const existing = [...tagSelect.options].map((o) => o.value);
    if (ids.join(",") !== existing.join(",")) {
      tagSelect.innerHTML = ids.map((id) => `<option value="${id}">${id}</option>`).join("");
      if (ids.includes(current)) tagSelect.value = current;
    }
  });

  // ---- create record ----
  async function createRecord() {
    const tagId = tagSelect.value;
    if (tagId === "") {
      msg.textContent = "目前沒有偵測到 tag，無法建立資料";
      return;
    }
    const actual = [Number(inX.value), Number(inY.value), Number(inZ.value)];
    if (actual.some((v) => Number.isNaN(v))) {
      msg.textContent = "請輸入有效的 X / Y / Z 座標";
      return;
    }
    try {
      const res = await fetch("/api/experiment/record", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tag_id: Number(tagId), actual }),
      });
      const data = await res.json();
      if (!res.ok) {
        msg.textContent = data.error || "建立失敗";
        return;
      }
      msg.textContent = "";
      App.toast(`已建立 tag ${data.tag_id} 的資料`);
      loadRecords();
    } catch (e) {
      msg.textContent = "建立失敗（網路錯誤）";
    }
  }

  // ---- records table ----
  function renderRecords(records) {
    countEl.textContent = records.length;
    if (!records.length) {
      recordsBody.innerHTML = '<tr class="empty"><td colspan="12">尚無資料</td></tr>';
      return;
    }
    recordsBody.innerHTML = records
      .map((r) => {
        // Highlight whichever method had the smaller error.
        let nCls = "", iCls = "";
        if (r.novel_error != null && r.ippe_error != null) {
          if (r.novel_error <= r.ippe_error) { nCls = "err-good"; iCls = "err-bad"; }
          else { nCls = "err-bad"; iCls = "err-good"; }
        }
        return `
          <tr>
            <td>${r.index}</td>
            <td class="mono">${r.timestamp.replace("T", " ")}</td>
            <td><span class="tag-id">${r.tag_id}</span></td>
            <td class="mono">${fmt3(r.actual)}</td>
            <td class="mono">${fmt3(r.novel)}</td>
            <td class="mono"><span class="${nCls}">${fmtErr(r.novel_error)}</span></td>
            <td class="mono">${fmt3(r.ippe)}</td>
            <td class="mono"><span class="${iCls}">${fmtErr(r.ippe_error)}</span></td>
            <td class="mono">${fmtAng(r.novel_euler)}</td>
            <td class="mono">${fmtAng(r.ippe_euler)}</td>
            <td class="mono">${fmt1(r.orientation_diff_deg)}</td>
            <td><button class="btn danger tiny" data-del="${r.index}">刪除</button></td>
          </tr>`;
      })
      .join("");
  }

  async function loadRecords() {
    try {
      const res = await fetch("/api/experiment/records", { cache: "no-store" });
      renderRecords(await res.json());
    } catch (e) { /* ignore */ }
  }

  recordsBody.addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-del]");
    if (!btn) return;
    await fetch("/api/experiment/record/" + btn.dataset.del, { method: "DELETE" });
    loadRecords();
  });

  exportBtn.addEventListener("click", () => {
    window.location.href = "/api/experiment/export.csv";
  });

  clearBtn.addEventListener("click", async () => {
    if (!confirm("確定要清除所有實驗資料？")) return;
    await fetch("/api/experiment/records", { method: "DELETE" });
    loadRecords();
  });

  createBtn.addEventListener("click", createRecord);

  document.addEventListener("tabchange", (e) => {
    if (e.detail === "experiment") loadRecords();
  });
  loadRecords();
})();
