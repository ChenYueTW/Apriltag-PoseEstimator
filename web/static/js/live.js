/* Live tab: render the detection table from the shared state poller. */

(() => {
  const body = document.getElementById("detections-body");

  function fmt3(v) {
    if (!v) return "—";
    return `(${v[0].toFixed(3)}, ${v[1].toFixed(3)}, ${v[2].toFixed(3)})`;
  }

  App.onState((state) => {
    const dets = state.detections || [];
    if (!dets.length) {
      body.innerHTML = '<tr class="empty"><td colspan="5">尚未偵測到 tag</td></tr>';
      return;
    }
    body.innerHTML = dets
      .map(
        (d) => `
        <tr>
          <td><span class="tag-id">${d.id}</span></td>
          <td class="mono">${fmt3(d.novel_pose)}</td>
          <td class="mono">${fmt3(d.ippe_pose)}</td>
          <td class="mono">${d.distance.toFixed(3)}</td>
          <td class="mono">${d.ippe_reproj_error != null ? d.ippe_reproj_error.toFixed(3) : "—"}</td>
        </tr>`
      )
      .join("");
  });
})();
