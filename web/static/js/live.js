/* Live tab: render the detection table from the shared state poller. */

(() => {
  const body = document.getElementById("detections-body");

  function fmt3(v) {
    return `(${v[0].toFixed(3)}, ${v[1].toFixed(3)}, ${v[2].toFixed(3)})`;
  }

  App.onState((state) => {
    const dets = state.detections || [];
    if (!dets.length) {
      body.innerHTML = '<tr class="empty"><td colspan="3">尚未偵測到 tag</td></tr>';
      return;
    }
    body.innerHTML = dets
      .map(
        (d) => `
        <tr>
          <td><span class="tag-id">${d.id}</span></td>
          <td class="mono">${fmt3(d.novel_pose)}</td>
          <td class="mono">${d.distance.toFixed(3)}</td>
        </tr>`
      )
      .join("");
  });
})();
