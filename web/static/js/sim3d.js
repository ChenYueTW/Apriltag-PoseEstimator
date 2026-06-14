/* 3D simulation tab (PhotonVision-style).

   Renders the camera + its view frustum at the configured extrinsics and draws
   each detected tag in the shared world frame: a blue sphere at the novel-method
   pose and a green oriented plate at the PnP/IPPE pose. World frame is Z-up.

   Three.js is bundled locally (web/static/lib) so it works offline. */

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

let inited = false;
let renderer, scene, camera, controls, container;
let sceneData = null;
const tagObjs = new Map();

const DEFAULT_TAG_SIZE = 0.1651;
const COLOR_NOVEL = 0x2f81f7;
const COLOR_IPPE = 0x3fb950;
const COLOR_CAM = 0xd29922;

function setStatus(msg) {
  const el = document.getElementById("sim-status");
  if (el) el.textContent = msg;
}

function v3(arr) {
  return new THREE.Vector3(arr[0], arr[1], arr[2]);
}

// ---------------------------------------------------------------- scene setup
function init() {
  container = document.getElementById("sim-canvas");
  THREE.Object3D.DEFAULT_UP = new THREE.Vector3(0, 0, 1);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0d12);

  const w = container.clientWidth || 800;
  const h = container.clientHeight || 500;
  camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(1.6, -1.6, 1.2);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(w, h);
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.6, 0, 0.2);
  controls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const dir = new THREE.DirectionalLight(0xffffff, 0.6);
  dir.position.set(2, -2, 4);
  scene.add(dir);

  // Ground grid on the world z = 0 plane (GridHelper defaults to the XZ plane).
  const grid = new THREE.GridHelper(4, 40, 0x39424e, 0x222a33);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  // World origin axes (x=red, y=green, z=blue).
  scene.add(new THREE.AxesHelper(0.3));

  window.addEventListener("resize", onResize);
  inited = true;
  animate();
}

function onResize() {
  if (!inited || !container.clientWidth) return;
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

// --------------------------------------------------------------- camera gizmo
function buildCameraGizmo(s) {
  const group = new THREE.Group();
  const apex = v3(s.camera_pose);
  const fwd = v3(s.forward).normalize();
  const xax = v3(s.x_axis).normalize();
  const yax = v3(s.y_axis).normalize();

  // Camera body, oriented to the camera basis (right=x_hat, up=y_hat,
  // depth=forward) so it visibly tilts up by the pitch. Elongated along the
  // forward axis so the tilt is obvious.
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.06, 0.045, 0.10),
    new THREE.MeshStandardMaterial({ color: COLOR_CAM })
  );
  body.position.copy(apex);
  // The camera basis (right, up, forward) is left-handed, so build a proper
  // right-handed rotation (right' = up x forward) for a valid quaternion;
  // local +Z then points along forward (tilted up by the pitch).
  const bodyRight = new THREE.Vector3().crossVectors(yax, fwd).normalize();
  body.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(bodyRight, yax, fwd));
  group.add(body);

  // Frustum: rays through the 4 image corners, length L.
  const L = 0.5;
  const corners = [
    [0, 0], [s.width, 0], [s.width, s.height], [0, s.height],
  ].map(([px, py]) => {
    const xn = (px - s.cx) / s.fx;
    const yn = (py - s.cy) / s.fy;
    // dir = forward + xn*x_axis - yn*y_axis  (matches the backend optical->world)
    const dir = fwd.clone()
      .add(xax.clone().multiplyScalar(xn))
      .add(yax.clone().multiplyScalar(-yn))
      .normalize();
    return apex.clone().add(dir.multiplyScalar(L));
  });

  const mat = new THREE.LineBasicMaterial({ color: COLOR_CAM });
  const pts = [];
  corners.forEach((c) => { pts.push(apex.clone(), c.clone()); });        // apex -> corners
  for (let i = 0; i < 4; i++) { pts.push(corners[i], corners[(i + 1) % 4]); } // far rect
  const frustum = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), mat);
  group.add(frustum);

  // Forward (optical axis, blue) and up (green) arrows, so the camera's
  // orientation - including that it tilts upward by the pitch - is unmistakable.
  group.add(new THREE.ArrowHelper(fwd, apex, L * 1.3, 0x58a6ff, 0.08, 0.05));
  group.add(new THREE.ArrowHelper(yax, apex, 0.30, 0x3fb950, 0.07, 0.045));

  scene.add(group);
}

// --------------------------------------------------------------- tag objects
function makeLabel(text) {
  const canvas = document.createElement("canvas");
  canvas.width = 128; canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(20,26,33,0.85)";
  ctx.fillRect(0, 0, 128, 64);
  ctx.fillStyle = "#e6edf3";
  ctx.font = "bold 34px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("ID " + text, 64, 34);
  const tex = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
  sprite.scale.set(0.22, 0.11, 1);
  return sprite;
}

function makeLine(color) {
  const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
}

function setLine(line, a, b) {
  line.geometry.setFromPoints([v3(a), v3(b)]);
  line.geometry.attributes.position.needsUpdate = true;
}

function createTagObj(id) {
  const tagSize = (sceneData && sceneData.tag_size) || DEFAULT_TAG_SIZE;
  const group = new THREE.Group();

  // Novel pose marker (sphere)
  const novel = new THREE.Mesh(
    new THREE.SphereGeometry(0.025, 16, 16),
    new THREE.MeshStandardMaterial({ color: COLOR_NOVEL })
  );
  group.add(novel);

  // IPPE oriented plate, textured with the actual AprilTag id image.
  const ippe = new THREE.Group();
  const tex = new THREE.TextureLoader().load(`/api/tag_image/${id}.png`);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.NearestFilter;  // crisp tag pixels
  const plate = new THREE.Mesh(
    new THREE.PlaneGeometry(tagSize, tagSize),
    new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide })
  );
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.PlaneGeometry(tagSize, tagSize)),
    new THREE.LineBasicMaterial({ color: COLOR_IPPE })
  );
  ippe.add(plate, edges);
  group.add(ippe);

  const normal = makeLine(COLOR_IPPE);     // tag surface normal
  const camLine = makeLine(0x55606b);      // camera -> tag
  const label = makeLabel(id);
  group.add(normal, camLine, label);

  scene.add(group);
  return { group, novel, ippe, normal, camLine, label };
}

function updateTags(state) {
  if (!inited || !sceneData) return;
  const seen = new Set();

  for (const d of state.detections || []) {
    seen.add(d.id);
    let o = tagObjs.get(d.id);
    if (!o) { o = createTagObj(d.id); tagObjs.set(d.id, o); }

    if (d.novel_pose) {
      o.novel.visible = true;
      o.novel.position.set(...d.novel_pose);
      o.label.visible = true;
      o.label.position.set(d.novel_pose[0], d.novel_pose[1], d.novel_pose[2] + 0.12);
      o.camLine.visible = true;
      setLine(o.camLine, sceneData.camera_pose, d.novel_pose);
    } else {
      o.novel.visible = false;
      o.label.visible = false;
      o.camLine.visible = false;
    }

    if (d.ippe_pose && d.ippe_quat) {
      o.ippe.visible = true;
      o.ippe.position.set(...d.ippe_pose);
      o.ippe.quaternion.set(d.ippe_quat[0], d.ippe_quat[1], d.ippe_quat[2], d.ippe_quat[3]);
      const n = new THREE.Vector3(0, 0, 1)
        .applyQuaternion(o.ippe.quaternion)
        .multiplyScalar((sceneData.tag_size || DEFAULT_TAG_SIZE));
      o.normal.visible = true;
      setLine(o.normal, d.ippe_pose, [d.ippe_pose[0] + n.x, d.ippe_pose[1] + n.y, d.ippe_pose[2] + n.z]);
    } else {
      o.ippe.visible = false;
      o.normal.visible = false;
    }
  }

  for (const [id, o] of tagObjs) {
    if (!seen.has(id)) { scene.remove(o.group); tagObjs.delete(id); }
  }
}

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  renderer.render(scene, camera);
}

async function loadScene() {
  try {
    const res = await fetch("/api/scene", { cache: "no-store" });
    sceneData = await res.json();
    buildCameraGizmo(sceneData);
    controls.target.copy(
      v3(sceneData.camera_pose).add(v3(sceneData.forward).normalize().multiplyScalar(0.7))
    );
    setStatus(`相機 pitch ${sceneData.pitch_deg.toFixed(1)}° · tag ${(sceneData.tag_size * 100).toFixed(1)} cm`);
  } catch (e) {
    setStatus("無法載入場景設定");
  }
}

// Lazy-init when the simulation tab is first shown (canvas needs a non-zero size).
document.addEventListener("tabchange", (e) => {
  if (e.detail !== "sim") return;
  if (!inited) { init(); loadScene(); }
  else onResize();
});

if (window.App) window.App.onState(updateTags);

// Lightweight introspection hook (handy for debugging the scene graph).
// Pass a state object to inject it through updateTags() for testing.
window.__sim = (injectState) => {
  if (injectState) updateTags(injectState);
  // Camera body forward direction (local +Z in world) – should match forward_hat.
  let bodyForward = null;
  if (scene) {
    const gizmo = scene.children.find(
      (c) => c.isGroup && c.children.some((x) => x.isMesh && x.geometry.type === "BoxGeometry")
    );
    const body = gizmo && gizmo.children.find((x) => x.isMesh && x.geometry.type === "BoxGeometry");
    if (body) {
      const q = body.quaternion;
      bodyForward = [
        2 * (q.x * q.z + q.w * q.y),
        2 * (q.y * q.z - q.w * q.x),
        1 - 2 * (q.x * q.x + q.y * q.y),
      ];
    }
  }
  return {
    inited,
    sceneLoaded: !!sceneData,
    sceneChildren: scene ? scene.children.length : 0,
    tags: tagObjs.size,
    hasCanvas: !!(container && container.querySelector("canvas")),
    bodyForward,
  };
};
