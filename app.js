import {
  FilesetResolver,
  HandLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const MOTOR_NAMES = [
  "L1 Başparmak", "L2 İşaret", "L3 Kendi Etrafında", "L4 Bilek Sağa/Sol",
  "R1 Başparmak", "R2 İşaret", "R3 Kendi Etrafında", "R4 Bilek Sağa/Sol"
];

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const offlineBadge = document.getElementById("offlineBadge");
const wsBadge = document.getElementById("wsBadge");
const wsDot = document.getElementById("wsDot");
const handStatus = document.getElementById("handStatus");
const fpsLabel = document.getElementById("fpsLabel");
const leftHandBadge = document.getElementById("leftHandBadge");
const rightHandBadge = document.getElementById("rightHandBadge");

const btnStartCam = document.getElementById("btnStartCam");
const btnFlipCam = document.getElementById("btnFlipCam");
const btnZeroRef = document.getElementById("btnZeroRef");
const btnHoldActive = document.getElementById("btnHoldActive");
const chkLatchMode = document.getElementById("chkLatchMode");
const chkSwapHands = document.getElementById("chkSwapHands");
const rngAlpha = document.getElementById("rngAlpha");
const rngDeadband = document.getElementById("rngDeadband");
const lblAlpha = document.getElementById("lblAlpha");
const lblDeadband = document.getElementById("lblDeadband");

const valEls = [
  document.getElementById("valL1"), document.getElementById("valL2"),
  document.getElementById("valL3"), document.getElementById("valL4"),
  document.getElementById("valR1"), document.getElementById("valR2"),
  document.getElementById("valR3"), document.getElementById("valR4")
];

// PID Analiz elemanları
const pidMotorSelect = document.getElementById("pidMotorSelect");
const pidChart = document.getElementById("pidChart");
const pidCtx = pidChart.getContext("2d");
const pidSpEl = document.getElementById("pidSp");
const pidActEl = document.getElementById("pidAct");
const pidErrEl = document.getElementById("pidErr");
const pidOsEl = document.getElementById("pidOs");
const inpKp = document.getElementById("inpKp");
const inpKi = document.getElementById("inpKi");
const inpKd = document.getElementById("inpKd");
const inpMaxPwm = document.getElementById("inpMaxPwm");
const btnApplyPidOne = document.getElementById("btnApplyPidOne");
const btnApplyPidAll = document.getElementById("btnApplyPidAll");
const btnStepTest = document.getElementById("btnStepTest");
const btnStepTestNeg = document.getElementById("btnStepTestNeg");
const btnClearPlot = document.getElementById("btnClearPlot");

// Manuel ve kalibrasyon elemanları
const wsUrlInput = document.getElementById("wsUrlInput");
const btnConnectWs = document.getElementById("btnConnectWs");
const btnTrustCert = document.getElementById("btnTrustCert");
const btnSendManual = document.getElementById("btnSendManual");
const btnCenterAll = document.getElementById("btnCenterAll");
const calibContainer = document.getElementById("calibContainer");

let handLandmarker = null;
let drawingUtils = new DrawingUtils(ctx);
let facingMode = "user";
let stream = null;
let running = false;
let isArmed = false;
let ws = null;
let lastSendTime = 0;

let smoothed = [0, 0, 0, 0, 0, 0, 0, 0];
let lastSent = [0, 0, 0, 0, 0, 0, 0, 0];

function createHandState() {
  return {
    refFrame: null,       // M4 bilek sağ/sol için ilk referans çerçevesi
    refRoll: null,        // M3 (9->13 noktalarından) ilk referans açısı
    thumbZero: null,      // Temas anındaki başparmak baz açısı
    indexZero: null       // Temas anındaki işaret parmağı baz açısı
  };
}

let handState = {
  left: createHandState(),
  right: createHandState()
};

const DEFAULT_CALIB = Array.from({ length: 8 }, () => ({ invert: false, gain: 1.0 }));
let calib = JSON.parse(localStorage.getItem("marmara_calib_v7") || "null") || DEFAULT_CALIB;

function saveCalib() {
  localStorage.setItem("marmara_calib_v7", JSON.stringify(calib));
}

function renderCalibUI() {
  calibContainer.innerHTML = "";
  MOTOR_NAMES.forEach((name, i) => {
    const row = document.createElement("div");
    row.className = "calib-row";
    row.innerHTML = `
      <span>${name}</span>
      <label class="inline-chk"><input type="checkbox" data-cidx="${i}" class="cal-inv" ${calib[i].invert ? "checked" : ""}/> Ters</label>
      <label class="inline-num">Çarpan <input type="number" data-cidx="${i}" class="cal-gain" step="0.1" min="0.2" max="5" value="${calib[i].gain}"/></label>
    `;
    calibContainer.appendChild(row);
  });
  calibContainer.querySelectorAll(".cal-inv").forEach((el) => {
    el.addEventListener("change", () => {
      calib[Number(el.dataset.cidx)].invert = el.checked;
      saveCalib();
    });
  });
  calibContainer.querySelectorAll(".cal-gain").forEach((el) => {
    el.addEventListener("change", () => {
      calib[Number(el.dataset.cidx)].gain = Number(el.value) || 1.0;
      saveCalib();
    });
  });
}
renderCalibUI();

const MAX_HIST = 150;
let pidHistory = [];
let stepStart = null;
let stepTarget = null;
let stepPeak = 0;

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab, .panel").forEach((el) => el.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./sw.js").catch(console.error);
}

rngAlpha.addEventListener("input", () => (lblAlpha.textContent = rngAlpha.value));
rngDeadband.addEventListener("input", () => (lblDeadband.textContent = `${rngDeadband.value}°`));

document.querySelectorAll(".man-range").forEach((r) => {
  r.addEventListener("input", () => {
    const idx = Number(r.dataset.idx);
    document.getElementById(`manLbl${idx}`).textContent = `${r.value}°`;
  });
});

btnZeroRef.addEventListener("click", () => {
  handState.left = createHandState();
  handState.right = createHandState();
  smoothed = [0, 0, 0, 0, 0, 0, 0, 0];
  valEls.forEach((el) => (el.textContent = "0°"));
  if (isArmed) sendTargets(smoothed, true);
});

function setArmed(state) {
  isArmed = state;
  btnHoldActive.classList.toggle("armed", state);
  btnHoldActive.textContent = state ? "8 MOTOR AKTİF (Canlı)" : "Motorlar Kilitli (Basılı Tut / Aç)";
  if (!state) sendTargets(smoothed, false);
}

btnHoldActive.addEventListener("pointerdown", () => { if (!chkLatchMode.checked) setArmed(true); });
btnHoldActive.addEventListener("pointerup", () => { if (!chkLatchMode.checked) setArmed(false); });
btnHoldActive.addEventListener("pointerleave", () => { if (!chkLatchMode.checked && isArmed) setArmed(false); });
btnHoldActive.addEventListener("click", () => { if (chkLatchMode.checked) setArmed(!isArmed); });

async function createLandmarker(vision, delegate) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.45,
    minHandPresenceConfidence: 0.45,
    minTrackingConfidence: 0.45
  });
}

async function initMediaPipe() {
  try {
    offlineBadge.textContent = "El modeli yükleniyor (~13 MB)...";
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    try {
      handLandmarker = await createLandmarker(vision, "GPU");
      offlineBadge.textContent = "Çevrimdışı Hazır (GPU) ✓";
    } catch (gpuErr) {
      console.warn("GPU delegate başlatılamadı, CPU moduna geçiliyor:", gpuErr);
      handLandmarker = await createLandmarker(vision, "CPU");
      offlineBadge.textContent = "Çevrimdışı Hazır (CPU) ✓";
    }
    offlineBadge.classList.add("ok");
  } catch (err) {
    offlineBadge.textContent = `Model hatası: ${err?.message || "İlk açılışta internet gerekir"}`;
    console.error(err);
  }
}

initMediaPipe();

function connectWS() {
  if (ws) ws.close();
  const url = wsUrlInput.value.trim();
  wsBadge.textContent = "ESP32: Bağlanıyor...";
  ws = new WebSocket(url);

  ws.onopen = () => {
    wsBadge.textContent = "ESP32: Bağlı ✓";
    wsBadge.classList.add("ok");
    wsDot.classList.add("ok");
    sendWs({ type: "watch", ch: Number(pidMotorSelect.value) });
  };
  ws.onclose = () => {
    wsBadge.textContent = "ESP32: Bağlı Değil";
    wsBadge.classList.remove("ok");
    wsDot.classList.remove("ok");
  };
  ws.onerror = () => {
    wsBadge.textContent = "ESP32: SSL/Bağlantı Hatası";
  };
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "telemetry") onTelemetry(msg);
    } catch (_) {}
  };
}

function sendWs(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify(obj));
}

function sendTargets(angles, active) {
  sendWs({
    type: "targets",
    m: angles.map((v) => Number(v.toFixed(1))),
    active
  });
}

btnConnectWs.addEventListener("click", connectWS);
btnTrustCert.addEventListener("click", () => window.open("https://192.168.4.1", "_blank"));

btnSendManual.addEventListener("click", () => {
  const vals = Array.from(document.querySelectorAll(".man-range")).map((r) => Number(r.value));
  smoothed = [...vals];
  vals.forEach((v, i) => (valEls[i].textContent = `${Math.round(v)}°`));
  sendTargets(vals, true);
});

btnCenterAll.addEventListener("click", () => {
  document.querySelectorAll(".man-range").forEach((r) => {
    r.value = 0;
    document.getElementById(`manLbl${r.dataset.idx}`).textContent = "0°";
  });
  smoothed = [0, 0, 0, 0, 0, 0, 0, 0];
  valEls.forEach((el) => (el.textContent = "0°"));
  sendTargets(smoothed, true);
});

pidMotorSelect.addEventListener("change", () => {
  pidHistory = [];
  stepTarget = null;
  stepStart = null;
  sendWs({ type: "watch", ch: Number(pidMotorSelect.value) });
  drawPidChart();
});

function applyPid(ch) {
  sendWs({
    type: "pid",
    ch,
    kp: Number(inpKp.value),
    ki: Number(inpKi.value),
    kd: Number(inpKd.value),
    maxPwm: Number(inpMaxPwm.value)
  });
}

btnApplyPidOne.addEventListener("click", () => applyPid(Number(pidMotorSelect.value)));
btnApplyPidAll.addEventListener("click", () => applyPid("all"));

function triggerStep(delta) {
  pidHistory = [];
  stepTarget = null;
  stepStart = null;
  stepPeak = 0;
  sendWs({ type: "step", ch: Number(pidMotorSelect.value), delta });
}

btnStepTest.addEventListener("click", () => triggerStep(30));
btnStepTestNeg.addEventListener("click", () => triggerStep(-30));
btnClearPlot.addEventListener("click", () => {
  pidHistory = [];
  stepTarget = null;
  stepStart = null;
  drawPidChart();
});

function onTelemetry(t) {
  pidSpEl.textContent = `${Number(t.sp).toFixed(1)}°`;
  pidActEl.textContent = `${Number(t.act).toFixed(1)}°`;
  pidErrEl.textContent = `${Number(t.err).toFixed(1)}°`;

  if (stepTarget === null) {
    stepTarget = t.sp;
    stepStart = t.act;
    stepPeak = t.act;
  } else if (t.sp !== stepTarget) {
    stepStart = t.act;
    stepTarget = t.sp;
    stepPeak = t.act;
  } else {
    const dir = stepTarget >= stepStart ? 1 : -1;
    if (dir > 0) stepPeak = Math.max(stepPeak, t.act);
    else stepPeak = Math.min(stepPeak, t.act);
  }

  const stepSize = Math.abs(stepTarget - stepStart);
  const overshootDeg = stepTarget >= stepStart ? (stepPeak - stepTarget) : (stepTarget - stepPeak);
  const osPct = stepSize > 1 ? Math.max(0, (overshootDeg / stepSize) * 100) : 0;
  pidOsEl.textContent = `${osPct.toFixed(1)}%`;

  pidHistory.push({ sp: t.sp, act: t.act, err: t.err, out: t.out });
  if (pidHistory.length > MAX_HIST) pidHistory.shift();
  drawPidChart();
}

function drawPidChart() {
  const w = pidChart.width;
  const h = pidChart.height;
  pidCtx.clearRect(0, 0, w, h);

  pidCtx.strokeStyle = "#1e293b";
  pidCtx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (h / 4) * i;
    pidCtx.beginPath();
    pidCtx.moveTo(0, y);
    pidCtx.lineTo(w, y);
    pidCtx.stroke();
  }
  const zeroY = (10 + h * 0.58) / 2;
  pidCtx.strokeStyle = "#475569";
  pidCtx.setLineDash([4, 4]);
  pidCtx.beginPath();
  pidCtx.moveTo(0, zeroY);
  pidCtx.lineTo(w, zeroY);
  pidCtx.stroke();
  pidCtx.setLineDash([]);

  if (pidHistory.length < 2) return;

  const plotLine = (key, color, minV, maxV, yTop, yBot) => {
    pidCtx.strokeStyle = color;
    pidCtx.lineWidth = 2;
    pidCtx.beginPath();
    pidHistory.forEach((p, i) => {
      const x = (i / (MAX_HIST - 1)) * w;
      const norm = clamp((p[key] - minV) / (maxV - minV), 0, 1);
      const y = yBot - norm * (yBot - yTop);
      if (i === 0) pidCtx.moveTo(x, y);
      else pidCtx.lineTo(x, y);
    });
    pidCtx.stroke();
  };

  plotLine("sp", "#38bdf8", -180, 180, 10, h * 0.58);
  plotLine("act", "#22c55e", -180, 180, 10, h * 0.58);
  plotLine("err", "#f59e0b", -90, 90, h * 0.64, h - 10);
  plotLine("out", "#ef4444", -255, 255, h * 0.64, h - 10);
}

// --- 3D Vektör Yardımcıları ---
function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
function vec(a, b) { return { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z }; }
function dot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
function normLen(a) { return Math.hypot(a.x, a.y, a.z) || 1e-6; }
function normalize(a) { const l = normLen(a); return { x: a.x / l, y: a.y / l, z: a.z / l }; }
function cross(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x
  };
}
function angleBetween(u, v) {
  const d = dot(u, v);
  return (Math.acos(clamp(d / (normLen(u) * normLen(v)), -1, 1)) * 180) / Math.PI;
}
function wrapDelta(d) {
  while (d > 180) d -= 360;
  while (d < -180) d += 360;
  return d;
}

function buildHandFrame(wl) {
  const fwd = normalize(vec(wl[0], wl[9]));
  const rawAcross = vec(wl[5], wl[17]);
  const norm = normalize(cross(fwd, rawAcross));
  const across = normalize(cross(norm, fwd));
  return { fwd, across, norm };
}

function computeV7Angles(wl, state) {
  const frame = buildHandFrame(wl);
  const palmWidth = normLen(vec(wl[5], wl[17])) || 0.08;

  // --- M1 (Başparmak -) & M2 (İşaret +): Bağımsız eklem açıları + temas sıfırlama ---
  const tipGap = normLen(vec(wl[4], wl[8]));
  const touchThresh = palmWidth * 0.42; // Parmak ucu yastıkları değdiğinde ~3.3cm

  // Başparmak kendi açısı: 1->4 ile sabit avuç 0->5 arasındaki açı (işaret parmağından bağımsız)
  const thumbRaw = angleBetween(vec(wl[1], wl[4]), vec(wl[0], wl[5]));
  // İşaret parmağı kendi açısı: 5->8 ile sabit 5->1 hattı arasındaki açı (başparmaktan bağımsız)
  const indexRaw = angleBetween(vec(wl[5], wl[8]), vec(wl[5], wl[1]));

  if (state.thumbZero === null) state.thumbZero = thumbRaw;
  if (state.indexZero === null) state.indexZero = indexRaw;

  let m1 = 0;
  let m2 = 0;
  let touching = tipGap <= touchThresh;

  if (touching) {
    // Uçlar değiyorsa ikisi de kesin 0° ve temas açısı kalibre edilir
    state.thumbZero = thumbRaw;
    state.indexZero = indexRaw;
    m1 = 0;
    m2 = 0;
  } else {
    m1 = -clamp(Math.max(0, thumbRaw - state.thumbZero) * 2.2, 0, 180);
    m2 = clamp(Math.max(0, indexRaw - state.indexZero) * 2.2, 0, 180);
  }

  // --- M3 (Kendi Etrafında Dönme): Doğrudan ortadaki 9 (orta kök) ve 13 (yüzük kök) noktaları ---
  const midKnuckleVec = vec(wl[9], wl[13]);
  const rollRaw = (Math.atan2(midKnuckleVec.z, midKnuckleVec.x) * 180) / Math.PI;
  if (state.refRoll === null) state.refRoll = rollRaw;
  const m3 = clamp(wrapDelta(rollRaw - state.refRoll), -180, 180);

  // --- M4 (Bilek Sağa-Sola): V6 ile aynı (dokunulmadı) ---
  if (!state.refFrame) state.refFrame = frame;
  const yawRad = Math.atan2(dot(frame.fwd, state.refFrame.across), dot(frame.fwd, state.refFrame.fwd));
  const m4 = clamp((yawRad * 180) / Math.PI, -180, 180);

  return { angles: [m1, m2, m3, m4], touching };
}

async function startCamera() {
  if (!handLandmarker) { alert("El tanıma modeli henüz yükleniyor, birkaç saniye bekleyin."); return; }
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  running = true;
  requestAnimationFrame(loop);
}

btnStartCam.addEventListener("click", startCamera);
btnFlipCam.addEventListener("click", () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  if (running) startCamera();
});

let frameCount = 0;
let fpsTimer = performance.now();

function loop(now) {
  if (!running) return;
  frameCount++;
  if (now - fpsTimer >= 1000) {
    fpsLabel.textContent = `${frameCount} FPS`;
    frameCount = 0;
    fpsTimer = now;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (video.readyState >= 2) {
    const res = handLandmarker.detectForVideo(video, now);
    let sawLeft = false;
    let sawRight = false;

    if (res.landmarks && res.landmarks.length > 0 && res.worldLandmarks) {
      const alpha = Number(rngAlpha.value);
      const deadband = Number(rngDeadband.value);

      res.landmarks.forEach((lm, idx) => {
        const wl = res.worldLandmarks[idx];
        const rawLabel = res.handedness?.[idx]?.[0]?.categoryName || "Right";
        let isLeft = rawLabel === "Left";
        if (chkSwapHands.checked) isLeft = !isLeft;

        const color = isLeft ? "#38bdf8" : "#4ade80";
        drawingUtils.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color, lineWidth: 3 });
        drawingUtils.drawLandmarks(lm, { color: "#f8fafc", radius: 4 });

        const st = isLeft ? handState.left : handState.right;
        if (isLeft) sawLeft = true; else sawRight = true;

        const { angles, touching } = computeV7Angles(wl, st);
        const offset = isLeft ? 0 : 4;

        for (let j = 0; j < 4; j++) {
          const ch = offset + j;
          // Uçlar değdiğinde M1 ve M2 kalıntı bırakmadan doğrudan 0°'a oturur
          if (touching && (j === 0 || j === 1)) {
            smoothed[ch] = 0;
          } else {
            let val = angles[j];
            if (calib[ch].invert) val = -val;
            val = clamp(val * calib[ch].gain, -180, 180);
            smoothed[ch] = alpha * val + (1 - alpha) * smoothed[ch];
          }
          valEls[ch].textContent = `${Math.round(smoothed[ch])}°`;
        }
      });

      leftHandBadge.textContent = sawLeft ? "Algılandı" : "Yok";
      leftHandBadge.className = sawLeft ? "mini-status ok" : "mini-status";
      rightHandBadge.textContent = sawRight ? "Algılandı" : "Yok";
      rightHandBadge.className = sawRight ? "mini-status ok" : "mini-status";
      handStatus.textContent = `Sol: ${sawLeft ? "✓" : "✗"} | Sağ: ${sawRight ? "✓" : "✗"}`;
      handStatus.className = "pill ok";

      if (isArmed && now - lastSendTime >= 33) {
        const changed = smoothed.some((v, i) => Math.abs(v - lastSent[i]) >= deadband);
        if (changed) {
          sendTargets(smoothed, true);
          lastSent = [...smoothed];
        } else {
          sendTargets(lastSent, true);
        }
        lastSendTime = now;
      }
    } else {
      leftHandBadge.textContent = "Yok";
      leftHandBadge.className = "mini-status";
      rightHandBadge.textContent = "Yok";
      rightHandBadge.className = "mini-status";
      handStatus.textContent = "Eller bekleniyor (Kilitli)";
      handStatus.className = "pill warn";
      if (isArmed) sendTargets(lastSent, false);
    }
  }
  requestAnimationFrame(loop);
}
