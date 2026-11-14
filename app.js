import {
  FilesetResolver,
  HandLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

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
const btnHoldActive = document.getElementById("btnHoldActive");
const chkLatchMode = document.getElementById("chkLatchMode");
const chkSwapHands = document.getElementById("chkSwapHands");
const rngAlpha = document.getElementById("rngAlpha");
const rngDeadband = document.getElementById("rngDeadband");
const lblAlpha = document.getElementById("lblAlpha");
const lblDeadband = document.getElementById("lblDeadband");

// 0..3: Sol El (L1..L4), 4..7: Sağ El (R1..R4)
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
const btnClearPlot = document.getElementById("btnClearPlot");

// Manuel ve bağlantı elemanları
const wsUrlInput = document.getElementById("wsUrlInput");
const btnConnectWs = document.getElementById("btnConnectWs");
const btnTrustCert = document.getElementById("btnTrustCert");
const btnSendManual = document.getElementById("btnSendManual");
const btnCenterAll = document.getElementById("btnCenterAll");

let handLandmarker = null;
let drawingUtils = new DrawingUtils(ctx);
let facingMode = "user";
let stream = null;
let running = false;
let isArmed = false;
let ws = null;
let lastSendTime = 0;

let smoothed = [90, 90, 90, 90, 90, 90, 90, 90];
let lastSent = [90, 90, 90, 90, 90, 90, 90, 90];

// PID grafik geçmişi (son 150 örnek)
const MAX_HIST = 150;
let pidHistory = [];
let stepTarget = null;
let stepPeak = 0;

// Sekme geçişleri
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

async function initMediaPipe() {
  try {
    offlineBadge.textContent = "El modeli yükleniyor (~13 MB)...";
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.6
    });
    offlineBadge.textContent = "Çevrimdışı Kullanıma Hazır ✓";
    offlineBadge.classList.add("ok");
  } catch (err) {
    offlineBadge.textContent = "Model yüklenemedi (İlk açılışta internet gerekir)";
    console.error(err);
  }
}

initMediaPipe();

// WebSocket bağlantısı ve telemetri alıcı
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
      if (msg.type === "telemetry") {
        onTelemetry(msg);
      }
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
    r.value = 90;
    document.getElementById(`manLbl${r.dataset.idx}`).textContent = "90°";
  });
  smoothed = [90, 90, 90, 90, 90, 90, 90, 90];
  valEls.forEach((el) => (el.textContent = "90°"));
  sendTargets(smoothed, true);
});

// PID kontrolleri
pidMotorSelect.addEventListener("change", () => {
  pidHistory = [];
  stepTarget = null;
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
btnStepTest.addEventListener("click", () => {
  pidHistory = [];
  const ch = Number(pidMotorSelect.value);
  stepTarget = null;
  stepPeak = 0;
  sendWs({ type: "step", ch, delta: 30 });
});
btnClearPlot.addEventListener("click", () => {
  pidHistory = [];
  stepTarget = null;
  drawPidChart();
});

function onTelemetry(t) {
  pidSpEl.textContent = `${Number(t.sp).toFixed(1)}°`;
  pidActEl.textContent = `${Number(t.act).toFixed(1)}°`;
  pidErrEl.textContent = `${Number(t.err).toFixed(1)}°`;

  if (stepTarget === null) stepTarget = t.sp;
  if (t.sp !== stepTarget) {
    stepTarget = t.sp;
    stepPeak = t.act;
  } else {
    stepPeak = Math.max(stepPeak, t.act);
  }
  const os = stepTarget > 1 ? Math.max(0, ((stepPeak - stepTarget) / Math.abs(stepTarget)) * 100) : 0;
  pidOsEl.textContent = `${os.toFixed(1)}%`;

  pidHistory.push({ sp: t.sp, act: t.act, err: t.err, out: t.out });
  if (pidHistory.length > MAX_HIST) pidHistory.shift();
  drawPidChart();
}

function drawPidChart() {
  const w = pidChart.width;
  const h = pidChart.height;
  pidCtx.clearRect(0, 0, w, h);

  // Izgara çizgileri
  pidCtx.strokeStyle = "#1e293b";
  pidCtx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (h / 4) * i;
    pidCtx.beginPath();
    pidCtx.moveTo(0, y);
    pidCtx.lineTo(w, y);
    pidCtx.stroke();
  }

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

  // Üst yarı: Açı (0..180°)
  plotLine("sp", "#38bdf8", 0, 180, 10, h * 0.58);
  plotLine("act", "#22c55e", 0, 180, 10, h * 0.58);

  // Alt yarı: Hata (-90..90°) ve PWM (-255..255)
  plotLine("err", "#f59e0b", -90, 90, h * 0.64, h - 10);
  plotLine("out", "#ef4444", -255, 255, h * 0.64, h - 10);
}

// 3D El Açısı Hesaplama
function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
function vec(a, b) { return { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z }; }
function angleBetween(u, v) {
  const dot = u.x * v.x + u.y * v.y + u.z * v.z;
  const magU = Math.hypot(u.x, u.y, u.z) || 1e-6;
  const magV = Math.hypot(v.x, v.y, v.z) || 1e-6;
  return (Math.acos(clamp(dot / (magU * magV), -1, 1)) * 180) / Math.PI;
}

function computeHandAngles(wl) {
  const thumbFlex = angleBetween(vec(wl[1], wl[2]), vec(wl[2], wl[4]));
  const m1 = clamp((thumbFlex / 80) * 180, 0, 180);
  const indexFlex = angleBetween(vec(wl[0], wl[5]), vec(wl[5], wl[8]));
  const m2 = clamp((indexFlex / 130) * 180, 0, 180);
  const palmAcross = vec(wl[5], wl[17]);
  const rollDeg = (Math.atan2(palmAcross.z, palmAcross.x) * 180) / Math.PI;
  const m3 = clamp(90 + rollDeg, 0, 180);
  const wristAxis = vec(wl[0], wl[9]);
  const yawDeg = (Math.atan2(wristAxis.x, -wristAxis.y) * 180) / Math.PI;
  const m4 = clamp(90 + yawDeg * 1.5, 0, 180);
  return [m1, m2, m3, m4];
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

        const raw = computeHandAngles(wl);
        const offset = isLeft ? 0 : 4;
        if (isLeft) sawLeft = true; else sawRight = true;

        for (let j = 0; j < 4; j++) {
          const ch = offset + j;
          smoothed[ch] = alpha * raw[j] + (1 - alpha) * smoothed[ch];
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
