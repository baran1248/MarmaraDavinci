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

const btnStartCam = document.getElementById("btnStartCam");
const btnFlipCam = document.getElementById("btnFlipCam");
const btnHoldActive = document.getElementById("btnHoldActive");
const chkLatchMode = document.getElementById("chkLatchMode");
const rngAlpha = document.getElementById("rngAlpha");
const rngDeadband = document.getElementById("rngDeadband");
const lblAlpha = document.getElementById("lblAlpha");
const lblDeadband = document.getElementById("lblDeadband");

const valEls = [
  document.getElementById("valM1"),
  document.getElementById("valM2"),
  document.getElementById("valM3"),
  document.getElementById("valM4")
];

const wsUrlInput = document.getElementById("wsUrlInput");
const btnConnectWs = document.getElementById("btnConnectWs");
const btnTrustCert = document.getElementById("btnTrustCert");
const btnSendManual = document.getElementById("btnSendManual");

let handLandmarker = null;
let drawingUtils = new DrawingUtils(ctx);
let facingMode = "user";
let stream = null;
let running = false;
let isArmed = false;
let ws = null;
let lastSendTime = 0;

let smoothed = [90, 90, 90, 90];
let lastSent = [90, 90, 90, 90];

// Sekme geçişleri
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab, .panel").forEach((el) => el.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

// Service Worker kaydı
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./sw.js").catch(console.error);
}

// Slider etiketleri
rngAlpha.addEventListener("input", () => (lblAlpha.textContent = rngAlpha.value));
rngDeadband.addEventListener("input", () => (lblDeadband.textContent = `${rngDeadband.value}°`));

document.querySelectorAll(".man-range").forEach((r) => {
  r.addEventListener("input", () => {
    const idx = Number(r.dataset.idx);
    document.getElementById(`manLbl${idx + 1}`).textContent = `${r.value}°`;
  });
});

// Dead-man switch (Güvenlik kilidi)
function setArmed(state) {
  isArmed = state;
  btnHoldActive.classList.toggle("armed", state);
  btnHoldActive.textContent = state ? "MOTORLAR AKTİF (Canlı)" : "Motorlar Kilitli (Basılı Tut / Aç)";
  if (!state) sendAngles(smoothed, false);
}

btnHoldActive.addEventListener("pointerdown", () => {
  if (!chkLatchMode.checked) setArmed(true);
});
btnHoldActive.addEventListener("pointerup", () => {
  if (!chkLatchMode.checked) setArmed(false);
});
btnHoldActive.addEventListener("pointerleave", () => {
  if (!chkLatchMode.checked && isArmed) setArmed(false);
});
btnHoldActive.addEventListener("click", () => {
  if (chkLatchMode.checked) setArmed(!isArmed);
});

// MediaPipe yükleme ve önbellek doğrulama
async function initMediaPipe() {
  try {
    offlineBadge.textContent = "El modeli yükleniyor (~13 MB)...";
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 1,
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

// WebSocket bağlantısı
function connectWS() {
  if (ws) ws.close();
  const url = wsUrlInput.value.trim();
  wsBadge.textContent = "ESP32: Bağlanıyor...";
  ws = new WebSocket(url);

  ws.onopen = () => {
    wsBadge.textContent = "ESP32: Bağlı ✓";
    wsBadge.classList.add("ok");
    wsDot.classList.add("ok");
  };
  ws.onclose = () => {
    wsBadge.textContent = "ESP32: Bağlı Değil";
    wsBadge.classList.remove("ok");
    wsDot.classList.remove("ok");
  };
  ws.onerror = () => {
    wsBadge.textContent = "ESP32: SSL/Bağlantı Hatası";
  };
}

btnConnectWs.addEventListener("click", connectWS);
btnTrustCert.addEventListener("click", () => {
  window.open("https://192.168.4.1", "_blank");
});

btnSendManual.addEventListener("click", () => {
  const vals = Array.from(document.querySelectorAll(".man-range")).map((r) => Number(r.value));
  sendAngles(vals, true);
});

function sendAngles(angles, active) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const payload = JSON.stringify({
    m1: Number(angles[0].toFixed(1)),
    m2: Number(angles[1].toFixed(1)),
    m3: Number(angles[2].toFixed(1)),
    m4: Number(angles[3].toFixed(1)),
    active
  });
  ws.send(payload);
}

// Yardımcı vektör açısı fonksiyonları (3D worldLandmarks)
function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
}

function angleBetween(u, v) {
  const dot = u.x * v.x + u.y * v.y + u.z * v.z;
  const magU = Math.hypot(u.x, u.y, u.z) || 1e-6;
  const magV = Math.hypot(v.x, v.y, v.z) || 1e-6;
  return (Math.acos(clamp(dot / (magU * magV), -1, 1)) * 180) / Math.PI;
}

function computeMotorAngles(wl) {
  // M1: Başparmak bükülme (CMC 1 -> MCP 2 vs MCP 2 -> TIP 4)
  const thumbFlex = angleBetween(vec(wl[1], wl[2]), vec(wl[2], wl[4]));
  const m1 = clamp((thumbFlex / 80) * 180, 0, 180);

  // M2: İşaret parmağı bükülme (WRIST 0 -> INDEX_MCP 5 vs INDEX_MCP 5 -> INDEX_TIP 8)
  const indexFlex = angleBetween(vec(wl[0], wl[5]), vec(wl[5], wl[8]));
  const m2 = clamp((indexFlex / 130) * 180, 0, 180);

  // M3: Bilek döndürme (Roll: INDEX_MCP 5 -> PINKY_MCP 17 ekseninin Z-X açısı)
  const palmAcross = vec(wl[5], wl[17]);
  const rollDeg = (Math.atan2(palmAcross.z, palmAcross.x) * 180) / Math.PI;
  const m3 = clamp(90 + rollDeg, 0, 180);

  // M4: Bilek sağa-sola eğme (Yaw: WRIST 0 -> MIDDLE_MCP 9 ekseninin X-Y açısı)
  const wristAxis = vec(wl[0], wl[9]);
  const yawDeg = (Math.atan2(wristAxis.x, -wristAxis.y) * 180) / Math.PI;
  const m4 = clamp(90 + yawDeg * 1.5, 0, 180);

  return [m1, m2, m3, m4];
}

// Kamera başlatma
async function startCamera() {
  if (!handLandmarker) {
    alert("El tanıma modeli henüz yükleniyor, birkaç saniye bekleyin.");
    return;
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
  }
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
    if (res.landmarks && res.landmarks.length > 0 && res.worldLandmarks && res.worldLandmarks.length > 0) {
      handStatus.textContent = "El Algılandı ✓";
      handStatus.className = "pill ok";

      const lm = res.landmarks[0];
      drawingUtils.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: "#38bdf8", lineWidth: 3 });
      drawingUtils.drawLandmarks(lm, { color: "#f8fafc", radius: 4 });

      const raw = computeMotorAngles(res.worldLandmarks[0]);
      const alpha = Number(rngAlpha.value);
      const deadband = Number(rngDeadband.value);

      for (let i = 0; i < 4; i++) {
        smoothed[i] = alpha * raw[i] + (1 - alpha) * smoothed[i];
        valEls[i].textContent = `${Math.round(smoothed[i])}°`;
      }

      if (isArmed && now - lastSendTime >= 33) {
        const changed = smoothed.some((v, i) => Math.abs(v - lastSent[i]) >= deadband);
        if (changed) {
          sendAngles(smoothed, true);
          lastSent = [...smoothed];
        } else {
          // Heartbeat: bağlantı kesilmesin diye mevcut konumu aktif tut
          sendAngles(lastSent, true);
        }
        lastSendTime = now;
      }
    } else {
      handStatus.textContent = "El bekleniyor (Motorlar Kilitli)";
      handStatus.className = "pill warn";
      if (isArmed) sendAngles(lastSent, false);
    }
  }
  requestAnimationFrame(loop);
}
