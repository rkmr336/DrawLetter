/* ============================================================
   script.js — DrawLetter Air-Writing Logic
   - Hand tracking with MediaPipe Tasks Vision HandLandmarker
   - Gestures: 
       * Index up only: PEN DOWN (draws)
       * Index + Middle up: PEN UP (hover/move between strokes)
       * Open Palm: CLEAR (held 1s)
   - Coordinate mirroring: x = (1 - lm.x) * width
   - Exponential Moving Average (alpha = 0.5)
   - Stroke storage: Array of stroke point segments
   - Auto-predict: Debounced 1.5s after pen up
   ============================================================ */

'use strict';

// ── Constants ────────────────────────────────────────────────────────────────
const API_PREDICT = '/api/predict';
const API_HEALTH = '/api/health';
const AUTO_PREDICT_DELAY_MS = 1500;  // 1.5s after pen up / no hand
const MIN_POINTS_REQUIRED = 12;
const MIN_BBOX_SIZE = 28;             // Minimum pixel width/height of air-written letter
const EMA_ALPHA = 0.5;                // Smoothing factor for index fingertip
const CLEAR_HOLD_MS = 1000;           // Hold open palm for 1s to clear
const MAX_JUMP_DIST_RATIO = 0.35;     // Jump rejection threshold relative to canvas width
const BOX_SIZE_RATIO = 0.85;          // Guide box size ratio relative to video height

// ── DOM Elements ─────────────────────────────────────────────────────────────
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('trailCanvas');
const ctx = canvas.getContext('2d');
const exportCanvas = document.getElementById('exportCanvas');
const exportCtx = exportCanvas.getContext('2d');

const startCameraBtn = document.getElementById('startCameraBtn');
const startOverlay = document.getElementById('startOverlay');
const cameraNotice = document.getElementById('cameraNotice');
const feedbackToast = document.getElementById('feedbackToast');
const fpsText = document.getElementById('fpsText');
const gestureChip = document.getElementById('gestureChip');
const gestureChipText = document.getElementById('gestureChipText');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const autoPredictToggle = document.getElementById('autoPredictToggle');
const showSkeletonToggle = document.getElementById('showSkeletonToggle');

const backendBadge = document.getElementById('backendBadge');
const backendBadgeText = document.getElementById('backendBadgeText');
const predictionEmpty = document.getElementById('predictionEmpty');
const predictionActive = document.getElementById('predictionActive');
const predLetter = document.getElementById('predLetter');
const predConfidence = document.getElementById('predConfidence');
const predMeter = document.getElementById('predMeter');
const top3List = document.getElementById('top3List');
const previewBox = document.getElementById('previewBox');
const previewImg = document.getElementById('previewImg');
const cardLoading = document.getElementById('cardLoading');

// ── State Variables ──────────────────────────────────────────────────────────
let handLandmarker = null;
let isCameraRunning = false;
let lastVideoTime = -1;
let currentGesture = 'NONE';           // 'NONE', 'PEN_DOWN', 'PEN_UP', 'CLEAR'
let gestureDebounceCandidate = 'NONE';
let gestureDebounceCount = 0;
const GESTURE_DEBOUNCE_FRAMES = 3;

// Stroke representation: Array of strokes, each is an Array of {x, y}
let strokes = [];
let currentStroke = [];

// Smoothed fingertip position
let smoothedPoint = null;
let lastRawPoint = null;

// Clear gesture timer
let openPalmStartTime = null;

// Auto-predict timer
let autoPredictTimer = null;
let shouldClearOnNextPenDown = false;
let isPredicting = false;

// FPS tracking
let frameCount = 0;
let lastFpsUpdateTime = performance.now();

// ── Show Temporary Toast Feedback ────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, durationMs = 2000) {
  if (toastTimer) clearTimeout(toastTimer);
  feedbackToast.textContent = msg;
  feedbackToast.style.display = 'block';
  toastTimer = setTimeout(() => {
    feedbackToast.style.display = 'none';
  }, durationMs);
}

// ── Check Backend Health ─────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(API_HEALTH);
    const data = await res.json();
    if (data.status === 'ok') {
      backendBadge.className = 'status-badge ok';
      const acc = data.test_accuracy != null ? ` (${(data.test_accuracy * 100).toFixed(1)}% acc)` : '';
      backendBadgeText.textContent = `Model ready${acc}`;
    } else {
      backendBadge.className = 'status-badge err';
      backendBadgeText.textContent = 'Model error';
    }
  } catch (err) {
    backendBadge.className = 'status-badge err';
    backendBadgeText.textContent = 'API Offline';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

// ── MediaPipe Vision Initialization ──────────────────────────────────────────
async function initMediaPipe() {
  try {
    let FilesetResolver, HandLandmarker;

    // First attempt: local self-hosted vendor files
    try {
      const vision = await import('./vendor/mediapipe/vision_bundle.mjs');
      FilesetResolver = vision.FilesetResolver;
      HandLandmarker = vision.HandLandmarker;
      
      const wasmFileset = await FilesetResolver.forVisionTasks('./vendor/mediapipe/wasm');
      
      handLandmarker = await HandLandmarker.createFromOptions(wasmFileset, {
        baseOptions: {
          modelAssetPath: './models/hand_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 1,
      });
      console.log('[MediaPipe] Initialized with local vendor files (GPU delegate)');
    } catch (localErr) {
      console.warn('[MediaPipe] Local initialization fallback to CPU or CDN:', localErr);
      // Fallback try with CPU delegate locally
      try {
        const vision = await import('./vendor/mediapipe/vision_bundle.mjs');
        const wasmFileset = await vision.FilesetResolver.forVisionTasks('./vendor/mediapipe/wasm');
        handLandmarker = await vision.HandLandmarker.createFromOptions(wasmFileset, {
          baseOptions: {
            modelAssetPath: './models/hand_landmarker.task',
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          numHands: 1,
        });
        console.log('[MediaPipe] Initialized with local vendor files (CPU delegate)');
      } catch (cdnErr) {
        // Fallback to pinned CDN
        const cdnVision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14');
        const wasmFileset = await cdnVision.FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );
        handLandmarker = await cdnVision.HandLandmarker.createFromOptions(wasmFileset, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 1,
        });
        console.log('[MediaPipe] Initialized via jsDelivr CDN');
      }
    }
    return true;
  } catch (err) {
    console.error('[MediaPipe] Fatal error loading model:', err);
    cameraNotice.textContent = 'Error loading hand tracking models. Check browser compatibility and network.';
    return false;
  }
}

// ── Start Webcam ─────────────────────────────────────────────────────────────
startCameraBtn.addEventListener('click', async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraNotice.textContent = 'Camera API not supported or context insecure (requires localhost or HTTPS).';
    return;
  }

  startCameraBtn.disabled = true;
  startCameraBtn.textContent = 'Loading tracking model...';

  const ready = await initMediaPipe();
  if (!ready) {
    startCameraBtn.disabled = false;
    startCameraBtn.textContent = 'Retry Start';
    return;
  }

  startCameraBtn.textContent = 'Requesting camera...';

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    });

    webcam.srcObject = stream;
    webcam.addEventListener('loadeddata', () => {
      isCameraRunning = true;
      startOverlay.style.display = 'none';
      resizeCanvases();
      requestAnimationFrame(predictWebcamLoop);
    });
  } catch (err) {
    console.error('Camera access error:', err);
    startCameraBtn.disabled = false;
    startCameraBtn.textContent = 'Start camera';

    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      cameraNotice.textContent = 'Camera permission denied. Please allow camera access in your browser address bar.';
    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      cameraNotice.textContent = 'No camera found on this device.';
    } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      cameraNotice.textContent = 'Camera is already in use by another application.';
    } else {
      cameraNotice.textContent = `Camera error: ${err.message || err.name}`;
    }
  }
});

function resizeCanvases() {
  const rect = webcam.getBoundingClientRect();
  const w = rect.width || 640;
  const h = rect.height || 480;
  canvas.width = w;
  canvas.height = h;
}
window.addEventListener('resize', resizeCanvases);

// ── Gesture Classification ───────────────────────────────────────────────────
/**
 * Classifies gesture based on 21 landmarks:
 *  Index fingertip: 8, PIP: 6
 *  Middle fingertip: 12, PIP: 10
 *  Ring fingertip: 16, PIP: 14
 *  Pinky fingertip: 20, PIP: 18
 * y is smaller when finger is higher up (towards top of frame).
 */
function classifyHandGesture(landmarks) {
  // Check extension: tip.y < pip.y (above PIP)
  const isIndexUp  = landmarks[8].y < landmarks[6].y;
  const isMiddleUp = landmarks[12].y < landmarks[10].y;
  const isRingUp   = landmarks[16].y < landmarks[14].y;
  const isPinkyUp  = landmarks[20].y < landmarks[18].y;

  // Open Palm: Index, Middle, Ring, Pinky all extended
  if (isIndexUp && isMiddleUp && isRingUp && isPinkyUp) {
    return 'CLEAR';
  }

  // Peace sign: Index + Middle up, Ring + Pinky down
  if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp) {
    return 'PEN_UP';
  }

  // Pointing / Pen down: Only Index up
  if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) {
    return 'PEN_DOWN';
  }

  return 'PEN_UP'; // Default to hovering if unclear
}

// ── Guide Box Bounds Calculation ─────────────────────────────────────────────
function getGuideBox(w, h) {
  const size = Math.round(h * BOX_SIZE_RATIO);
  const x = Math.round((w - size) / 2);
  const y = Math.round((h - size) / 2);
  return { x, y, size, w: size, h: size };
}

function isPointInsideBox(pt, box) {
  return pt && pt.x >= box.x && pt.x <= (box.x + box.size) &&
         pt.y >= box.y && pt.y <= (box.y + box.size);
}

// ── Update Chip UI ───────────────────────────────────────────────────────────
function updateGestureChip(gesture, isOutside = false) {
  gestureChip.className = 'gesture-state-chip';
  if (isOutside) {
    gestureChip.classList.add('outside');
    gestureChipText.textContent = 'Move your finger inside the box';
    return;
  }
  if (gesture === 'PEN_DOWN') {
    gestureChip.classList.add('writing');
    gestureChipText.textContent = 'Pen down (writing)';
  } else if (gesture === 'PEN_UP') {
    gestureChip.classList.add('penup');
    gestureChipText.textContent = 'Pen up (moving)';
  } else if (gesture === 'CLEAR') {
    gestureChip.classList.add('clearing');
    gestureChipText.textContent = 'Clearing... (hold)';
  } else {
    gestureChipText.textContent = 'No hand';
  }
}

// ── Main Webcam Detection Loop ───────────────────────────────────────────────
async function predictWebcamLoop() {
  if (!isCameraRunning) return;

  // Calculate FPS
  frameCount++;
  const now = performance.now();
  if (now - lastFpsUpdateTime >= 1000) {
    const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdateTime));
    fpsText.textContent = `${fps} FPS`;
    frameCount = 0;
    lastFpsUpdateTime = now;
  }

  if (webcam.currentTime !== lastVideoTime && handLandmarker) {
    lastVideoTime = webcam.currentTime;
    const results = handLandmarker.detectForVideo(webcam, now);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const guideBox = getGuideBox(canvas.width, canvas.height);
    let isFingertipInside = false;

    if (results && results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];
      const detected = classifyHandGesture(landmarks);

      // Debounce gesture across 3 frames
      if (detected === gestureDebounceCandidate) {
        gestureDebounceCount++;
        if (gestureDebounceCount >= GESTURE_DEBOUNCE_FRAMES) {
          currentGesture = detected;
        }
      } else {
        gestureDebounceCandidate = detected;
        gestureDebounceCount = 1;
      }

      // Compute mirrored coordinates for Index Fingertip (landmark 8)
      // x mirrored: (1 - lm.x) * width
      const rawX = (1 - landmarks[8].x) * canvas.width;
      const rawY = landmarks[8].y * canvas.height;

      // Smooth point using Exponential Moving Average (EMA)
      if (!smoothedPoint) {
        smoothedPoint = { x: rawX, y: rawY };
      } else {
        // Jump rejection check
        const dist = Math.hypot(rawX - smoothedPoint.x, rawY - smoothedPoint.y);
        if (dist < canvas.width * MAX_JUMP_DIST_RATIO) {
          smoothedPoint.x = EMA_ALPHA * rawX + (1 - EMA_ALPHA) * smoothedPoint.x;
          smoothedPoint.y = EMA_ALPHA * rawY + (1 - EMA_ALPHA) * smoothedPoint.y;
        } else {
          // Reset on large jump
          smoothedPoint = { x: rawX, y: rawY };
        }
      }

      isFingertipInside = isPointInsideBox(smoothedPoint, guideBox);

      // Handle gestures with spatial filtering
      handleGestureState(currentGesture, smoothedPoint, isFingertipInside);

      // Optionally draw hand skeleton
      if (showSkeletonToggle.checked) {
        drawSkeleton(ctx, landmarks, canvas.width, canvas.height);
      }
    } else {
      // No hand in view
      if (currentGesture !== 'NONE') {
        onPenUp();
      }
      currentGesture = 'NONE';
      gestureDebounceCandidate = 'NONE';
      gestureDebounceCount = 0;
      smoothedPoint = null;
      openPalmStartTime = null;
      updateGestureChip('NONE');
    }

    // Draw the "Write inside the box" guide box with semi-transparent overlay
    const isPenActiveInBox = (currentGesture === 'PEN_DOWN') && isFingertipInside;
    drawGuideBox(ctx, guideBox, canvas.width, canvas.height, isPenActiveInBox);

    // Draw fingertip indicator dot if hand is present
    if (smoothedPoint) {
      drawFingertipMarker(ctx, smoothedPoint, currentGesture, isFingertipInside);
    }

    // Always redraw all recorded air strokes
    drawStrokes(ctx, strokes, currentStroke);
  }

  requestAnimationFrame(predictWebcamLoop);
}

// ── Gesture Handling & Stroke Collection ─────────────────────────────────────
function handleGestureState(gesture, point, isInsideBox) {
  // Palm clear gesture works anywhere on screen
  if (gesture === 'CLEAR') {
    updateGestureChip(gesture);
    if (!openPalmStartTime) {
      openPalmStartTime = performance.now();
    } else if (performance.now() - openPalmStartTime >= CLEAR_HOLD_MS) {
      clearAllStrokes();
      showToast('Cleared!');
      openPalmStartTime = null;
    }
    return;
  } else {
    openPalmStartTime = null;
  }

  // Peace sign (PEN_UP) works anywhere on screen
  if (gesture === 'PEN_UP') {
    updateGestureChip(gesture);
    onPenUp();
    return;
  }

  // Pen Down (Drawing)
  if (gesture === 'PEN_DOWN') {
    if (!isInsideBox) {
      // Finger outside the box: stroke breaks immediately, prompt user
      if (currentStroke.length > 0) {
        strokes.push([...currentStroke]);
        currentStroke = [];
      }
      updateGestureChip(gesture, true); // show "Move your finger inside the box"
      return;
    }

    // Inside box
    updateGestureChip(gesture, false);
    cancelAutoPredict();

    // If new pen-down after a prediction, automatically start fresh
    if (shouldClearOnNextPenDown) {
      clearAllStrokes();
      shouldClearOnNextPenDown = false;
    }

    // Append point to current active stroke
    currentStroke.push({ x: point.x, y: point.y });
    predictBtn.disabled = false;
  }
}

function onPenUp() {
  if (currentStroke.length > 0) {
    strokes.push([...currentStroke]);
    currentStroke = [];
  }

  // Schedule auto-prediction if enabled
  if (autoPredictToggle.checked && getTotalPointCount() >= MIN_POINTS_REQUIRED && !isPredicting) {
    scheduleAutoPredict();
  }
}

function getTotalPointCount() {
  let count = currentStroke.length;
  for (const s of strokes) {
    count += s.length;
  }
  return count;
}

// ── Auto-Predict Timer ───────────────────────────────────────────────────────
function scheduleAutoPredict() {
  cancelAutoPredict();
  autoPredictTimer = setTimeout(() => {
    if (getTotalPointCount() >= MIN_POINTS_REQUIRED && !isPredicting) {
      exportAndPredict();
    }
  }, AUTO_PREDICT_DELAY_MS);
}

function cancelAutoPredict() {
  if (autoPredictTimer) {
    clearTimeout(autoPredictTimer);
    autoPredictTimer = null;
  }
}

// ── Canvas Rendering Helpers ─────────────────────────────────────────────────
function drawGuideBox(targetCtx, box, canvasW, canvasH, isActive) {
  targetCtx.save();

  // 1. Semi-transparent dark overlay outside the box
  targetCtx.fillStyle = 'rgba(14, 17, 22, 0.45)';
  // Top region
  targetCtx.fillRect(0, 0, canvasW, box.y);
  // Bottom region
  targetCtx.fillRect(0, box.y + box.size, canvasW, canvasH - (box.y + box.size));
  // Left region
  targetCtx.fillRect(0, box.y, box.x, box.size);
  // Right region
  targetCtx.fillRect(box.x + box.size, box.y, canvasW - (box.x + box.size), box.size);

  // 2. Box Border (rounded rect)
  const radius = 16;
  targetCtx.beginPath();
  targetCtx.roundRect(box.x, box.y, box.size, box.size, radius);
  targetCtx.strokeStyle = isActive ? '#FFB547' : 'rgba(255, 181, 71, 0.4)';
  targetCtx.lineWidth = isActive ? 3 : 2;
  targetCtx.stroke();

  // 3. Corner accent marks (thick brackets at four corners)
  const cornerLen = 22;
  targetCtx.strokeStyle = '#FFB547';
  targetCtx.lineWidth = 4;
  targetCtx.lineCap = 'round';

  // Top-left
  targetCtx.beginPath();
  targetCtx.moveTo(box.x, box.y + cornerLen);
  targetCtx.lineTo(box.x, box.y + 6);
  targetCtx.arcTo(box.x, box.y, box.x + 6, box.y, 6);
  targetCtx.lineTo(box.x + cornerLen, box.y);
  targetCtx.stroke();

  // Top-right
  targetCtx.beginPath();
  targetCtx.moveTo(box.x + box.size - cornerLen, box.y);
  targetCtx.lineTo(box.x + box.size - 6, box.y);
  targetCtx.arcTo(box.x + box.size, box.y, box.x + box.size, box.y + 6, 6);
  targetCtx.lineTo(box.x + box.size, box.y + cornerLen);
  targetCtx.stroke();

  // Bottom-left
  targetCtx.beginPath();
  targetCtx.moveTo(box.x, box.y + box.size - cornerLen);
  targetCtx.lineTo(box.x, box.y + box.size - 6);
  targetCtx.arcTo(box.x, box.y + box.size, box.x + 6, box.y + box.size, 6);
  targetCtx.lineTo(box.x + cornerLen, box.y + box.size);
  targetCtx.stroke();

  // Bottom-right
  targetCtx.beginPath();
  targetCtx.moveTo(box.x + box.size - cornerLen, box.y + box.size);
  targetCtx.lineTo(box.x + box.size - 6, box.y + box.size);
  targetCtx.arcTo(box.x + box.size, box.y + box.size, box.x + box.size, box.y + box.size - 6, 6);
  targetCtx.lineTo(box.x + box.size, box.y + box.size - cornerLen);
  targetCtx.stroke();

  // 4. Subtle guide label: "Write inside the box"
  targetCtx.font = '600 13px Inter, sans-serif';
  targetCtx.textAlign = 'center';
  targetCtx.textBaseline = 'bottom';
  targetCtx.fillStyle = isActive ? 'rgba(255, 181, 71, 0.9)' : 'rgba(255, 255, 255, 0.45)';
  targetCtx.fillText('Write inside the box', box.x + box.size / 2, box.y - 8);

  targetCtx.restore();
}

function drawFingertipMarker(targetCtx, point, gesture, isInside = true) {
  targetCtx.save();
  targetCtx.beginPath();
  targetCtx.arc(point.x, point.y, 8, 0, Math.PI * 2);
  if (!isInside && gesture === 'PEN_DOWN') {
    targetCtx.fillStyle = '#F87171';
  } else {
    targetCtx.fillStyle = gesture === 'PEN_DOWN' ? '#FFB547' : '#60A5FA';
  }
  targetCtx.shadowColor = targetCtx.fillStyle;
  targetCtx.shadowBlur = 10;
  targetCtx.fill();
  targetCtx.restore();
}

function drawStrokes(targetCtx, completedStrokes, inProgressStroke) {
  targetCtx.save();
  targetCtx.strokeStyle = '#FFB547';
  targetCtx.lineWidth = 14;
  targetCtx.lineCap = 'round';
  targetCtx.lineJoin = 'round';
  targetCtx.shadowColor = 'rgba(255, 181, 71, 0.4)';
  targetCtx.shadowBlur = 8;

  const allStrokes = [...completedStrokes];
  if (inProgressStroke && inProgressStroke.length > 0) {
    allStrokes.push(inProgressStroke);
  }

  for (const stroke of allStrokes) {
    if (stroke.length < 2) {
      if (stroke.length === 1) {
        targetCtx.beginPath();
        targetCtx.arc(stroke[0].x, stroke[0].y, 7, 0, Math.PI * 2);
        targetCtx.fillStyle = '#FFB547';
        targetCtx.fill();
      }
      continue;
    }
    targetCtx.beginPath();
    targetCtx.moveTo(stroke[0].x, stroke[0].y);
    for (let i = 1; i < stroke.length; i++) {
      targetCtx.lineTo(stroke[i].x, stroke[i].y);
    }
    targetCtx.stroke();
  }
  targetCtx.restore();
}

function drawSkeleton(targetCtx, landmarks, w, h) {
  const connections = [
    [0,1],[1,2],[2,3],[3,4],        // Thumb
    [0,5],[5,6],[6,7],[7,8],        // Index
    [5,9],[9,10],[10,11],[11,12],   // Middle
    [9,13],[13,14],[14,15],[15,16], // Ring
    [13,17],[17,18],[18,19],[19,20],[0,17] // Pinky & Palm
  ];

  targetCtx.save();
  targetCtx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
  targetCtx.lineWidth = 2;

  for (const [i, j] of connections) {
    const x1 = (1 - landmarks[i].x) * w;
    const y1 = landmarks[i].y * h;
    const x2 = (1 - landmarks[j].x) * w;
    const y2 = landmarks[j].y * h;
    targetCtx.beginPath();
    targetCtx.moveTo(x1, y1);
    targetCtx.lineTo(x2, y2);
    targetCtx.stroke();
  }

  targetCtx.fillStyle = 'rgba(255, 255, 255, 0.5)';
  for (let i = 0; i < landmarks.length; i++) {
    const x = (1 - landmarks[i].x) * w;
    const y = landmarks[i].y * h;
    targetCtx.beginPath();
    targetCtx.arc(x, y, 3, 0, Math.PI * 2);
    targetCtx.fill();
  }
  targetCtx.restore();
}

// ── Clear Action ─────────────────────────────────────────────────────────────
function clearAllStrokes() {
  strokes = [];
  currentStroke = [];
  cancelAutoPredict();
  predictBtn.disabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}
clearBtn.addEventListener('click', () => {
  clearAllStrokes();
  showToast('Cleared');
});

// ── Export & Predict ─────────────────────────────────────────────────────────
async function exportAndPredict() {
  if (isPredicting) return;

  // Flush any in-progress points
  if (currentStroke.length > 0) {
    strokes.push([...currentStroke]);
    currentStroke = [];
  }

  const allPoints = strokes.flat();
  if (allPoints.length < MIN_POINTS_REQUIRED) {
    showToast('Write a bigger letter');
    return;
  }

  // Compute bounding box
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  for (const pt of allPoints) {
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
  }

  const boxW = maxX - minX;
  const boxH = maxY - minY;

  if (boxW < MIN_BBOX_SIZE || boxH < MIN_BBOX_SIZE) {
    showToast('Write a bigger letter');
    return;
  }

  // Render onto offscreen 400x400 canvas (white stroke on black background)
  exportCtx.fillStyle = '#000000';
  exportCtx.fillRect(0, 0, 400, 400);

  // Scale & center strokes within 400x400 with padding
  const padding = 40;
  const targetArea = 400 - padding * 2;
  const scale = targetArea / Math.max(boxW, boxH);
  const offsetX = (400 - boxW * scale) / 2 - minX * scale;
  const offsetY = (400 - boxH * scale) / 2 - minY * scale;

  // Stroke width matched to EMNIST ink density: 16% of larger side (min 12px, max 34px)
  const calcLineWidth = Math.min(34, Math.max(12, Math.max(boxW, boxH) * scale * 0.16));

  exportCtx.save();
  exportCtx.strokeStyle = '#FFFFFF';
  exportCtx.lineWidth = calcLineWidth;
  exportCtx.lineCap = 'round';
  exportCtx.lineJoin = 'round';

  for (const s of strokes) {
    if (s.length < 2) {
      if (s.length === 1) {
        exportCtx.beginPath();
        exportCtx.arc(s[0].x * scale + offsetX, s[0].y * scale + offsetY, calcLineWidth / 2, 0, Math.PI * 2);
        exportCtx.fillStyle = '#FFFFFF';
        exportCtx.fill();
      }
      continue;
    }
    exportCtx.beginPath();
    exportCtx.moveTo(s[0].x * scale + offsetX, s[0].y * scale + offsetY);
    for (let i = 1; i < s.length; i++) {
      exportCtx.lineTo(s[i].x * scale + offsetX, s[i].y * scale + offsetY);
    }
    exportCtx.stroke();
  }
  exportCtx.restore();

  const dataUrl = exportCanvas.toDataURL('image/png');

  // Trigger API call
  isPredicting = true;
  cardLoading.style.display = 'flex';

  try {
    const res = await fetch(API_PREDICT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl, include_preview: true }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Prediction failed' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderPrediction(data);
    shouldClearOnNextPenDown = true;
  } catch (err) {
    showToast(err.message || 'Inference error');
  } finally {
    isPredicting = false;
    cardLoading.style.display = 'none';
  }
}
predictBtn.addEventListener('click', exportAndPredict);

// ── Render Results in Prediction Card ────────────────────────────────────────
function renderPrediction(data) {
  predictionEmpty.style.display = 'none';
  predictionActive.style.display = 'flex';

  // Hero display
  predLetter.textContent = data.prediction || '?';
  const confPct = Math.round((data.confidence || 0) * 100);
  predConfidence.textContent = `${confPct}%`;
  predMeter.style.width = `${confPct}%`;

  // Top 3 list
  top3List.innerHTML = '';
  if (data.top3 && data.top3.length > 0) {
    for (const item of data.top3) {
      const pct = Math.round((item.prob || 0) * 100);
      const row = document.createElement('div');
      row.className = 'top3-row';
      row.innerHTML = `
        <span class="top3-label">${item.label}</span>
        <div class="top3-bar-track">
          <div class="top3-bar-fill" style="width: ${pct}%;"></div>
        </div>
        <span class="top3-percent">${pct}%</span>
      `;
      top3List.appendChild(row);
    }
  }

  // Processed 28x28 preview
  if (data.processed_preview) {
    previewImg.src = data.processed_preview;
    previewBox.style.display = 'flex';
  }
}

// ── Keyboard Shortcuts ───────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    exportAndPredict();
  } else if (e.key === 'Escape') {
    e.preventDefault();
    clearAllStrokes();
    showToast('Cleared');
  } else if (e.code === 'Space') {
    e.preventDefault();
    // Toggle pen state manually for accessibility / testing
    currentGesture = currentGesture === 'PEN_DOWN' ? 'PEN_UP' : 'PEN_DOWN';
    updateGestureChip(currentGesture);
    if (currentGesture === 'PEN_UP') onPenUp();
  }
});
