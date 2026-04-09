/* ================================================================
   CONFORMAL PREDICTION + REJECTION OPTION — rejection.js
   9-Step Interactive Simulation with Animation
   ================================================================ */
'use strict';

/* ─── CONFIG ──────────────────────────────────────────────────── */
const CFG = {
  nPerClass: 50,           // 50 × 3 = 150 total
  trainRatio: 0.60,
  calibRatio: 0.20,        // test = 0.20 → ~30 test pts
  knn: 7,
  alpha: 0.10,
  gridSize: 55,            // decision boundary grid
  wMin: -5.5,
  wMax:  5.5,
  // Clusters intentionally overlap to create ~20% base error → rejection benefit is visible
  classes: [
    { id: 0, name: 'A', color: '#6e8efb', cx: -1.4, cy:  0.5, sx: 1.3, sy: 1.2 },
    { id: 1, name: 'B', color: '#a78bfa', cx:  1.4, cy: -1.1, sx: 1.2, sy: 1.3 },
    { id: 2, name: 'C', color: '#38bdf8', cx:  0.3, cy:  1.9, sx: 1.1, sy: 1.0 },
  ],
  STEP_DURATIONS: [3200, 2800, 4500, 4000, 3500, 3500, 4000, 5000, 8000],
};

const STEPS_META = [
  { id: 1, icon: '📊', title: 'Veri Üretimi',
    desc: '3 sınıflı 2 boyutlu sentetik veri seti oluşturuluyor. Her sınıf bir Gaussian kümesinden 50 nokta üretiyor.',
    insight: '50 × 3 = 150 toplam nokta · 3 Gaussian kümesi' },
  { id: 2, icon: '✂️', title: 'Veri Bölme',
    desc: 'Veri seti eğitim (%60), kalibrasyon (%20) ve test (%20) olarak rastgele ayrılıyor.',
    insight: '%60 eğitim · %20 kalibrasyon · %20 test' },
  { id: 3, icon: '🧠', title: 'Model Eğitimi',
    desc: 'KNN (k=7) sınıflandırıcı yalnızca eğitim verisi üzerinde fitleniyor. Karar sınırı hesaplanıyor.',
    insight: 'Model hiç kalibrasyon verisini görmez!' },
  { id: 4, icon: '🎯', title: 'Conformal Prediction',
    desc: 'Kalibrasyon verisi ile uyumsuzluk pozisyonları hesaplanır. Eşik değeri q̂ belirlenir.',
    insight: 'score = 1 − P(true_class) · q̂ = (1−α) yüzdelik dilim' },
  { id: 5, icon: '🔵', title: 'Prediction Setleri',
    desc: 'Her test noktası için prediction set oluşturulur: (1 − P(c)) ≤ q̂ olan tüm sınıflar dahil.',
    insight: '● Tek etiket = emin · ◎ Çok etiket = belirsiz · ✕ Boş = anomali' },
  { id: 6, icon: '🌡️', title: 'Güven Hesaplama',
    desc: 'Her test noktasının güveni 1 − (2. en yüksek sınıf olasılığı) olarak hesaplanır. Yüksek değer = model 1. seçiminden emindir.',
    insight: 'Düşük güven → Mavi · Yüksek güven → Yeşil · Aralık: [0.5, 1.0]' },
  { id: 7, icon: '📋', title: 'Güvene Göre Sıralama',
    desc: 'Test noktaları güven değerine göre düşükten yükseğe sıralanır. En düşük güvenliler reddedilmeye aday.',
    insight: 'Sıralamada sol = düşük güven = reddet adayı' },
  { id: 8, icon: '🚫', title: 'Reddetme Mekanizması',
    desc: 'Reddetme oranı arttıkça en düşük güvenli noktalar çıkarılır. Geri kalanların doğruluğu artar.',
    insight: '🔴 Reddedilen · 🟢 Kabul Edilen + Doğru · 🟡 Kabul + Yanlış' },
  { id: 9, icon: '✨', title: 'Sonuç & İçgörü',
    desc: 'Trade-off analizi: reddetme eşiğini kaydırarak doğruluk kazanımını ve kapsama garantisini keşfet.',
    insight: 'Uncertainty-Aware AI · Güvenilir · Şeffaf · Kalibre' },
];

/* ─── STATE ───────────────────────────────────────────────────── */
let S = {
  step: 1, playing: false,
  animT: 0, animStart: 0, animFrame: null,
  speed: 1, rejRate: 0, alpha: 0.10,
  showDecision: true, showPredSets: true, showConfLabels: true,
  // Data
  all: [], train: [], calib: [], test: [],
  // Model
  boundary: null,
  // CP
  calibScores: [], qhat: 0,
  testProbs: [], testPredSets: [], testConfidence: [],
  testCorrect: [], testSorted: [],
  metricCurve: [],
  // Charts
  scoreChart: null, accuracyChart: null,
};

/* ─── UTILS ───────────────────────────────────────────────────── */
function randn() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function quantile(arr, q) {
  const s = [...arr].sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return lo === hi ? s[lo] : s[lo] * (hi - pos) + s[hi] * (pos - lo);
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}
function easeOut(t) { return 1 - (1 - t) ** 3; }
function easeInOut(t) { return t < .5 ? 4*t*t*t : 1 - (-2*t+2)**3/2; }

/* ─── DATA GENERATION ─────────────────────────────────────────── */
function generateData() {
  const all = [];
  CFG.classes.forEach(cls => {
    for (let i = 0; i < CFG.nPerClass; i++) {
      all.push({ wx: cls.cx + cls.sx * randn(), wy: cls.cy + cls.sy * randn(), cls: cls.id });
    }
  });
  // Fisher-Yates shuffle
  for (let i = all.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [all[i], all[j]] = [all[j], all[i]];
  }
  return all;
}

function splitData(all) {
  const n = all.length;
  const nTrain = Math.floor(n * CFG.trainRatio);
  const nCalib  = Math.floor(n * CFG.calibRatio);
  return {
    train: all.slice(0, nTrain).map(p => ({ ...p, split: 'train' })),
    calib: all.slice(nTrain, nTrain + nCalib).map(p => ({ ...p, split: 'calib' })),
    test:  all.slice(nTrain + nCalib).map(p => ({ ...p, split: 'test' })),
  };
}

/* ─── KNN MODEL ───────────────────────────────────────────────── */
function knnPredict(wx, wy, train, k = CFG.knn) {
  const dists = train.map(p => ({ d: (p.wx-wx)**2 + (p.wy-wy)**2, cls: p.cls }));
  dists.sort((a, b) => a.d - b.d);
  const votes = [0, 0, 0];
  dists.slice(0, k).forEach(n => votes[n.cls]++);
  const total = k + 0.3;
  const probs = votes.map(v => (v + 0.1) / total);
  const sum = probs.reduce((s,v) => s+v, 0);
  return probs.map(p => p / sum);
}

function computeDecisionBoundary(train) {
  const G = CFG.gridSize;
  const grid = [];
  for (let gy = 0; gy <= G; gy++) {
    for (let gx = 0; gx <= G; gx++) {
      const wx = lerp(CFG.wMin, CFG.wMax, gx / G);
      const wy = lerp(CFG.wMin, CFG.wMax, 1 - gy / G);
      const probs = knnPredict(wx, wy, train);
      const cls = probs.indexOf(Math.max(...probs));
      grid.push({ gx, gy, cls, prob: Math.max(...probs) });
    }
  }
  return grid;
}

/* ─── CONFORMAL PREDICTION ────────────────────────────────────── */
function ncScore(probs, trueClass) { return 1 - probs[trueClass]; }

function computeCalibScores(calib, train) {
  return calib.map(p => ncScore(knnPredict(p.wx, p.wy, train), p.cls));
}

function computeQhat(scores, alpha) {
  const n = scores.length;
  const level = Math.min((1 - alpha) * (1 + 1/n), 1);
  return quantile(scores, level);
}

function computePredSet(probs, qhat) {
  return probs.map((p, i) => (1 - p) <= qhat ? i : -1).filter(i => i >= 0);
}

function computeMetricCurve(test, testCorrect, testSorted) {
  const n = test.length;
  const curve = [];
  for (let r = 0; r <= 0.8; r += 0.02) {
    const nRej = Math.floor(r * n);
    const rejSet = new Set(testSorted.slice(0, nRej));
    const accepted = test.map((_,i) => i).filter(i => !rejSet.has(i));
    const nCorr = accepted.filter(i => testCorrect[i]).length;
    curve.push({ r, acc: accepted.length > 0 ? nCorr / accepted.length : 0, n: accepted.length });
  }
  return curve;
}

/* ─── CONFIDENCE: 1 minus second-largest probability ─────────── */
function computeConfidence(probs) {
  // Higher = model is more "separated" from its 2nd best guess
  const sorted = [...probs].sort((a, b) => b - a);
  return 1 - sorted[1]; // range roughly [0.5, 1.0] for 3-class
}

/* ─── CANVAS SETUP ────────────────────────────────────────────── */
let mainCanvas, mainCtx, auxCanvas, auxCtx;
const M = { top: 28, right: 28, bottom: 28, left: 36 };

function setupCanvas(id, logH) {
  const el = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const w = el.parentElement.clientWidth || 600;
  el.width  = w * dpr;
  el.height = logH * dpr;
  el.style.width  = '100%';
  el.style.height = logH + 'px';
  const ctx = el.getContext('2d');
  ctx.scale(dpr, dpr);
  el.lw = w; el.lh = logH;
  return { canvas: el, ctx };
}

function w2c(wx, wy, canvas) {
  const W = canvas.lw - M.left - M.right;
  const H = canvas.lh - M.top  - M.bottom;
  return {
    cx: M.left + (wx - CFG.wMin) / (CFG.wMax - CFG.wMin) * W,
    cy: M.top  + (1 - (wy - CFG.wMin) / (CFG.wMax - CFG.wMin)) * H,
  };
}

function c2w(cx, cy, canvas) {
  const W = canvas.lw - M.left - M.right;
  const H = canvas.lh - M.top  - M.bottom;
  return {
    wx: CFG.wMin + (cx - M.left) / W * (CFG.wMax - CFG.wMin),
    wy: CFG.wMin + (1 - (cy - M.top) / H) * (CFG.wMax - CFG.wMin),
  };
}

/* ─── CONFIDENCE COLOR ────────────────────────────────────────── */
function confColor(conf) {
  // New formula range: ~0.5 (uncertain) → 1.0 (certain)
  const t = clamp((conf - 0.50) / 0.50, 0, 1);
  const r = Math.round(lerp(110, 52,  t));
  const g = Math.round(lerp(142, 211, t));
  const b = Math.round(lerp(251, 153, t));
  return `rgb(${r},${g},${b})`;
}

/* ─── DRAWING PRIMITIVES ──────────────────────────────────────── */
function clearC(ctx, canvas) {
  ctx.clearRect(0, 0, canvas.lw, canvas.lh);
}

function drawGrid(ctx, canvas) {
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let v = -4; v <= 4; v += 2) {
    const {cx} = w2c(v, 0, canvas);
    const {cy} = w2c(0, v, canvas);
    ctx.beginPath(); ctx.moveTo(cx, M.top); ctx.lineTo(cx, canvas.lh - M.bottom); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(M.left, cy); ctx.lineTo(canvas.lw - M.right, cy); ctx.stroke();
  }
}

function drawBoundary(ctx, canvas, alpha01) {
  if (!S.boundary || !S.showDecision) return;
  const G = CFG.gridSize;
  const W = canvas.lw - M.left - M.right;
  const H = canvas.lh - M.top  - M.bottom;
  const cW = W / G, cH = H / G;
  S.boundary.forEach(cell => {
    const cls = CFG.classes[cell.cls];
    const op = alpha01 * lerp(0.04, 0.18, (cell.prob - 0.33) / 0.67);
    ctx.fillStyle = hexToRgba(cls.color, op);
    ctx.fillRect(M.left + cell.gx * cW, M.top + cell.gy * cH, cW + 1, cH + 1);
  });
}

function dot(ctx, cx, cy, { color, r=5, alpha=1, stroke=null, strokeW=1.5, fill=true }) {
  ctx.globalAlpha = clamp(alpha, 0, 1);
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2);
  if (fill) { ctx.fillStyle = color; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = strokeW; ctx.stroke(); }
  ctx.globalAlpha = 1;
}

function cross(ctx, cx, cy, sz=7, color='#f87171', alpha=1) {
  ctx.globalAlpha = clamp(alpha, 0, 1);
  ctx.strokeStyle = color; ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(cx-sz, cy-sz); ctx.lineTo(cx+sz, cy+sz);
  ctx.moveTo(cx+sz, cy-sz); ctx.lineTo(cx-sz, cy+sz);
  ctx.stroke(); ctx.globalAlpha = 1;
}

function ring(ctx, cx, cy, r, color, alpha=1, lw=1.8) {
  ctx.globalAlpha = clamp(alpha, 0, 1);
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2);
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
  ctx.globalAlpha = 1;
}

function label(ctx, text, x, y, { color='rgba(255,255,255,0.65)', size=10, font='JetBrains Mono', align='left' }={}) {
  ctx.fillStyle = color; ctx.font = `${size}px ${font}`;
  ctx.textAlign = align; ctx.fillText(text, x, y); ctx.textAlign = 'left';
}

/* ─── STEP RENDERERS ──────────────────────────────────────────── */

/* STEP 1 — Data generation */
function renderStep1(t) {
  clearC(mainCtx, mainCanvas); drawGrid(mainCtx, mainCanvas);
  const nShow = Math.floor(easeOut(t) * S.all.length);
  for (let i = 0; i < nShow; i++) {
    const p = S.all[i];
    const { cx, cy } = w2c(p.wx, p.wy, mainCanvas);
    const cls = CFG.classes[p.cls];
    const al = i < nShow - 1 ? 0.75 : clamp((t * S.all.length - i) * 3, 0.1, 0.75);
    dot(mainCtx, cx, cy, { color: cls.color, r: 4.5, alpha: al, stroke: hexToRgba(cls.color, 0.5), strokeW: 1 });
  }
  // Class labels
  CFG.classes.forEach(cls => {
    const {cx, cy} = w2c(cls.cx, cls.cy, mainCanvas);
    label(mainCtx, `Sınıf ${cls.name}`, cx, cy - 16, { color: cls.color, size: 11 });
  });
}

/* STEP 2 — Data split */
const SPLIT_COLORS = { train: '#38bdf8', calib: '#fbbf24', test: '#34d399' };
function renderStep2(t) {
  clearC(mainCtx, mainCanvas); drawGrid(mainCtx, mainCanvas);
  S.all.forEach(p => {
    const { cx, cy } = w2c(p.wx, p.wy, mainCanvas);
    const sColor = SPLIT_COLORS[p.split];
    const clsColor = CFG.classes[p.cls].color;
    const animT = easeInOut(clamp(t * 1.5, 0, 1));
    // Interpolate color
    const c = blendColor(clsColor, sColor, animT);
    const sz = p.split === 'test' ? 5.5 : 4;
    dot(mainCtx, cx, cy, { color: c, r: sz, alpha: 0.82, stroke: hexToRgba(sColor, 0.5 * animT), strokeW: 1.5 });
  });
  // Legend
  const lx = mainCanvas.lw - M.right - 110, ly = M.top + 10;
  drawSplitLegend(mainCtx, lx, ly, easeOut(t));
}

function blendColor(hex1, hex2, t) {
  const r1=parseInt(hex1.slice(1,3),16), g1=parseInt(hex1.slice(3,5),16), b1=parseInt(hex1.slice(5,7),16);
  const r2=parseInt(hex2.slice(1,3),16), g2=parseInt(hex2.slice(3,5),16), b2=parseInt(hex2.slice(5,7),16);
  return `rgb(${Math.round(lerp(r1,r2,t))},${Math.round(lerp(g1,g2,t))},${Math.round(lerp(b1,b2,t))})`;
}

function drawSplitLegend(ctx, lx, ly, alpha) {
  const items = [
    { color: SPLIT_COLORS.train, label: `Eğitim (${S.train.length})` },
    { color: SPLIT_COLORS.calib, label: `Kalibrasyon (${S.calib.length})` },
    { color: SPLIT_COLORS.test,  label: `Test (${S.test.length})` },
  ];
  ctx.globalAlpha = alpha;
  ctx.fillStyle = 'rgba(15,19,32,0.88)';
  ctx.beginPath(); ctx.roundRect(lx - 8, ly - 6, 126, 72, 8); ctx.fill();
  items.forEach((it, i) => {
    const y = ly + i * 22 + 10;
    ctx.beginPath(); ctx.arc(lx + 6, y, 5, 0, Math.PI*2);
    ctx.fillStyle = it.color; ctx.fill();
    ctx.fillStyle = '#8892b0'; ctx.font = '11px Inter';
    ctx.fillText(it.label, lx + 16, y + 4);
  });
  ctx.globalAlpha = 1;
}

/* STEP 3 — Model training */
function renderStep3(t) {
  clearC(mainCtx, mainCanvas); drawGrid(mainCtx, mainCanvas);
  drawBoundary(mainCtx, mainCanvas, easeOut(clamp(t * 1.4, 0, 1)));
  S.train.forEach(p => {
    const { cx, cy } = w2c(p.wx, p.wy, mainCanvas);
    dot(mainCtx, cx, cy, { color: CFG.classes[p.cls].color, r: 4, alpha: 0.65, stroke: 'rgba(255,255,255,0.08)', strokeW: 0.5 });
  });
  // Progress indicator
  if (t < 0.9) {
    const pct = Math.round(t / 0.9 * 100);
    label(mainCtx, `🧠 Karar sınırı hesaplanıyor… ${pct}%`, M.left + 6, M.top + 18, { color: 'rgba(110,142,251,0.8)', size: 12 });
  } else {
    label(mainCtx, '✅ Model hazır', M.left + 6, M.top + 18, { color: 'rgba(52,211,153,0.9)', size: 12 });
  }
}

/* STEP 4 — Conformal prediction (show calib points with score colors) */
function renderStep4(t) {
  clearC(mainCtx, mainCanvas); drawGrid(mainCtx, mainCanvas);
  drawBoundary(mainCtx, mainCanvas, 0.6);
  // Train (faded)
  S.train.forEach(p => {
    const { cx, cy } = w2c(p.wx, p.wy, mainCanvas);
    dot(mainCtx, cx, cy, { color: CFG.classes[p.cls].color, r: 3, alpha: 0.25 });
  });
  // Calib points colored by whether score > qhat
  const delay = 0.3;
  S.calib.forEach((p, i) => {
    const { cx, cy } = w2c(p.wx, p.wy, mainCanvas);
    const al = easeOut(clamp((t - delay - i * 0.008) / 0.4, 0, 1));
    if (al <= 0) return;
    const score = S.calibScores[i];
    const color = score > S.qhat ? '#f87171' : '#34d399';
    dot(mainCtx, cx, cy, { color, r: 5.5, alpha: al * 0.9, stroke: 'rgba(255,255,255,0.25)', strokeW: 1 });
  });
  // qhat label
  const qa = easeOut(clamp((t - 0.6) / 0.4, 0, 1));
  mainCtx.globalAlpha = qa;
  mainCtx.fillStyle = 'rgba(10,13,20,0.85)';
  mainCtx.beginPath(); mainCtx.roundRect(M.left + 4, M.top + 4, 230, 44, 8); mainCtx.fill();
  label(mainCtx, `q̂ = ${S.qhat.toFixed(4)}`, M.left + 12, M.top + 20, { color: '#fbbf24', size: 13 });
  label(mainCtx, `α = ${S.alpha.toFixed(2)} → ${Math.round((1-S.alpha)*100)}% kapsama garantisi`, M.left + 12, M.top + 36, { color: '#8892b0', size: 10 });
  mainCtx.globalAlpha = 1;
}

/* STEP 5 — Prediction sets */
function renderStep5(t) {
  clearC(mainCtx, mainCanvas); drawGrid(mainCtx, mainCanvas);
  if (S.showDecision) drawBoundary(mainCtx, mainCanvas, 0.45);
  S.train.forEach(p => {
    const {cx, cy} = w2c(p.wx, p.wy, mainCanvas);
    dot(mainCtx, cx, cy, { color: CFG.classes[p.cls].color, r: 3, alpha: 0.2 });
  });
  S.test.forEach((p, i) => {
    const { cx, cy } = w2c(p.wx, p.wy, mainCanvas);
    const al = easeOut(clamp((t - i * 0.012) / 0.5, 0, 1));
    if (al <= 0) return;
    if (!S.showPredSets) {
      dot(mainCtx, cx, cy, { color: CFG.classes[p.cls].color, r: 5, alpha: al * 0.85 });
      return;
    }
    const predSet = S.testPredSets[i];
    if (predSet.length === 0) {
      dot(mainCtx, cx, cy, { color: '#1c1f2e', r: 6, alpha: al, stroke: '#64748b', strokeW: 2 });
      cross(mainCtx, cx, cy, 5, '#94a3b8', al);
      // Size label: Ø (empty set)
      if (al > 0.6) {
        mainCtx.globalAlpha = al * 0.7;
        mainCtx.fillStyle = '#94a3b8'; mainCtx.font = 'bold 8px Inter'; mainCtx.textAlign = 'left';
        mainCtx.fillText('Ø', cx + 9, cy - 5);
        mainCtx.globalAlpha = 1;
      }
    } else if (predSet.length === 1) {
      const pc = CFG.classes[predSet[0]];
      dot(mainCtx, cx, cy, { color: pc.color, r: 5.5, alpha: al, stroke: 'rgba(255,255,255,0.6)', strokeW: 1.8 });
      // Size label: [1]
      if (al > 0.6) {
        mainCtx.globalAlpha = al * 0.75;
        mainCtx.fillStyle = pc.color; mainCtx.font = 'bold 8px Inter'; mainCtx.textAlign = 'left';
        mainCtx.fillText('[1]', cx + 9, cy - 4);
        mainCtx.globalAlpha = 1;
      }
    } else {
      // Multi-label: concentric rings
      dot(mainCtx, cx, cy, { color: CFG.classes[p.cls].color, r: 4, alpha: al });
      predSet.forEach((cid, ri) => {
        ring(mainCtx, cx, cy, 6 + ri * 5.5, CFG.classes[cid].color, al * 0.85, 1.8);
      });
      // Size label: [2] or [3]
      if (al > 0.6) {
        const hw = predSet.length === 3 ? '#fb923c' : '#fbbf24';
        mainCtx.globalAlpha = al * 0.9;
        mainCtx.fillStyle = hw; mainCtx.font = 'bold 8px Inter'; mainCtx.textAlign = 'left';
        mainCtx.fillText(`[${predSet.length}]`, cx + 9, cy - 4);
        mainCtx.globalAlpha = 1;
      }
    }
  });
  // Legend for set sizes
  if (t > 0.8) {
    const la = easeOut(clamp((t - 0.8) / 0.2, 0, 1));
    mainCtx.globalAlpha = la;
    mainCtx.fillStyle = 'rgba(10,13,20,0.82)';
    mainCtx.beginPath(); mainCtx.roundRect(M.left + 4, M.top + 4, 220, 56, 8); mainCtx.fill();
    const items = [
      { text: '[1]  Tek etiket — kesin tahmin', color: '#6e8efb' },
      { text: '[2]  İki etiket — sınır noktası', color: '#fbbf24' },
      { text: '[3]  Üç etiket — çok belirsiz', color: '#fb923c' },
    ];
    items.forEach((it, idx) => {
      mainCtx.fillStyle = it.color; mainCtx.font = '9px Inter';
      mainCtx.fillText(it.text, M.left + 12, M.top + 18 + idx * 14);
    });
    mainCtx.globalAlpha = 1;
  }
}

/* STEP 6 — Confidence coloring */
function renderStep6(t) {
  clearC(mainCtx, mainCanvas); drawGrid(mainCtx, mainCanvas);
  if (S.showDecision) drawBoundary(mainCtx, mainCanvas, 0.4);
  S.train.forEach(p => {
    const {cx, cy} = w2c(p.wx, p.wy, mainCanvas);
    dot(mainCtx, cx, cy, { color: '#8892b0', r: 2.5, alpha: 0.15 });
  });
  S.test.forEach((p, i) => {
    const { cx, cy } = w2c(p.wx, p.wy, mainCanvas);
    const al = easeOut(clamp((t - i * 0.012) / 0.5, 0, 1));
    const conf = S.testConfidence[i];
    const color = confColor(conf);
    dot(mainCtx, cx, cy, { color, r: 6, alpha: al * 0.88, stroke: 'rgba(255,255,255,0.25)', strokeW: 1.2 });
    if (al > 0.8 && S.showConfLabels) {
      label(mainCtx, (conf*100).toFixed(0)+'%', cx+8, cy+3, { color: 'rgba(255,255,255,0.6)', size: 9 });
    }
  });
  // Color scale bar
  drawConfColorBar(mainCtx, mainCanvas, easeOut(clamp(t*1.5,0,1)));
}

function drawConfColorBar(ctx, canvas, alpha) {
  const bx = M.left + 4, by = canvas.lh - M.bottom - 28;
  const bw = 100, bh = 12;
  ctx.globalAlpha = alpha;
  const grad = ctx.createLinearGradient(bx, 0, bx + bw, 0);
  grad.addColorStop(0, confColor(0.33));
  grad.addColorStop(1, confColor(1.0));
  ctx.fillStyle = grad; ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, 4); ctx.fill();
  ctx.fillStyle = '#8892b0'; ctx.font = '9px Inter';
  ctx.fillText('Düşük', bx, by - 3);
  ctx.textAlign = 'right'; ctx.fillText('Yüksek Güven', bx + bw, by - 3); ctx.textAlign = 'left';
  ctx.globalAlpha = 1;
}

/* STEP 7 — Sorting (main + aux) */
function renderStep7(t) {
  renderStep6(1.0);
  renderSortedBar(t);
}

/* AUX: Vertical bar chart — confidence scores sorted low→high */
function renderSortedBar(t) {
  if (!auxCanvas || !auxCtx) return;
  clearC(auxCtx, auxCanvas);
  const n = S.testSorted.length;
  if (n === 0) return;

  const ml = 32, mr = 10, mt = 22, mb = 18;
  const W = auxCanvas.lw - ml - mr;
  const H = auxCanvas.lh - mt - mb;
  const spacing = W / n;
  const barW = Math.max(spacing - 1.5, 2);
  const rejN = Math.floor(S.rejRate * n);

  // Rejection zone background — clearly visible red zone
  if (rejN > 0) {
    const zoneW = rejN * spacing;
    auxCtx.fillStyle = 'rgba(248,113,113,0.18)';
    auxCtx.fillRect(ml, mt, zoneW, H);
    // Left border
    auxCtx.strokeStyle = 'rgba(248,113,113,0.5)'; auxCtx.lineWidth = 1.5;
    auxCtx.beginPath(); auxCtx.moveTo(ml, mt); auxCtx.lineTo(ml, mt + H); auxCtx.stroke();
    // "REDDET" label inside zone if wide enough
    if (zoneW > 32) {
      auxCtx.fillStyle = 'rgba(248,113,113,0.6)'; auxCtx.font = 'bold 8px Inter'; auxCtx.textAlign = 'center';
      auxCtx.fillText('REDDET', ml + zoneW / 2, mt - 3);
    }
  }

  // Y-axis gridlines + labels
  [0.5, 0.625, 0.75, 0.875, 1.0].forEach(v => {
    const normV = (v - 0.5) / 0.5;
    const gy = mt + H * (1 - normV);
    auxCtx.strokeStyle = 'rgba(255,255,255,0.05)'; auxCtx.lineWidth = 1;
    auxCtx.beginPath(); auxCtx.moveTo(ml, gy); auxCtx.lineTo(ml + W, gy); auxCtx.stroke();
    auxCtx.fillStyle = '#3a4255'; auxCtx.font = '8px JetBrains Mono';
    auxCtx.textAlign = 'right';
    auxCtx.fillText(v.toFixed(2), ml - 3, gy + 3);
  });
  auxCtx.textAlign = 'left';

  // Bars
  S.testSorted.forEach((idx, rank) => {
    const conf = S.testConfidence[idx];
    const color = confColor(conf);
    const al = easeOut(clamp((t * (n + 6) - rank) / 6, 0, 1));
    const normConf = clamp((conf - 0.5) / 0.5, 0, 1);
    const bh = H * normConf * al;
    const x = ml + rank * spacing;
    const rejected = rank < rejN;
    // Bar: red for rejected, confidence color for accepted
    if (rejected) {
      auxCtx.fillStyle = `rgba(248,113,113,${al * 0.75})`;
    } else {
      auxCtx.fillStyle = hexToRgba(color, al * 0.85);
    }
    auxCtx.fillRect(x + 0.5, mt + H - bh, barW, bh);
    // Status stripe at bottom
    auxCtx.fillStyle = rejected ? 'rgba(248,113,113,0.9)' : 'rgba(52,211,153,0.55)';
    auxCtx.fillRect(x + 0.5, mt + H + 1, barW, 3);
    // Top dot
    if (al > 0.5 && bh > 4) {
      auxCtx.beginPath();
      auxCtx.arc(x + barW / 2, mt + H - bh, 2, 0, Math.PI * 2);
      auxCtx.fillStyle = rejected ? 'rgba(248,113,113,0.9)' : hexToRgba(color, al);
      auxCtx.fill();
    }
  });

  // Rejection threshold vertical line
  if (rejN > 0 && rejN <= n) {
    const threshX = ml + rejN * spacing;
    auxCtx.strokeStyle = 'rgba(248,113,113,0.95)'; auxCtx.lineWidth = 1.8;
    auxCtx.setLineDash([4, 3]);
    auxCtx.beginPath(); auxCtx.moveTo(threshX, mt - 6); auxCtx.lineTo(threshX, mt + H + 5); auxCtx.stroke();
    auxCtx.setLineDash([]);
    const labelX = threshX > ml + W * 0.75 ? threshX - 46 : threshX + 3;
    auxCtx.fillStyle = 'rgba(248,113,113,0.95)'; auxCtx.font = 'bold 9px Inter';
    auxCtx.fillText(`${rejN}/${n} red. (${Math.round(S.rejRate * 100)}%)`, labelX, mt + 11);
  }

  // Y-axis title (rotated)
  auxCtx.save();
  auxCtx.translate(11, mt + H / 2);
  auxCtx.rotate(-Math.PI / 2);
  auxCtx.font = '9px Inter'; auxCtx.textAlign = 'center'; auxCtx.fillStyle = '#5a647a';
  auxCtx.fillText('Güven', 0, 0);
  auxCtx.restore();

  // Header label
  auxCtx.fillStyle = '#5a647a'; auxCtx.font = '9px Inter'; auxCtx.textAlign = 'left';
  auxCtx.fillText(`${n} gözlem · güven = 1 − 2. en yüksek P`, ml, mt - 8);
  auxCtx.textAlign = 'right';
  auxCtx.fillText('← Reddet · Kabul →', ml + W, mt - 8);
  auxCtx.textAlign = 'left';
}

/* STEP 8 — Rejection mechanism */
function renderStep8(t) {
  clearC(mainCtx, mainCanvas); drawGrid(mainCtx, mainCanvas);
  if (S.showDecision) drawBoundary(mainCtx, mainCanvas, 0.3);
  S.train.forEach(p => {
    const {cx, cy} = w2c(p.wx, p.wy, mainCanvas);
    dot(mainCtx, cx, cy, { color: '#8892b0', r: 2.5, alpha: 0.12 });
  });
  const nRej = Math.floor(S.rejRate * S.test.length);
  const rejSet = new Set(S.testSorted.slice(0, nRej));
  let nAccepted = 0, nCorrectAccepted = 0;
  S.test.forEach((p, i) => {
    const { cx, cy } = w2c(p.wx, p.wy, mainCanvas);
    const rejected = rejSet.has(i);
    if (rejected) {
      dot(mainCtx, cx, cy, { color: '#f87171', r: 5, alpha: 0.55, stroke: '#f87171', strokeW: 1 });
      cross(mainCtx, cx, cy, 4, '#f87171', 0.75);
    } else {
      nAccepted++;
      const correct = S.testCorrect[i];
      if (correct) nCorrectAccepted++;
      const color = correct ? '#34d399' : '#fbbf24';
      dot(mainCtx, cx, cy, { color, r: 6, alpha: 0.9, stroke: 'rgba(255,255,255,0.5)', strokeW: 1.8 });
    }
  });
  // Stat overlay
  const acc = nAccepted > 0 ? nCorrectAccepted / nAccepted : 0;
  mainCtx.fillStyle = 'rgba(10,13,20,0.88)';
  mainCtx.beginPath(); mainCtx.roundRect(M.left + 4, M.top + 4, 240, 56, 8); mainCtx.fill();
  label(mainCtx, `✅ Kabul: ${nAccepted} · Doğruluk: ${(acc*100).toFixed(1)}%`, M.left + 10, M.top + 22, { color: '#34d399', size: 12 });
  label(mainCtx, `❌ Reddedilen: ${nRej} (${(S.rejRate*100).toFixed(0)}%)`, M.left + 10, M.top + 40, { color: '#f87171', size: 12 });
  label(mainCtx, `🟡 Yanlış kabul: ${nAccepted - nCorrectAccepted}`, M.left + 10, M.top + 52, { color: '#fbbf24', size: 10, font: 'Inter' });

  // Also render sorted bar in aux
  renderSortedBar(1.0);
}

/* STEP 9 — Metrics + accuracy chart */
function renderStep9(t) {
  renderStep8(1.0);
  // Accuracy chart renders via Chart.js
  const nShow = Math.floor(easeOut(t) * S.metricCurve.length);
  renderAccuracyChart(nShow);
}

/* STEP 9 (Final) — Comprehensive Trade-Off Dashboard */
function renderStep10(t) {
  renderStep8(1.0);
  const al = easeInOut(clamp(t * 1.5, 0, 1));
  if (al <= 0) return;
  const W = mainCanvas.lw, H = mainCanvas.lh;

  // Dark overlay
  mainCtx.globalAlpha = al * 0.93;
  mainCtx.fillStyle = 'rgba(8,10,18,1)';
  mainCtx.fillRect(0, 0, W, H);
  mainCtx.globalAlpha = 1;

  // Ambient glow left
  mainCtx.globalAlpha = al * 0.5;
  const radL = mainCtx.createRadialGradient(W*0.25, H*0.5, 0, W*0.25, H*0.5, W*0.32);
  radL.addColorStop(0, 'rgba(110,142,251,0.12)'); radL.addColorStop(1, 'transparent');
  mainCtx.fillStyle = radL; mainCtx.fillRect(0, 0, W, H);
  mainCtx.globalAlpha = 1;

  mainCtx.globalAlpha = al;

  // ── LEFT PANEL: Quote + 4 stat boxes ──
  const qx = 12, qy = 14, qw = W * 0.46, qh = H - 28;
  mainCtx.fillStyle = 'rgba(20,26,50,0.95)';
  mainCtx.strokeStyle = 'rgba(110,142,251,0.25)'; mainCtx.lineWidth = 1.5;
  mainCtx.beginPath(); mainCtx.roundRect(qx, qy, qw, qh, 14);
  mainCtx.fill(); mainCtx.stroke();

  mainCtx.fillStyle = 'rgba(110,142,251,0.3)'; mainCtx.font = '28px Inter';
  mainCtx.textAlign = 'left';
  mainCtx.fillText('"', qx + 14, qy + 38);
  mainCtx.fillStyle = '#e8eaf6'; mainCtx.font = 'bold 12.5px Inter';
  mainCtx.textAlign = 'center';
  const qcx = qx + qw / 2;
  mainCtx.fillText('Bir model sadece tahmin yapmamalı —', qcx, qy + 68);
  mainCtx.fillText('ne zaman tahmin yapmaması', qcx, qy + 88);
  mainCtx.fillText('gerektiğini de bilmeli."', qcx, qy + 108);
  mainCtx.fillStyle = '#6e8efb'; mainCtx.font = '500 10px Inter';
  mainCtx.fillText('— Uncertainty-Aware AI Prensibi', qcx, qy + 130);

  // Divider
  mainCtx.strokeStyle = 'rgba(110,142,251,0.12)'; mainCtx.lineWidth = 1;
  mainCtx.beginPath(); mainCtx.moveTo(qx + 16, qy + 148); mainCtx.lineTo(qx + qw - 16, qy + 148); mainCtx.stroke();

  // Principle bullets
  const bullets = [
    { icon: '📈', text: 'Reddetme ↑  →  Doğruluk ↑', color: '#34d399' },
    { icon: '📉', text: 'Reddetme ↑  →  Tahmin Edilen Gözlem ↓', color: '#a78bfa' },
    { icon: '🎯', text: `Kapsama Garantisi ≥ ${Math.round((1-S.alpha)*100)}%`, color: '#6e8efb' },
  ];
  bullets.forEach((b, i) => {
    const by = qy + 166 + i * 22;
    mainCtx.fillStyle = b.color; mainCtx.font = '10px Inter';
    mainCtx.textAlign = 'left';
    mainCtx.fillText(b.icon + '  ' + b.text, qx + 18, by);
  });

  // 4 stat boxes
  const nRej = Math.floor(S.rejRate * S.test.length);
  const rejSet = new Set(S.testSorted.slice(0, nRej));
  const accepted = S.test.map((_,i) => i).filter(i => !rejSet.has(i));
  const nCorr = accepted.filter(i => S.testCorrect[i]).length;
  const acc = accepted.length > 0 ? nCorr / accepted.length : 0;
  const baseAcc = S.metricCurve[0]?.acc ?? 0;
  const gain = acc - baseAcc;

  const bby = qy + qh - 78, sw = (qw - 26) / 4;
  const stats = [
    { label: 'CP Kapsama', val: `≥${Math.round((1-S.alpha)*100)}%`, color: '#6e8efb' },
    { label: 'Doğruluk', val: `${(acc*100).toFixed(1)}%`, color: '#34d399' },
    { label: 'Kazanım', val: `${gain>=0?'+':''}${(gain*100).toFixed(1)}%`, color: gain >= 0 ? '#34d399' : '#f87171' },
    { label: 'Tahmin Edilen', val: `${accepted.length}`, color: '#a78bfa' },
  ];
  stats.forEach((st, i) => {
    const sx = qx + 6 + i * (sw + 4);
    mainCtx.fillStyle = hexToRgba(st.color, 0.12);
    mainCtx.beginPath(); mainCtx.roundRect(sx, bby, sw, 54, 7); mainCtx.fill();
    mainCtx.strokeStyle = hexToRgba(st.color, 0.3); mainCtx.lineWidth = 1; mainCtx.stroke();
    mainCtx.fillStyle = st.color; mainCtx.font = `bold 13px JetBrains Mono`;
    mainCtx.textAlign = 'center';
    mainCtx.fillText(st.val, sx + sw / 2, bby + 28);
    mainCtx.fillStyle = '#8892b0'; mainCtx.font = '8px Inter';
    mainCtx.fillText(st.label, sx + sw / 2, bby + 44);
  });

  // ── RIGHT PANEL: Dual-axis Trade-Off Chart ──
  const cx = W * 0.50 + 4, cw2 = W - cx - 12;
  const cy = qy, ch2 = qh;
  mainCtx.fillStyle = 'rgba(20,26,50,0.95)';
  mainCtx.strokeStyle = 'rgba(52,211,153,0.2)'; mainCtx.lineWidth = 1.5;
  mainCtx.beginPath(); mainCtx.roundRect(cx, cy, cw2, ch2, 14);
  mainCtx.fill(); mainCtx.stroke();

  mainCtx.fillStyle = '#e8eaf6'; mainCtx.font = 'bold 10.5px Inter'; mainCtx.textAlign = 'center';
  mainCtx.fillText('Reddetme → Doğruluk & Kapsama Trade-Off', cx + cw2 / 2, cy + 18);

  drawFinalChart(mainCtx, cx + 8, cy + 26, cw2 - 16, ch2 - 40, al);

  mainCtx.textAlign = 'left';
  mainCtx.globalAlpha = 1;
}

/* Single-axis canvas chart: Accuracy vs Rejection Rate */
function drawFinalChart(ctx, ox, oy, ow, oh, alpha) {
  if (!S.metricCurve.length) return;
  ctx.globalAlpha = alpha;
  const ml = 32, mr = 12, mt = 14, mb = 26;
  const cw = ow - ml - mr, ch = oh - mt - mb;

  // Horizontal grid + Y labels (accuracy %)
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
    const gy = oy + mt + ch * (1 - v);
    ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ox + ml, gy); ctx.lineTo(ox + ml + cw, gy); ctx.stroke();
    ctx.fillStyle = 'rgba(52,211,153,0.65)'; ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'right';
    ctx.fillText((v * 100).toFixed(0) + '%', ox + ml - 4, gy + 3);
  });
  ctx.textAlign = 'left';

  // Baseline accuracy (no rejection) — red dashed
  const baseAcc = S.metricCurve[0]?.acc ?? 0;
  const bly = oy + mt + ch * (1 - baseAcc);
  ctx.strokeStyle = 'rgba(248,113,113,0.45)'; ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(ox + ml, bly); ctx.lineTo(ox + ml + cw, bly); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(248,113,113,0.75)'; ctx.font = '7.5px Inter'; ctx.textAlign = 'right';
  ctx.fillText(`τ=0: ${(baseAcc * 100).toFixed(0)}%`, ox + ml + cw, bly - 3);
  ctx.textAlign = 'left';

  // Conformal coverage guarantee — blue dashed
  const covY = oy + mt + ch * (1 - (1 - S.alpha));
  ctx.strokeStyle = 'rgba(110,142,251,0.4)'; ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(ox + ml, covY); ctx.lineTo(ox + ml + cw, covY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(110,142,251,0.75)'; ctx.font = '7.5px Inter';
  ctx.fillText(`≥${Math.round((1 - S.alpha) * 100)}% kapsama`, ox + ml + 3, covY - 3);

  // Green area fill under accuracy curve
  ctx.fillStyle = 'rgba(52,211,153,0.09)';
  ctx.beginPath();
  S.metricCurve.forEach((p, i) => {
    const px = ox + ml + (p.r / 0.8) * cw;
    const py = oy + mt + ch * (1 - p.acc);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.lineTo(ox + ml + cw, oy + mt + ch);
  ctx.lineTo(ox + ml, oy + mt + ch);
  ctx.closePath(); ctx.fill();

  // Accuracy curve — green solid
  ctx.strokeStyle = '#34d399'; ctx.lineWidth = 2.5;
  ctx.beginPath();
  S.metricCurve.forEach((p, i) => {
    const px = ox + ml + (p.r / 0.8) * cw;
    const py = oy + mt + ch * (1 - p.acc);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();

  // Current rejection threshold — orange vertical + point
  const rx = ox + ml + (S.rejRate / 0.8) * cw;
  ctx.strokeStyle = 'rgba(251,146,60,0.85)'; ctx.lineWidth = 1.8;
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(rx, oy + mt); ctx.lineTo(rx, oy + mt + ch); ctx.stroke();
  ctx.setLineDash([]);

  const closest = S.metricCurve.reduce((best, p) =>
    Math.abs(p.r - S.rejRate) < Math.abs(best.r - S.rejRate) ? p : best
  );
  const ptX = ox + ml + (closest.r / 0.8) * cw;
  const ptY = oy + mt + ch * (1 - closest.acc);
  ctx.beginPath(); ctx.arc(ptX, ptY, 5.5, 0, Math.PI * 2);
  ctx.fillStyle = '#fb923c'; ctx.fill();
  ctx.fillStyle = '#fb923c'; ctx.font = 'bold 10px JetBrains Mono'; ctx.textAlign = 'center';
  const labelAbove = ptY > oy + mt + 22;
  ctx.fillText(`${(closest.acc * 100).toFixed(1)}%`, ptX, labelAbove ? ptY - 10 : ptY + 17);
  ctx.textAlign = 'left';

  // Y axis label (rotated)
  ctx.save();
  ctx.translate(ox + 11, oy + mt + ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.font = '8px Inter'; ctx.textAlign = 'center'; ctx.fillStyle = 'rgba(52,211,153,0.7)';
  ctx.fillText('Doğruluk (%)', 0, 0);
  ctx.restore();

  // X axis ticks + label
  [0, 0.2, 0.4, 0.6, 0.8].forEach(v => {
    const gx = ox + ml + (v / 0.8) * cw;
    ctx.fillStyle = '#3a4255'; ctx.font = '7px JetBrains Mono'; ctx.textAlign = 'center';
    ctx.fillText((v * 100).toFixed(0) + '%', gx, oy + mt + ch + 11);
  });
  ctx.fillStyle = '#5a647a'; ctx.font = '8px Inter'; ctx.textAlign = 'center';
  ctx.fillText('Reddetme Oranı (τ) →', ox + ml + cw / 2, oy + oh - 2);
  ctx.textAlign = 'left';

  ctx.globalAlpha = 1;
}

/* ─── CHART.JS: Score Histogram ───────────────────────────────── */
function renderScoreChart() {
  const el = document.getElementById('scoreChart');
  if (!el) return;
  const maxS = Math.max(...S.calibScores, S.qhat * 1.5);
  const bins = 18;
  const bw = maxS / bins;
  const counts = new Array(bins).fill(0);
  S.calibScores.forEach(s => {
    counts[Math.min(Math.floor(s / bw), bins-1)]++;
  });
  const labels = Array.from({ length: bins }, (_, i) => ((i+0.5)*bw).toFixed(2));
  const bgColors = labels.map((_, i) => (i+0.5)*bw <= S.qhat ? 'rgba(110,142,251,0.65)' : 'rgba(248,113,113,0.55)');

  if (S.scoreChart) S.scoreChart.destroy();
  S.scoreChart = new Chart(el, {
    type: 'bar',
    data: { labels, datasets: [{ data: counts, backgroundColor: bgColors, borderWidth: 0 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 600 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(21,27,46,0.95)', borderColor: 'rgba(110,142,251,.3)', borderWidth:1,
          titleColor: '#e8eaf6', bodyColor: '#8892b0',
          callbacks: { title: () => 'Uyumsuzluk Skoru', label: ctx => `${ctx.parsed.y} nokta` }
        }
      },
      scales: {
        x: { ticks: { color: '#5a647a', font:{size:9}, maxTicksLimit:6 }, grid: { display:false } },
        y: { ticks: { color: '#5a647a', font:{size:9} }, grid:{ color:'rgba(255,255,255,0.04)' } }
      }
    }
  });
}

/* ─── CHART.JS: Accuracy Curve ────────────────────────────────── */
function renderAccuracyChart(nShow) {
  const el = document.getElementById('accuracyChart');
  if (!el || !S.metricCurve.length) return;
  const visible = S.metricCurve.slice(0, nShow);
  if (S.accuracyChart) S.accuracyChart.destroy();
  S.accuracyChart = new Chart(el, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Doğruluk',
          data: visible.map(p => ({ x: p.r*100, y: p.acc*100 })),
          borderColor: '#34d399', backgroundColor: 'rgba(52,211,153,0.08)',
          fill: true, borderWidth: 2.5, pointRadius: 0, tension: 0.4,
        },
        {
          label: 'Mevcut Eşik',
          data: [{ x: S.rejRate*100, y: 0 }, { x: S.rejRate*100, y: 100 }],
          borderColor: 'rgba(251,146,60,0.7)', borderDash: [4,4],
          borderWidth: 1.5, pointRadius: 0, tension: 0,
        },
        {
          label: 'Baseline (0 red.)',
          data: [{ x:0, y: S.metricCurve[0]?.acc*100 ?? 70 }, { x:80, y: S.metricCurve[0]?.acc*100 ?? 70 }],
          borderColor: 'rgba(248,113,113,0.3)', borderDash:[6,4],
          borderWidth:1, pointRadius:0, tension:0,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { labels: { color:'#8892b0', font:{size:10}, boxWidth:10 } },
        tooltip: { backgroundColor:'rgba(21,27,46,0.95)', borderColor:'rgba(110,142,251,.3)', borderWidth:1, titleColor:'#e8eaf6', bodyColor:'#8892b0' }
      },
      scales: {
        x: {
          type:'linear', min:0, max:65,
          ticks:{ color:'#5a647a', font:{size:9}, callback: v=>v+'%' },
          grid:{ color:'rgba(255,255,255,0.04)' },
          title:{ display:true, text:'Reddetme (%)', color:'#8892b0', font:{size:10} }
        },
        y: {
          ticks:{ color:'#5a647a', font:{size:9}, callback: v=>v+'%' },
          grid:{ color:'rgba(255,255,255,0.04)' },
          title:{ display:true, text:'Doğruluk (%)', color:'#8892b0', font:{size:10} }
        }
      }
    }
  });
}

/* ─── STEP SYSTEM ─────────────────────────────────────────────── */
// 9 steps: step 9 maps to renderStep10 (the final dashboard)
const RENDERERS = [null, renderStep1, renderStep2, renderStep3, renderStep4,
  renderStep5, renderStep6, renderStep7, renderStep8, renderStep10];

function renderCurrent() {
  const fn = RENDERERS[S.step];
  if (fn) fn(S.animT);
}

function updateUI() {
  // Step bar
  document.querySelectorAll('.step-item').forEach((el, i) => {
    el.classList.toggle('active', i + 1 === S.step);
    el.classList.toggle('done', i + 1 < S.step);
    if (i + 1 < S.step) el.querySelector('.step-bubble').textContent = '✓';
    else el.querySelector('.step-bubble').textContent = i + 1;
  });
  // Explain box
  const meta = STEPS_META[S.step - 1];
  document.getElementById('stepIcon').textContent = meta.icon;
  document.getElementById('stepTitle').textContent = meta.title;
  document.getElementById('stepDesc').textContent = meta.desc;
  document.getElementById('stepInsight').textContent = meta.insight;
  // Progress
  document.getElementById('progressFill').style.width = ((S.step-1)/8*100) + '%';
  document.getElementById('stepCounter').textContent = `Adım ${S.step} / 9`;
  document.getElementById('ctrlStepInfo').textContent = `Adım ${S.step} / 9`;
  document.getElementById('hStatStep').textContent = `${S.step} / 9`;
  // Charts
  const showScore = S.step >= 4;
  document.getElementById('scoreChartWrap').style.display = showScore ? 'block' : 'none';
  // Accuracy chart shown only on final step (canvas-drawn)
  document.getElementById('accuracyChartWrap').style.display = 'none';
  document.getElementById('auxPanel').style.display = (S.step >= 7 && S.step <= 9) ? 'block' : 'none';
  if (showScore && S.calibScores.length > 0) renderScoreChart();
  // Legend
  updateLegend();
  // Metrics
  updateMetrics();
}

function updateLegend() {
  const el = document.getElementById('dynamicLegend');
  let html = '';
  if (S.step <= 1) {
    CFG.classes.forEach(c => {
      html += `<div class="legend-item"><div class="legend-dot" style="background:${c.color}"></div>Sınıf ${c.name}</div>`;
    });
  } else if (S.step === 2) {
    html += `<div class="legend-item"><div class="legend-dot" style="background:#38bdf8"></div>Eğitim (${S.train.length})</div>`;
    html += `<div class="legend-item"><div class="legend-dot" style="background:#fbbf24"></div>Kalibrasyon (${S.calib.length})</div>`;
    html += `<div class="legend-item"><div class="legend-dot" style="background:#34d399"></div>Test (${S.test.length})</div>`;
  } else if (S.step === 4) {
    html += `<div class="legend-item"><div class="legend-dot" style="background:#34d399"></div>Score ≤ q̂ (kapsamda)</div>`;
    html += `<div class="legend-item"><div class="legend-dot" style="background:#f87171"></div>Score > q̂ (dışında)</div>`;
  } else if (S.step === 5) {
    html += `<div class="legend-item"><div class="legend-dot" style="background:#6e8efb"></div>Tek etiketli (emin)</div>`;
    html += `<div class="legend-item"><div class="legend-ring" style="border-color:#a78bfa"></div>Çok etiketli (belirsiz)</div>`;
    html += `<div class="legend-item"><div class="legend-x" style="color:#94a3b8">✕</div>Boş küme (anomali)</div>`;
  } else if (S.step >= 8) {
    html += `<div class="legend-item"><div class="legend-dot" style="background:#34d399"></div>Kabul + Doğru</div>`;
    html += `<div class="legend-item"><div class="legend-dot" style="background:#fbbf24"></div>Kabul + Yanlış</div>`;
    html += `<div class="legend-item"><div class="legend-dot" style="background:#f87171"></div>Reddedilen</div>`;
  }
  el.innerHTML = html;
}

function updateMetrics() {
  if (!S.test.length) return;
  const nRej = Math.floor(S.rejRate * S.test.length);
  const rejSet = new Set(S.testSorted.slice(0, nRej));
  const accepted = S.test.map((_,i) => i).filter(i => !rejSet.has(i));
  const nCorr = accepted.filter(i => S.testCorrect[i]).length;
  const acc = accepted.length > 0 ? nCorr / accepted.length : 0;

  document.getElementById('mAccuracy').textContent = (acc*100).toFixed(1) + '%';
  document.getElementById('mRejRate').textContent = (S.rejRate*100).toFixed(0) + '%';
  document.getElementById('mQhat').textContent = S.qhat.toFixed(4);
  document.getElementById('mAccepted').textContent = accepted.length;
  document.getElementById('mTrainN').textContent = S.train.length;
  document.getElementById('mCalibN').textContent = S.calib.length;
  document.getElementById('mTestN').textContent = S.test.length;
  document.getElementById('mAlpha').textContent = S.alpha.toFixed(2);
  document.getElementById('mCovTarget').textContent = Math.round((1-S.alpha)*100) + '%';
  document.getElementById('hStatAcc').textContent = (acc*100).toFixed(1) + '%';
  document.getElementById('hStatRej').textContent = (S.rejRate*100).toFixed(0) + '%';
}

function setStep(n, fromPlay = false) {
  S.step = clamp(n, 1, 9);
  S.animT = fromPlay ? 0 : 1.0;
  S.animStart = performance.now();
  updateUI();
  renderCurrent();
}

/* ─── ANIMATION LOOP ──────────────────────────────────────────── */
function animLoop(ts) {
  const elapsed = ts - S.animStart;
  const dur = CFG.STEP_DURATIONS[S.step - 1] / S.speed;
  S.animT = clamp(elapsed / dur, 0, 1);

  // Auto-animate rejection rate during step 8 playback
  if (S.step === 8 && S.playing) {
    const autoRate = easeInOut(S.animT) * 0.50;
    S.rejRate = autoRate;
    syncRejSliders(autoRate);
  }

  renderCurrent();
  updateMetrics();

  if (S.playing) {
    if (S.animT >= 1) {
      if (S.step < 9) {
        const nextStep = S.step + 1;
        // Reset rejection rate when leaving step 8 (before final dashboard)
        if (S.step === 8) { S.rejRate = 0; syncRejSliders(0); }
        setTimeout(() => {
          if (!S.playing) return;
          S.step = nextStep;
          S.animT = 0;
          S.animStart = performance.now();
          updateUI();
        }, 700 / S.speed);
        S.animFrame = requestAnimationFrame(animLoop);
      } else {
        S.playing = false;
        updatePlayBtn();
      }
    } else {
      S.animFrame = requestAnimationFrame(animLoop);
    }
  }
}

/* Sync both rejection sliders + display labels */
function syncRejSliders(rate) {
  const s1 = document.getElementById('rejSlider');
  const s2 = document.getElementById('rejThresholdCtrl');
  if (s1) { s1.value = rate; updateSliderFill(s1); }
  if (s2) { s2.value = rate; updateSliderFill(s2); }
  const pct = Math.round(rate * 100) + '%';
  const v1 = document.getElementById('rejVal');
  const v2 = document.getElementById('rejThresholdVal');
  if (v1) v1.textContent = pct;
  if (v2) v2.textContent = pct;
  document.getElementById('hStatRej').textContent = pct;
  document.getElementById('mRejRate').textContent = pct;
}

function startAnim() {
  cancelAnimationFrame(S.animFrame);
  S.animStart = performance.now();
  S.animFrame = requestAnimationFrame(animLoop);
}

function play() {
  S.playing = true;
  if (S.animT >= 1 && S.step < 9) {
    S.step++;
    S.animT = 0;
    S.animStart = performance.now();
    updateUI();
  }
  startAnim();
  updatePlayBtn();
}

function pause() {
  S.playing = false;
  cancelAnimationFrame(S.animFrame);
  updatePlayBtn();
}

function updatePlayBtn() {
  const btn = document.getElementById('btnPlay');
  if (S.playing) {
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Duraklat`;
    btn.classList.remove('primary');
    btn.style.background = '';
    btn.style.color = 'var(--accent4)';
    btn.style.borderColor = 'rgba(251,146,60,.5)';
  } else {
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Oynat`;
    btn.classList.add('primary');
    btn.style.color = '';
    btn.style.borderColor = '';
  }
}

/* ─── REJECTION ANIMATION ─────────────────────────────────────── */
let rejAnimRunning = false;
function animateRejection() {
  if (rejAnimRunning) { rejAnimRunning = false; return; }
  pause();
  setStep(8);
  S.rejRate = 0;
  rejAnimRunning = true;
  const AUTOBTN = document.getElementById('btnAutoReject');
  AUTOBTN.textContent = '⏹ Durdur';

  const tick = () => {
    if (!rejAnimRunning) { AUTOBTN.textContent = '🔴 Reddetme Animasyonu'; return; }
    S.rejRate = Math.min(S.rejRate + 0.005, 0.55);
    const rejSlider = document.getElementById('rejSlider');
    rejSlider.value = S.rejRate;
    document.getElementById('rejVal').textContent = Math.round(S.rejRate * 100) + '%';
    updateSliderFill(rejSlider);
    document.getElementById('hStatRej').textContent = Math.round(S.rejRate * 100) + '%';
    renderStep8(1);
    updateMetrics();
    if (S.rejRate < 0.55) requestAnimationFrame(tick);
    else { rejAnimRunning = false; AUTOBTN.textContent = '🔴 Reddetme Animasyonu'; }
  };
  requestAnimationFrame(tick);
}

/* ─── TOOLTIP ─────────────────────────────────────────────────── */
function handleHover(e) {
  if (!S.testProbs.length) return;
  const rect = mainCanvas.getBoundingClientRect();
  const scaleX = mainCanvas.lw / rect.width;
  const scaleY = mainCanvas.lh / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;
  const { wx, wy } = c2w(px, py, mainCanvas);

  let nearest = -1, minD = Infinity;
  S.test.forEach((p, i) => {
    const d = (p.wx-wx)**2 + (p.wy-wy)**2;
    if (d < minD) { minD = d; nearest = i; }
  });

  const tt = document.getElementById('tooltip');
  if (nearest >= 0 && minD < 0.6 && S.step >= 4) {
    const p = S.test[nearest];
    const probs = S.testProbs[nearest];
    const conf = S.testConfidence[nearest];
    const predSet = S.testPredSets[nearest];
    const nRej = Math.floor(S.rejRate * S.test.length);
    const rejSet = new Set(S.testSorted.slice(0, nRej));
    const rejected = rejSet.has(nearest);
    tt.innerHTML = `
      <div class="tt-title">🔍 Test Noktası #${nearest + 1}</div>
      <hr class="tt-divider">
      <div class="tt-row"><span>Gerçek Sınıf</span><span style="color:${CFG.classes[p.cls].color}">${CFG.classes[p.cls].name}</span></div>
      <div class="tt-row"><span>P(A)</span><span>${(probs[0]*100).toFixed(1)}%</span></div>
      <div class="tt-row"><span>P(B)</span><span>${(probs[1]*100).toFixed(1)}%</span></div>
      <div class="tt-row"><span>P(C)</span><span>${(probs[2]*100).toFixed(1)}%</span></div>
      <hr class="tt-divider">
      <div class="tt-row"><span>Güven (1−2P)</span><span style="color:${confColor(conf)}">${(conf*100).toFixed(1)}%</span></div>
      <div class="tt-row"><span>NC Skoru</span><span>${(1-conf).toFixed(3)}</span></div>
      <div class="tt-row"><span>Pred. Set</span><span>{${predSet.map(i=>CFG.classes[i].name).join(',')}}</span></div>
      <div class="tt-row"><span>Doğru?</span><span>${S.testCorrect[nearest] ? '✅ Evet' : '❌ Hayır'}</span></div>
      ${S.step >= 8 ? `<div class="tt-row"><span>Durum</span><span style="color:${rejected?'#f87171':'#34d399'}">${rejected?'Reddedildi':'Kabul'}</span></div>` : ''}
    `;
    tt.style.display = 'block';
    tt.style.left = (e.clientX + 14) + 'px';
    tt.style.top = Math.min(e.clientY - 20, window.innerHeight - 260) + 'px';
  } else {
    tt.style.display = 'none';
  }
}

/* ─── SLIDER FILL ─────────────────────────────────────────────── */
function updateSliderFill(input) {
  const min = +input.min || 0, max = +input.max || 1, val = +input.value;
  input.style.setProperty('--pct', ((val - min) / (max - min) * 100).toFixed(1) + '%');
}

/* ─── INITIALIZATION ──────────────────────────────────────────── */
function initData() {
  const raw = generateData();
  const { train, calib, test } = splitData(raw);
  // Mark split on all
  const allWithSplit = [
    ...train.map(p => ({ ...p, split: 'train' })),
    ...calib.map(p => ({ ...p, split: 'calib' })),
    ...test.map(p => ({ ...p, split: 'test' })),
  ];
  S.all = allWithSplit;
  S.train = train; S.calib = calib; S.test = test;
  document.getElementById('hStatTotal').textContent = raw.length;
}

function initModel() {
  document.getElementById('loadMsg').textContent = 'Karar sınırı hesaplanıyor…';
  return new Promise(resolve => {
    setTimeout(() => {
      S.boundary = computeDecisionBoundary(S.train);
      S.calibScores = computeCalibScores(S.calib, S.train);
      S.qhat = computeQhat(S.calibScores, S.alpha);
      S.testProbs = S.test.map(p => knnPredict(p.wx, p.wy, S.train));
      S.testPredSets = S.testProbs.map(probs => computePredSet(probs, S.qhat));
      S.testConfidence = S.testProbs.map(probs => computeConfidence(probs));
      S.testCorrect = S.test.map((p, i) => S.testProbs[i].indexOf(Math.max(...S.testProbs[i])) === p.cls);
      // Sort by confidence ascending (lowest first = rejected first)
      S.testSorted = S.test.map((_,i) => i).sort((a,b) => S.testConfidence[a] - S.testConfidence[b]);
      S.metricCurve = computeMetricCurve(S.test, S.testCorrect, S.testSorted);
      resolve();
    }, 60);
  });
}

async function fullInit() {
  document.getElementById('loadOverlay').style.display = 'flex';
  S.playing = false; S.rejRate = 0; S.step = 1; S.animT = 0;

  initData();
  await initModel();

  document.getElementById('loadOverlay').style.display = 'none';
  setStep(1, true);
  updateUI();
  startAnim();
}

/* ─── EVENT WIRING ────────────────────────────────────────────── */
function wireEvents() {
  document.getElementById('btnPlay').addEventListener('click', () => S.playing ? pause() : play());
  document.getElementById('btnPrev').addEventListener('click', () => { pause(); if (S.step > 1) setStep(S.step - 1); });
  document.getElementById('btnNext').addEventListener('click', () => { if (S.step < 10) { pause(); setStep(S.step + 1); } });
  document.getElementById('btnReset').addEventListener('click', () => { rejAnimRunning = false; fullInit(); });
  document.getElementById('btnAutoReject').addEventListener('click', animateRejection);
  document.getElementById('btnNewData').addEventListener('click', () => { rejAnimRunning = false; fullInit(); });

  const rejSlider = document.getElementById('rejSlider');
  rejSlider.addEventListener('input', e => {
    S.rejRate = +e.target.value;
    syncRejSliders(S.rejRate);
    updateMetrics();
    // Update from step 7 onward (bar chart + main canvas)
    if (S.step >= 7) {
      renderCurrent();
      renderSortedBar(1.0); // always keep aux canvas in sync
    }
  });

  const rejCtrl = document.getElementById('rejThresholdCtrl');
  if (rejCtrl) {
    rejCtrl.addEventListener('input', e => {
      S.rejRate = +e.target.value;
      syncRejSliders(S.rejRate);
      updateMetrics();
      if (S.step >= 7) {
        renderCurrent();
        renderSortedBar(1.0);
      }
    });
  }

  const alphaSlider = document.getElementById('alphaSlider');
  alphaSlider.addEventListener('input', e => {
    S.alpha = +e.target.value;
    document.getElementById('alphaVal').textContent = S.alpha.toFixed(2);
    updateSliderFill(e.target);
    if (S.calib.length) {
      S.qhat = computeQhat(S.calibScores, S.alpha);
      S.testPredSets = S.testProbs.map(probs => computePredSet(probs, S.qhat));
      S.metricCurve = computeMetricCurve(S.test, S.testCorrect, S.testSorted);
      updateMetrics();
      renderCurrent();
      if (S.step >= 4) renderScoreChart();
      if (S.step >= 7) renderSortedBar(1.0); // keep aux bar chart in sync
    }
  });

  document.getElementById('speedSelect').addEventListener('change', e => { S.speed = +e.target.value; });
  document.getElementById('toggleDecision').addEventListener('change', e => { S.showDecision = e.target.checked; renderCurrent(); });
  document.getElementById('togglePredSets').addEventListener('change', e => { S.showPredSets = e.target.checked; renderCurrent(); });
  document.getElementById('toggleConfidence').addEventListener('change', e => { S.showConfLabels = e.target.checked; renderCurrent(); });

  mainCanvas.addEventListener('mousemove', handleHover);
  mainCanvas.addEventListener('mouseleave', () => { document.getElementById('tooltip').style.display = 'none'; });

  document.querySelectorAll('.step-item').forEach((el, i) => {
    el.addEventListener('click', () => { pause(); setStep(i + 1); });
  });

  // Init slider fills
  [rejSlider, alphaSlider].forEach(updateSliderFill);
  const rejCtrl2 = document.getElementById('rejThresholdCtrl');
  if (rejCtrl2) updateSliderFill(rejCtrl2);

  window.addEventListener('resize', () => {
    const mc = setupCanvas('mainCanvas', 420);
    mainCanvas = mc.canvas; mainCtx = mc.ctx;
    const ac = setupCanvas('auxCanvas', 140);
    auxCanvas = ac.canvas; auxCtx = ac.ctx;
    renderCurrent();
  });
}

/* ─── BUILD STEP BAR ──────────────────────────────────────────── */
function buildStepBar() {
  const bar = document.getElementById('stepBarInner');
  STEPS_META.forEach((step, i) => {
    const el = document.createElement('div');
    el.className = 'step-item';
    el.innerHTML = `<div class="step-bubble">${step.id}</div><div class="step-label">${step.title}</div>`;
    bar.appendChild(el);
    if (i < STEPS_META.length - 1) {
      const conn = document.createElement('div');
      conn.className = 'step-conn';
      bar.appendChild(conn);
    }
  });
}

/* ─── ENTRY POINT ─────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', async () => {
  buildStepBar();
  const mc = setupCanvas('mainCanvas', 420);
  mainCanvas = mc.canvas; mainCtx = mc.ctx;
  const ac = setupCanvas('auxCanvas', 140);
  auxCanvas = ac.canvas; auxCtx = ac.ctx;
  wireEvents();
  await fullInit();
});
