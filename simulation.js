/* ============================================================
   CONFORMAL PREDICTION SİMÜLASYONU - simulation.js
   Split Conformal Prediction with interactive visualization
   ============================================================ */

'use strict';

// ─── Chart.js CDN loader ────────────────────────────────────
(function loadChartJS() {
  const s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js';
  s.onload = () => init();
  document.head.appendChild(s);
})();

// ─── State ──────────────────────────────────────────────────
let state = {
  mainChart: null,
  scoreChart: null,
  alphaChart: null,
  animRunning: false,
  animFrame: null,
  data: null,
};

// ─── Utility: random normal (Box-Muller) ────────────────────
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function linspace(a, b, n) {
  const step = (b - a) / (n - 1);
  return Array.from({ length: n }, (_, i) => a + i * step);
}

// ─── Data Generators ────────────────────────────────────────
function trueFunc(x, type) {
  switch (type) {
    case 'linear':        return 1.5 * x + 2;
    case 'sinusoidal':    return 3 * Math.sin(x * 1.5) + x * 0.4;
    case 'quadratic':     return 0.4 * x * x - x + 1;
    case 'heteroscedastic': return 2 * x + 1;
    default:              return x;
  }
}

function generateData(n, noiseLevel, funcType) {
  const xs = Array.from({ length: n }, () => (Math.random() - 0.5) * 8);
  const ys = xs.map(x => {
    const sigma = funcType === 'heteroscedastic'
      ? noiseLevel * (0.4 + 0.3 * Math.abs(x))
      : noiseLevel;
    return trueFunc(x, funcType) + sigma * randn();
  });
  // Sort by x for nice plotting
  const pairs = xs.map((x, i) => ({ x, y: ys[i] })).sort((a, b) => a.x - b.x);
  return { xs: pairs.map(p => p.x), ys: pairs.map(p => p.y) };
}

// ─── Model Fitting ───────────────────────────────────────────
function fitModel(xs, ys, modelType) {
  if (modelType === 'linreg' || modelType === 'poly2' || modelType === 'poly3') {
    const degree = modelType === 'linreg' ? 1 : modelType === 'poly2' ? 2 : 3;
    const coeffs = polyFit(xs, ys, degree);
    return { predict: x => polyEval(coeffs, x), coeffs, type: 'poly' };
  }
  if (modelType === 'moving') {
    return { predict: x => movingAvgPredict(xs, ys, x, 30), type: 'moving' };
  }
  return { predict: x => 0 };
}

// Polynomial regression via normal equations (simplified, robust for small degree)
function polyFit(xs, ys, degree) {
  const n = xs.length;
  const d = degree + 1;
  // Build Vandermonde matrix columns
  const X = xs.map(x => Array.from({ length: d }, (_, k) => Math.pow(x, k)));
  // X^T X
  const XtX = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) =>
      X.reduce((s, row) => s + row[i] * row[j], 0)));
  // X^T y
  const Xty = Array.from({ length: d }, (_, i) =>
    X.reduce((s, row, r) => s + row[i] * ys[r], 0));
  // Solve via Gaussian elimination
  return gaussElim(XtX, Xty);
}

function gaussElim(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    // Pivot
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-12) continue;
    for (let k = col; k <= n; k++) M[col][k] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = M[row][col];
      for (let k = col; k <= n; k++) M[row][k] -= factor * M[col][k];
    }
  }
  return M.map(row => row[n]);
}

function polyEval(coeffs, x) {
  return coeffs.reduce((s, c, k) => s + c * Math.pow(x, k), 0);
}

function movingAvgPredict(xs, ys, x, k) {
  // k-nearest neighbors average
  const dists = xs.map((xi, i) => ({ d: Math.abs(xi - x), y: ys[i] }));
  dists.sort((a, b) => a.d - b.d);
  const neighbors = dists.slice(0, k);
  return neighbors.reduce((s, p) => s + p.y, 0) / neighbors.length;
}

// ─── Nonconformity Scores ────────────────────────────────────
function computeScore(x, y, yhat, scoreType, residStd) {
  const err = y - yhat;
  switch (scoreType) {
    case 'absolute': return Math.abs(err);
    case 'normalized': return Math.abs(err) / (residStd + 0.01);
    case 'signed': return err; // uses quantile interval
    default: return Math.abs(err);
  }
}

// ─── Quantile ───────────────────────────────────────────────
function quantile(arr, q) {
  const sorted = [...arr].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo);
}

// ─── MAIN SIMULATION ─────────────────────────────────────────
function runSimulation() {
  const n         = +document.getElementById('nTotal').value;
  const calibFrac = +document.getElementById('calibRatio').value;
  const noiseLevel= +document.getElementById('noiseLevel').value;
  const funcType  = document.getElementById('funcType').value;
  const alpha     = +document.getElementById('alphaVal').value;
  const scoreType = document.getElementById('scoreType').value;
  const modelType = document.getElementById('modelType').value;
  const showCalib = document.getElementById('showCalib').checked;
  const showTrain = document.getElementById('showTrain').checked;
  const showBand  = document.getElementById('showBand').checked;
  const showOut   = document.getElementById('showOutlier').checked;

  // ── Step 1: Generate & split data
  activateStep(1);
  const { xs, ys } = generateData(n, noiseLevel, funcType);
  const nCalib = Math.floor(n * calibFrac);
  const nTrain = n - nCalib;

  const trainXs = xs.slice(0, nTrain);
  const trainYs = ys.slice(0, nTrain);
  const calibXs = xs.slice(nTrain);
  const calibYs = ys.slice(nTrain);

  updateStepDesc(1, `${nTrain} eğitim, ${nCalib} kalibrasyon örneği`);

  // ── Step 2: Fit model
  activateStep(2);
  const model = fitModel(trainXs, trainYs, modelType);
  updateStepDesc(2, `${modelType === 'linreg' ? 'Doğrusal regresyon' : modelType === 'poly2' ? 'Polinom (k=2)' : modelType === 'poly3' ? 'Polinom (k=3)' : 'KNN hareketli ort.'} fitlendi`);

  // ── Step 3: Calibration scores
  activateStep(3);
  const calibPreds = calibXs.map(x => model.predict(x));
  const residStd = std(calibYs.map((y, i) => y - calibPreds[i]));
  const calibScores = calibXs.map((x, i) =>
    computeScore(x, calibYs[i], calibPreds[i], scoreType, residStd));
  updateStepDesc(3, `${nCalib} uyumsuzluk skoru; ${scoreType === 'absolute' ? 'mutlak hata' : scoreType === 'normalized' ? 'normalize hata' : 'işaretli hata'}`);

  // ── Step 4: Compute qhat
  activateStep(4);
  const quantileLevel = Math.min((1 - alpha) * (1 + 1 / nCalib), 1);
  let qhat;
  if (scoreType === 'signed') {
    // Two-sided: use ±qhat
    const absScores = calibScores.map(Math.abs);
    qhat = quantile(absScores, quantileLevel);
  } else {
    qhat = quantile(calibScores, quantileLevel);
  }
  updateStepDesc(4, `q̂ = ${qhat.toFixed(3)} (α=${alpha}, quantile=${(quantileLevel*100).toFixed(1)}%)`);

  // ── Step 5: Build prediction sets on a fine grid
  activateStep(5);
  const gridXs = linspace(Math.min(...xs), Math.max(...xs), 200);
  const gridPreds = gridXs.map(x => model.predict(x));
  const gridLow  = gridPreds.map(p => p - qhat);
  const gridHigh = gridPreds.map(p => p + qhat);
  updateStepDesc(5, `Ĉ(x) = [ŷ(x) − q̂, ŷ(x) + q̂]`);

  // ── Compute empirical coverage on calibration set
  let covered = 0;
  calibXs.forEach((x, i) => {
    const pred = model.predict(x);
    const lo = pred - qhat, hi = pred + qhat;
    if (calibYs[i] >= lo && calibYs[i] <= hi) covered++;
  });
  const coverage = covered / nCalib;
  const avgWidth = 2 * qhat;

  // Store for reuse
  state.data = { xs, ys, trainXs, trainYs, calibXs, calibYs,
    gridXs, gridPreds, gridLow, gridHigh, calibPreds, qhat,
    coverage, avgWidth, alpha, nCalib };

  // ── Update UI
  updateHeroStats(coverage, 1 - alpha, n, avgWidth);
  updateMetrics({ nCalib, qhat, coverage, alpha, avgWidth });
  updateCoverageBadge(coverage, 1 - alpha);

  // ── Render charts
  renderMainChart({
    xs, ys, trainXs, trainYs, calibXs, calibYs,
    gridXs, gridPreds, gridLow, gridHigh, calibPreds, qhat,
    showCalib, showTrain, showBand, showOut, coverage, alpha
  });
  renderScoreChart(calibScores, qhat, alpha);
  renderAlphaChart(trainXs, trainYs, calibXs, calibYs, model, scoreType, residStd);
}

// ─── std dev ─────────────────────────────────────────────────
function std(arr) {
  const m = arr.reduce((s, v) => s + v, 0) / arr.length;
  const v = arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length;
  return Math.sqrt(v);
}

// ─── Step UI ─────────────────────────────────────────────────
function activateStep(n) {
  document.querySelectorAll('.step').forEach((el, i) => {
    el.classList.toggle('active', i < n);
  });
}

function updateStepDesc(n, text) {
  const el = document.getElementById(`stepDesc${n}`);
  if (el) el.textContent = text;
}

// ─── Hero Stats ──────────────────────────────────────────────
function updateHeroStats(coverage, target, n, width) {
  animateVal('statCoverage', (coverage * 100).toFixed(1) + '%');
  animateVal('statTarget', (target * 100).toFixed(0) + '%');
  animateVal('statN', n);
  animateVal('statWidth', width.toFixed(3));
}

function animateVal(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.opacity = '0.3';
  el.style.transform = 'translateY(6px)';
  requestAnimationFrame(() => {
    el.textContent = val;
    el.style.transition = 'opacity .4s, transform .4s';
    el.style.opacity = '1';
    el.style.transform = 'translateY(0)';
  });
}

// ─── Metrics ─────────────────────────────────────────────────
function updateMetrics({ nCalib, qhat, coverage, alpha, avgWidth }) {
  document.getElementById('mCalibN').textContent  = nCalib;
  document.getElementById('mQhat').textContent    = qhat.toFixed(4);
  document.getElementById('mCoverage').textContent= (coverage * 100).toFixed(2) + '%';
  document.getElementById('mExpected').textContent= ((1 - alpha) * 100).toFixed(0) + '%';

  const dev = ((coverage - (1 - alpha)) * 100).toFixed(2);
  const devEl = document.getElementById('mDeviation');
  devEl.textContent = (dev > 0 ? '+' : '') + dev + '%';
  devEl.style.color = Math.abs(+dev) < 3 ? 'var(--success)' : 'var(--danger)';

  document.getElementById('mBandWidth').textContent = (2 * qhat).toFixed(4);
}

// ─── Coverage Badge ──────────────────────────────────────────
function updateCoverageBadge(coverage, target) {
  const badge = document.getElementById('coverageBadge');
  badge.textContent = `${(coverage * 100).toFixed(1)}% kapsama`;
  badge.classList.toggle('warn', coverage < target - 0.05);
}

// ─── CHART 1: Main prediction band ──────────────────────────
function renderMainChart({ xs, ys, trainXs, trainYs, calibXs, calibYs,
    gridXs, gridPreds, gridLow, gridHigh, calibPreds, qhat,
    showCalib, showTrain, showBand, showOut, coverage, alpha }) {

  const ctx = document.getElementById('mainChart').getContext('2d');

  const datasets = [];

  // Band fill
  if (showBand) {
    datasets.push({
      label: 'Üst Sınır',
      data: gridXs.map((x, i) => ({ x, y: gridHigh[i] })),
      type: 'line', borderColor: 'transparent',
      backgroundColor: 'rgba(110, 142, 251, 0.12)',
      fill: '+1',
      pointRadius: 0, tension: 0.3, order: 10,
    });
    datasets.push({
      label: 'Alt Sınır',
      data: gridXs.map((x, i) => ({ x, y: gridLow[i] })),
      type: 'line', borderColor: 'rgba(110,142,251,0.35)',
      backgroundColor: 'rgba(110, 142, 251, 0.12)',
      fill: false,
      pointRadius: 0, tension: 0.3,
      borderDash: [4, 4], order: 10,
      borderWidth: 1.5,
    });
    // Model line
    datasets.push({
      label: 'Model (ŷ)',
      data: gridXs.map((x, i) => ({ x, y: gridPreds[i] })),
      type: 'line', borderColor: '#6e8efb',
      backgroundColor: 'transparent',
      fill: false,
      pointRadius: 0, tension: 0.3,
      borderWidth: 2.5, order: 9,
    });
  }

  // Training points
  if (showTrain) {
    datasets.push({
      label: 'Eğitim',
      data: trainXs.map((x, i) => ({ x, y: trainYs[i] })),
      type: 'scatter',
      backgroundColor: 'rgba(56, 189, 248, 0.5)',
      borderColor: 'rgba(56, 189, 248, 0.8)',
      borderWidth: 1, pointRadius: 4, order: 5,
    });
  }

  // Calibration points (split by covered/not)
  if (showCalib) {
    const calibCovered = [], calibOutlie = [];
    calibXs.forEach((x, i) => {
      const pred = calibPreds[i];
      const inside = calibYs[i] >= pred - qhat && calibYs[i] <= pred + qhat;
      if (inside) calibCovered.push({ x, y: calibYs[i] });
      else calibOutlie.push({ x, y: calibYs[i] });
    });

    datasets.push({
      label: 'Kalibrasyon (içinde)',
      data: calibCovered,
      type: 'scatter',
      backgroundColor: 'rgba(52, 211, 153, 0.6)',
      borderColor: 'rgba(52, 211, 153, 0.9)',
      borderWidth: 1, pointRadius: 5, order: 6,
    });

    if (showOut) {
      datasets.push({
        label: 'Kalibrasyon (dışında)',
        data: calibOutlie,
        type: 'scatter',
        backgroundColor: 'rgba(248, 113, 113, 0.7)',
        borderColor: 'rgba(248, 113, 113, 1)',
        borderWidth: 1.5, pointRadius: 6,
        pointStyle: 'crossRot', order: 7,
      });
    }
  }

  if (state.mainChart) state.mainChart.destroy();
  state.mainChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600, easing: 'easeOutQuart' },
      plugins: {
        legend: {
          labels: {
            color: '#8892b0',
            font: { family: 'Inter', size: 11 },
            boxWidth: 12, boxHeight: 12
          }
        },
        tooltip: {
          backgroundColor: 'rgba(26,33,56,0.95)',
          borderColor: 'rgba(110,142,251,0.3)',
          borderWidth: 1,
          titleColor: '#e8eaf6',
          bodyColor: '#8892b0',
          callbacks: {
            label: ctx => `(${ctx.parsed.x.toFixed(2)}, ${ctx.parsed.y.toFixed(2)})`
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#5a647a', font: { family: 'JetBrains Mono', size: 11 } },
          title: { display: true, text: 'x', color: '#8892b0', font: { size: 12 } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#5a647a', font: { family: 'JetBrains Mono', size: 11 } },
          title: { display: true, text: 'y', color: '#8892b0', font: { size: 12 } }
        }
      }
    }
  });
}

// ─── CHART 2: Score Distribution ────────────────────────────
function renderScoreChart(calibScores, qhat, alpha) {
  const ctx = document.getElementById('scoreChart').getContext('2d');

  // Build histogram
  const minS = 0, maxS = Math.max(...calibScores) * 1.05;
  const bins = 30;
  const binSize = (maxS - minS) / bins;
  const counts = new Array(bins).fill(0);
  calibScores.forEach(s => {
    const idx = Math.min(Math.floor((s - minS) / binSize), bins - 1);
    counts[idx]++;
  });
  const labels = Array.from({ length: bins }, (_, i) => (minS + (i + 0.5) * binSize).toFixed(2));
  const bgColors = labels.map(l => parseFloat(l) <= qhat
    ? 'rgba(110,142,251,0.55)'
    : 'rgba(248,113,113,0.55)');

  if (state.scoreChart) state.scoreChart.destroy();
  state.scoreChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Frekans',
        data: counts,
        backgroundColor: bgColors,
        borderColor: bgColors.map(c => c.replace('0.55', '0.85')),
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(26,33,56,0.95)',
          borderColor: 'rgba(110,142,251,0.3)',
          borderWidth: 1,
          titleColor: '#e8eaf6',
          bodyColor: '#8892b0',
        },
        annotation: {}
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#5a647a', font: { family: 'JetBrains Mono', size: 10 },
            maxTicksLimit: 6 },
          title: { display: true, text: 'Uyumsuzluk Skoru', color: '#8892b0', font: { size: 11 } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#5a647a', font: { size: 10 } },
          title: { display: true, text: 'Sayı', color: '#8892b0', font: { size: 11 } }
        }
      }
    }
  });
}

// ─── CHART 3: Coverage vs alpha curve ───────────────────────
function renderAlphaChart(trainXs, trainYs, calibXs, calibYs, model, scoreType, residStd) {
  const ctx = document.getElementById('alphaChart').getContext('2d');

  const alphas = linspace(0.01, 0.5, 50);
  const calibPreds = calibXs.map(x => model.predict(x));
  const calibScores = calibXs.map((x, i) =>
    computeScore(x, calibYs[i], calibPreds[i], scoreType, residStd));
  const nCalib = calibXs.length;

  const coverages = alphas.map(a => {
    const ql = Math.min((1 - a) * (1 + 1 / nCalib), 1);
    const q = quantile(calibScores, ql);
    let covered = 0;
    calibXs.forEach((x, i) => {
      const pred = calibPreds[i];
      if (calibYs[i] >= pred - q && calibYs[i] <= pred + q) covered++;
    });
    return covered / nCalib;
  });

  const currentAlpha = +document.getElementById('alphaVal').value;

  if (state.alphaChart) state.alphaChart.destroy();
  state.alphaChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: alphas.map(a => a.toFixed(2)),
      datasets: [
        {
          label: 'Gerçek Kapsama',
          data: coverages.map((c, i) => ({ x: alphas[i], y: c })),
          borderColor: '#6e8efb',
          backgroundColor: 'rgba(110,142,251,0.08)',
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
        },
        {
          label: '1−α (Hedef)',
          data: alphas.map(a => ({ x: a, y: 1 - a })),
          borderColor: 'rgba(52,211,153,0.6)',
          borderDash: [5, 5],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0,
        },
        {
          label: 'Seçili α',
          data: [{ x: currentAlpha, y: 0 }, { x: currentAlpha, y: 1 }],
          borderColor: 'rgba(251,146,60,0.7)',
          borderWidth: 1.5,
          borderDash: [3, 3],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: {
          labels: {
            color: '#8892b0',
            font: { family: 'Inter', size: 11 },
            boxWidth: 12, boxHeight: 12
          }
        },
        tooltip: {
          backgroundColor: 'rgba(26,33,56,0.95)',
          borderColor: 'rgba(110,142,251,0.3)',
          borderWidth: 1,
          titleColor: '#e8eaf6',
          bodyColor: '#8892b0',
        }
      },
      scales: {
        x: {
          type: 'linear',
          min: 0, max: 0.5,
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#5a647a', font: { family: 'JetBrains Mono', size: 10 } },
          title: { display: true, text: 'α', color: '#8892b0', font: { size: 12 } }
        },
        y: {
          min: 0, max: 1,
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: {
            color: '#5a647a', font: { family: 'JetBrains Mono', size: 10 },
            callback: v => (v * 100).toFixed(0) + '%'
          },
          title: { display: true, text: 'Kapsama', color: '#8892b0', font: { size: 11 } }
        }
      }
    }
  });
}

// ─── Slider range fill update ─────────────────────────────────
function updateSliderFill(input) {
  const min = +input.min, max = +input.max, val = +input.value;
  const pct = ((val - min) / (max - min) * 100).toFixed(1) + '%';
  input.style.setProperty('--pct', pct);
}

// ─── Event Wiring ─────────────────────────────────────────────
function init() {
  const sliders = [
    { id: 'nTotal',     display: 'nTotalVal',     fmt: v => v },
    { id: 'calibRatio', display: 'calibRatioVal',  fmt: v => Math.round(v * 100) + '%' },
    { id: 'noiseLevel', display: 'noiseLevelVal',  fmt: v => parseFloat(v).toFixed(1) },
    { id: 'alphaVal',   display: 'alphaValText',   fmt: v => `${parseFloat(v).toFixed(2)} → ${Math.round((1-v)*100)}%` },
  ];

  sliders.forEach(({ id, display, fmt }) => {
    const input = document.getElementById(id);
    const out   = document.getElementById(display);
    updateSliderFill(input);
    input.addEventListener('input', () => {
      out.textContent = fmt(input.value);
      updateSliderFill(input);
      debounceRun();
    });
  });

  ['funcType', 'scoreType', 'modelType'].forEach(id => {
    document.getElementById(id).addEventListener('change', debounceRun);
  });

  ['showCalib','showTrain','showBand','showOutlier'].forEach(id => {
    document.getElementById(id).addEventListener('change', debounceRun);
  });

  document.getElementById('btnResample').addEventListener('click', () => {
    runSimulation();
    // button pulse
    const btn = document.getElementById('btnResample');
    btn.style.transform = 'scale(0.96)';
    setTimeout(() => btn.style.transform = '', 150);
  });

  document.getElementById('btnAnimate').addEventListener('click', toggleAnimation);

  // Initial run
  runSimulation();
}

// ─── Debounce ─────────────────────────────────────────────────
let _debTimer = null;
function debounceRun() {
  clearTimeout(_debTimer);
  _debTimer = setTimeout(runSimulation, 120);
}

// ─── Alpha Animation ──────────────────────────────────────────
function toggleAnimation() {
  const btn = document.getElementById('btnAnimate');
  if (state.animRunning) {
    state.animRunning = false;
    cancelAnimationFrame(state.animFrame);
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg> α Animasyonu`;
    btn.classList.remove('running');
    return;
  }
  state.animRunning = true;
  btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Durdur`;
  btn.classList.add('running');

  let t = 0;
  const alphaInput = document.getElementById('alphaVal');
  const alphaOut   = document.getElementById('alphaValText');

  function step() {
    if (!state.animRunning) return;
    t += 0.008;
    const alpha = 0.01 + 0.49 * (0.5 + 0.5 * Math.sin(t));
    alphaInput.value = alpha.toFixed(3);
    alphaOut.textContent = `${alpha.toFixed(2)} → ${Math.round((1-alpha)*100)}%`;
    updateSliderFill(alphaInput);
    runSimulation();
    state.animFrame = requestAnimationFrame(step);
  }
  state.animFrame = requestAnimationFrame(step);
}
