/**
 * Cluster Trends visualization
 *
 * Three charts showing how attention head clusters vary across
 * layer depth and model size:
 * 1. Layer depth x cluster fraction (scatter/line)
 * 2. Model size x cluster fraction (scatter)
 * 3. Cluster entropy by layer depth (line)
 */

// ── Config ──────────────────────────────────────────────────────
const urlParams = new URLSearchParams(window.location.search);
const CONFIG = {
  dataUrl: urlParams.get("data") || "../../features/cluster_trends.json",
  defaultK: urlParams.get("k") || null,
};

// ── State ───────────────────────────────────────────────────────
let DATA = null;
let currentK = null;

/** @type {Chart|null} */ let layerChart = null;
/** @type {Chart|null} */ let sizeChart = null;
/** @type {Chart|null} */ let entropyChart = null;

// Which models/families are enabled (true = visible)
/** @type {Object<string, boolean>} */ let modelEnabled = {};
/** @type {Object<string, boolean>} */ let familyEnabled = {};

// ── Colors ──────────────────────────────────────────────────────

const GOLDEN_ANGLE = 137.508;

/**
 * Generate a distinct color for a cluster index.
 * @param {number} idx
 * @returns {string} CSS hsl color
 */
function clusterColor(idx) {
  const hue = (idx * GOLDEN_ANGLE) % 360;
  const sat = 65 + (idx % 3) * 10;
  const lit = 50 + (idx % 2) * 8;
  return `hsl(${hue}, ${sat}%, ${lit}%)`;
}

/**
 * Generate a color for a model index (different palette from clusters).
 * @param {number} idx
 * @param {number} total
 * @returns {string} CSS hsl color
 */
function modelColor(idx, total) {
  const hue = (idx / Math.max(total, 1)) * 360;
  return `hsl(${hue}, 70%, 60%)`;
}

// ── Data loading ────────────────────────────────────────────────

async function loadData() {
  const resp = await fetch(CONFIG.dataUrl);
  if (!resp.ok) {
    throw new Error(`Failed to fetch ${CONFIG.dataUrl}: ${resp.status}`);
  }
  DATA = await resp.json();
}

// ── Toggle helpers ──────────────────────────────────────────────

/**
 * Create model toggle buttons inside a container.
 * @param {string} containerId
 * @param {() => void} onChange - called when a toggle changes
 */
function buildModelToggles(containerId, onChange) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const models = Object.keys(DATA.models).sort();
  for (const m of models) {
    const btn = document.createElement("span");
    btn.className = "model-toggle";
    btn.textContent = m;
    btn.dataset.model = m;
    if (!modelEnabled[m]) btn.classList.add("disabled");
    btn.addEventListener("click", () => {
      modelEnabled[m] = !modelEnabled[m];
      btn.classList.toggle("disabled", !modelEnabled[m]);
      onChange();
    });
    container.appendChild(btn);
  }
}

/**
 * Create family toggle buttons inside a container.
 * @param {string} containerId
 * @param {() => void} onChange
 */
function buildFamilyToggles(containerId, onChange) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const families = [
    ...new Set(Object.values(DATA.models).map((m) => m.family)),
  ].sort();
  for (const f of families) {
    const btn = document.createElement("span");
    btn.className = "family-toggle";
    btn.textContent = f;
    btn.dataset.family = f;
    if (!familyEnabled[f]) btn.classList.add("disabled");
    btn.addEventListener("click", () => {
      familyEnabled[f] = !familyEnabled[f];
      btn.classList.toggle("disabled", !familyEnabled[f]);
      onChange();
    });
    container.appendChild(btn);
  }
}

// ── Chart 1: Layer depth x cluster fraction ─────────────────────

function buildLayerChart() {
  const key = `k${currentK}`;
  const records = DATA.by_layer[key] || [];
  const showLines = document.getElementById("show-lines").checked;

  // Group by cluster -> array of {x: depth, y: frac, model}
  /** @type {Object<number, Array<{x:number, y:number, model:string}>>} */
  const byCluster = {};
  for (const r of records) {
    if (!modelEnabled[r.model]) continue;
    if (!byCluster[r.cluster]) byCluster[r.cluster] = [];
    byCluster[r.cluster].push({ x: r.depth, y: r.frac, model: r.model });
  }

  const clusterIds = Object.keys(byCluster)
    .map(Number)
    .sort((a, b) => a - b);

  // For each cluster, create one dataset per model (so lines connect within a model)
  const datasets = [];
  const models = Object.keys(DATA.models).sort();
  for (const cid of clusterIds) {
    const points = byCluster[cid] || [];
    // group by model
    /** @type {Object<string, Array<{x:number, y:number}>>} */
    const byModel = {};
    for (const p of points) {
      if (!byModel[p.model]) byModel[p.model] = [];
      byModel[p.model].push({ x: p.x, y: p.y });
    }
    const color = clusterColor(cid);
    let isFirst = true;
    for (const m of models) {
      const pts = byModel[m];
      if (!pts) continue;
      pts.sort((a, b) => a.x - b.x);
      datasets.push({
        label: isFirst ? `Cluster ${cid}` : "",
        data: pts,
        backgroundColor: color,
        borderColor: color,
        borderWidth: showLines ? 1.5 : 0,
        pointRadius: 3,
        pointHoverRadius: 5,
        showLine: showLines,
        tension: 0,
        _clusterId: cid,
        _model: m,
      });
      isFirst = false;
    }
  }

  const ctx = document.getElementById("layer-chart").getContext("2d");
  if (layerChart) layerChart.destroy();

  layerChart = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "nearest",
        intersect: true,
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Normalized Layer Depth",
            color: "#a0a0b0",
          },
          min: -0.02,
          max: 1.02,
          ticks: { color: "#808090" },
          grid: { color: "#1a2a40" },
        },
        y: {
          title: { display: true, text: "Fraction of Heads", color: "#a0a0b0" },
          min: 0,
          ticks: { color: "#808090" },
          grid: { color: "#1a2a40" },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "right",
          labels: {
            color: "#c0c0d0",
            font: { size: 11 },
            // Only show one entry per cluster (not per model)
            filter: (item) => item.text !== "",
          },
          onClick: (_e, legendItem, legend) => {
            // Toggle all datasets for this cluster
            const cid =
              legend.chart.data.datasets[legendItem.datasetIndex]._clusterId;
            const isHidden = !legendItem.hidden;
            for (let i = 0; i < legend.chart.data.datasets.length; i++) {
              if (legend.chart.data.datasets[i]._clusterId === cid) {
                legend.chart.setDatasetVisibility(i, isHidden);
              }
            }
            legend.chart.update();
          },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const ds = ctx.dataset;
              return `${ds._model} | Cluster ${ds._clusterId} | depth=${ctx.parsed.x.toFixed(2)} frac=${ctx.parsed.y.toFixed(3)}`;
            },
          },
        },
      },
    },
  });
}

// ── Chart 2: Model size x cluster fraction ──────────────────────

function buildSizeChart() {
  const key = `k${currentK}`;
  const records = DATA.by_model[key] || [];

  // Group by cluster
  /** @type {Object<number, Array<{x:number, y:number, model:string}>>} */
  const byCluster = {};
  for (const r of records) {
    const meta = DATA.models[r.model];
    if (!meta) continue;
    if (!familyEnabled[meta.family]) continue;
    if (!byCluster[r.cluster]) byCluster[r.cluster] = [];
    byCluster[r.cluster].push({
      x: meta.n_params,
      y: r.frac,
      model: r.model,
    });
  }

  const clusterIds = Object.keys(byCluster)
    .map(Number)
    .sort((a, b) => a - b);

  const datasets = clusterIds.map((cid) => {
    const pts = (byCluster[cid] || []).sort((a, b) => a.x - b.x);
    const color = clusterColor(cid);
    return {
      label: `Cluster ${cid}`,
      data: pts,
      backgroundColor: color,
      borderColor: color,
      borderWidth: 0,
      pointRadius: 4,
      pointHoverRadius: 6,
      showLine: false,
      _clusterId: cid,
    };
  });

  const ctx = document.getElementById("size-chart").getContext("2d");
  if (sizeChart) sizeChart.destroy();

  sizeChart = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "nearest",
        intersect: true,
      },
      scales: {
        x: {
          type: "logarithmic",
          title: { display: true, text: "Parameters", color: "#a0a0b0" },
          ticks: {
            color: "#808090",
            callback: (val) => {
              if (val >= 1e9) return (val / 1e9).toFixed(1) + "B";
              if (val >= 1e6) return (val / 1e6).toFixed(0) + "M";
              if (val >= 1e3) return (val / 1e3).toFixed(0) + "K";
              return val;
            },
          },
          grid: { color: "#1a2a40" },
        },
        y: {
          title: { display: true, text: "Fraction of Heads", color: "#a0a0b0" },
          min: 0,
          ticks: { color: "#808090" },
          grid: { color: "#1a2a40" },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "right",
          labels: { color: "#c0c0d0", font: { size: 11 } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const pt = ctx.raw;
              const ds = ctx.dataset;
              return `${pt.model} | Cluster ${ds._clusterId} | frac=${pt.y.toFixed(3)}`;
            },
          },
        },
      },
    },
  });
}

// ── Chart 3: Cluster entropy by layer ───────────────────────────

function buildEntropyChart() {
  const key = `k${currentK}`;
  const records = DATA.entropy_by_layer[key] || [];

  // Group by model
  /** @type {Object<string, Array<{x:number, y:number}>>} */
  const byModel = {};
  for (const r of records) {
    if (!modelEnabled[r.model]) continue;
    if (!byModel[r.model]) byModel[r.model] = [];
    byModel[r.model].push({ x: r.depth, y: r.entropy });
  }

  const models = Object.keys(byModel).sort();
  const datasets = models.map((m, i) => {
    const pts = byModel[m].sort((a, b) => a.x - b.x);
    const color = modelColor(i, models.length);
    return {
      label: m,
      data: pts,
      backgroundColor: color,
      borderColor: color,
      borderWidth: 2,
      pointRadius: 2,
      pointHoverRadius: 4,
      showLine: true,
      tension: 0.2,
      fill: false,
    };
  });

  const ctx = document.getElementById("entropy-chart").getContext("2d");
  if (entropyChart) entropyChart.destroy();

  entropyChart = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "nearest",
        intersect: false,
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Normalized Layer Depth",
            color: "#a0a0b0",
          },
          min: -0.02,
          max: 1.02,
          ticks: { color: "#808090" },
          grid: { color: "#1a2a40" },
        },
        y: {
          title: {
            display: true,
            text: "Shannon Entropy (bits)",
            color: "#a0a0b0",
          },
          min: 0,
          ticks: { color: "#808090" },
          grid: { color: "#1a2a40" },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "right",
          labels: { color: "#c0c0d0", font: { size: 11 } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const ds = ctx.dataset;
              return `${ds.label} | depth=${ctx.parsed.x.toFixed(2)} entropy=${ctx.parsed.y.toFixed(3)}`;
            },
          },
        },
      },
    },
  });
}

// ── Rebuild all ─────────────────────────────────────────────────

function rebuildAll() {
  buildLayerChart();
  buildSizeChart();
  buildEntropyChart();
}

// ── Init ────────────────────────────────────────────────────────

async function init() {
  try {
    await loadData();
  } catch (err) {
    document.getElementById("loading").textContent =
      `Error loading data: ${err.message}`;
    return;
  }

  // Initialize toggle states
  for (const m of Object.keys(DATA.models)) {
    modelEnabled[m] = true;
  }
  const families = [
    ...new Set(Object.values(DATA.models).map((m) => m.family)),
  ];
  for (const f of families) {
    familyEnabled[f] = true;
  }

  // Populate K selector
  const kSelect = document.getElementById("k-select");
  for (const k of DATA.k_values) {
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = `K = ${k}`;
    kSelect.appendChild(opt);
  }

  // Set default K
  if (CONFIG.defaultK && DATA.k_values.includes(Number(CONFIG.defaultK))) {
    currentK = Number(CONFIG.defaultK);
  } else {
    // pick a sensible default: prefer 10 if available
    currentK = DATA.k_values.includes(10) ? 10 : DATA.k_values[0];
  }
  kSelect.value = currentK;

  // Event listeners
  kSelect.addEventListener("change", () => {
    currentK = Number(kSelect.value);
    rebuildAll();
  });

  document.getElementById("show-lines").addEventListener("change", () => {
    buildLayerChart();
  });

  // Build toggle buttons
  buildModelToggles("layer-model-toggles", () => {
    buildLayerChart();
    buildEntropyChart();
  });
  buildFamilyToggles("size-family-toggles", () => {
    buildSizeChart();
  });
  buildModelToggles("entropy-model-toggles", () => {
    buildEntropyChart();
    buildLayerChart();
  });

  // Show charts, hide loading
  document.getElementById("loading").style.display = "none";
  document.getElementById("charts-container").style.display = "block";

  // Draw
  rebuildAll();
}

init();
