/**
 * Cluster Trends visualization
 *
 * Three charts showing how attention head clusters vary across
 * layer depth and model size:
 * 1. Layer depth x cluster fraction (scatter/line)
 * 2. Model size x cluster fraction (scatter)
 * 3. Cluster entropy by layer depth (line)
 *
 * Supports both precomputed K-based data and dynamic cut-height clustering.
 */

// ── Config ──────────────────────────────────────────────────────
const urlParams = new URLSearchParams(window.location.search);
const CONFIG = {
  dataUrl: urlParams.get("data") || "../../features/cluster_trends.json",
  defaultK: urlParams.get("k") || null,
  clusteringMetaUrl:
    urlParams.get("meta") || "../../features/clustering/clustering_meta.json",
  linkageUrl:
    urlParams.get("linkage") || "../../features/clustering/linkage.json",
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

// Cut-height clustering state
/** @type {number[][]|null} */ let linkageData = null;
/** @type {string[]|null} */ let clsValues = null;
/** @type {boolean} */ let usingCutHeight = true;

// Current records used by chart builders (either precomputed or dynamic)
let currentRecords = { by_layer: [], by_model: [], entropy_by_layer: [] };

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

async function loadClusteringData() {
  const [metaResp, linkageResp] = await Promise.all([
    fetch(CONFIG.clusteringMetaUrl),
    fetch(CONFIG.linkageUrl),
  ]);
  if (!metaResp.ok || !linkageResp.ok) {
    console.warn("Clustering data not available for cut-height mode");
    return false;
  }
  const meta = await metaResp.json();
  clsValues = meta.cls_values;
  linkageData = await linkageResp.json();
  return true;
}

// ── Cut-height clustering ───────────────────────────────────────

/**
 * Compute cluster assignments by cutting the dendrogram at a given height.
 * Same union-find algorithm as ClusteringLoader / gridView.
 * @param {number} cutHeight
 * @returns {Object<string, number>} head ID -> cluster ID
 */
function computeAssignmentsByCutHeight(cutHeight) {
  if (!linkageData || !clsValues) return {};

  const n = clsValues.length;
  const parent = Array.from({ length: 2 * n - 1 }, (_, i) => i);

  function find(x) {
    if (parent[x] !== x) parent[x] = find(parent[x]);
    return parent[x];
  }

  function union(x, y, newParent) {
    parent[find(x)] = newParent;
    parent[find(y)] = newParent;
  }

  for (let i = 0; i < linkageData.length; i++) {
    const [idx1, idx2, distance] = linkageData[i];
    if (distance <= cutHeight) {
      union(Math.floor(idx1), Math.floor(idx2), n + i);
    }
  }

  const rootToCluster = {};
  let nextCluster = 0;
  const assignments = {};

  for (let i = 0; i < n; i++) {
    const root = find(i);
    if (!(root in rootToCluster)) {
      rootToCluster[root] = nextCluster++;
    }
    assignments[clsValues[i]] = rootToCluster[root];
  }

  return assignments;
}

/**
 * Shannon entropy from a list of counts.
 * @param {number[]} counts
 * @returns {number}
 */
function shannonEntropy(counts) {
  const total = counts.reduce((a, b) => a + b, 0);
  if (total === 0) return 0;
  let entropy = 0;
  for (const c of counts) {
    if (c > 0) {
      const p = c / total;
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}

/**
 * From head assignments, compute trend records in the same format
 * as the precomputed DATA (by_layer, by_model, entropy_by_layer).
 * @param {Object<string, number>} assignments - head ID -> cluster ID
 * @returns {{by_layer: Array, by_model: Array, entropy_by_layer: Array}}
 */
function computeTrendRecords(assignments) {
  // Parse cls_values "model:Llayer:Hhead" into per-head metadata
  const heads = [];
  for (const cls of clsValues) {
    const parts = cls.split(":");
    const model = parts[0];
    const layer = parseInt(parts[1].substring(1));
    const cluster = assignments[cls];
    if (cluster !== undefined) {
      heads.push({ model, layer, cluster });
    }
  }

  const byLayerRecords = [];
  const entropyRecords = [];
  const byModelRecords = [];

  // Group heads by model
  const headsByModel = {};
  for (const h of heads) {
    if (!headsByModel[h.model]) headsByModel[h.model] = [];
    headsByModel[h.model].push(h);
  }

  for (const [modelName, modelHeads] of Object.entries(headsByModel)) {
    const meta = DATA.models[modelName];
    if (!meta) continue;

    const nLayers = meta.n_layers;
    const nHeadsPerLayer = meta.n_heads;

    // --- by_layer: per (model, layer, cluster) ---
    const layerGroups = {};
    for (const h of modelHeads) {
      if (!layerGroups[h.layer]) layerGroups[h.layer] = {};
      const lg = layerGroups[h.layer];
      lg[h.cluster] = (lg[h.cluster] || 0) + 1;
    }

    for (const [layerStr, clusterCounts] of Object.entries(layerGroups)) {
      const layerIdx = parseInt(layerStr);
      const depth = layerIdx / Math.max(nLayers - 1, 1);

      for (const [clusterStr, count] of Object.entries(clusterCounts)) {
        const frac = count / nHeadsPerLayer;
        byLayerRecords.push({
          model: modelName,
          layer: layerIdx,
          depth: Math.round(depth * 10000) / 10000,
          cluster: parseInt(clusterStr),
          count: count,
          frac: Math.round(frac * 10000) / 10000,
        });
      }

      // Shannon entropy at this layer
      const counts = Object.values(clusterCounts);
      const ent = shannonEntropy(counts);
      entropyRecords.push({
        model: modelName,
        layer: layerIdx,
        depth: Math.round(depth * 10000) / 10000,
        entropy: Math.round(ent * 10000) / 10000,
      });
    }

    // --- by_model: per (model, cluster) ---
    const totalHeads = modelHeads.length;
    const modelClusterCounts = {};
    for (const h of modelHeads) {
      modelClusterCounts[h.cluster] = (modelClusterCounts[h.cluster] || 0) + 1;
    }

    for (const [clusterStr, count] of Object.entries(modelClusterCounts)) {
      const frac = count / totalHeads;
      byModelRecords.push({
        model: modelName,
        cluster: parseInt(clusterStr),
        count: count,
        frac: Math.round(frac * 10000) / 10000,
      });
    }
  }

  return {
    by_layer: byLayerRecords,
    by_model: byModelRecords,
    entropy_by_layer: entropyRecords,
  };
}

/**
 * Update currentRecords from a cut height and rebuild charts.
 * @param {number} cutHeight
 */
function updateFromCutHeight(cutHeight) {
  usingCutHeight = true;
  const assignments = computeAssignmentsByCutHeight(cutHeight);
  currentRecords = computeTrendRecords(assignments);
  rebuildAll();
}

/**
 * Update currentRecords from a precomputed K value and rebuild charts.
 * @param {number} k
 */
function updateFromK(k) {
  usingCutHeight = false;
  currentK = k;
  const key = `k${k}`;
  currentRecords = {
    by_layer: DATA.by_layer[key] || [],
    by_model: DATA.by_model[key] || [],
    entropy_by_layer: DATA.entropy_by_layer[key] || [],
  };
  rebuildAll();
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
  const records = currentRecords.by_layer;
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
  const records = currentRecords.by_model;

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
  const records = currentRecords.entropy_by_layer;

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

  // Set default K (used when switching to K mode)
  if (CONFIG.defaultK && DATA.k_values.includes(Number(CONFIG.defaultK))) {
    currentK = Number(CONFIG.defaultK);
  } else {
    currentK = DATA.k_values.includes(10) ? 10 : DATA.k_values[0];
  }
  kSelect.value = currentK;

  // Load clustering data for cut-height mode
  const clusteringAvailable = await loadClusteringData();

  // Cut height slider (bidirectional range + number input)
  const cutSlider = document.getElementById("cut-height");
  const cutInput = document.getElementById("cut-height-input");

  if (clusteringAvailable) {
    cutSlider.addEventListener("input", (e) => {
      const h = parseFloat(e.target.value);
      cutInput.value = h.toFixed(2);
      updateFromCutHeight(h);
    });
    cutInput.addEventListener("change", (e) => {
      const h = Math.max(0, Math.min(10, parseFloat(e.target.value) || 0));
      cutInput.value = h.toFixed(2);
      cutSlider.value = h;
      updateFromCutHeight(h);
    });
  } else {
    cutSlider.disabled = true;
    cutInput.disabled = true;
  }

  // K selector: switch to precomputed mode
  kSelect.addEventListener("change", () => {
    updateFromK(Number(kSelect.value));
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

  // Initial draw: use cut height = 5 if clustering available, else K
  if (clusteringAvailable) {
    updateFromCutHeight(5);
  } else {
    updateFromK(currentK);
  }
}

init();
