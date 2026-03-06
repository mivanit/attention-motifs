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

// Which families/models are enabled (true = visible)
/** @type {Object<string, boolean>} */ let familyEnabled = {};
/** @type {Object<string, boolean>} */ let modelEnabled = {};

// Cut-height clustering state
/** @type {number[][]|null} */ let linkageData = null;
/** @type {string[]|null} */ let clsValues = null;

// Current records used by chart builders (either precomputed or dynamic)
let currentRecords = { by_layer: [], by_model: [], entropy_by_layer: [] };

// Cluster labels state
/** @type {Object} */ let allClusterLabels = {};
/** @type {Object<number, {name: string, desc: string|null}>} */ let resolvedLabels =
  {};

// ── Colors ──────────────────────────────────────────────────────
// clusterColor() and clusterColorAlpha() are provided by cluster_utils.js

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
 * Get a legend/tooltip label for a cluster, using name if available.
 * @param {number} cid
 * @returns {string}
 */
function clusterLegendLabel(cid) {
  return clusterDisplayName(cid, resolvedLabels);
}

/**
 * Get the desc (long description) for a cluster, if available.
 * @param {number} cid
 * @returns {string|null}
 */
function clusterDesc(cid) {
  const label = resolvedLabels[cid];
  return label ? label.desc : null;
}

/**
 * Update currentRecords from a cut height and rebuild charts.
 * @param {number} cutHeight
 */
function updateFromCutHeight(cutHeight) {
  gridState.currentCutHeight = cutHeight;
  ClusteringConfig.setCutHeight(cutHeight);
  const assignments = computeAssignmentsByCutHeight(cutHeight);
  resolvedLabels = resolveClusterLabels(
    allClusterLabels,
    cutHeight,
    assignments,
  );
  currentRecords = computeTrendRecords(assignments);
  rebuildAll();
}

/**
 * Update currentRecords from a precomputed K value and rebuild charts.
 * @param {number} k
 */
function updateFromK(k) {
  currentK = k;
  const key = `k${k}`;
  resolvedLabels = {}; // labels are keyed by cut height, not K
  currentRecords = {
    by_layer: DATA.by_layer[key] || [],
    by_model: DATA.by_model[key] || [],
    entropy_by_layer: DATA.entropy_by_layer[key] || [],
  };
  rebuildAll();
}

// ── Toggle helpers ──────────────────────────────────────────────
// buildToggleButtons() is provided by cluster_utils.js

/**
 * Check whether a model passes the current family filter.
 * @param {string} modelName
 * @returns {boolean}
 */
function isModelEnabled(modelName) {
  return modelEnabled[modelName] === true;
}

// ── Chart 1: Layer depth x cluster fraction ─────────────────────

function buildLayerChart() {
  const records = currentRecords.by_layer;
  const showDist = document.getElementById("layer-distribution").checked;
  const showLines = document.getElementById("show-lines").checked;

  // Disable "Lines" checkbox when distribution is active
  document.getElementById("show-lines").disabled = showDist;

  // Filter by enabled families
  const filtered = records.filter((r) => isModelEnabled(r.model));

  if (showDist) {
    buildLayerChartDistribution(filtered);
  } else {
    buildLayerChartScatter(filtered, showLines);
  }
}

/**
 * Scatter/line mode for the layer chart (original behavior).
 * @param {Array} filtered - family-filtered by_layer records
 * @param {boolean} showLines
 */
function buildLayerChartScatter(filtered, showLines) {
  // Group by cluster -> array of {x: depth, y: frac, model}
  /** @type {Object<number, Array<{x:number, y:number, model:string}>>} */
  const byCluster = {};
  for (const r of filtered) {
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
        label: isFirst ? clusterLegendLabel(cid) : "",
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
      interaction: { mode: "nearest", intersect: true },
      scales: {
        x: {
          title: {
            display: true,
            text: "Normalized Layer Depth",
            color: "#555",
          },
          min: -0.02,
          max: 1.02,
          ticks: { color: "#888" },
          grid: { color: "rgba(0,0,0,0.08)" },
        },
        y: {
          title: { display: true, text: "Fraction of Heads", color: "#555" },
          min: 0,
          ticks: { color: "#888" },
          grid: { color: "rgba(0,0,0,0.08)" },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "right",
          labels: {
            color: "#555",
            font: { size: 11 },
            filter: (item) => item.text !== "",
          },
          onClick: (_e, legendItem, legend) => {
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
              const line = `${ds._model} | ${clusterLegendLabel(ds._clusterId)} | depth=${ctx.parsed.x.toFixed(2)} frac=${ctx.parsed.y.toFixed(3)}`;
              const desc = clusterDesc(ds._clusterId);
              return desc ? [line, desc] : line;
            },
          },
        },
      },
    },
  });
}

/**
 * Distribution mode for the layer chart.
 * Per cluster: min/max/mean band across models at each depth.
 * @param {Array} filtered - family-filtered by_layer records
 */
function buildLayerChartDistribution(filtered) {
  // Group by cluster
  /** @type {Object<number, Array<{depth:number, frac:number}>>} */
  const byCluster = {};
  for (const r of filtered) {
    if (!byCluster[r.cluster]) byCluster[r.cluster] = [];
    byCluster[r.cluster].push({ depth: r.depth, frac: r.frac });
  }

  const clusterIds = Object.keys(byCluster)
    .map(Number)
    .sort((a, b) => a - b);

  const datasets = [];

  for (const cid of clusterIds) {
    const points = byCluster[cid];
    // Group by depth -> collect frac values
    /** @type {Object<string, number[]>} */
    const byDepth = {};
    for (const p of points) {
      const dk = p.depth.toFixed(4);
      if (!byDepth[dk]) byDepth[dk] = [];
      byDepth[dk].push(p.frac);
    }

    const depths = Object.keys(byDepth)
      .map(Number)
      .sort((a, b) => a - b);
    const maxPts = depths.map((d) => {
      const vals = byDepth[d.toFixed(4)];
      return { x: d, y: Math.max(...vals) };
    });
    const minPts = depths.map((d) => {
      const vals = byDepth[d.toFixed(4)];
      return { x: d, y: Math.min(...vals) };
    });
    const meanPts = depths.map((d) => {
      const vals = byDepth[d.toFixed(4)];
      return { x: d, y: vals.reduce((a, b) => a + b, 0) / vals.length };
    });

    const color = clusterColor(cid);
    const fillColor = clusterColorAlpha(cid, 0.15);

    // Max line (fill down to next dataset = min line)
    datasets.push({
      label: clusterLegendLabel(cid),
      data: maxPts,
      borderColor: color,
      borderWidth: 1,
      borderDash: [4, 2],
      pointRadius: 0,
      showLine: true,
      tension: 0.2,
      fill: "+1",
      backgroundColor: fillColor,
      _clusterId: cid,
      _role: "max",
    });
    // Min line
    datasets.push({
      label: "",
      data: minPts,
      borderColor: color,
      borderWidth: 1,
      borderDash: [4, 2],
      pointRadius: 0,
      showLine: true,
      tension: 0.2,
      fill: false,
      _clusterId: cid,
      _role: "min",
    });
    // Mean line (solid, thicker)
    datasets.push({
      label: "",
      data: meanPts,
      borderColor: color,
      borderWidth: 2.5,
      pointRadius: 1,
      pointHoverRadius: 4,
      showLine: true,
      tension: 0.2,
      fill: false,
      _clusterId: cid,
      _role: "mean",
    });
  }

  const ctx = document.getElementById("layer-chart").getContext("2d");
  if (layerChart) layerChart.destroy();

  layerChart = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: {
          title: {
            display: true,
            text: "Normalized Layer Depth",
            color: "#555",
          },
          min: -0.02,
          max: 1.02,
          ticks: { color: "#888" },
          grid: { color: "rgba(0,0,0,0.08)" },
        },
        y: {
          title: { display: true, text: "Fraction of Heads", color: "#555" },
          min: 0,
          ticks: { color: "#888" },
          grid: { color: "rgba(0,0,0,0.08)" },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "right",
          labels: {
            color: "#555",
            font: { size: 11 },
            filter: (item) => item.text !== "",
          },
          onClick: (_e, legendItem, legend) => {
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
              const line = `${clusterLegendLabel(ds._clusterId)} (${ds._role}) | depth=${ctx.parsed.x.toFixed(2)} frac=${ctx.parsed.y.toFixed(3)}`;
              const desc = clusterDesc(ds._clusterId);
              return desc ? [line, desc] : line;
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
  const showDist = document.getElementById("size-distribution").checked;

  // Filter by enabled families
  const filtered = records.filter((r) => isModelEnabled(r.model));

  if (showDist) {
    buildSizeChartDistribution(filtered);
  } else {
    buildSizeChartScatter(filtered);
  }
}

/** @type {(val: number) => string} */
function formatParams(val) {
  if (val >= 1e9) return (val / 1e9).toFixed(1) + "B";
  if (val >= 1e6) return (val / 1e6).toFixed(0) + "M";
  if (val >= 1e3) return (val / 1e3).toFixed(0) + "K";
  return String(val);
}

/**
 * Scatter mode for the size chart (original behavior).
 * @param {Array} filtered - family-filtered by_model records
 */
function buildSizeChartScatter(filtered) {
  // Group by cluster
  /** @type {Object<number, Array<{x:number, y:number, model:string}>>} */
  const byCluster = {};
  for (const r of filtered) {
    const meta = DATA.models[r.model];
    if (!meta) continue;
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
      label: clusterLegendLabel(cid),
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
      interaction: { mode: "nearest", intersect: true },
      scales: {
        x: {
          type: "logarithmic",
          title: { display: true, text: "Parameters", color: "#555" },
          ticks: { color: "#888", callback: formatParams },
          grid: { color: "rgba(0,0,0,0.08)" },
        },
        y: {
          title: { display: true, text: "Fraction of Heads", color: "#555" },
          min: 0,
          ticks: { color: "#888" },
          grid: { color: "rgba(0,0,0,0.08)" },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "right",
          labels: { color: "#555", font: { size: 11 } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const pt = ctx.raw;
              const ds = ctx.dataset;
              const line = `${pt.model} | ${clusterLegendLabel(ds._clusterId)} | frac=${pt.y.toFixed(3)}`;
              const desc = clusterDesc(ds._clusterId);
              return desc ? [line, desc] : line;
            },
          },
        },
      },
    },
  });
}

/**
 * Distribution mode for the size chart.
 * Per cluster: min/max/mean band across models at each model size.
 * @param {Array} filtered - family-filtered by_model records
 */
function buildSizeChartDistribution(filtered) {
  // Group by cluster first
  /** @type {Object<number, Array<{size:number, frac:number}>>} */
  const byCluster = {};
  for (const r of filtered) {
    const meta = DATA.models[r.model];
    if (!meta) continue;
    if (!byCluster[r.cluster]) byCluster[r.cluster] = [];
    byCluster[r.cluster].push({ size: meta.n_params, frac: r.frac });
  }

  const clusterIds = Object.keys(byCluster)
    .map(Number)
    .sort((a, b) => a - b);

  const datasets = [];

  for (const cid of clusterIds) {
    const points = byCluster[cid];
    // Group by model size -> collect frac values
    /** @type {Object<string, number[]>} */
    const bySize = {};
    for (const p of points) {
      const sk = String(p.size);
      if (!bySize[sk]) bySize[sk] = [];
      bySize[sk].push(p.frac);
    }

    const sizes = Object.keys(bySize)
      .map(Number)
      .sort((a, b) => a - b);
    const maxPts = sizes.map((s) => ({
      x: s,
      y: Math.max(...bySize[String(s)]),
    }));
    const minPts = sizes.map((s) => ({
      x: s,
      y: Math.min(...bySize[String(s)]),
    }));
    const meanPts = sizes.map((s) => {
      const vals = bySize[String(s)];
      return { x: s, y: vals.reduce((a, b) => a + b, 0) / vals.length };
    });

    const color = clusterColor(cid);
    const fillColor = clusterColorAlpha(cid, 0.15);

    // Max line (fill down to next dataset = min line)
    datasets.push({
      label: clusterLegendLabel(cid),
      data: maxPts,
      borderColor: color,
      borderWidth: 1,
      borderDash: [4, 2],
      pointRadius: 0,
      showLine: true,
      tension: 0.2,
      fill: "+1",
      backgroundColor: fillColor,
      _clusterId: cid,
      _role: "max",
    });
    // Min line
    datasets.push({
      label: "",
      data: minPts,
      borderColor: color,
      borderWidth: 1,
      borderDash: [4, 2],
      pointRadius: 0,
      showLine: true,
      tension: 0.2,
      fill: false,
      _clusterId: cid,
      _role: "min",
    });
    // Mean line (solid, thicker)
    datasets.push({
      label: "",
      data: meanPts,
      borderColor: color,
      borderWidth: 2.5,
      pointRadius: 1,
      pointHoverRadius: 4,
      showLine: true,
      tension: 0.2,
      fill: false,
      _clusterId: cid,
      _role: "mean",
    });
  }

  const ctx = document.getElementById("size-chart").getContext("2d");
  if (sizeChart) sizeChart.destroy();

  sizeChart = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: {
          type: "logarithmic",
          title: { display: true, text: "Parameters", color: "#555" },
          ticks: { color: "#888", callback: formatParams },
          grid: { color: "rgba(0,0,0,0.08)" },
        },
        y: {
          title: { display: true, text: "Fraction of Heads", color: "#555" },
          min: 0,
          ticks: { color: "#888" },
          grid: { color: "rgba(0,0,0,0.08)" },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "right",
          labels: {
            color: "#555",
            font: { size: 11 },
            filter: (item) => item.text !== "",
          },
          onClick: (_e, legendItem, legend) => {
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
              const line = `${clusterLegendLabel(ds._clusterId)} (${ds._role}) | ${formatParams(ctx.parsed.x)} | frac=${ctx.parsed.y.toFixed(3)}`;
              const desc = clusterDesc(ds._clusterId);
              return desc ? [line, desc] : line;
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

  // Group by model (filtered by family)
  /** @type {Object<string, Array<{x:number, y:number}>>} */
  const byModel = {};
  for (const r of records) {
    if (!isModelEnabled(r.model)) continue;
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
            color: "#555",
          },
          min: -0.02,
          max: 1.02,
          ticks: { color: "#888" },
          grid: { color: "rgba(0,0,0,0.08)" },
        },
        y: {
          title: {
            display: true,
            text: "Shannon Entropy (bits)",
            color: "#555",
          },
          min: 0,
          ticks: { color: "#888" },
          grid: { color: "rgba(0,0,0,0.08)" },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "right",
          labels: { color: "#555", font: { size: 11 } },
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

// ── Cluster chips ───────────────────────────────────────────────

/**
 * Render cluster chips above the charts.
 * Each chip links to the clustering page with the current cut height
 * and highlights that cluster.
 */
function renderClusterChips() {
  const container = document.getElementById("cluster-chips");
  if (!container) return;

  // Compute cluster sizes from by_model records (all models, not filtered)
  /** @type {Object<number, number>} */
  const clusterSizes = {};
  for (const r of currentRecords.by_model) {
    clusterSizes[r.cluster] = (clusterSizes[r.cluster] || 0) + r.count;
  }

  const sorted = Object.entries(clusterSizes)
    .map(([cid, size]) => ({ clusterId: parseInt(cid), size }))
    .filter((c) => c.clusterId !== -1)
    .sort((a, b) => b.size - a.size);

  const top = sorted.slice(0, 20);
  const hasMore = sorted.length > 20;

  container.innerHTML =
    `<span class="top-clusters-label">Clusters (${sorted.length}):</span>` +
    top
      .map(({ clusterId, size }) => {
        const color = clusterColor(clusterId);
        const label = resolvedLabels[clusterId];
        const name = (label && label.name) || "";
        const desc = (label && label.desc) || "";
        const labelHtml = name
          ? `<span class="cluster-chip-label" title="${desc.replace(/"/g, "&quot;")}">${name}</span>`
          : "";
        return `
        <div class="cluster-chip" data-cluster-id="${clusterId}">
          <span class="cluster-chip-color" style="background-color: ${color}"></span>
          <span class="cluster-chip-size">${size}</span>
          ${labelHtml}
        </div>`;
      })
      .join("") +
    (hasMore
      ? `<span class="top-clusters-ellipsis">... +${sorted.length - 20} more</span>`
      : "");

  container.classList.add("visible");

  // Click → navigate to clustering page with shared config
  container.querySelectorAll(".cluster-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const clusterId = parseInt(chip.dataset.clusterId);
      if (gridState.currentCutHeight != null) {
        ClusteringConfig.setCutHeight(gridState.currentCutHeight);
      }
      ClusteringConfig.setHighlightCluster(clusterId);
      window.location.href = "../clustering/index.html";
    });
  });
}

// Track current cut height for chip navigation
/** @type {{currentCutHeight: number|null}} */
const gridState = { currentCutHeight: null };

// ── Family / model toggles ──────────────────────────────────────

/**
 * Build nested family→model toggle buttons.
 * Click family = expand/collapse model list.
 * Shift+click family = toggle all models in family.
 * Click model = toggle individual model.
 * @param {string} containerId
 * @param {Object<string, string[]>} familyModels - family name -> sorted model names
 */
function buildFamilyModelToggles(containerId, familyModels) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  for (const [family, models] of Object.entries(familyModels)) {
    const group = document.createElement("div");
    group.className = "family-group";

    // Family header button
    const familyBtn = document.createElement("span");
    familyBtn.className = "family-toggle";
    familyBtn.textContent = family;
    updateFamilyBtnState(familyBtn, family, models);

    familyBtn.addEventListener("click", (e) => {
      if (e.shiftKey) {
        // Shift+click: toggle all models in family
        const allEnabled = models.every((m) => modelEnabled[m]);
        for (const m of models) {
          modelEnabled[m] = !allEnabled;
        }
        familyEnabled[family] = !allEnabled;
        updateFamilyBtnState(familyBtn, family, models);
        updateModelBtnStates(modelRow, models);
        rebuildAll();
      } else {
        // Click: expand/collapse
        modelRow.classList.toggle("expanded");
      }
    });

    // Model row (initially collapsed)
    const modelRow = document.createElement("div");
    modelRow.className = "model-toggles";

    for (const m of models) {
      const btn = document.createElement("span");
      btn.className = "model-toggle";
      btn.textContent = m;
      if (!modelEnabled[m]) btn.classList.add("disabled");

      btn.addEventListener("click", () => {
        modelEnabled[m] = !modelEnabled[m];
        btn.classList.toggle("disabled", !modelEnabled[m]);
        // Update family state
        familyEnabled[family] = models.some((mm) => modelEnabled[mm]);
        updateFamilyBtnState(familyBtn, family, models);
        rebuildAll();
      });
      modelRow.appendChild(btn);
    }

    group.appendChild(familyBtn);
    group.appendChild(modelRow);
    container.appendChild(group);
  }
}

/**
 * Update family button visual state based on model enabled states.
 * @param {HTMLElement} btn
 * @param {string} family
 * @param {string[]} models
 */
function updateFamilyBtnState(btn, _family, models) {
  const enabledCount = models.filter((m) => modelEnabled[m]).length;
  btn.classList.remove("disabled", "partial");
  if (enabledCount === 0) {
    btn.classList.add("disabled");
  } else if (enabledCount < models.length) {
    btn.classList.add("partial");
  }
}

/**
 * Update model button states in a model row.
 * @param {HTMLElement} modelRow
 * @param {string[]} models
 */
function updateModelBtnStates(modelRow, models) {
  const btns = modelRow.querySelectorAll(".model-toggle");
  btns.forEach((btn, i) => {
    btn.classList.toggle("disabled", !modelEnabled[models[i]]);
  });
}

// ── Rebuild all ─────────────────────────────────────────────────

function rebuildAll() {
  buildLayerChart();
  buildSizeChart();
  buildEntropyChart();
  renderClusterChips();
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

  // Initialize family and model toggle states
  const families = [
    ...new Set(Object.values(DATA.models).map((m) => m.family)),
  ].sort();
  const defaultFamilies = ["pythia", "gpt2"];

  /** @type {Object<string, string[]>} */
  const familyModels = {};
  for (const [modelName, meta] of Object.entries(DATA.models)) {
    if (!familyModels[meta.family]) familyModels[meta.family] = [];
    familyModels[meta.family].push(modelName);
  }
  for (const f of families) {
    familyModels[f].sort();
    const enabled = defaultFamilies.includes(f);
    familyEnabled[f] = enabled;
    for (const m of familyModels[f]) {
      modelEnabled[m] = enabled;
    }
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

  // Load cluster labels
  const labelsUrl = "../../features/clustering/cluster_labels.json";
  allClusterLabels = await loadClusterLabels(labelsUrl);

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

  // Distribution toggles
  document
    .getElementById("layer-distribution")
    .addEventListener("change", () => {
      buildLayerChart();
    });
  document
    .getElementById("size-distribution")
    .addEventListener("change", () => {
      buildSizeChart();
    });

  // Build nested family→model toggles
  buildFamilyModelToggles("global-family-toggles", familyModels);

  // Show charts, hide loading
  document.getElementById("loading").style.display = "none";
  document.getElementById("charts-container").style.display = "block";

  // Initial draw: use shared config cut height if available
  if (clusteringAvailable) {
    const savedCutHeight = ClusteringConfig.getCutHeight();
    const startCutHeight = savedCutHeight ?? 5;
    cutSlider.value = startCutHeight;
    cutInput.value = startCutHeight.toFixed(2);
    updateFromCutHeight(startCutHeight);
  } else {
    updateFromK(currentK);
  }
}

init();
