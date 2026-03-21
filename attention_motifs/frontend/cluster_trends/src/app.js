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
 * All charts render as SVG via D3.js for PDF-quality export.
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

// ── Chart margins ───────────────────────────────────────────────
const CHART_MARGIN = { top: 14, right: 160, bottom: 46, left: 56 };

// ── State ───────────────────────────────────────────────────────
let DATA = null;
let currentK = null;

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

// Cluster visibility state for legend toggling
/** @type {Object<number, boolean>} */ let clusterVisible = {};

// Multi-method state
/** @type {string} */ let currentMethod = "hierarchical";

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
// Uses shared computeClustersByHeight() from cluster_engine.js

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
  const assignments = computeClustersByHeight(
    linkageData,
    clsValues,
    cutHeight,
  );
  resolvedLabels = resolveClusterLabels(
    allClusterLabels,
    cutHeight,
    assignments,
  );
  currentRecords = computeTrendRecords(assignments);
  clusterVisible = {}; // Reset visibility on new clustering
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
  clusterVisible = {}; // Reset visibility on new clustering
  rebuildAll();
}

/**
 * Update currentRecords from a flat clustering method's precomputed parameter.
 * Reads from DATA.hdbscan or DATA.leiden sub-objects.
 * @param {string} method - "hdbscan" or "leiden"
 * @param {string} paramKey - The parameter key (e.g. "5", "0.5")
 */
function updateFromFlatParam(method, paramKey) {
  const methodData = DATA[method];
  if (!methodData) return;

  // Build the trend key: for HDBSCAN "hdbscan.mcs{v}", for Leiden "leiden.r{v}"
  let trendKey;
  if (method === "hdbscan") {
    trendKey = `hdbscan.mcs${paramKey}`;
  } else if (method === "leiden") {
    trendKey = `leiden.r${paramKey}`;
  } else {
    return;
  }

  resolvedLabels = {};
  currentRecords = {
    by_layer: methodData.by_layer[trendKey] || [],
    by_model: methodData.by_model[trendKey] || [],
    entropy_by_layer: methodData.entropy_by_layer[trendKey] || [],
  };
  clusterVisible = {};
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
 * Scatter/line mode for the layer chart.
 * @param {Array} filtered - family-filtered by_layer records
 * @param {boolean} showLines
 */
function buildLayerChartScatter(filtered, showLines) {
  const containerId = "layer-chart-container";
  const { svg, g, innerWidth, innerHeight } = createChartSVG(
    containerId,
    CHART_MARGIN,
  );
  const tooltip = createTooltip(containerId);

  // Group by cluster
  /** @type {Object<number, Array<{x:number, y:number, model:string}>>} */
  const byCluster = {};
  for (const r of filtered) {
    if (!byCluster[r.cluster]) byCluster[r.cluster] = [];
    byCluster[r.cluster].push({ x: r.depth, y: r.frac, model: r.model });
  }

  const clusterIds = Object.keys(byCluster)
    .map(Number)
    .sort((a, b) => a - b);

  // Initialize visibility
  for (const cid of clusterIds) {
    if (clusterVisible[cid] === undefined) clusterVisible[cid] = true;
  }

  const yMax = d3.max(filtered, (r) => r.frac) || 1;

  const xScale = d3.scaleLinear().domain([-0.02, 1.02]).range([0, innerWidth]);
  const yScale = d3
    .scaleLinear()
    .domain([0, yMax * 1.05])
    .range([innerHeight, 0]);

  createXAxis(g, xScale, {
    height: innerHeight,
    label: "Normalized Layer Depth",
    gridHeight: innerHeight,
  });
  createYAxis(g, yScale, {
    label: "Fraction of Heads",
    gridWidth: innerWidth,
  });

  // Plot data per cluster
  const models = Object.keys(DATA.models).sort();
  for (const cid of clusterIds) {
    const points = byCluster[cid] || [];
    const color = clusterColor(cid);
    const clusterG = g
      .append("g")
      .attr("class", `cluster-group cluster-${cid}`)
      .style("display", clusterVisible[cid] ? null : "none");

    // Group by model for lines
    /** @type {Object<string, Array<{x:number, y:number}>>} */
    const byModel = {};
    for (const p of points) {
      if (!byModel[p.model]) byModel[p.model] = [];
      byModel[p.model].push({ x: p.x, y: p.y });
    }

    for (const m of models) {
      const pts = byModel[m];
      if (!pts) continue;
      pts.sort((a, b) => a.x - b.x);

      // Lines
      if (showLines && pts.length > 1) {
        const line = d3
          .line()
          .x((d) => xScale(d.x))
          .y((d) => yScale(d.y));
        clusterG
          .append("path")
          .datum(pts)
          .attr("fill", "none")
          .attr("stroke", color)
          .attr("stroke-width", 1.5)
          .attr("d", line);
      }

      // Points
      clusterG
        .selectAll(null)
        .data(pts.map((p) => ({ ...p, model: m, cid })))
        .enter()
        .append("circle")
        .attr("cx", (d) => xScale(d.x))
        .attr("cy", (d) => yScale(d.y))
        .attr("r", 3)
        .attr("fill", color)
        .on("mouseover", (event, d) => {
          d3.select(event.target).attr("r", 5);
          const label = clusterLegendLabel(d.cid);
          let html = `${d.model} | ${label} | depth=${d.x.toFixed(2)} frac=${d.y.toFixed(3)}`;
          const desc = clusterDesc(d.cid);
          if (desc) html += `<br>${desc}`;
          tooltip.show(event, html);
        })
        .on("mouseout", (event) => {
          d3.select(event.target).attr("r", 3);
          tooltip.hide();
        });
    }
  }

  // Legend
  createLegend(svg, {
    items: clusterIds.map((cid) => ({
      label: clusterLegendLabel(cid),
      color: clusterColor(cid),
      _cid: cid,
    })),
    x: CHART_MARGIN.left + innerWidth + 12,
    y: CHART_MARGIN.top,
    onClick: (_label, _idx, item) => {
      const cid = item._cid;
      clusterVisible[cid] = !clusterVisible[cid];
      g.select(`.cluster-${cid}`).style(
        "display",
        clusterVisible[cid] ? null : "none",
      );
    },
  });

  addExportButton(containerId, "layer-depth-cluster-fraction");
}

/**
 * Distribution mode for the layer chart.
 * Per cluster: min/max/mean band across models at each depth.
 * @param {Array} filtered - family-filtered by_layer records
 */
function buildLayerChartDistribution(filtered) {
  const containerId = "layer-chart-container";
  const { svg, g, innerWidth, innerHeight } = createChartSVG(
    containerId,
    CHART_MARGIN,
  );
  const tooltip = createTooltip(containerId);

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

  for (const cid of clusterIds) {
    if (clusterVisible[cid] === undefined) clusterVisible[cid] = true;
  }

  const yMax =
    d3.max(
      clusterIds.flatMap((cid) => (byCluster[cid] || []).map((d) => d.frac)),
    ) || 1;

  const xScale = d3.scaleLinear().domain([-0.02, 1.02]).range([0, innerWidth]);
  const yScale = d3
    .scaleLinear()
    .domain([0, yMax * 1.05])
    .range([innerHeight, 0]);

  createXAxis(g, xScale, {
    height: innerHeight,
    label: "Normalized Layer Depth",
    gridHeight: innerHeight,
  });
  createYAxis(g, yScale, {
    label: "Fraction of Heads",
    gridWidth: innerWidth,
  });

  const curve = d3.curveCatmullRom.alpha(0.5);

  for (const cid of clusterIds) {
    const points = byCluster[cid];
    const color = clusterColor(cid);
    const fillColor = clusterColorAlpha(cid, 0.15);

    // Group by depth
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
    const stats = depths.map((d) => {
      const vals = byDepth[d.toFixed(4)];
      return {
        depth: d,
        min: Math.min(...vals),
        max: Math.max(...vals),
        mean: vals.reduce((a, b) => a + b, 0) / vals.length,
      };
    });

    const clusterG = g
      .append("g")
      .attr("class", `cluster-group cluster-${cid}`)
      .style("display", clusterVisible[cid] ? null : "none");

    // Filled area between min and max
    const area = d3
      .area()
      .x((d) => xScale(d.depth))
      .y0((d) => yScale(d.min))
      .y1((d) => yScale(d.max))
      .curve(curve);

    clusterG
      .append("path")
      .datum(stats)
      .attr("fill", fillColor)
      .attr("stroke", "none")
      .attr("d", area);

    // Max line (dashed)
    const maxLine = d3
      .line()
      .x((d) => xScale(d.depth))
      .y((d) => yScale(d.max))
      .curve(curve);
    clusterG
      .append("path")
      .datum(stats)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,2")
      .attr("d", maxLine);

    // Min line (dashed)
    const minLine = d3
      .line()
      .x((d) => xScale(d.depth))
      .y((d) => yScale(d.min))
      .curve(curve);
    clusterG
      .append("path")
      .datum(stats)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,2")
      .attr("d", minLine);

    // Mean line (solid, thicker)
    const meanLine = d3
      .line()
      .x((d) => xScale(d.depth))
      .y((d) => yScale(d.mean))
      .curve(curve);
    clusterG
      .append("path")
      .datum(stats)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 2.5)
      .attr("d", meanLine);

    // Mean line hover points
    clusterG
      .selectAll(null)
      .data(stats.map((s) => ({ ...s, cid })))
      .enter()
      .append("circle")
      .attr("cx", (d) => xScale(d.depth))
      .attr("cy", (d) => yScale(d.mean))
      .attr("r", 1)
      .attr("fill", color)
      .on("mouseover", (event, d) => {
        d3.select(event.target).attr("r", 4);
        const label = clusterLegendLabel(d.cid);
        let html = `${label} (mean) | depth=${d.depth.toFixed(2)} frac=${d.mean.toFixed(3)}`;
        const desc = clusterDesc(d.cid);
        if (desc) html += `<br>${desc}`;
        tooltip.show(event, html);
      })
      .on("mouseout", (event) => {
        d3.select(event.target).attr("r", 1);
        tooltip.hide();
      });
  }

  // Legend
  createLegend(svg, {
    items: clusterIds.map((cid) => ({
      label: clusterLegendLabel(cid),
      color: clusterColor(cid),
      _cid: cid,
    })),
    x: CHART_MARGIN.left + innerWidth + 12,
    y: CHART_MARGIN.top,
    onClick: (_label, _idx, item) => {
      const cid = item._cid;
      clusterVisible[cid] = !clusterVisible[cid];
      g.select(`.cluster-${cid}`).style(
        "display",
        clusterVisible[cid] ? null : "none",
      );
    },
  });

  addExportButton(containerId, "layer-depth-cluster-distribution");
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
 * Scatter mode for the size chart.
 * @param {Array} filtered - family-filtered by_model records
 */
function buildSizeChartScatter(filtered) {
  const containerId = "size-chart-container";
  const { svg, g, innerWidth, innerHeight } = createChartSVG(
    containerId,
    CHART_MARGIN,
  );
  const tooltip = createTooltip(containerId);

  // Group by cluster
  /** @type {Object<number, Array<{x:number, y:number, model:string}>>} */
  const byCluster = {};
  for (const r of filtered) {
    const meta = DATA.models[r.model];
    if (!meta || !meta.n_params) continue;
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

  for (const cid of clusterIds) {
    if (clusterVisible[cid] === undefined) clusterVisible[cid] = true;
  }

  const allPoints = clusterIds.flatMap((cid) => byCluster[cid] || []);

  if (allPoints.length === 0) {
    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight / 2)
      .attr("text-anchor", "middle")
      .attr("fill", "#999")
      .text("No data for current filters");
    return;
  }

  const xExtent = d3.extent(allPoints, (d) => d.x);
  const yMax = d3.max(allPoints, (d) => d.y) || 1;

  const xScale = d3
    .scaleLog()
    .domain([xExtent[0] * 0.8, xExtent[1] * 1.2])
    .range([0, innerWidth]);
  const yScale = d3
    .scaleLinear()
    .domain([0, yMax * 1.05])
    .range([innerHeight, 0]);

  createXAxis(g, xScale, {
    height: innerHeight,
    label: "Parameters",
    tickFormat: formatParams,
    gridHeight: innerHeight,
  });
  createYAxis(g, yScale, {
    label: "Fraction of Heads",
    gridWidth: innerWidth,
  });

  for (const cid of clusterIds) {
    const pts = (byCluster[cid] || []).sort((a, b) => a.x - b.x);
    const color = clusterColor(cid);

    const clusterG = g
      .append("g")
      .attr("class", `cluster-group cluster-${cid}`)
      .style("display", clusterVisible[cid] ? null : "none");

    clusterG
      .selectAll(null)
      .data(pts.map((p) => ({ ...p, cid })))
      .enter()
      .append("circle")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 4)
      .attr("fill", color)
      .on("mouseover", (event, d) => {
        d3.select(event.target).attr("r", 6);
        const label = clusterLegendLabel(d.cid);
        let html = `${d.model} | ${label} | frac=${d.y.toFixed(3)}`;
        const desc = clusterDesc(d.cid);
        if (desc) html += `<br>${desc}`;
        tooltip.show(event, html);
      })
      .on("mouseout", (event) => {
        d3.select(event.target).attr("r", 4);
        tooltip.hide();
      });
  }

  // Legend
  createLegend(svg, {
    items: clusterIds.map((cid) => ({
      label: clusterLegendLabel(cid),
      color: clusterColor(cid),
      _cid: cid,
    })),
    x: CHART_MARGIN.left + innerWidth + 12,
    y: CHART_MARGIN.top,
    onClick: (_label, _idx, item) => {
      const cid = item._cid;
      clusterVisible[cid] = !clusterVisible[cid];
      g.select(`.cluster-${cid}`).style(
        "display",
        clusterVisible[cid] ? null : "none",
      );
    },
  });

  addExportButton(containerId, "model-size-cluster-fraction");
}

/**
 * Distribution mode for the size chart.
 * Per cluster: min/max/mean band across models at each model size.
 * @param {Array} filtered - family-filtered by_model records
 */
function buildSizeChartDistribution(filtered) {
  const containerId = "size-chart-container";
  const { svg, g, innerWidth, innerHeight } = createChartSVG(
    containerId,
    CHART_MARGIN,
  );
  const tooltip = createTooltip(containerId);

  // Group by cluster first
  /** @type {Object<number, Array<{size:number, frac:number}>>} */
  const byCluster = {};
  for (const r of filtered) {
    const meta = DATA.models[r.model];
    if (!meta || !meta.n_params) continue;
    if (!byCluster[r.cluster]) byCluster[r.cluster] = [];
    byCluster[r.cluster].push({ size: meta.n_params, frac: r.frac });
  }

  const clusterIds = Object.keys(byCluster)
    .map(Number)
    .sort((a, b) => a - b);

  for (const cid of clusterIds) {
    if (clusterVisible[cid] === undefined) clusterVisible[cid] = true;
  }

  const allPoints = clusterIds.flatMap((cid) => byCluster[cid] || []);

  if (allPoints.length === 0) {
    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight / 2)
      .attr("text-anchor", "middle")
      .attr("fill", "#999")
      .text("No data for current filters");
    return;
  }

  const xExtent = d3.extent(allPoints, (d) => d.size);
  const yMax = d3.max(allPoints, (d) => d.frac) || 1;

  const xScale = d3
    .scaleLog()
    .domain([xExtent[0] * 0.8, xExtent[1] * 1.2])
    .range([0, innerWidth]);
  const yScale = d3
    .scaleLinear()
    .domain([0, yMax * 1.05])
    .range([innerHeight, 0]);

  createXAxis(g, xScale, {
    height: innerHeight,
    label: "Parameters",
    tickFormat: formatParams,
    gridHeight: innerHeight,
  });
  createYAxis(g, yScale, {
    label: "Fraction of Heads",
    gridWidth: innerWidth,
  });

  const curve = d3.curveCatmullRom.alpha(0.5);

  for (const cid of clusterIds) {
    const points = byCluster[cid];
    const color = clusterColor(cid);
    const fillColor = clusterColorAlpha(cid, 0.15);

    // Group by model size
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
    const stats = sizes.map((s) => {
      const vals = bySize[String(s)];
      return {
        size: s,
        min: Math.min(...vals),
        max: Math.max(...vals),
        mean: vals.reduce((a, b) => a + b, 0) / vals.length,
      };
    });

    const clusterG = g
      .append("g")
      .attr("class", `cluster-group cluster-${cid}`)
      .style("display", clusterVisible[cid] ? null : "none");

    // Filled area
    const area = d3
      .area()
      .x((d) => xScale(d.size))
      .y0((d) => yScale(d.min))
      .y1((d) => yScale(d.max))
      .curve(curve);
    clusterG
      .append("path")
      .datum(stats)
      .attr("fill", fillColor)
      .attr("stroke", "none")
      .attr("d", area);

    // Max line (dashed)
    clusterG
      .append("path")
      .datum(stats)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,2")
      .attr(
        "d",
        d3
          .line()
          .x((d) => xScale(d.size))
          .y((d) => yScale(d.max))
          .curve(curve),
      );

    // Min line (dashed)
    clusterG
      .append("path")
      .datum(stats)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,2")
      .attr(
        "d",
        d3
          .line()
          .x((d) => xScale(d.size))
          .y((d) => yScale(d.min))
          .curve(curve),
      );

    // Mean line (solid)
    clusterG
      .append("path")
      .datum(stats)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 2.5)
      .attr(
        "d",
        d3
          .line()
          .x((d) => xScale(d.size))
          .y((d) => yScale(d.mean))
          .curve(curve),
      );

    // Hover points on mean
    clusterG
      .selectAll(null)
      .data(stats.map((s) => ({ ...s, cid })))
      .enter()
      .append("circle")
      .attr("cx", (d) => xScale(d.size))
      .attr("cy", (d) => yScale(d.mean))
      .attr("r", 1)
      .attr("fill", color)
      .on("mouseover", (event, d) => {
        d3.select(event.target).attr("r", 4);
        const label = clusterLegendLabel(d.cid);
        let html = `${label} (mean) | ${formatParams(d.size)} | frac=${d.mean.toFixed(3)}`;
        const desc = clusterDesc(d.cid);
        if (desc) html += `<br>${desc}`;
        tooltip.show(event, html);
      })
      .on("mouseout", (event) => {
        d3.select(event.target).attr("r", 1);
        tooltip.hide();
      });
  }

  // Legend
  createLegend(svg, {
    items: clusterIds.map((cid) => ({
      label: clusterLegendLabel(cid),
      color: clusterColor(cid),
      _cid: cid,
    })),
    x: CHART_MARGIN.left + innerWidth + 12,
    y: CHART_MARGIN.top,
    onClick: (_label, _idx, item) => {
      const cid = item._cid;
      clusterVisible[cid] = !clusterVisible[cid];
      g.select(`.cluster-${cid}`).style(
        "display",
        clusterVisible[cid] ? null : "none",
      );
    },
  });

  addExportButton(containerId, "model-size-cluster-distribution");
}

// ── Chart 3: Cluster entropy by layer ───────────────────────────

function buildEntropyChart() {
  const containerId = "entropy-chart-container";
  const records = currentRecords.entropy_by_layer;

  const { svg, g, innerWidth, innerHeight } = createChartSVG(
    containerId,
    CHART_MARGIN,
  );
  const tooltip = createTooltip(containerId);

  // Group by model (filtered by family)
  /** @type {Object<string, Array<{x:number, y:number}>>} */
  const byModel = {};
  for (const r of records) {
    if (!isModelEnabled(r.model)) continue;
    if (!byModel[r.model]) byModel[r.model] = [];
    byModel[r.model].push({ x: r.depth, y: r.entropy });
  }

  const models = Object.keys(byModel).sort();
  const yMax = d3.max(models.flatMap((m) => byModel[m].map((d) => d.y))) || 1;

  const xScale = d3.scaleLinear().domain([-0.02, 1.02]).range([0, innerWidth]);
  const yScale = d3
    .scaleLinear()
    .domain([0, yMax * 1.05])
    .range([innerHeight, 0]);

  createXAxis(g, xScale, {
    height: innerHeight,
    label: "Normalized Layer Depth",
    gridHeight: innerHeight,
  });
  createYAxis(g, yScale, {
    label: "Shannon Entropy (bits)",
    gridWidth: innerWidth,
  });

  const curve = d3.curveCatmullRom.alpha(0.5);

  models.forEach((m, i) => {
    const pts = byModel[m].sort((a, b) => a.x - b.x);
    const color = modelColor(i, models.length);

    const modelG = g.append("g").attr("class", "model-group");

    // Line
    const line = d3
      .line()
      .x((d) => xScale(d.x))
      .y((d) => yScale(d.y))
      .curve(curve);
    modelG
      .append("path")
      .datum(pts)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 2)
      .attr("d", line);

    // Points
    modelG
      .selectAll(null)
      .data(pts.map((p) => ({ ...p, model: m })))
      .enter()
      .append("circle")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 2)
      .attr("fill", color)
      .on("mouseover", (event, d) => {
        d3.select(event.target).attr("r", 4);
        tooltip.show(
          event,
          `${d.model} | depth=${d.x.toFixed(2)} entropy=${d.y.toFixed(3)}`,
        );
      })
      .on("mouseout", (event) => {
        d3.select(event.target).attr("r", 2);
        tooltip.hide();
      });
  });

  // Legend
  createLegend(svg, {
    items: models.map((m, i) => ({
      label: m,
      color: modelColor(i, models.length),
    })),
    x: CHART_MARGIN.left + innerWidth + 12,
    y: CHART_MARGIN.top,
  });

  addExportButton(containerId, "cluster-entropy-by-layer");
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
    console.error("Cluster trends init failed:", err);
    document.getElementById("loading").textContent =
      `Error loading data: ${err.message}`;
    return;
  }

  console.log(
    `Cluster trends loaded: ${Object.keys(DATA.models).length} models, ` +
      `${DATA.k_values.length} K values, methods: ${(DATA.methods || []).join(", ")}`,
  );

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

  // --- Method selector ---
  const methodSelect = document.getElementById("method-select");
  const hierControls = document.getElementById("hierarchical-controls");
  const flatParamControls = document.getElementById("flat-param-controls");
  const flatParamSelect = document.getElementById("flat-param-select");
  const flatParamLabel = document.getElementById("flat-param-label");

  const availableMethods = DATA.methods || ["hierarchical"];

  // Filter to methods that have data
  const methodsWithData = availableMethods.filter(
    (m) => m === "hierarchical" || DATA[m] !== undefined,
  );

  if (methodSelect) {
    methodSelect.innerHTML = "";
    for (const m of methodsWithData) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      methodSelect.appendChild(opt);
    }

    const savedMethod = ClusteringConfig.getMethod();
    currentMethod =
      savedMethod && methodsWithData.includes(savedMethod)
        ? savedMethod
        : "hierarchical";
    methodSelect.value = currentMethod;
  }

  function updateMethodControls() {
    const isHier = currentMethod === "hierarchical";
    if (hierControls) hierControls.style.display = isHier ? "" : "none";
    if (flatParamControls)
      flatParamControls.style.display = isHier ? "none" : "";

    if (!isHier && flatParamSelect) {
      const methodData = DATA[currentMethod];
      if (methodData) {
        if (flatParamLabel)
          flatParamLabel.textContent = methodData.param_name + ":";
        flatParamSelect.innerHTML = "";
        for (const pv of methodData.param_values) {
          const opt = document.createElement("option");
          opt.value = String(pv);
          opt.textContent = String(pv);
          flatParamSelect.appendChild(opt);
        }
        const savedPK = ClusteringConfig.getParamKey();
        flatParamSelect.value =
          savedPK && methodData.param_values.map(String).includes(savedPK)
            ? savedPK
            : String(methodData.param_values[0]);
      }
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

  // Cut height control (shared component)
  const cutHeightCtrl = createCutHeightControl({
    container: document.getElementById("cut-height-container"),
    maxHeight: 10,
    step: 0.01,
    initialValue: 5,
    label: "Cut height:",
    onChange: (h) => updateFromCutHeight(h),
  });
  if (!clusteringAvailable) {
    cutHeightCtrl.setDisabled(true);
  }

  // K selector: switch to precomputed mode
  kSelect.addEventListener("change", () => {
    updateFromK(Number(kSelect.value));
  });

  // Method selector
  if (methodSelect) {
    methodSelect.addEventListener("change", () => {
      currentMethod = methodSelect.value;
      ClusteringConfig.setMethod(currentMethod);
      updateMethodControls();

      if (currentMethod === "hierarchical") {
        if (clusteringAvailable) {
          const h = cutHeightCtrl.getValue ? cutHeightCtrl.getValue() : 5;
          updateFromCutHeight(h);
        } else {
          updateFromK(currentK);
        }
      } else {
        const pk = flatParamSelect ? flatParamSelect.value : null;
        if (pk) {
          ClusteringConfig.setParamKey(pk);
          updateFromFlatParam(currentMethod, pk);
        }
      }
    });
  }

  // Flat param selector
  if (flatParamSelect) {
    flatParamSelect.addEventListener("change", () => {
      const pk = flatParamSelect.value;
      ClusteringConfig.setParamKey(pk);
      updateFromFlatParam(currentMethod, pk);
    });
  }

  updateMethodControls();

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

  // Initial draw based on current method
  if (currentMethod === "hierarchical") {
    if (clusteringAvailable) {
      const savedCutHeight = ClusteringConfig.getCutHeight();
      const startCutHeight = savedCutHeight ?? 5;
      cutHeightCtrl.setValue(startCutHeight);
      updateFromCutHeight(startCutHeight);
    } else {
      updateFromK(currentK);
    }
  } else {
    const pk = flatParamSelect ? flatParamSelect.value : null;
    if (pk) updateFromFlatParam(currentMethod, pk);
  }
}

init();
