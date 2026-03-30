// === Configuration ===

const urlParams = new URLSearchParams(window.location.search);

// CONFIG object used by ClusteringLoader and HeadDistances
const CONFIG = {
  dataUrl: urlParams.get("data") || "ablation_results.json",
  clustering_methods_url: "../features/clustering_methods.json",
  clustering_meta_url: "../features/clustering/clustering_meta.json",
  clustering_linkage_url: "../features/clustering/linkage.json",
  cluster_labels_url: "../features/clustering/cluster_labels.json",
  clustering_hdbscan_meta_url:
    "../features/clustering_hdbscan/clustering_meta.json",
  clustering_hdbscan_partitions_url:
    "../features/clustering_hdbscan/partitions.json",
  clustering_hdbscan_labels_url:
    "../features/clustering_hdbscan/cluster_labels.json",
  clustering_leiden_meta_url:
    "../features/clustering_leiden/clustering_meta.json",
  clustering_leiden_partitions_url:
    "../features/clustering_leiden/partitions.json",
  clustering_leiden_labels_url:
    "../features/clustering_leiden/cluster_labels.json",
  attnpedia_url: "../vis/attnpedia/ap.json",
  headDistsnpy_url: "../features/head_dists_raw/distances.npy",
  headDistsmeta_url: "../features/head_dists_raw/dists_meta.json",
};

const DEFAULT_HEAD = "gpt2-small:L5:H5";
const DEFAULT_CUT_HEIGHT = 5.0;

// === Metric definitions ===

const DISTRIBUTION_METRICS = [
  "prefix_score",
  "copying_score",
  "loss_increase",
  "ov_copying_score",
  "icl_degradation",
];

const METRIC_DISPLAY_NAMES = {
  loss_increase: "Loss Increase (\u0394 Loss)",
  prefix_score: "Prefix Matching Score",
  copying_score: "Copying Score",
  ov_copying_score: "OV Copying Score",
  icl_degradation: "ICL Degradation",
};

const SCATTER_MARGIN = { top: 28, right: 20, bottom: 48, left: 54 };

// === Helpers ===

function escapeHTML(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function median(arr) {
  if (arr.length === 0) return NaN;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

function pearsonR(xs, ys) {
  const n = xs.length;
  if (n < 2) return NaN;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0,
    dx2 = 0,
    dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom === 0 ? 0 : num / denom;
}

// === State ===

let allData = null;
let clustering = null;
let headDistances = null;
let currentDistances = null; // Map<head_name, distance>
let clusteringAvailable = false;

let targetHead = urlParams.get("target") || DEFAULT_HEAD;
let selectedCluster = null;
let modelFilter = "all";
let selectedMethod = null;
let xAxisMode = "head"; // "head" | "cluster"
let targetClusterId = null; // cluster ID for cluster distance mode
let clusterAggregation = "mean"; // "mean" | "median"

// === Model family helpers ===

function getModelFamily(modelName) {
  if (modelName.startsWith("gpt2") || modelName === "distillgpt2")
    return "gpt2";
  if (modelName.startsWith("pythia")) return "pythia";
  return "other";
}

function matchesModelFilter(modelName) {
  if (modelFilter === "all") return true;
  return getModelFamily(modelName) === modelFilter;
}

// === Initialization ===

async function init() {
  clustering = new ClusteringLoader();
  headDistances = new HeadDistances();

  let ablationResp;
  try {
    ablationResp = await fetch(CONFIG.dataUrl);
  } catch {
    showError("Failed to load data. Serve this directory over HTTP.");
    return;
  }
  if (!ablationResp.ok) {
    showError(`No data found at ${CONFIG.dataUrl}.`);
    return;
  }

  // Load ablation data, clustering, and head distances metadata in parallel
  const [ablationData, clusteringReady] = await Promise.all([
    ablationResp.json(),
    clustering
      ._ensureLoaded()
      .then(() => clustering.isAvailable())
      .catch(() => false),
  ]);

  allData = ablationData;
  clusteringAvailable = clusteringReady;

  // Load head distances metadata
  try {
    await headDistances._ensureMetaLoaded();
  } catch {
    showError("Failed to load head distances metadata.");
    return;
  }

  if (!clusteringReady) {
    const el = document.getElementById("clustering-controls");
    if (el)
      el.innerHTML =
        '<p style="color:#c33;font-size:13px;font-weight:bold;">\u26A0 Clustering data failed to load. Cluster features disabled.</p>';
  }

  if (!allData.models || Object.keys(allData.models).length === 0) {
    showError("No model results found in data.");
    return;
  }

  // Set up clustering
  if (clusteringReady) {
    await setupClusteringControls();
  }

  // Collect all ablation methods
  const methods = new Set();
  for (const modelData of Object.values(allData.models)) {
    for (const r of modelData.results) {
      methods.add(r.ablation_method);
    }
  }
  selectedMethod = [...methods].sort()[0] || null;
  setupMethodSelect([...methods].sort());

  // Set up UI
  setupTargetHeadInput();
  setupXAxisModeControls();
  if (clusteringReady) populateTargetClusterDropdown();
  setupModelFamilyButtons();
  setupChartWidthSlider();
  document
    .getElementById("log-x-toggle")
    .addEventListener("change", () => renderAll());
  if (clusteringReady) setupMinClusterSizeSlider();

  // Initial render
  renderClusterChips();
  setupClusterChipHandler();
  await loadDistancesAndRender();
}

function showError(msg) {
  document.querySelector(".container").innerHTML =
    `<p style="color:#999;text-align:center;padding:60px;">${msg}</p>`;
}

// === Target Head Input ===

function setupTargetHeadInput() {
  const datalist = document.getElementById("head-datalist");
  const allHeads = headDistances.head_dists_meta.cls_values;
  datalist.innerHTML = allHeads.map((h) => `<option value="${h}">`).join("");

  const input = document.getElementById("target-head-input");
  input.value = targetHead;

  // Validate initial value
  if (!allHeads.includes(targetHead)) {
    input.classList.add("invalid");
  }

  input.addEventListener("change", async () => {
    const newTarget = input.value.trim();
    if (!allHeads.includes(newTarget)) {
      input.classList.add("invalid");
      return;
    }
    input.classList.remove("invalid");
    targetHead = newTarget;

    // Update URL without reload
    const url = new URL(window.location);
    url.searchParams.set("target", targetHead);
    window.history.replaceState({}, "", url);

    await loadDistancesAndRender();
  });
}

// === X-Axis Mode Controls ===

function setupXAxisModeControls() {
  const modeSelect = document.getElementById("xaxis-mode-select");
  const headControls = document.getElementById("xaxis-head-controls");
  const clusterControls = document.getElementById("xaxis-cluster-controls");
  const aggSelect = document.getElementById("cluster-agg-select");
  const targetClusterSelect = document.getElementById("target-cluster-select");

  // Disable cluster option if clustering is unavailable
  if (!clusteringAvailable) {
    const clusterOption = modeSelect.querySelector('option[value="cluster"]');
    if (clusterOption) clusterOption.disabled = true;
  }

  modeSelect.addEventListener("change", async () => {
    xAxisMode = modeSelect.value;
    headControls.style.display = xAxisMode === "head" ? "" : "none";
    clusterControls.style.display = xAxisMode === "cluster" ? "" : "none";
    if (xAxisMode === "cluster" && targetClusterId === null) {
      // Auto-select first cluster
      const firstOpt = targetClusterSelect.querySelector("option");
      if (firstOpt) {
        targetClusterId = parseInt(firstOpt.value);
        targetClusterSelect.value = firstOpt.value;
      }
    }
    await loadDistancesAndRender();
  });

  aggSelect.addEventListener("change", async () => {
    clusterAggregation = aggSelect.value;
    if (xAxisMode === "cluster") {
      await loadDistancesAndRender();
    }
  });

  targetClusterSelect.addEventListener("change", async () => {
    targetClusterId = parseInt(targetClusterSelect.value);
    if (xAxisMode === "cluster") {
      await loadDistancesAndRender();
    }
  });
}

function populateTargetClusterDropdown() {
  const select = document.getElementById("target-cluster-select");
  if (!clustering || !clustering._assignments) return;

  const sizes = {};
  for (const [, cid] of Object.entries(clustering._assignments)) {
    if (cid === -1) continue;
    sizes[cid] = (sizes[cid] || 0) + 1;
  }

  const sorted = Object.entries(sizes).sort((a, b) => b[1] - a[1]);

  const prevValue = targetClusterId;
  select.innerHTML = sorted
    .map(([cid, size]) => {
      const id = parseInt(cid);
      const label = clustering.getClusterLabel(id);
      const name = label && label.name ? label.name : `${id}`;
      return `<option value="${id}">${name} (${size} heads)</option>`;
    })
    .join("");

  // Preserve previous selection if still valid
  if (prevValue !== null && sizes[prevValue] !== undefined) {
    select.value = prevValue;
  } else if (sorted.length > 0) {
    targetClusterId = parseInt(sorted[0][0]);
    select.value = targetClusterId;
  }
}

async function computeClusterDistances(clusterId) {
  // Get heads in the target cluster
  const headsInCluster = [];
  for (const [headName, cid] of Object.entries(clustering._assignments)) {
    if (cid === clusterId) headsInCluster.push(headName);
  }

  if (headsInCluster.length === 0) return new Map();

  const allDistHeads = headDistances.head_dists_meta.cls_values;

  // Filter to cluster members that exist in the distance matrix
  const validClusterHeads = headsInCluster.filter((h) =>
    allDistHeads.includes(h),
  );
  if (validClusterHeads.length === 0) return new Map();

  // Load distance rows for all cluster members in parallel
  const clusterIndices = validClusterHeads.map((h) =>
    headDistances.get_head_idx(h),
  );
  const rows = await Promise.all(
    clusterIndices.map((idx) => headDistances._loadRow(idx)),
  );

  // For each ablation head, compute aggregated distance to cluster
  const allAblationHeads = getAllHeadNames();
  const validHeads = allAblationHeads.filter((h) => allDistHeads.includes(h));

  const distances = new Map();
  const aggFn =
    clusterAggregation === "median"
      ? median
      : (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;

  for (const headName of validHeads) {
    const headIdx = headDistances.get_head_idx(headName);
    const dists = rows.map((row) => row.data[headIdx]);
    distances.set(headName, aggFn(dists));
  }

  return distances;
}

// === Distance Loading ===

function showLoading(visible) {
  document.getElementById("loading-indicator").style.display = visible
    ? ""
    : "none";
  document.getElementById("charts-grid").style.display = visible ? "none" : "";
}

function getAllHeadNames() {
  const heads = new Set();
  for (const modelData of Object.values(allData.models)) {
    for (const r of modelData.results) {
      heads.add(r.head);
    }
  }
  return [...heads];
}

async function loadDistancesAndRender() {
  showLoading(true);

  try {
    if (xAxisMode === "head") {
      // Validate target head exists in distance matrix
      const allDistHeads = headDistances.head_dists_meta.cls_values;
      if (!allDistHeads.includes(targetHead)) {
        showLoading(false);
        showError(
          `Target head "${escapeHTML(targetHead)}" not found in distance matrix.`,
        );
        return;
      }

      const allAblationHeads = getAllHeadNames();
      const validHeads = allAblationHeads.filter((h) =>
        allDistHeads.includes(h),
      );

      const distResults = await headDistances.getHeadDistances(
        targetHead,
        validHeads,
      );

      currentDistances = new Map();
      for (const { head_name, distance } of distResults) {
        currentDistances.set(head_name, distance);
      }
    } else {
      // Cluster mode
      if (targetClusterId === null) {
        showLoading(false);
        return;
      }
      currentDistances = await computeClusterDistances(targetClusterId);
    }
  } catch (err) {
    showLoading(false);
    showError(`Failed to load distances: ${err.message}`);
    return;
  }

  showLoading(false);
  createChartContainers();
  renderAll();
}

// === UI Setup (clustering, model filters) ===

async function setupClusteringControls() {
  const availableMethods = clustering.getAvailableMethods();
  const methodSelect = document.getElementById("cluster-method-select");

  methodSelect.innerHTML = availableMethods
    .map((m) => `<option value="${m}">${m}</option>`)
    .join("");

  const savedMethod = ClusteringConfig.getMethod();
  const initialMethod =
    savedMethod && availableMethods.includes(savedMethod)
      ? savedMethod
      : availableMethods.includes("leiden")
        ? "leiden"
        : availableMethods[0];
  methodSelect.value = initialMethod;

  if (initialMethod === "hierarchical") {
    const savedCutHeight = ClusteringConfig.getCutHeight();
    const initialCutHeight = savedCutHeight || DEFAULT_CUT_HEIGHT;
    await clustering.setCutHeight(initialCutHeight);
    setupCutHeightSlider(initialCutHeight);
  } else {
    clustering.setMethod(initialMethod);
  }

  updateClusteringControlVisibility();
  updateClusterStats();

  methodSelect.addEventListener("change", async () => {
    clustering.setMethod(methodSelect.value);
    if (clustering.isHierarchical()) {
      const savedCutHeight = ClusteringConfig.getCutHeight();
      const h = savedCutHeight || DEFAULT_CUT_HEIGHT;
      await clustering.setCutHeight(h);
      setupCutHeightSlider(h);
    }
    updateClusteringControlVisibility();
    updateClusterStats();
    selectedCluster = null;
    populateTargetClusterDropdown();
    renderClusterChips();
    if (xAxisMode === "cluster") {
      await loadDistancesAndRender();
    } else {
      renderAll();
    }
  });
}

let _cutHeightSliderBound = false;

function setupCutHeightSlider(initialValue) {
  const slider = document.getElementById("cut-height-slider");
  const maxH = clustering.getMaxCutHeight() || 20;
  slider.min = 0;
  slider.max = maxH;
  slider.step = (maxH / 1000).toFixed(4);
  slider.value = initialValue;
  document.getElementById("cut-height-value").textContent =
    initialValue.toFixed(2);

  if (_cutHeightSliderBound) return;
  _cutHeightSliderBound = true;

  let _sliderTimeout = null;
  slider.addEventListener("input", () => {
    const h = parseFloat(slider.value);
    document.getElementById("cut-height-value").textContent = h.toFixed(2);
    clearTimeout(_sliderTimeout);
    _sliderTimeout = setTimeout(async () => {
      await clustering.setCutHeight(h);
      ClusteringConfig.setCutHeight(h);
      updateClusterStats();
      selectedCluster = null;
      populateTargetClusterDropdown();
      renderClusterChips();
      if (xAxisMode === "cluster") {
        await loadDistancesAndRender();
      } else {
        renderAll();
      }
    }, 50);
  });
}

function updateClusteringControlVisibility() {
  const isHier = clustering.isHierarchical();
  document.getElementById("cluster-hier-controls").style.display = isHier
    ? ""
    : "none";
  document.getElementById("cluster-param-controls").style.display = isHier
    ? "none"
    : "";

  if (!isHier) {
    setupParamSelect();
  }
}

function setupParamSelect() {
  const info = clustering.getFlatParamInfo();
  if (!info) return;

  const paramLabel = document.getElementById("cluster-param-label");
  const oldSelect = document.getElementById("cluster-param-select");

  paramLabel.textContent = info.paramName + ":";

  const paramSelect = oldSelect.cloneNode(false);
  paramSelect.id = "cluster-param-select";
  oldSelect.parentNode.replaceChild(paramSelect, oldSelect);

  paramSelect.innerHTML = info.paramKeys
    .map((pk) => {
      const meta = info.meta[pk];
      const extra = meta
        ? ` (${meta.n_clusters} clusters${meta.n_outliers ? `, ${meta.n_outliers} outliers` : ""})`
        : "";
      return `<option value="${pk}">${pk}${extra}</option>`;
    })
    .join("");

  const savedKey = ClusteringConfig.getParamKey();
  paramSelect.value =
    savedKey && info.paramKeys.includes(savedKey)
      ? savedKey
      : info.paramKeys[0];

  clustering.setParamKey(paramSelect.value);

  paramSelect.addEventListener("change", async () => {
    await clustering.setParamKey(paramSelect.value);
    updateClusterStats();
    selectedCluster = null;
    populateTargetClusterDropdown();
    renderClusterChips();
    if (xAxisMode === "cluster") {
      await loadDistancesAndRender();
    } else {
      renderAll();
    }
  });
}

function updateClusterStats() {
  const n = clustering.getNClusters();
  const unclustered = clustering.getUnclusteredCount();
  let text = `(${n} clusters`;
  if (unclustered > 0) text += `, ${unclustered} unclustered`;
  text += ")";
  document.getElementById("n-clusters-label").textContent = text;
}

function setupMethodSelect(methods) {
  const select = document.getElementById("method-select");
  select.innerHTML = methods
    .map(
      (m) =>
        `<option value="${m}" ${m === selectedMethod ? "selected" : ""}>${m}</option>`,
    )
    .join("");
  select.addEventListener("change", () => {
    selectedMethod = select.value;
    renderAll();
  });
}

function setupModelFamilyButtons() {
  const container = document.getElementById("model-family-buttons");
  const families = ["all", "gpt2", "pythia"];
  const labels = { all: "All", gpt2: "GPT-2", pythia: "Pythia" };
  container.innerHTML = families
    .map(
      (f) =>
        `<button class="family-btn${f === modelFilter ? " active" : ""}" data-family="${f}">${labels[f]}</button>`,
    )
    .join("");
  container.addEventListener("click", (e) => {
    const btn = e.target.closest(".family-btn");
    if (!btn) return;
    modelFilter = btn.dataset.family;
    container
      .querySelectorAll(".family-btn")
      .forEach((b) =>
        b.classList.toggle("active", b.dataset.family === modelFilter),
      );
    renderAll();
  });
}

function setupMinClusterSizeSlider() {
  const slider = document.getElementById("min-cluster-size");
  const label = document.getElementById("min-cluster-size-value");
  if (!slider || !clustering) return;

  const initial = clustering.getMinClusterSize();
  slider.value = initial;
  label.textContent = initial;

  slider.addEventListener("input", async () => {
    const n = parseInt(slider.value);
    label.textContent = n;
    await clustering.setMinClusterSize(n);
    updateClusterStats();
    selectedCluster = null;
    populateTargetClusterDropdown();
    renderClusterChips();
    if (xAxisMode === "cluster") {
      await loadDistancesAndRender();
    } else {
      renderAll();
    }
  });
}

function setupChartWidthSlider() {
  const slider = document.getElementById("chart-width-slider");
  const label = document.getElementById("chart-width-value");
  slider.addEventListener("input", () => {
    const val = slider.value;
    label.textContent = val + "px";
    document
      .querySelector(".charts-grid")
      .style.setProperty("--chart-min-width", val + "px");
    renderAll();
  });
}

// === Cluster Chips ===

function renderClusterChips() {
  const container = document.getElementById("cluster-chips");
  if (!clustering || !clustering._is_loaded || !clustering._meta) {
    container.innerHTML =
      '<span style="color:#999">Clustering data not available</span>';
    return;
  }

  const sizes = {};
  for (const [headId, cid] of Object.entries(clustering._assignments)) {
    if (cid === -1) continue;
    sizes[cid] = (sizes[cid] || 0) + 1;
  }

  const sorted = Object.entries(sizes).sort((a, b) => b[1] - a[1]);
  const totalHeads = Object.values(sizes).reduce((a, b) => a + b, 0);

  const showAllChip = `<button class="cluster-chip${selectedCluster === null ? " selected" : ""}" data-cluster="all" title="Show all clusters">All <span class="chip-count">${totalHeads}</span></button>`;

  const clusterChips = sorted
    .map(([cid, size]) => {
      const id = parseInt(cid);
      const color = clustering.getClusterColor(id);
      const label = clustering.getClusterLabel(id);
      const name = label && label.name ? label.name : `${id}`;
      const isSelected = id === selectedCluster;
      return `<button class="cluster-chip${isSelected ? " selected" : ""}" data-cluster="${id}" style="--chip-color: ${color}" title="Cluster ${id}: ${size} heads">${name} <span class="chip-count">${size}</span></button>`;
    })
    .join("");

  container.innerHTML = showAllChip + clusterChips;
}

function setupClusterChipHandler() {
  const container = document.getElementById("cluster-chips");
  container.addEventListener("click", (e) => {
    const btn = e.target.closest(".cluster-chip");
    if (!btn) return;
    const val = btn.dataset.cluster;
    selectedCluster = val === "all" ? null : parseInt(val);
    renderClusterChips();
    renderAll();
  });

  container.addEventListener("auxclick", (e) => {
    if (e.button !== 1) return;
    const btn = e.target.closest(".cluster-chip");
    if (!btn) return;
    const val = btn.dataset.cluster;
    if (val === "all") return;
    e.preventDefault();
    const id = parseInt(val);
    ClusteringConfig.setHighlightCluster(id);
    window.open(`../vis/clustering/index.html?highlight=${id}`, "_blank");
  });
}

// === Data helpers ===

function getFilteredResults() {
  if (!allData) return [];
  const results = [];
  for (const [modelName, modelData] of Object.entries(allData.models)) {
    if (!matchesModelFilter(modelName)) continue;
    for (const r of modelData.results) {
      if (selectedMethod && r.ablation_method !== selectedMethod) continue;
      results.push({ ...r, model: modelName });
    }
  }
  return results;
}

// === Render All ===

function renderAll() {
  renderCharts();
}

// === Chart Rendering ===

function createChartContainers() {
  const grid = document.getElementById("charts-grid");
  grid.innerHTML = "";
  for (const metric of DISTRIBUTION_METRICS) {
    const div = document.createElement("div");
    div.className = "chart-cell";
    div.id = `chart-${metric}`;
    grid.appendChild(div);
  }
}

function renderCharts() {
  if (!currentDistances) return;

  const results = getFilteredResults();

  for (const metric of DISTRIBUTION_METRICS) {
    renderScatterplot(metric, results);
  }
}

function renderScatterplot(metric, results) {
  const containerId = `chart-${metric}`;
  const container = document.getElementById(containerId);
  if (!container) return;

  const { svg, g, innerWidth, innerHeight } = createChartSVG(
    containerId,
    SCATTER_MARGIN,
  );

  // Build data points, one per head
  const points = [];
  const seenHeads = new Set();
  for (const r of results) {
    if (seenHeads.has(r.head)) continue;
    seenHeads.add(r.head);

    const dist = currentDistances.get(r.head);
    const val = r[metric];
    if (dist == null || val == null || isNaN(val)) continue;

    const cid =
      clustering && clustering._assignments
        ? (clustering._assignments[r.head] ?? -1)
        : -1;
    const color =
      cid >= 0 && clustering ? clustering.getClusterColor(cid) : "#999";

    points.push({
      head: r.head,
      distance: dist,
      value: val,
      clusterId: cid,
      color: color,
    });
  }

  if (points.length === 0) {
    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight / 2)
      .attr("text-anchor", "middle")
      .attr("fill", "#999")
      .attr("font-size", 13)
      .text("No data");
    return;
  }

  // Scales
  const logX = document.getElementById("log-x-toggle").checked;
  const xMax = d3.max(points, (d) => d.distance) * 1.05;
  let xScale;
  if (logX) {
    const xMin =
      d3.min(
        points.filter((d) => d.distance > 0),
        (d) => d.distance,
      ) || 0.1;
    xScale = d3
      .scaleLog()
      .domain([xMin * 0.9, xMax || 1])
      .range([0, innerWidth]);
  } else {
    xScale = d3
      .scaleLinear()
      .domain([0, xMax || 1])
      .range([0, innerWidth]);
  }

  const yExtent = d3.extent(points, (d) => d.value);
  const yPad = (yExtent[1] - yExtent[0]) * 0.05 || 0.1;
  const yScale = d3
    .scaleLinear()
    .domain([yExtent[0] - yPad, yExtent[1] + yPad])
    .nice()
    .range([innerHeight, 0]);

  // Axes
  createXAxis(g, xScale, {
    height: innerHeight,
    label:
      xAxisMode === "head"
        ? `Distance to ${targetHead}`
        : `${clusterAggregation === "mean" ? "Mean" : "Median"} Dist. to Cluster ${targetClusterId}`,
    gridHeight: innerHeight,
    ticks: 6,
  });
  createYAxis(g, yScale, {
    label: METRIC_DISPLAY_NAMES[metric] || metric,
    gridWidth: innerWidth,
    ticks: 5,
  });

  // Title
  svg
    .append("text")
    .attr("x", SCATTER_MARGIN.left + innerWidth / 2)
    .attr("y", 18)
    .attr("text-anchor", "middle")
    .attr("font-size", CHART_STYLES.titleFontSize)
    .attr("font-weight", "bold")
    .attr("fill", "#333")
    .text(METRIC_DISPLAY_NAMES[metric] || metric);

  // Tooltip
  const tooltip = createTooltip(containerId);

  // Sort points so selected-cluster dots draw on top
  const sortedPoints = [...points].sort((a, b) => {
    const aActive =
      selectedCluster === null || a.clusterId === selectedCluster ? 1 : 0;
    const bActive =
      selectedCluster === null || b.clusterId === selectedCluster ? 1 : 0;
    return aActive - bActive;
  });

  // Draw dots
  g.selectAll(".scatter-dot")
    .data(sortedPoints)
    .enter()
    .append("circle")
    .attr("class", "scatter-dot")
    .attr("cx", (d) =>
      xScale(logX ? Math.max(d.distance, xScale.domain()[0]) : d.distance),
    )
    .attr("cy", (d) => yScale(d.value))
    .attr("r", 3.5)
    .attr("fill", (d) => {
      if (selectedCluster !== null && d.clusterId !== selectedCluster) {
        return "#ddd";
      }
      return d.color;
    })
    .attr("stroke", (d) => {
      if (selectedCluster !== null && d.clusterId !== selectedCluster) {
        return "#ccc";
      }
      return d3.color(d.color)?.darker(0.5)?.toString() || "#666";
    })
    .attr("stroke-width", 0.5)
    .attr("opacity", (d) => {
      if (selectedCluster !== null && d.clusterId !== selectedCluster) {
        return 0.2;
      }
      return 0.75;
    })
    .on("mouseover", function (event, d) {
      d3.select(this).attr("r", 6).attr("opacity", 1);
      tooltip.show(
        event,
        `<b>${escapeHTML(d.head)}</b><br>` +
          `Distance: ${d.distance.toFixed(3)}<br>` +
          `${METRIC_DISPLAY_NAMES[metric] || metric}: ${d.value.toFixed(4)}<br>` +
          `Cluster: ${d.clusterId >= 0 ? d.clusterId : "none"}`,
      );
    })
    .on("mousemove", function (event, d) {
      tooltip.show(
        event,
        `<b>${escapeHTML(d.head)}</b><br>` +
          `Distance: ${d.distance.toFixed(3)}<br>` +
          `${METRIC_DISPLAY_NAMES[metric] || metric}: ${d.value.toFixed(4)}<br>` +
          `Cluster: ${d.clusterId >= 0 ? d.clusterId : "none"}`,
      );
    })
    .on("mouseout", function () {
      const d = d3.select(this).datum();
      const isActive =
        selectedCluster === null || d.clusterId === selectedCluster;
      d3.select(this)
        .attr("r", 3.5)
        .attr("opacity", isActive ? 0.75 : 0.2);
      tooltip.hide();
    });

  // Highlight target head with red ring (head mode only)
  if (xAxisMode === "head") {
    const targetPoint = points.find((d) => d.head === targetHead);
    if (targetPoint) {
      g.append("circle")
        .attr(
          "cx",
          xScale(
            logX
              ? Math.max(targetPoint.distance, xScale.domain()[0])
              : targetPoint.distance,
          ),
        )
        .attr("cy", yScale(targetPoint.value))
        .attr("r", 7)
        .attr("fill", "none")
        .attr("stroke", "#e74c3c")
        .attr("stroke-width", 2)
        .attr("class", "no-export");
    }
  }

  // Compute Pearson r for active points
  const activePoints =
    selectedCluster !== null
      ? points.filter((d) => d.clusterId === selectedCluster)
      : points;

  if (activePoints.length > 2) {
    const r = pearsonR(
      activePoints.map((d) => d.distance),
      activePoints.map((d) => d.value),
    );
    svg
      .append("text")
      .attr("x", SCATTER_MARGIN.left + innerWidth - 4)
      .attr("y", SCATTER_MARGIN.top + 14)
      .attr("text-anchor", "end")
      .attr("font-size", 11)
      .attr("fill", Math.abs(r) >= 0.3 ? "#c0392b" : "#888")
      .text(`r = ${r.toFixed(3)} (n=${activePoints.length})`);
  }

  // SVG export
  addExportButton(
    containerId,
    `correlation_${metric}_${targetHead.replace(/:/g, "_")}`,
  );
}

// === Entry point ===

window.addEventListener("DOMContentLoaded", init);
