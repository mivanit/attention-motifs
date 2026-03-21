// === Configuration ===

const urlParams = new URLSearchParams(window.location.search);

// CONFIG object used by ClusteringLoader and AttentionPedia
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
};

const DEFAULT_HEAD = "gpt2-small:L5:H5";
const DEFAULT_CUT_HEIGHT = 5.0;

// === Metric definitions ===

const METRIC_GROUPS = {
  Loss: ["baseline_repeated_loss", "ablated_repeated_loss", "loss_increase"],
  "Induction Scores": ["prefix_score", "copying_score", "ov_copying_score"],
  "Prefix (legacy)": ["prefix_score_legacy"],
  ICL: ["baseline_icl_score", "ablated_icl_score", "icl_degradation"],
};

const DEFAULT_VISIBLE_GROUPS = new Set(["Loss", "Induction Scores", "ICL"]);

const DISTRIBUTION_METRICS = [
  "loss_increase",
  "prefix_score",
  "copying_score",
  "ov_copying_score",
  "icl_degradation",
];

const DELTA_COLUMNS = {
  loss_increase: { higher_is_worse: true, thresholds: [0.1, 0.5] },
  icl_degradation: { higher_is_worse: true, thresholds: [0.05, 0.2] },
  prefix_score: { higher_is_worse: false, thresholds: [0.1, 0.3] },
  prefix_score_legacy: { higher_is_worse: false, thresholds: [0.1, 0.3] },
  copying_score: { higher_is_worse: false, thresholds: [0.5, 2.0] },
  ov_copying_score: { higher_is_worse: false, thresholds: [0.05, 0.2] },
};

const COLUMN_LABELS = {
  head: "Head",
  cluster: "Cluster",
  classifications: "Class",
  ablation_method: "Method",
  baseline_repeated_loss: "Base Loss",
  ablated_repeated_loss: "Abl. Loss",
  loss_increase: "\u0394 Loss",
  prefix_score: "Prefix",
  prefix_score_legacy: "Pfx Leg.",
  copying_score: "Copying",
  ov_copying_score: "OV Copy",
  baseline_icl_score: "Base ICL",
  ablated_icl_score: "Abl. ICL",
  icl_degradation: "\u0394 ICL",
};

const METRIC_DISPLAY_NAMES = {
  loss_increase: "Loss Increase (\u0394 Loss)",
  prefix_score: "Prefix Matching Score",
  copying_score: "Copying Score",
  ov_copying_score: "OV Copying Score",
  icl_degradation: "ICL Degradation",
};

// === Chart margins ===
const ABLATION_CHART_MARGIN = { top: 28, right: 20, bottom: 40, left: 50 };

// === Helpers ===

function escapeHTML(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// === State ===

let allData = null;
let clustering = null;
let attnpedia = null;
let apData = null;
let clusteringAvailable = false;

let selectedCluster = null; // null = "show all", number = single cluster
let hoveredCluster = null; // used in show-all mode for highlight
let modelFilter = "all"; // "all" | "gpt2" | "pythia"
let selectedMethod = null; // null = first available
let chartType = "histogram"; // "histogram" | "boxplot"
let logScale = false; // log frequency axis for histograms
let sortByMean = false; // sort boxplot clusters by mean value
let visibleGroups = new Set(DEFAULT_VISIBLE_GROUPS);
let tableClusterFilter = false; // filter table to selected cluster only
let dataTable = null;

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
  // Load data sources in parallel
  clustering = new ClusteringLoader();

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

  // Load ablation data and clustering in parallel
  const [ablationData, clusteringReady, apResp] = await Promise.all([
    ablationResp.json(),
    clustering._ensureLoaded().then(() => clustering.isAvailable()),
    fetch(CONFIG.attnpedia_url)
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null),
  ]);

  allData = ablationData;
  apData = apResp;
  clusteringAvailable = clusteringReady;

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

  // Set up initial clustering state
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
  setupModelFamilyButtons();
  setupChartTypeButtons();
  setupLogScaleButton();
  setupSortByMeanButton();
  setupChartWidthSlider();
  setupColumnToggles();

  // Initial render
  renderClusterChips();
  setupClusterChipHandler();
  createChartContainers();
  renderCharts();
  renderConfigSummary();
  renderTable();
}

function showError(msg) {
  document.querySelector(".container").innerHTML =
    `<p style="color:#999;text-align:center;padding:60px;">${msg}</p>`;
}

// === UI Setup ===

async function setupClusteringControls() {
  const availableMethods = clustering.getAvailableMethods();
  const methodSelect = document.getElementById("cluster-method-select");

  // Populate method dropdown
  methodSelect.innerHTML = availableMethods
    .map((m) => `<option value="${m}">${m}</option>`)
    .join("");

  // Restore saved method or default to first available
  const savedMethod = ClusteringConfig.getMethod();
  const initialMethod =
    savedMethod && availableMethods.includes(savedMethod)
      ? savedMethod
      : availableMethods[0];
  methodSelect.value = initialMethod;

  // Initialize the clustering method
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

  // Bind method change
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
    renderClusterChips();
    renderCharts();
    renderTable();
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

      // Update selected cluster to follow DEFAULT_HEAD
      const newClusterId = clustering._assignments[DEFAULT_HEAD];
      if (newClusterId !== undefined) selectedCluster = newClusterId;

      renderClusterChips();
      renderCharts();
      renderTable();
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

  // Replace select element to avoid duplicate listeners
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

  // Apply initial param key
  clustering.setParamKey(paramSelect.value);

  paramSelect.addEventListener("change", async () => {
    await clustering.setParamKey(paramSelect.value);
    updateClusterStats();
    selectedCluster = null;
    renderClusterChips();
    renderCharts();
    renderTable();
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

async function selectDefaultCluster() {
  const defaultClusterId = clustering._assignments[DEFAULT_HEAD];
  selectedCluster = defaultClusterId !== undefined ? defaultClusterId : null;
  if (selectedCluster === null) {
    const sizes = await clustering.getClusterSizes();
    const sorted = Object.entries(sizes).sort((a, b) => b[1] - a[1]);
    if (sorted.length > 0) selectedCluster = parseInt(sorted[0][0]);
  }
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
    renderCharts();
    renderTable();
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
    renderCharts();
    renderTable();
  });
}

function setupChartTypeButtons() {
  const container = document.getElementById("chart-type-buttons");
  const types = [
    { id: "histogram", label: "Histogram" },
    { id: "boxplot", label: "Box Plot" },
  ];
  container.innerHTML = types
    .map(
      (t) =>
        `<button class="chart-type-btn${t.id === chartType ? " active" : ""}" data-type="${t.id}">${t.label}</button>`,
    )
    .join("");
  container.addEventListener("click", (e) => {
    const btn = e.target.closest(".chart-type-btn");
    if (!btn) return;
    chartType = btn.dataset.type;
    container
      .querySelectorAll(".chart-type-btn")
      .forEach((b) =>
        b.classList.toggle("active", b.dataset.type === chartType),
      );
    renderCharts();
  });
}

function setupLogScaleButton() {
  const container = document.getElementById("log-scale-button");
  const btn = document.createElement("button");
  btn.className = "chart-type-btn" + (logScale ? " active" : "");
  btn.textContent = "Log";
  btn.addEventListener("click", () => {
    logScale = !logScale;
    btn.classList.toggle("active", logScale);
    renderCharts();
  });
  container.appendChild(btn);
}

function setupSortByMeanButton() {
  const container = document.getElementById("sort-mean-button");
  const btn = document.createElement("button");
  btn.className = "chart-type-btn" + (sortByMean ? " active" : "");
  btn.textContent = "Sort Mean";
  btn.addEventListener("click", () => {
    sortByMean = !sortByMean;
    btn.classList.toggle("active", sortByMean);
    renderCharts();
  });
  container.appendChild(btn);
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
    renderCharts();
  });
}

function setupColumnToggles() {
  const groups = Object.keys(METRIC_GROUPS);
  document.getElementById("column-toggles").innerHTML = groups
    .map(
      (g) =>
        `<label class="chip"><input type="checkbox" ${visibleGroups.has(g) ? "checked" : ""} onchange="toggleGroup('${g}')"> ${g}</label>`,
    )
    .join("");
}

function toggleGroup(group) {
  if (visibleGroups.has(group)) visibleGroups.delete(group);
  else visibleGroups.add(group);
  renderTable();
}

function toggleTableClusterFilter() {
  tableClusterFilter = document.getElementById("table-cluster-filter").checked;
  renderTable();
}

// === Cluster Chips ===

function renderClusterChips() {
  const container = document.getElementById("cluster-chips");
  if (!clustering || !clustering._is_loaded || !clustering._meta) {
    container.innerHTML =
      '<span style="color:#999">Clustering data not available</span>';
    return;
  }

  // Get cluster sizes
  const sizes = {};
  for (const [headId, cid] of Object.entries(clustering._assignments)) {
    if (cid === -1) continue;
    sizes[cid] = (sizes[cid] || 0) + 1;
  }

  // Sort by size descending
  const sorted = Object.entries(sizes).sort((a, b) => b[1] - a[1]);
  const totalHeads = Object.values(sizes).reduce((a, b) => a + b, 0);

  // "Show All" chip + per-cluster chips
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
    renderCharts();
    renderTable();
  });
}

// === Data helpers ===

/**
 * Get all results as flat array of {head, model, ...metrics},
 * filtered by model family and method.
 */
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

/**
 * Split results into in-cluster and out-of-cluster arrays.
 */
function splitByCluster(results) {
  if (selectedCluster === null || !clustering || !clustering._is_loaded) {
    return { inCluster: [], outCluster: results };
  }
  const inCluster = [];
  const outCluster = [];
  for (const r of results) {
    const cid = clustering._assignments[r.head];
    if (cid === selectedCluster) {
      inCluster.push(r);
    } else {
      outCluster.push(r);
    }
  }
  return { inCluster, outCluster };
}

/**
 * Extract numeric values for a metric from results, skipping null/undefined.
 */
function extractValues(results, metric) {
  return results.map((r) => r[metric]).filter((v) => v != null && !isNaN(v));
}

/**
 * Group results by cluster ID. Returns array of { clusterId, color, label, results }.
 * Sorted by cluster size descending. Skips unclustered heads (cid === -1).
 */
function groupByCluster(results) {
  if (!clustering || !clustering._is_loaded) return [];
  const groups = {};
  for (const r of results) {
    const cid = clustering._assignments[r.head];
    if (cid === undefined || cid === -1) continue;
    if (!groups[cid]) groups[cid] = [];
    groups[cid].push(r);
  }
  return Object.entries(groups)
    .sort((a, b) => b[1].length - a[1].length)
    .map(([cid, res]) => {
      const id = parseInt(cid);
      const clLabel = clustering.getClusterLabel(id);
      const name = clLabel && clLabel.name ? clLabel.name : `${id}`;
      return {
        clusterId: id,
        color: clustering.getClusterColor(id),
        label: name,
        results: res,
      };
    });
}

// === KS Test ===

/**
 * Two-sample Kolmogorov-Smirnov test.
 * Returns { D, pValue } where D is the KS statistic and pValue is the
 * asymptotic approximation.
 */
function ksTest(sample1, sample2) {
  const n1 = sample1.length;
  const n2 = sample2.length;
  if (n1 === 0 || n2 === 0) return { D: NaN, pValue: NaN };

  const s1 = [...sample1].sort((a, b) => a - b);
  const s2 = [...sample2].sort((a, b) => a - b);

  // Merge and walk both CDFs
  let i = 0,
    j = 0,
    maxD = 0;
  while (i < n1 || j < n2) {
    const v1 = i < n1 ? s1[i] : Infinity;
    const v2 = j < n2 ? s2[j] : Infinity;
    if (v1 <= v2) i++;
    if (v2 <= v1) j++;
    const d = Math.abs(i / n1 - j / n2);
    if (d > maxD) maxD = d;
  }

  // Asymptotic p-value: P ≈ 2 * exp(-2 * n_eff * D^2)
  const nEff = (n1 * n2) / (n1 + n2);
  const pValue = 2 * Math.exp(-2 * nEff * maxD * maxD);

  return { D: maxD, pValue: Math.min(pValue, 1) };
}

function formatKsStat(ks) {
  if (isNaN(ks.D)) return "";
  return `D = ${ks.D.toFixed(2)}`;
}

function isKsSignificant(ks) {
  return ks.D >= 0.15;
}

// === Charts (D3 SVG) ===

function createChartContainers() {
  const grid = document.getElementById("charts-grid");
  grid.innerHTML = "";
  for (const metric of DISTRIBUTION_METRICS) {
    const div = document.createElement("div");
    div.className = "chart-cell";
    div.id = `chart-container-${metric}`;
    grid.appendChild(div);
  }
}

function renderCharts() {
  const results = getFilteredResults();
  // Deduplicate by head (one entry per head, using first matching method)
  const seenHeads = new Set();
  const uniqueResults = [];
  for (const r of results) {
    if (!seenHeads.has(r.head)) {
      seenHeads.add(r.head);
      uniqueResults.push(r);
    }
  }

  const showAll = selectedCluster === null && clusteringAvailable;

  if (showAll) {
    const clusterGroups = groupByCluster(uniqueResults);
    for (const metric of DISTRIBUTION_METRICS) {
      const groupData = clusterGroups
        .map((cg) => ({
          ...cg,
          values: extractValues(cg.results, metric),
        }))
        .filter((cg) => cg.values.length > 0);

      if (groupData.length === 0) {
        const container = document.getElementById(`chart-container-${metric}`);
        if (container) container.innerHTML = "";
        continue;
      }

      if (chartType === "histogram") {
        renderHistogramAll(metric, groupData);
      } else {
        renderBoxPlotAll(metric, groupData);
      }
    }
  } else {
    const { inCluster, outCluster } = splitByCluster(uniqueResults);
    for (const metric of DISTRIBUTION_METRICS) {
      const inValues = extractValues(inCluster, metric);
      const outValues = extractValues(outCluster, metric);

      if (inValues.length === 0 && outValues.length === 0) {
        const container = document.getElementById(`chart-container-${metric}`);
        if (container) container.innerHTML = "";
        continue;
      }

      if (chartType === "histogram") {
        renderHistogram(metric, inValues, outValues);
      } else {
        renderBoxPlot(metric, inValues, outValues);
      }
    }
  }
}

function renderHistogram(metric, inValues, outValues) {
  const containerId = `chart-container-${metric}`;
  const margin = ABLATION_CHART_MARGIN;
  const { svg, g, innerWidth, innerHeight } = createChartSVG(
    containerId,
    margin,
  );

  const allValues = [...inValues, ...outValues];
  if (allValues.length === 0) return;

  const rawMin = Math.min(...allValues);
  const rawMax = Math.max(...allValues);
  const min = rawMin === rawMax ? rawMin - 0.5 : rawMin;
  const max = rawMin === rawMax ? rawMax + 0.5 : rawMax;
  const nBins = rawMin === rawMax ? 1 : 20;
  const binWidth = (max - min) / nBins;

  function histogram(values) {
    const counts = new Array(nBins).fill(0);
    for (const v of values) {
      let bin = Math.floor((v - min) / binWidth);
      if (bin >= nBins) bin = nBins - 1;
      if (bin < 0) bin = 0;
      counts[bin]++;
    }
    return counts;
  }

  const inCounts = histogram(inValues);
  const outCounts = histogram(outValues);

  // Normalize to frequency so different-sized groups are comparable
  const inTotal = inValues.length || 1;
  const outTotal = outValues.length || 1;
  const inFreqs = inCounts.map((c) => c / inTotal);
  const outFreqs = outCounts.map((c) => c / outTotal);
  const maxFreq = Math.max(...inFreqs, ...outFreqs, 0.01);

  const xScale = d3.scaleLinear().domain([min, max]).range([0, innerWidth]);
  const epsilon = maxFreq * 0.005 || 0.001;
  const yScale = logScale
    ? d3
        .scaleLog()
        .domain([epsilon, maxFreq * 1.1])
        .range([innerHeight, 0])
        .clamp(true)
    : d3
        .scaleLinear()
        .domain([0, maxFreq * 1.1])
        .range([innerHeight, 0]);

  // Clamp zero-frequency bins for log scale
  if (logScale) {
    for (let i = 0; i < inFreqs.length; i++)
      if (inFreqs[i] === 0) inFreqs[i] = epsilon;
    for (let i = 0; i < outFreqs.length; i++)
      if (outFreqs[i] === 0) outFreqs[i] = epsilon;
  }

  createXAxis(g, xScale, {
    height: innerHeight,
    label: metric,
    ticks: 6,
  });
  createYAxis(g, yScale, {
    label: logScale ? "Frequency (log)" : "Frequency",
    ticks: 5,
  });

  const clusterColorStr =
    selectedCluster !== null && clusteringAvailable
      ? clustering.getClusterColor(selectedCluster)
      : "hsl(210, 70%, 50%)";
  const clusterLabel = getClusterChipLabel();

  // Build line points at bin midpoints
  const binMidpoints = Array.from(
    { length: nBins },
    (_, i) => min + (i + 0.5) * binWidth,
  );

  function drawDistributionLine(freqs, strokeColor, fillColor) {
    const areaGen = d3
      .area()
      .x((_, i) => xScale(binMidpoints[i]))
      .y0(innerHeight)
      .y1((d) => yScale(d))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(freqs)
      .attr("d", areaGen)
      .attr("fill", fillColor)
      .attr("stroke", "none");

    const lineGen = d3
      .line()
      .x((_, i) => xScale(binMidpoints[i]))
      .y((d) => yScale(d))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(freqs)
      .attr("d", lineGen)
      .attr("fill", "none")
      .attr("stroke", strokeColor)
      .attr("stroke-width", 2);
  }

  // Draw out-cluster first (behind), then in-cluster on top
  drawDistributionLine(
    outFreqs,
    "rgba(150, 150, 150, 0.7)",
    "rgba(150, 150, 150, 0.15)",
  );
  drawDistributionLine(
    inFreqs,
    clusterColorToRGBA(clusterColorStr, 1),
    clusterColorToRGBA(clusterColorStr, 0.2),
  );

  // Title
  svg
    .append("text")
    .attr("x", margin.left + innerWidth / 2)
    .attr("y", 16)
    .attr("text-anchor", "middle")
    .attr("font-size", CHART_STYLES.titleFontSize)
    .attr("font-weight", "bold")
    .attr("fill", "#333")
    .text(METRIC_DISPLAY_NAMES[metric] || metric);

  // KS D statistic
  const ks = ksTest(inValues, outValues);
  const ksText = formatKsStat(ks);
  if (ksText) {
    svg
      .append("text")
      .attr("x", margin.left + innerWidth - 4)
      .attr("y", margin.top + 14)
      .attr("text-anchor", "end")
      .attr("font-size", 11)
      .attr("fill", isKsSignificant(ks) ? "#c33" : "#888")
      .text(ksText);
  }

  // Legend (line swatches)
  const legendG = svg
    .append("g")
    .attr("transform", `translate(${margin.left + 8},${margin.top + 4})`);

  legendG
    .append("line")
    .attr("x1", 0)
    .attr("y1", 5)
    .attr("x2", 14)
    .attr("y2", 5)
    .attr("stroke", clusterColorToRGBA(clusterColorStr, 1))
    .attr("stroke-width", 2);
  legendG
    .append("text")
    .attr("x", 18)
    .attr("y", 9)
    .attr("font-size", CHART_STYLES.legendFontSize)
    .attr("fill", CHART_STYLES.labelColor)
    .text(`${clusterLabel} (${inValues.length})`);

  legendG
    .append("line")
    .attr("x1", 0)
    .attr("y1", 19)
    .attr("x2", 14)
    .attr("y2", 19)
    .attr("stroke", "rgba(150, 150, 150, 0.7)")
    .attr("stroke-width", 2);
  legendG
    .append("text")
    .attr("x", 18)
    .attr("y", 23)
    .attr("font-size", CHART_STYLES.legendFontSize)
    .attr("fill", CHART_STYLES.labelColor)
    .text(`Other (${outValues.length})`);

  addExportButton(containerId, `ablation-histogram-${metric}`);
}

function renderBoxPlot(metric, inValues, outValues) {
  const containerId = `chart-container-${metric}`;
  const margin = ABLATION_CHART_MARGIN;
  const { svg, g, innerWidth, innerHeight } = createChartSVG(
    containerId,
    margin,
  );

  if (inValues.length === 0 && outValues.length === 0) return;

  const inStats = computeBoxplotStats(inValues);
  const outStats = computeBoxplotStats(outValues);

  // Compute Y domain from combined data
  const allVals = [...inValues, ...outValues];
  const yMin = Math.min(...allVals);
  const yMax = Math.max(...allVals);
  const yPad = (yMax - yMin) * 0.1 || 1;

  const yScale = d3
    .scaleLinear()
    .domain([yMin - yPad, yMax + yPad])
    .range([innerHeight, 0]);

  createYAxis(g, yScale, { ticks: 6 });

  const clusterColorStr =
    selectedCluster !== null && clusteringAvailable
      ? clustering.getClusterColor(selectedCluster)
      : "hsl(210, 70%, 50%)";
  const clusterLabel = getClusterChipLabel();

  const boxWidth = Math.min(innerWidth * 0.3, 60);
  const centerX = innerWidth / 2;
  const gap = 10;

  // In-cluster boxplot (left)
  if (inValues.length > 0) {
    drawBoxplot(g, {
      stats: inStats,
      x: centerX - boxWidth - gap / 2,
      width: boxWidth,
      yScale,
      fillColor: clusterColorToRGBA(clusterColorStr, 0.4),
      strokeColor: clusterColorToRGBA(clusterColorStr, 1),
    });
  }

  // Out-cluster boxplot (right)
  if (outValues.length > 0) {
    drawBoxplot(g, {
      stats: outStats,
      x: centerX + gap / 2,
      width: boxWidth,
      yScale,
      fillColor: "rgba(150, 150, 150, 0.25)",
      strokeColor: "rgba(150, 150, 150, 0.7)",
    });
  }

  // X-axis labels below boxes
  if (inValues.length > 0) {
    g.append("text")
      .attr("x", centerX - boxWidth / 2 - gap / 2)
      .attr("y", innerHeight + 20)
      .attr("text-anchor", "middle")
      .attr("font-size", CHART_STYLES.labelFontSize)
      .attr("fill", CHART_STYLES.labelColor)
      .text(clusterLabel);
  }
  if (outValues.length > 0) {
    g.append("text")
      .attr("x", centerX + boxWidth / 2 + gap / 2)
      .attr("y", innerHeight + 20)
      .attr("text-anchor", "middle")
      .attr("font-size", CHART_STYLES.labelFontSize)
      .attr("fill", CHART_STYLES.labelColor)
      .text("Other");
  }

  // Title
  svg
    .append("text")
    .attr("x", margin.left + innerWidth / 2)
    .attr("y", 16)
    .attr("text-anchor", "middle")
    .attr("font-size", CHART_STYLES.titleFontSize)
    .attr("font-weight", "bold")
    .attr("fill", "#333")
    .text(METRIC_DISPLAY_NAMES[metric] || metric);

  // KS D statistic
  const ks = ksTest(inValues, outValues);
  const ksText = formatKsStat(ks);
  if (ksText) {
    svg
      .append("text")
      .attr("x", margin.left + innerWidth - 4)
      .attr("y", margin.top + 14)
      .attr("text-anchor", "end")
      .attr("font-size", 11)
      .attr("fill", isKsSignificant(ks) ? "#c33" : "#888")
      .text(ksText);
  }

  // Legend
  const legendG = svg
    .append("g")
    .attr("transform", `translate(${margin.left + 8},${margin.top + 4})`);

  legendG
    .append("rect")
    .attr("width", 10)
    .attr("height", 10)
    .attr("rx", 2)
    .attr("fill", clusterColorToRGBA(clusterColorStr, 0.4));
  legendG
    .append("text")
    .attr("x", 14)
    .attr("y", 9)
    .attr("font-size", CHART_STYLES.legendFontSize)
    .attr("fill", CHART_STYLES.labelColor)
    .text(`${clusterLabel} (${inValues.length})`);

  legendG
    .append("rect")
    .attr("y", 14)
    .attr("width", 10)
    .attr("height", 10)
    .attr("rx", 2)
    .attr("fill", "rgba(150, 150, 150, 0.25)");
  legendG
    .append("text")
    .attr("x", 14)
    .attr("y", 23)
    .attr("font-size", CHART_STYLES.legendFontSize)
    .attr("fill", CHART_STYLES.labelColor)
    .text(`Other (${outValues.length})`);

  addExportButton(containerId, `ablation-boxplot-${metric}`);
}

// === Show-All mode renderers ===

/**
 * Apply hover highlight: emphasize the hovered group, dim all others.
 * @param {d3.Selection} svg - root SVG
 * @param {number|null} activeClusterId - cluster to highlight (null = reset)
 */
function applyHoverHighlight(svg, activeClusterId) {
  svg.selectAll("[data-cluster-id]").each(function () {
    const el = d3.select(this);
    const cid = parseInt(el.attr("data-cluster-id"));
    if (activeClusterId === null) {
      el.attr("opacity", 0.5);
    } else if (cid === activeClusterId) {
      el.attr("opacity", 1);
    } else {
      el.attr("opacity", 0.1);
    }
  });
}

function attachClusterHover(svg) {
  svg.selectAll("[data-cluster-id]").on("mouseenter", function () {
    const cid = parseInt(d3.select(this).attr("data-cluster-id"));
    applyHoverHighlight(svg, cid);
  });
  svg.on("mouseleave", () => applyHoverHighlight(svg, null));
}

function renderHistogramAll(metric, groupData) {
  const containerId = `chart-container-${metric}`;
  const margin = ABLATION_CHART_MARGIN;
  const { svg, g, innerWidth, innerHeight } = createChartSVG(
    containerId,
    margin,
  );

  // Compute shared domain from all groups
  const allValues = groupData.flatMap((cg) => cg.values);
  if (allValues.length === 0) return;

  const rawMin = Math.min(...allValues);
  const rawMax = Math.max(...allValues);
  const min = rawMin === rawMax ? rawMin - 0.5 : rawMin;
  const max = rawMin === rawMax ? rawMax + 0.5 : rawMax;
  const nBins = rawMin === rawMax ? 1 : 20;
  const binWidth = (max - min) / nBins;

  function histogram(values) {
    const counts = new Array(nBins).fill(0);
    for (const v of values) {
      let bin = Math.floor((v - min) / binWidth);
      if (bin >= nBins) bin = nBins - 1;
      if (bin < 0) bin = 0;
      counts[bin]++;
    }
    return counts;
  }

  // Compute frequencies for all groups and find max
  const groupFreqs = groupData.map((cg) => {
    const counts = histogram(cg.values);
    const total = cg.values.length || 1;
    return counts.map((c) => c / total);
  });
  const maxFreq = Math.max(...groupFreqs.flat(), 0.01);

  const xScale = d3.scaleLinear().domain([min, max]).range([0, innerWidth]);
  const epsilon = maxFreq * 0.005 || 0.001;
  const yScale = logScale
    ? d3
        .scaleLog()
        .domain([epsilon, maxFreq * 1.1])
        .range([innerHeight, 0])
        .clamp(true)
    : d3
        .scaleLinear()
        .domain([0, maxFreq * 1.1])
        .range([innerHeight, 0]);

  // Clamp zero-frequency bins for log scale
  if (logScale) {
    for (const freqs of groupFreqs) {
      for (let i = 0; i < freqs.length; i++)
        if (freqs[i] === 0) freqs[i] = epsilon;
    }
  }

  createXAxis(g, xScale, { height: innerHeight, label: metric, ticks: 6 });
  createYAxis(g, yScale, {
    label: logScale ? "Frequency (log)" : "Frequency",
    ticks: 5,
  });

  const binMidpoints = Array.from(
    { length: nBins },
    (_, i) => min + (i + 0.5) * binWidth,
  );

  // Draw one line+area per cluster
  for (let idx = 0; idx < groupData.length; idx++) {
    const cg = groupData[idx];
    const freqs = groupFreqs[idx];
    const strokeColor = clusterColorToRGBA(cg.color, 1);
    const fillColor = clusterColorToRGBA(cg.color, 0.15);

    const clusterG = g
      .append("g")
      .attr("data-cluster-id", cg.clusterId)
      .attr("opacity", 0.5)
      .style("cursor", "default");

    const areaGen = d3
      .area()
      .x((_, i) => xScale(binMidpoints[i]))
      .y0(innerHeight)
      .y1((d) => yScale(d))
      .curve(d3.curveMonotoneX);

    clusterG
      .append("path")
      .datum(freqs)
      .attr("d", areaGen)
      .attr("fill", fillColor)
      .attr("stroke", "none");

    const lineGen = d3
      .line()
      .x((_, i) => xScale(binMidpoints[i]))
      .y((d) => yScale(d))
      .curve(d3.curveMonotoneX);

    clusterG
      .append("path")
      .datum(freqs)
      .attr("d", lineGen)
      .attr("fill", "none")
      .attr("stroke", strokeColor)
      .attr("stroke-width", 2);

    // Invisible wider hit area for easier hover
    clusterG
      .append("path")
      .datum(freqs)
      .attr("d", lineGen)
      .attr("fill", "none")
      .attr("stroke", "transparent")
      .attr("stroke-width", 12);
  }

  // Pre-compute per-cluster KS stats (cluster vs all others)
  const clusterKsStats = {};
  for (const cg of groupData) {
    const otherValues = groupData
      .filter((other) => other.clusterId !== cg.clusterId)
      .flatMap((other) => other.values);
    clusterKsStats[cg.clusterId] = ksTest(cg.values, otherValues);
  }

  // Title
  svg
    .append("text")
    .attr("x", margin.left + innerWidth / 2)
    .attr("y", 16)
    .attr("text-anchor", "middle")
    .attr("font-size", CHART_STYLES.titleFontSize)
    .attr("font-weight", "bold")
    .attr("fill", "#333")
    .text(METRIC_DISPLAY_NAMES[metric] || metric);

  // KS D text element (shown on hover)
  const ksStatText = svg
    .append("text")
    .attr("x", margin.left + innerWidth - 4)
    .attr("y", margin.top + 14)
    .attr("text-anchor", "end")
    .attr("font-size", 11)
    .attr("opacity", 0);

  // Extended hover: also update KS D display
  svg.selectAll("[data-cluster-id]").on("mouseenter", function () {
    const cid = parseInt(d3.select(this).attr("data-cluster-id"));
    applyHoverHighlight(svg, cid);
    const ks = clusterKsStats[cid];
    const text = formatKsStat(ks);
    if (text) {
      ksStatText
        .text(text)
        .attr("fill", isKsSignificant(ks) ? "#c33" : "#888")
        .attr("opacity", 1);
    }
  });
  svg.on("mouseleave", () => {
    applyHoverHighlight(svg, null);
    ksStatText.attr("opacity", 0);
  });

  addExportButton(containerId, `ablation-histogram-all-${metric}`);
}

function renderBoxPlotAll(metric, groupData) {
  const containerId = `chart-container-${metric}`;
  const margin = { ...ABLATION_CHART_MARGIN, bottom: 60 };
  const { svg, g, innerWidth, innerHeight } = createChartSVG(
    containerId,
    margin,
  );

  if (groupData.length === 0) return;

  // Sort by mean if toggle is active
  if (sortByMean) {
    groupData = [...groupData].sort(
      (a, b) => d3.mean(a.values) - d3.mean(b.values),
    );
  }

  // Compute Y domain from all values
  const allValues = groupData.flatMap((cg) => cg.values);
  const yMin = Math.min(...allValues);
  const yMax = Math.max(...allValues);
  const yPad = (yMax - yMin) * 0.1 || 1;

  const yScale = d3
    .scaleLinear()
    .domain([yMin - yPad, yMax + yPad])
    .range([innerHeight, 0]);

  createYAxis(g, yScale, { ticks: 6 });

  // X positioning via scaleBand
  const xScale = d3
    .scaleBand()
    .domain(groupData.map((cg) => cg.clusterId))
    .range([0, innerWidth])
    .padding(0.2);

  const boxWidth = Math.min(xScale.bandwidth(), 50);

  for (const cg of groupData) {
    const stats = computeBoxplotStats(cg.values);
    const xPos = xScale(cg.clusterId) + (xScale.bandwidth() - boxWidth) / 2;
    const strokeColor = clusterColorToRGBA(cg.color, 1);
    const fillColor = clusterColorToRGBA(cg.color, 0.4);

    const clusterG = g
      .append("g")
      .attr("data-cluster-id", cg.clusterId)
      .attr("opacity", 0.5)
      .style("cursor", "default");

    drawBoxplot(clusterG, {
      stats,
      x: xPos,
      width: boxWidth,
      yScale,
      fillColor,
      strokeColor,
    });

    const xCenter = xScale(cg.clusterId) + xScale.bandwidth() / 2;

    // X-axis label: cluster name
    g.append("text")
      .attr("x", xCenter)
      .attr("y", innerHeight + 14)
      .attr("text-anchor", "middle")
      .attr("font-size", Math.min(CHART_STYLES.labelFontSize, 10))
      .attr("fill", CHART_STYLES.labelColor)
      .text(cg.label.length > 12 ? cg.label.slice(0, 11) + "\u2026" : cg.label);

    // X-axis label: KS D (cluster vs all others)
    const otherValues = groupData
      .filter((other) => other.clusterId !== cg.clusterId)
      .flatMap((other) => other.values);
    const ks = ksTest(cg.values, otherValues);
    const ksText = formatKsStat(ks);
    if (ksText) {
      g.append("text")
        .attr("x", xCenter)
        .attr("y", innerHeight + 26)
        .attr("text-anchor", "middle")
        .attr("font-size", 9)
        .attr("fill", isKsSignificant(ks) ? "#c33" : "#888")
        .text(ksText);
    }
  }

  // X-axis title
  g.append("text")
    .attr("x", innerWidth / 2)
    .attr("y", innerHeight + 46)
    .attr("text-anchor", "middle")
    .attr("font-size", CHART_STYLES.labelFontSize)
    .attr("fill", CHART_STYLES.labelColor)
    .text("Cluster ID");

  // Title
  svg
    .append("text")
    .attr("x", margin.left + innerWidth / 2)
    .attr("y", 16)
    .attr("text-anchor", "middle")
    .attr("font-size", CHART_STYLES.titleFontSize)
    .attr("font-weight", "bold")
    .attr("fill", "#333")
    .text(METRIC_DISPLAY_NAMES[metric] || metric);

  attachClusterHover(svg);
  addExportButton(containerId, `ablation-boxplot-all-${metric}`);
}

function getClusterChipLabel() {
  if (selectedCluster === null || !clusteringAvailable) return "Selected";
  const label = clustering.getClusterLabel(selectedCluster);
  if (label && label.name) return `Cluster ${selectedCluster}: ${label.name}`;
  return `Cluster ${selectedCluster}`;
}

/**
 * Convert an HSL/HSLA CSS string to rgba with given alpha.
 */
function clusterColorToRGBA(hslStr, alpha) {
  const m = hslStr.match(
    /hsla?\(([\d.]+),\s*([\d.]+)%,\s*([\d.]+)%(?:,\s*([\d.]+))?\)/,
  );
  if (!m) return `rgba(100, 100, 200, ${alpha})`;
  const h = parseFloat(m[1]) / 360;
  const s = parseFloat(m[2]) / 100;
  const l = parseFloat(m[3]) / 100;
  const a2 = s * Math.min(l, 1 - l);
  const f = (n) => {
    const k = (n + h * 12) % 12;
    return l - a2 * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  const r = Math.round(f(0) * 255);
  const g = Math.round(f(8) * 255);
  const b = Math.round(f(4) * 255);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// === Config Summary ===

function renderConfigSummary() {
  // Use first model's config
  const firstModel = Object.values(allData.models)[0];
  if (!firstModel || !firstModel.config) return;
  const cfg = firstModel.config;
  const items = [
    ["sequences", cfg.n_sequences],
    ["seq length", cfg.seq_length],
    ["repetitions", cfg.n_repetitions],
    ["methods", (cfg.ablation_methods || []).join(", ")],
    ["calibration", cfg.n_calibration_prompts],
    ["seed", cfg.seed],
    ["models", Object.keys(allData.models).length],
    [
      "total heads",
      new Set(
        Object.values(allData.models).flatMap((m) =>
          m.results.map((r) => r.head),
        ),
      ).size,
    ],
  ];
  document.getElementById("config-summary").innerHTML = items
    .map(
      ([k, v]) =>
        `<span class="config-item"><span class="config-key">${k}:</span> ${v}</span>`,
    )
    .join("");
}

// === Head Table ===

function getVisibleColumns() {
  const cols = ["head", "cluster"];
  if (hasAnyClassifications()) cols.push("classifications");
  cols.push("ablation_method");
  for (const [group, fields] of Object.entries(METRIC_GROUPS)) {
    if (visibleGroups.has(group)) cols.push(...fields);
  }
  return cols;
}

function hasAnyClassifications() {
  if (!apData || !apData.head_to_types) return false;
  const results = getFilteredResults();
  for (const r of results) {
    if (apData.head_to_types[r.head]?.length > 0) return true;
  }
  return false;
}

function buildTableColumns() {
  const visibleCols = getVisibleColumns();
  return visibleCols.map((col) => {
    const label = COLUMN_LABELS[col] || col;
    const base = { key: col, label };

    if (col === "head") {
      return {
        ...base,
        type: "string",
        renderer: (val) => {
          const encoded = String(val).replace(/:/g, "~");
          const a = document.createElement("a");
          a.href = `../vis/attnpedia/index.html?head_viewing=${encoded}`;
          a.target = "_blank";
          a.textContent = val;
          a.style.color = "#1565c0";
          a.style.textDecoration = "none";
          return a;
        },
      };
    }

    if (col === "cluster") {
      return {
        ...base,
        type: "string",
        id: "cluster",
        filterFunction: (filterValue) => {
          const ids = filterValue
            .split(",")
            .map((s) => parseInt(s.trim()))
            .filter((n) => !isNaN(n));
          if (ids.length === 0) return null;
          const idSet = new Set(ids);
          return (cellValue) => {
            const cid = parseInt(String(cellValue).split(":")[0]);
            return !isNaN(cid) && idSet.has(cid);
          };
        },
        renderer: (val, row) => {
          const cid = row._clusterId;
          if (cid === undefined || cid === null || cid === -1) return "\u2014";
          const color = clustering.getClusterColor(cid);
          const span = document.createElement("span");
          span.className = "cluster-badge";
          span.style.setProperty("--badge-color", color);
          span.textContent = val;
          return span;
        },
        sortFunction: (val, row) =>
          row._clusterId !== undefined ? row._clusterId : -999,
      };
    }

    if (col === "classifications") {
      return {
        ...base,
        type: "string",
        renderer: (val) => {
          if (!val) return "";
          const types = val.split(", ").filter((t) => t);
          if (types.length === 0) return "";
          const container = document.createElement("span");
          for (const t of types) {
            const badge = document.createElement("span");
            badge.className = "class-badge";
            badge.textContent = t;
            container.appendChild(badge);
            container.appendChild(document.createTextNode(" "));
          }
          return container;
        },
      };
    }

    if (col === "ablation_method") {
      return { ...base, type: "string" };
    }

    // Numeric metric column
    return {
      ...base,
      type: "number",
      align: "right",
      renderer: (val) => {
        const cls = getCellClass(col, val);
        const span = document.createElement("span");
        if (cls) span.className = cls;
        span.textContent = formatNumber(val);
        return span;
      },
    };
  });
}

function buildTableData() {
  let results = getFilteredResults();
  if (
    tableClusterFilter &&
    selectedCluster !== null &&
    clusteringAvailable &&
    clustering._is_loaded
  ) {
    results = results.filter(
      (r) => clustering._assignments[r.head] === selectedCluster,
    );
  }
  return results.map((r) => {
    const clusterId =
      clusteringAvailable && clustering._is_loaded
        ? clustering._assignments[r.head]
        : undefined;
    const classifications =
      apData && apData.head_to_types ? apData.head_to_types[r.head] || [] : [];

    // Build cluster display name
    let clusterName = "\u2014";
    if (clusterId !== undefined && clusterId !== null && clusterId !== -1) {
      const clLabel = clustering.getClusterLabel(clusterId);
      clusterName =
        clLabel && clLabel.name
          ? `${clusterId}: ${clLabel.name}`
          : `${clusterId}`;
    }

    return {
      ...r,
      _clusterId: clusterId,
      cluster: clusterName,
      classifications: classifications.join(", "),
    };
  });
}

function renderTable() {
  const container = document.getElementById("results-table-container");
  container.innerHTML = "";

  // Show/hide cluster filter toggle
  const clusterToggle = document.getElementById("cluster-filter-toggle");
  if (clusterToggle) {
    clusterToggle.style.display = selectedCluster !== null ? "" : "none";
    const cb = document.getElementById("table-cluster-filter");
    if (cb) cb.checked = tableClusterFilter;
  }

  const columns = buildTableColumns();
  const data = buildTableData();

  dataTable = new DataTable(container, {
    data,
    columns,
    pageSize: 50,
    showFilters: true,
    showInfo: true,
  });
}

// === Formatting ===

function formatNumber(val) {
  if (val == null) return "\u2014";
  if (val === 0) return "0";
  if (Math.abs(val) < 0.0001) return val.toExponential(2);
  return val.toFixed(4);
}

function getCellClass(column, value) {
  const spec = DELTA_COLUMNS[column];
  if (!spec || typeof value !== "number") return "";
  const abs = Math.abs(value);
  if (abs < spec.thresholds[0]) return "";
  const isStrong = abs >= spec.thresholds[1];
  const isPositive = value > 0;
  const isBad = spec.higher_is_worse ? isPositive : !isPositive;
  if (isBad) return isStrong ? "cell-bad-strong" : "cell-bad-mild";
  return isStrong ? "cell-good-strong" : "cell-good-mild";
}

// === Boot ===
document.addEventListener("DOMContentLoaded", init);
