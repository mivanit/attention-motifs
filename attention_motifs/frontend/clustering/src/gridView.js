/**
 * Grid View visualization for clustering
 *
 * Renders model grids with cells colored by cluster assignment.
 * Supports multi-selection of heads with pattern display in side pane.
 */

/**
 * Debounce a function call.
 * @param {Function} fn
 * @param {number} delay - Milliseconds
 * @returns {Function}
 */
function debounce(fn, delay) {
  let timer = null;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

let gridState = {
  linkage: null,
  clsValues: null,
  nHeads: 0,
  modelConfigs: {},
  selectedHeads: [],
  prompts: {},
  renderedHashes: null, // Set<string> or null (all rendered)
  scale: 1.0,
  patternsBaseUrl: "",
  tooltip: null,
  sortByCluster: false,
  patternSize: 100,
  promptIndices: {},
  logScale: false,
  currentAssignments: null,
  currentCutHeight: null,
  minClusterSize: 10,
  linkageMethod: "average",
  modelSizes: {},
  modelOrder: "data",
  originalModelOrder: [],
  modelDataFrame: null,
  clusterLabels: {}, // merged labels: cutHeightKey -> { clusterIdx: { name, desc, heads } }
  resolvedLabels: {}, // current cut height resolved: clusterId -> {name, desc}
  dendrogramFull: false,
  cutHeightControl: null, // shared cut-height control instance
  // Multi-method clustering
  currentMethod: "hierarchical", // "hierarchical" | "hdbscan" | "leiden"
  availableMethods: [], // methods with data available
  flatData: {}, // { method: { meta, partitions, labels } }
  currentParamKey: null, // current param key for flat methods
};

/**
 * Sync selected heads to URL search params (debounced).
 */
const syncSelectedHeadsToUrl = debounce(() => {
  const url = new URL(window.location);
  if (gridState.selectedHeads.length > 0) {
    url.searchParams.set("heads", gridState.selectedHeads.join(","));
  } else {
    url.searchParams.delete("heads");
  }
  history.replaceState(null, "", url);
}, 1000);

// =====================================================================
// Cluster Labels: persistence, matching, and resolution
// =====================================================================

const LABELS_STORAGE_KEY = "clustering_labels";

/**
 * Get localStorage key suffix for cut height
 * @param {number} cutHeight
 * @returns {string} e.g. "5.000"
 */
function getCutHeightKey(cutHeight) {
  return cutHeight.toFixed(3);
}

/**
 * Get the current label key based on the active clustering method.
 * Hierarchical: cutHeight.toFixed(3)
 * Flat methods: "{method}:{param_name}={value}"
 * @returns {string|null}
 */
function getCurrentLabelKey() {
  if (gridState.currentMethod === "hierarchical") {
    if (gridState.currentCutHeight === null) return null;
    return getCutHeightKey(gridState.currentCutHeight);
  }
  const flat = gridState.flatData[gridState.currentMethod];
  if (!flat || !gridState.currentParamKey) return null;
  return `${flat.meta.method}:${flat.meta.param_name}=${gridState.currentParamKey}`;
}

/**
 * Save current labels to localStorage
 */
function saveLabelsToLocalStorage() {
  try {
    localStorage.setItem(
      LABELS_STORAGE_KEY,
      JSON.stringify(gridState.clusterLabels),
    );
  } catch (e) {
    console.warn("Failed to save cluster labels to localStorage:", e);
  }
}

/**
 * Get labels object for the current clustering state
 * @returns {Object} Map of clusterIdx string -> { name, desc, heads }
 */
function getCurrentLabels() {
  const key = getCurrentLabelKey();
  if (!key) return {};
  return gridState.clusterLabels[key] || {};
}

/**
 * Update resolved labels for the current cut height and store in gridState.
 * Uses shared resolveClusterLabels() from cluster_utils.js.
 */
function updateResolvedLabels() {
  if (gridState.currentMethod === "hierarchical") {
    if (gridState.currentCutHeight === null) {
      gridState.resolvedLabels = {};
      return;
    }
    gridState.resolvedLabels = resolveClusterLabels(
      gridState.clusterLabels,
      gridState.currentCutHeight,
      window.CLUSTER_STATE.getAssignments(),
    );
  } else {
    const flat = gridState.flatData[gridState.currentMethod];
    if (flat && gridState.currentParamKey) {
      gridState.resolvedLabels = resolveClusterLabelsFlat(
        flat.labels,
        flat.meta.method,
        flat.meta.param_name,
        gridState.currentParamKey,
        window.CLUSTER_STATE.getAssignments(),
      );
    } else {
      gridState.resolvedLabels = {};
    }
  }
}

/**
 * Set a cluster label for the current cut height
 * @param {number} clusterId - Current cluster ID
 * @param {string|null} name - Short name (null/empty to clear)
 * @param {string|null} desc - Longer description (null/empty for none)
 */
function setClusterLabel(clusterId, name, desc) {
  const key = getCurrentLabelKey();
  if (!key) return;

  const trimName = name ? name.trim() : "";
  const trimDesc = desc ? desc.trim() : "";

  if (!trimName) {
    // Clear this cluster's label
    if (
      gridState.clusterLabels[key] &&
      gridState.clusterLabels[key][String(clusterId)]
    ) {
      gridState.clusterLabels[key][String(clusterId)].name = null;
      gridState.clusterLabels[key][String(clusterId)].desc = null;
      // If no cluster at this height has a name, remove the entire height
      const hasAny = Object.values(gridState.clusterLabels[key]).some(
        (e) => e && e.name,
      );
      if (!hasAny) {
        delete gridState.clusterLabels[key];
      }
    }
  } else {
    // Set the label with current heads snapshot
    if (!gridState.clusterLabels[key]) {
      gridState.clusterLabels[key] = {};
    }
    const heads = window.CLUSTER_STATE.getHeadsInCluster(clusterId);
    gridState.clusterLabels[key][String(clusterId)] = {
      name: trimName,
      desc: trimDesc || null,
      heads: heads,
    };

    // Populate all other clusters at this height with null name/desc
    const allClusterIds = Object.keys(window.CLUSTER_STATE.getClusterSizes());
    for (const cid of allClusterIds) {
      if (!(cid in gridState.clusterLabels[key])) {
        gridState.clusterLabels[key][cid] = {
          name: null,
          desc: null,
          heads: window.CLUSTER_STATE.getHeadsInCluster(parseInt(cid)),
        };
      }
    }
  }

  saveLabelsToLocalStorage();
  updateResolvedLabels();
  renderSliderTicks();
}

/**
 * Export all cluster labels as JSON download
 */
function exportClusterLabels() {
  const blob = new Blob([JSON.stringify(gridState.clusterLabels, null, "\t")], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "cluster_labels.json";
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Get all cut height keys that have at least one label
 * @returns {number[]} Sorted array of cut heights with labels
 */
function getLabeledCutHeights() {
  return Object.keys(gridState.clusterLabels)
    .filter((key) => {
      const entries = gridState.clusterLabels[key];
      if (!entries) return false;
      // Require at least one entry with a non-null name
      return Object.values(entries).some((e) => e && e.name);
    })
    .map((key) => parseFloat(key))
    .sort((a, b) => a - b);
}

/**
 * Render tick marks on the cut-height slider for labeled cut heights
 */
function renderSliderTicks() {
  if (!gridState.cutHeightControl) return;
  const ticksContainer = gridState.cutHeightControl.getTicksContainer();
  if (!ticksContainer) return;

  ticksContainer.innerHTML = "";
  const slider = gridState.cutHeightControl.getSliderElement();
  const maxHeight = parseFloat(slider.max) || 10;

  const labeledHeights = getLabeledCutHeights();
  for (const height of labeledHeights) {
    const pct = (height / maxHeight) * 100;
    const tick = document.createElement("div");
    tick.className = "slider-tick";
    tick.style.left = `${pct}%`;
    tick.title = `${height.toFixed(3)} (labeled)`;
    tick.addEventListener("click", () => {
      gridState.cutHeightControl.setValue(height);
      updateClustersByHeight(height);
    });
    ticksContainer.appendChild(tick);
  }
}

// =====================================================================

/**
 * Generate single pattern viewer URL for a specific head and prompt
 * @param {string} headId - Head ID in format "model:Llayer:Hhead"
 * @param {string} promptHash - The prompt hash
 * @returns {string} URL to single pattern viewer
 */
function getPatternViewerUrl(headId, promptHash) {
  const parts = headId.split(":");
  const model = parts[0];
  const layer = parts[1].substring(1); // Remove "L" prefix
  const head = parts[2].substring(1); // Remove "H" prefix

  // Format: single.html?prompt={hash}&head={model}.L{layer}.H{head}
  return `../../patterns/single.html?prompt=${promptHash}&head=${model}.L${layer}.H${head}`;
}

/**
 * Parse model sizes from TransformerLens CSV
 * @param {string} csvText - Raw CSV content
 * @returns {Object.<string, number>} Map of model name to n_params
 */
function parseModelSizes(csvText) {
  const sizes = {};
  const lines = csvText.trim().split("\n");

  // Find column indices from header
  const header = lines[0].split(",");
  const nameIdx = header.findIndex((h) => h.includes("default_alias"));
  const paramsIdx = header.findIndex((h) => h === "n_params.as_int");

  if (nameIdx === -1 || paramsIdx === -1) {
    console.warn("Could not find expected columns in CSV");
    return sizes;
  }

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",");
    const name = cols[nameIdx]?.trim();
    const params = parseInt(cols[paramsIdx]);
    if (name && !isNaN(params)) {
      sizes[name] = params;
    }
  }

  return sizes;
}

/**
 * Extract model family/type from model name
 * @param {string} modelName - e.g., "gpt2-small", "pythia-70m"
 * @returns {string} Model family - e.g., "gpt2", "pythia"
 */
function getModelFamily(modelName) {
  const parts = modelName.split("-");
  if (parts.length > 1) {
    return parts.slice(0, -1).join("-");
  }
  return modelName;
}

/**
 * Get model names in current sort order
 * @returns {string[]} Sorted model names
 */
function getSortedModelNames() {
  const models = gridState.originalModelOrder;

  if (gridState.modelOrder === "data") {
    return models;
  }

  if (gridState.modelOrder === "size") {
    return [...models].sort((a, b) => {
      const sizeA = gridState.modelSizes[a] ?? Infinity;
      const sizeB = gridState.modelSizes[b] ?? Infinity;
      return sizeA - sizeB;
    });
  }

  if (gridState.modelOrder === "family") {
    // Group by family, sort families alphabetically, sort by size within family
    const byFamily = {};
    for (const model of models) {
      const family = getModelFamily(model);
      if (!byFamily[family]) byFamily[family] = [];
      byFamily[family].push(model);
    }

    // Sort each family by size
    for (const family of Object.keys(byFamily)) {
      byFamily[family].sort((a, b) => {
        const sizeA = gridState.modelSizes[a] ?? Infinity;
        const sizeB = gridState.modelSizes[b] ?? Infinity;
        return sizeA - sizeB;
      });
    }

    // Sort families alphabetically and flatten
    const sortedFamilies = Object.keys(byFamily).sort();
    return sortedFamilies.flatMap((f) => byFamily[f]);
  }

  return models;
}

/**
 * Initialize the grid view visualization
 * @param {Object} config - Configuration object
 */
async function initGridView(config) {
  const container = document.getElementById("model-grids-container");
  container.innerHTML = '<div class="loading">Loading clustering data...</div>';

  gridState.patternsBaseUrl = config.patternsBaseUrl;

  try {
    // Load all data in parallel
    const [metaResponse, linkageResponse, modelsText] = await Promise.all([
      fetch(config.clusteringMetaUrl),
      fetch(config.linkageUrl),
      fetch(config.modelsUrl).then((r) => r.text()),
    ]);

    if (!metaResponse.ok || !linkageResponse.ok) {
      throw new Error("Failed to load clustering data");
    }

    const meta = await metaResponse.json();
    const linkage = await linkageResponse.json();

    gridState.clsValues = meta.cls_values;
    gridState.linkage = linkage;
    gridState.nHeads = meta.cls_values.length;

    // Parse model configs from JSONL
    gridState.modelConfigs = {};
    for (const line of modelsText.trim().split("\n")) {
      if (line.trim()) {
        const cfg = JSON.parse(line);
        gridState.modelConfigs[cfg.sanitized_name || cfg.model_name] = {
          n_layers: cfg.n_layers,
          n_heads: cfg.n_heads,
        };
      }
    }

    // Store original model order, filtering out models not in clustering data
    const modelsInClustering = new Set(
      gridState.clsValues.map((headId) => headId.split(":")[0]),
    );
    gridState.originalModelOrder = Object.keys(gridState.modelConfigs).filter(
      (m) => modelsInClustering.has(m),
    );
    const filteredConfigs = {};
    for (const m of gridState.originalModelOrder) {
      if (gridState.modelConfigs[m])
        filteredConfigs[m] = gridState.modelConfigs[m];
    }
    gridState.modelConfigs = filteredConfigs;

    // Fetch model data using DataFrame
    try {
      const csvUrl =
        "https://raw.githubusercontent.com/mivanit/transformerlens-model-table/refs/heads/main/docs/model_table.csv";
      const csvResponse = await fetch(csvUrl);
      const csvText = await csvResponse.text();
      gridState.modelDataFrame = DataFrame.from_csv(csvText);

      // Extract model sizes
      gridState.modelSizes = {};
      for (let i = 0; i < gridState.modelDataFrame.length; i++) {
        const row = gridState.modelDataFrame.row(i);
        const name = row["name.default_alias"];
        const params = row["n_params.as_int"];
        if (name && params) {
          gridState.modelSizes[name] = params;
        }
      }
    } catch (e) {
      console.warn("Failed to load model data:", e);
    }

    // Load rendered prompts filter (if available)
    try {
      const renderedUrl = `${config.patternsBaseUrl}/rendered_prompts.jsonl`;
      const renderedResp = await fetch(renderedUrl);
      if (renderedResp.ok) {
        const text = await renderedResp.text();
        const hashes = new Set();
        for (const line of text.trim().split("\n")) {
          if (!line.trim()) continue;
          try {
            const prompt = JSON.parse(line);
            if (prompt.hash) hashes.add(prompt.hash);
          } catch (e) {
            // skip malformed lines
          }
        }
        if (hashes.size > 0) {
          gridState.renderedHashes = hashes;
          console.log(
            `Clustering page: filtered to ${hashes.size} rendered prompts`,
          );
        }
      }
    } catch (e) {
      console.warn("rendered_prompts.jsonl not available for clustering page");
    }

    // Load cluster labels (server + localStorage, merged by shared util)
    const labelsUrl =
      config.clusterLabelsUrl ||
      "../../features/clustering/cluster_labels.json";
    gridState.clusterLabels = await loadClusterLabels(labelsUrl);

    // Detect available methods from manifest
    gridState.availableMethods = ["hierarchical"]; // hierarchical always if linkage loaded
    try {
      const methodsUrl = "../../features/clustering_methods.json";
      const mResp = await fetch(methodsUrl);
      if (mResp.ok) {
        const manifest = await mResp.json();
        gridState.availableMethods = manifest.methods || ["hierarchical"];
      }
    } catch (e) {
      // Fall back to hierarchical only
    }

    // Load flat clustering data (HDBSCAN, Leiden)
    for (const method of ["hdbscan", "leiden"]) {
      if (!gridState.availableMethods.includes(method)) continue;
      try {
        const flatMetaUrl = `../../features/clustering_${method}/clustering_meta.json`;
        const flatPartitionsUrl = `../../features/clustering_${method}/partitions.json`;
        const flatLabelsUrl = `../../features/clustering_${method}/cluster_labels.json`;
        const [fMetaResp, fPartResp] = await Promise.all([
          fetch(flatMetaUrl),
          fetch(flatPartitionsUrl),
        ]);
        if (fMetaResp.ok && fPartResp.ok) {
          const fMeta = await fMetaResp.json();
          const fPartitions = await fPartResp.json();
          const fLabels = await loadClusterLabels(flatLabelsUrl);
          gridState.flatData[method] = {
            meta: fMeta,
            partitions: fPartitions,
            labels: fLabels,
          };
          // Merge flat labels into clusterLabels for unified storage
          for (const [k, v] of Object.entries(fLabels)) {
            if (!gridState.clusterLabels[k]) {
              gridState.clusterLabels[k] = v;
            }
          }
        }
      } catch (e) {
        console.warn(`Failed to load ${method} clustering data:`, e);
      }
    }

    // Determine initial method and parameters
    const savedMethod = ClusteringConfig.getMethod();
    if (
      savedMethod &&
      (savedMethod === "hierarchical" ||
        gridState.flatData[savedMethod] !== undefined)
    ) {
      gridState.currentMethod = savedMethod;
    }

    // Determine initial cut height: shared config > labels > default
    let initialCutHeight = 5;
    const savedCutHeight = ClusteringConfig.getCutHeight();
    if (savedCutHeight !== null) {
      initialCutHeight = savedCutHeight;
    } else {
      const labeledHeights = getLabeledCutHeights();
      if (labeledHeights.length === 1) {
        initialCutHeight = labeledHeights[0];
      } else if (labeledHeights.length > 1) {
        initialCutHeight = labeledHeights[labeledHeights.length - 1]; // largest
      }
    }

    // Set up controls
    setupControls();

    // Set up tooltip
    setupTooltip();

    // Set up side pane
    setupSidePane();

    // Set up resizable divider
    setupResizableDivider();

    // Set up help tooltip with model data
    setupHelpTooltip();

    // Initial render
    const initialMinClusterSize = gridState.minClusterSize;
    document.getElementById("min-cluster-size").value = initialMinClusterSize;
    document.getElementById("min-cluster-size-input").value =
      initialMinClusterSize;

    if (gridState.currentMethod === "hierarchical") {
      gridState.cutHeightControl.setValue(initialCutHeight);
      updateClustersByHeight(initialCutHeight);
    } else {
      const flat = gridState.flatData[gridState.currentMethod];
      const savedParamKey = ClusteringConfig.getParamKey();
      const paramKey =
        savedParamKey && flat.meta.param_keys.includes(savedParamKey)
          ? savedParamKey
          : flat.meta.param_keys[0];
      // Update param dropdown
      const paramSelect = document.getElementById("clustering-param-select");
      if (paramSelect) paramSelect.value = paramKey;
      updateClustersByParam(paramKey);
    }

    // Restore selected heads from URL
    const initUrlParams = new URLSearchParams(window.location.search);
    const headsParam = initUrlParams.get("heads");
    if (headsParam) {
      const validHeads = new Set(gridState.clsValues);
      gridState.selectedHeads = headsParam
        .split(",")
        .filter((h) => h.trim() && validHeads.has(h));
      if (gridState.selectedHeads.length > 0) {
        updateSelectedCells();
        updateSidePane();
      }
    }

    // Handle highlight cluster from shared config (e.g. navigated from cluster_trends)
    const highlightId = ClusteringConfig.getHighlightCluster();
    if (highlightId !== null) {
      ClusteringConfig.clearHighlightCluster();
      selectCluster(highlightId);
    }
  } catch (error) {
    console.error("Error initializing grid view:", error);
    container.innerHTML = `<div class="error">Error loading data: ${error.message}</div>`;
  }
}

/**
 * Set up control elements
 */
function setupControls() {
  const minClusterSizeSlider = document.getElementById("min-cluster-size");
  const minClusterSizeInput = document.getElementById("min-cluster-size-input");
  const scaleSlider = document.getElementById("scale");
  const scaleValue = document.getElementById("scale-value");
  const exportBtn = document.getElementById("export-pattern-types");

  // --- Method selector ---
  const methodSelect = document.getElementById("clustering-method-select");
  const hierControls = document.getElementById("hierarchical-controls");
  const paramControls = document.getElementById("flat-param-controls");
  const paramSelect = document.getElementById("clustering-param-select");
  const paramLabel = document.getElementById("clustering-param-label");

  if (methodSelect) {
    methodSelect.innerHTML = "";
    const methodsWithData = gridState.availableMethods.filter(
      (m) => m === "hierarchical" || gridState.flatData[m] !== undefined,
    );
    for (const m of methodsWithData) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      methodSelect.appendChild(opt);
    }
    methodSelect.value = gridState.currentMethod;

    // Show/hide method-specific controls
    function updateMethodControls() {
      const isHier = gridState.currentMethod === "hierarchical";
      if (hierControls) hierControls.style.display = isHier ? "" : "none";
      if (paramControls) paramControls.style.display = isHier ? "none" : "";

      if (!isHier && paramSelect) {
        const flat = gridState.flatData[gridState.currentMethod];
        if (flat) {
          if (paramLabel) paramLabel.textContent = flat.meta.param_name + ":";
          paramSelect.innerHTML = "";
          for (const pk of flat.meta.param_keys) {
            const opt = document.createElement("option");
            opt.value = pk;
            const meta = flat.meta.partition_meta[pk];
            const extra = meta
              ? ` (${meta.n_clusters}cl${meta.n_outliers ? `, ${meta.n_outliers}out` : ""})`
              : "";
            opt.textContent = pk + extra;
            paramSelect.appendChild(opt);
          }
          const savedParamKey = ClusteringConfig.getParamKey();
          paramSelect.value =
            savedParamKey && flat.meta.param_keys.includes(savedParamKey)
              ? savedParamKey
              : flat.meta.param_keys[0];
        }
      }
    }

    updateMethodControls();

    methodSelect.addEventListener("change", () => {
      gridState.currentMethod = methodSelect.value;
      ClusteringConfig.setMethod(methodSelect.value);
      updateMethodControls();
      reapplyCurrentClustering();
    });

    if (paramSelect) {
      paramSelect.addEventListener("change", () => {
        updateClustersByParam(paramSelect.value);
      });
    }
  }

  // Get max height from linkage, capped at 10
  const maxHeight = Math.min(
    Math.max(...gridState.linkage.map((row) => row[2])),
    10,
  );

  // Create shared cut-height control (with tick marks for labeled heights)
  gridState.cutHeightControl = createCutHeightControl({
    container: document.getElementById("cut-height-container"),
    maxHeight,
    step: maxHeight / 1000,
    initialValue: 5,
    label: "Cut Height:",
    showTicks: true,
    onChange: (h) => updateClustersByHeight(h),
  });

  // min-cluster-size slider and input (bidirectional)
  minClusterSizeSlider.addEventListener("input", (e) => {
    const size = parseInt(e.target.value);
    minClusterSizeInput.value = size;
    gridState.minClusterSize = size;
    reapplyCurrentClustering();
  });
  minClusterSizeInput.addEventListener("change", (e) => {
    const size = Math.max(0, parseInt(e.target.value) || 0);
    minClusterSizeInput.value = size;
    minClusterSizeSlider.value = size;
    gridState.minClusterSize = size;
    reapplyCurrentClustering();
  });

  // Scale slider
  scaleSlider.addEventListener("input", (e) => {
    gridState.scale = parseFloat(e.target.value);
    scaleValue.textContent = gridState.scale.toFixed(1);
    document
      .getElementById("model-grids-container")
      .style.setProperty("--scale", gridState.scale);
  });

  // Sort toggle button
  const sortToggle = document.getElementById("sort-toggle");
  sortToggle.addEventListener("click", () => {
    gridState.sortByCluster = !gridState.sortByCluster;
    sortToggle.textContent = gridState.sortByCluster
      ? "By Cluster"
      : "By Index";
    sortToggle.classList.toggle("active", gridState.sortByCluster);
    renderModelGrids();
  });

  // Model order dropdown
  document.getElementById("model-order").addEventListener("change", (e) => {
    gridState.modelOrder = e.target.value;
    renderModelGrids();
  });

  // Export pattern types button
  exportBtn.addEventListener("click", exportPatternTypes);
}

/**
 * Set up tooltip element
 */
function setupTooltip() {
  const tooltip = document.createElement("div");
  tooltip.className = "tooltip";
  tooltip.style.display = "none";
  document.body.appendChild(tooltip);
  gridState.tooltip = tooltip;
}

/**
 * Set up side pane interactions
 */
function setupSidePane() {
  document.getElementById("clear-selection").addEventListener("click", () => {
    gridState.selectedHeads = [];
    updateSelectedCells();
    updateSidePane();
  });

  // Pattern size slider
  document.getElementById("pattern-size").addEventListener("input", (e) => {
    gridState.patternSize = parseInt(e.target.value);
    document.querySelectorAll(".pattern-image").forEach((img) => {
      img.style.width = gridState.patternSize + "px";
      img.style.height = gridState.patternSize + "px";
    });
  });

  // Randomize prompts button
  document.getElementById("randomize-prompts").addEventListener("click", () => {
    gridState.promptIndices = {};
    updateSidePane();
  });

  // Initial empty state
  updateSidePane();
}

/**
 * Set up resizable divider for split layout
 */
function setupResizableDivider() {
  const divider = document.getElementById("divider");
  const rightPane = document.getElementById("right-pane");
  let isDragging = false;

  divider.addEventListener("mousedown", (e) => {
    isDragging = true;
    divider.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const containerRect = divider.parentElement.getBoundingClientRect();
    const newRightWidth = containerRect.right - e.clientX;
    const clampedWidth = Math.max(300, Math.min(800, newRightWidth));
    rightPane.style.width = clampedWidth + "px";
  });

  document.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      divider.classList.remove("dragging");
      document.body.style.cursor = "";
    }
  });
}

/**
 * Set up help tooltip showing model data as YAML
 */
function setupHelpTooltip() {
  const helpIcon = document.getElementById("model-data-help");
  if (!helpIcon || !gridState.modelDataFrame) return;

  // Get data for models we have
  const modelData = {};
  for (const modelName of gridState.originalModelOrder) {
    // Find row in DataFrame
    for (let i = 0; i < gridState.modelDataFrame.length; i++) {
      const row = gridState.modelDataFrame.row(i);
      if (row["name.default_alias"] === modelName) {
        modelData[modelName] = {
          n_params: row["n_params.as_str"],
          n_layers: row["cfg.n_layers"],
          n_heads: row["cfg.n_heads"],
          d_model: row["cfg.d_model"],
        };
        break;
      }
    }
  }

  const tooltip = document.createElement("div");
  tooltip.className = "help-tooltip";
  tooltip.textContent = toYAML(modelData);
  helpIcon.appendChild(tooltip);
}

/**
 * Get YAML string with model data for a specific model
 * @param {string} modelName - Model name to look up
 * @returns {string|null} YAML string or null if not found
 */
function getModelDataYaml(modelName) {
  if (!gridState.modelDataFrame) return null;

  for (let i = 0; i < gridState.modelDataFrame.length; i++) {
    const row = gridState.modelDataFrame.row(i);
    if (row["name.default_alias"] === modelName) {
      return toYAML({
        n_params: row["n_params.as_str"],
        n_layers: row["cfg.n_layers"],
        n_heads: row["cfg.n_heads"],
        d_model: row["cfg.d_model"],
      });
    }
  }
  return null;
}

/**
 * Reapply current clustering with updated min-cluster-size
 */
function reapplyCurrentClustering() {
  if (gridState.currentMethod === "hierarchical") {
    if (gridState.currentCutHeight !== null) {
      updateClustersByHeight(gridState.currentCutHeight);
    }
  } else {
    if (gridState.currentParamKey !== null) {
      updateClustersByParam(gridState.currentParamKey);
    }
  }
}

/**
 * Update visualization for a flat clustering method parameter.
 * @param {string} paramKey - The parameter value key
 */
function updateClustersByParam(paramKey) {
  const flat = gridState.flatData[gridState.currentMethod];
  if (!flat) return;

  gridState.currentParamKey = paramKey;
  ClusteringConfig.setParamKey(paramKey);

  const rawAssignments = getFlatPartitionAssignments(flat.partitions, paramKey);
  const {
    assignments,
    smallClusters,
    nClusters: finalNClusters,
  } = applyMinSizeFilter(rawAssignments, gridState.minClusterSize);

  window.CLUSTER_STATE.setAssignments(assignments, finalNClusters);
  renderModelGrids();
  updateStats(assignments, finalNClusters, smallClusters);
}

/**
 * Update visualization for a given cut height.
 * Uses shared computeClustersByHeight() and applyMinSizeFilter() from cluster_engine.js.
 * @param {number} cutHeight - Height at which to cut
 */
function updateClustersByHeight(cutHeight) {
  gridState.currentCutHeight = cutHeight;
  ClusteringConfig.setCutHeight(cutHeight);

  const rawAssignments = computeClustersByHeight(
    gridState.linkage,
    gridState.clsValues,
    cutHeight,
  );
  const {
    assignments,
    smallClusters,
    nClusters: finalNClusters,
  } = applyMinSizeFilter(rawAssignments, gridState.minClusterSize);

  window.CLUSTER_STATE.setAssignments(assignments, finalNClusters);
  renderModelGrids();
  updateStats(assignments, finalNClusters, smallClusters);
}

/**
 * Export pattern types JSON
 */
function exportPatternTypes() {
  const assignments = window.CLUSTER_STATE.getAssignments();
  const nClusters = window.CLUSTER_STATE.getNumClusters();

  // Build cluster sizes
  const clusterSizes = {};
  for (const cid of Object.values(assignments)) {
    const key = String(cid);
    clusterSizes[key] = (clusterSizes[key] || 0) + 1;
  }

  // Build types array
  const clusterIds = [...new Set(Object.values(assignments))].sort(
    (a, b) => a - b,
  );
  const types = clusterIds.map((id) => ({
    id: id,
    name: id === -1 ? "misc" : "none",
    description: id === -1 ? "Merged from small clusters" : "none",
  }));

  const patternTypes = {
    meta: {
      cut_height: gridState.currentCutHeight,
      n_clusters: nClusters,
      linkage_method: gridState.linkageMethod,
      clustering_path: "browser-export",
      created_at: new Date().toISOString(),
      min_cluster_size: gridState.minClusterSize,
    },
    stats: {
      n_heads: Object.keys(assignments).length,
      cluster_sizes: clusterSizes,
    },
    types: types,
    assignments: assignments,
    selection: {
      heads: gridState.selectedHeads,
    },
  };

  // Trigger download
  const blob = new Blob([JSON.stringify(patternTypes, null, "\t")], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "pattern_types.json";
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Render all model grids
 */
function renderModelGrids() {
  const container = document.getElementById("model-grids-container");
  container.innerHTML = "";
  container.style.setProperty("--scale", gridState.scale);

  if (gridState.modelOrder === "family") {
    // Group by family and render each family as a row
    const byFamily = {};
    for (const model of gridState.originalModelOrder) {
      const family = getModelFamily(model);
      if (!byFamily[family]) byFamily[family] = [];
      byFamily[family].push(model);
    }

    // Sort each family by size
    for (const family of Object.keys(byFamily)) {
      byFamily[family].sort((a, b) => {
        const sizeA = gridState.modelSizes[a] ?? Infinity;
        const sizeB = gridState.modelSizes[b] ?? Infinity;
        return sizeA - sizeB;
      });
    }

    // Render each family on its own row
    const sortedFamilies = Object.keys(byFamily).sort();
    for (const family of sortedFamilies) {
      const familyRow = document.createElement("div");
      familyRow.className = "family-row";

      const label = document.createElement("div");
      label.className = "family-label";
      label.textContent = family;
      familyRow.appendChild(label);

      for (const modelName of byFamily[family]) {
        renderModelBox(familyRow, modelName);
      }

      container.appendChild(familyRow);
    }
  } else {
    for (const modelName of getSortedModelNames()) {
      renderModelBox(container, modelName);
    }
  }
}

/**
 * Render a single model box
 * @param {HTMLElement} container - Parent container
 * @param {string} modelName - Model name
 */
function renderModelBox(container, modelName) {
  const config = gridState.modelConfigs[modelName];
  const { n_layers, n_heads } = config;

  const box = document.createElement("div");
  box.className = "model-box";

  // Header
  const header = document.createElement("div");
  header.className = "model-box-header";

  const headerText = document.createElement("span");
  headerText.textContent = modelName;
  header.appendChild(headerText);

  // Add model-specific help icon
  const modelYaml = getModelDataYaml(modelName);
  if (modelYaml) {
    const helpIcon = document.createElement("span");
    helpIcon.className = "model-help-icon";
    helpIcon.textContent = "❓";

    const tooltip = document.createElement("div");
    tooltip.className = "help-tooltip";
    tooltip.textContent = modelYaml;
    helpIcon.appendChild(tooltip);

    header.appendChild(helpIcon);
  }

  box.appendChild(header);

  // Grid wrapper (layer labels + grid content)
  const wrapper = document.createElement("div");
  wrapper.className = "model-grid-wrapper";

  // Layer labels
  const layerLabels = document.createElement("div");
  layerLabels.className = "layer-labels";
  for (let l = 0; l < n_layers; l++) {
    const label = document.createElement("div");
    label.className = "layer-label";
    label.textContent = `L${l}`;
    layerLabels.appendChild(label);
  }
  wrapper.appendChild(layerLabels);

  // Grid content (head labels + grid)
  const gridContent = document.createElement("div");
  gridContent.className = "grid-content";

  // Head labels - only show in index mode
  const headLabels = document.createElement("div");
  headLabels.className = "head-labels";
  if (!gridState.sortByCluster) {
    for (let h = 0; h < n_heads; h++) {
      const label = document.createElement("div");
      label.className = "head-label";
      label.textContent = `${h}`;
      headLabels.appendChild(label);
    }
  }
  gridContent.appendChild(headLabels);

  // Grid
  const grid = document.createElement("div");
  grid.className = "model-grid";

  for (let l = 0; l < n_layers; l++) {
    const row = document.createElement("div");
    row.className = "grid-row";

    // Get head indices - sorted by cluster size if in cluster mode
    let headIndices = Array.from({ length: n_heads }, (_, i) => i);
    if (gridState.sortByCluster) {
      const clusterSizes = window.CLUSTER_STATE.getClusterSizes();
      headIndices.sort((a, b) => {
        const clusterA =
          window.CLUSTER_STATE.getClusterId(`${modelName}:L${l}:H${a}`) ?? 999;
        const clusterB =
          window.CLUSTER_STATE.getClusterId(`${modelName}:L${l}:H${b}`) ?? 999;
        // Sort by cluster size (descending), then by cluster ID for ties
        const sizeA = clusterSizes[clusterA] ?? 0;
        const sizeB = clusterSizes[clusterB] ?? 0;
        if (sizeA !== sizeB) return sizeB - sizeA; // Larger clusters first
        return clusterA - clusterB; // Tie-breaker by cluster ID
      });
    }

    for (const h of headIndices) {
      const headId = `${modelName}:L${l}:H${h}`;
      const cell = document.createElement("div");
      cell.className = "grid-cell";
      cell.dataset.headId = headId;

      // Set color from cluster state
      const color = window.CLUSTER_STATE.getColor(headId);
      cell.style.backgroundColor = color;

      // Check if selected
      if (gridState.selectedHeads.includes(headId)) {
        cell.classList.add("selected");
      }

      // Click handler - shift-click selects cluster across all models,
      // ctrl-click selects cluster in current model only
      cell.addEventListener("click", (e) => {
        if (e.ctrlKey || e.metaKey) {
          const clusterId = window.CLUSTER_STATE.getClusterId(headId);
          if (clusterId !== undefined) {
            selectClusterForModel(clusterId, modelName);
          }
        } else if (e.shiftKey) {
          const clusterId = window.CLUSTER_STATE.getClusterId(headId);
          if (clusterId !== undefined) {
            selectCluster(clusterId);
          }
        } else {
          toggleHeadSelection(headId);
        }
      });

      // Tooltip handlers
      cell.addEventListener("mouseenter", (e) => showTooltip(e, headId));
      cell.addEventListener("mousemove", (e) => moveTooltip(e));
      cell.addEventListener("mouseleave", () => hideTooltip());

      row.appendChild(cell);
    }

    grid.appendChild(row);
  }

  gridContent.appendChild(grid);
  wrapper.appendChild(gridContent);
  box.appendChild(wrapper);
  container.appendChild(box);
}

/**
 * Toggle head selection
 * @param {string} headId - Head ID to toggle
 */
function toggleHeadSelection(headId) {
  const idx = gridState.selectedHeads.indexOf(headId);
  if (idx >= 0) {
    gridState.selectedHeads.splice(idx, 1);
  } else {
    gridState.selectedHeads.push(headId);
  }
  updateSelectedCells();
  updateSidePane();
}

/**
 * Update selected cell styling
 */
function updateSelectedCells() {
  document.querySelectorAll(".grid-cell").forEach((cell) => {
    const headId = cell.dataset.headId;
    if (gridState.selectedHeads.includes(headId)) {
      cell.classList.add("selected");
    } else {
      cell.classList.remove("selected");
    }
  });

  // Update count
  document.getElementById("selected-count").textContent =
    `(${gridState.selectedHeads.length})`;

  // Sync selection to URL (debounced) and update cluster labels
  syncSelectedHeadsToUrl();
  renderClusterLabels();
}

/**
 * Shuffle array in place using Fisher-Yates algorithm
 */
function shuffleArray(array) {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/**
 * Update side pane with selected heads' patterns
 */
async function updateSidePane() {
  const patternImages = document.getElementById("pattern-images");
  const selectedCount = document.getElementById("selected-count");

  selectedCount.textContent = `(${gridState.selectedHeads.length})`;

  if (gridState.selectedHeads.length === 0) {
    patternImages.innerHTML =
      '<div class="empty-state">Click on cells to select heads and view their attention patterns. Shift-click to select all heads in a cluster.</div>';
    return;
  }

  patternImages.innerHTML = "";

  for (const headId of gridState.selectedHeads) {
    const section = document.createElement("div");
    section.className = "pattern-section";

    // Header with color swatch, clickable title, and arrow link
    const header = document.createElement("div");
    header.className = "pattern-section-header";

    const colorSwatch = document.createElement("div");
    colorSwatch.className = "pattern-section-color";
    colorSwatch.style.backgroundColor = window.CLUSTER_STATE.getColor(headId);
    header.appendChild(colorSwatch);

    // Clickable head name -> attentionpedia page
    const titleLink = document.createElement("a");
    titleLink.className = "pattern-section-title";
    titleLink.textContent = headId;
    titleLink.href = `../attnpedia/index.html?head=${encodeURIComponent(headId)}`;
    header.appendChild(titleLink);

    const clusterId = window.CLUSTER_STATE.getClusterId(headId);
    if (clusterId !== undefined) {
      const clusterBadge = document.createElement("span");
      clusterBadge.className = "pattern-section-cluster";
      const label = gridState.resolvedLabels[clusterId];
      clusterBadge.textContent =
        label && label.name
          ? `Cluster ${clusterId}: ${label.name}`
          : `Cluster ${clusterId}`;
      if (label && label.desc) clusterBadge.title = label.desc;
      header.appendChild(clusterBadge);
    }

    // Arrow link to attentionpedia
    const arrowLink = document.createElement("a");
    arrowLink.className = "pattern-arrow";
    arrowLink.href = `../attnpedia/index.html?head=${encodeURIComponent(headId)}`;
    arrowLink.textContent = "\u2192";
    arrowLink.title = "View in AttentionPedia";
    header.appendChild(arrowLink);

    section.appendChild(header);

    // Pattern row (horizontal scrolling)
    const patternRow = document.createElement("div");
    patternRow.className = "pattern-row";

    // Load prompts for this model if not cached
    const [modelName, layerPart, headPart] = headId.split(":");
    const layer = layerPart.substring(1);
    const head = headPart.substring(1);

    if (!gridState.prompts[modelName]) {
      try {
        const promptsUrl = `${gridState.patternsBaseUrl}/${modelName}/prompts.jsonl`;
        const response = await fetch(promptsUrl);
        if (response.ok) {
          const text = await response.text();
          let allPrompts = text
            .trim()
            .split("\n")
            .filter((line) => line.trim())
            .map((line) => JSON.parse(line));
          // Filter to only rendered prompts if available
          if (gridState.renderedHashes) {
            allPrompts = allPrompts.filter(
              (p) => p.hash && gridState.renderedHashes.has(p.hash),
            );
          }
          gridState.prompts[modelName] = allPrompts;
        } else {
          gridState.prompts[modelName] = [];
        }
      } catch (e) {
        console.warn(`Failed to load prompts for ${modelName}:`, e);
        gridState.prompts[modelName] = [];
      }
    }

    // Get or generate shuffled prompt indices for this model
    const prompts = gridState.prompts[modelName] || [];
    if (!gridState.promptIndices[modelName] && prompts.length > 0) {
      const indices = Array.from({ length: prompts.length }, (_, i) => i);
      gridState.promptIndices[modelName] = shuffleArray(indices);
    }

    // Show up to 8 patterns using shuffled order
    const promptOrder = gridState.promptIndices[modelName] || [];
    const maxPatterns = Math.min(8, promptOrder.length);

    for (let i = 0; i < maxPatterns; i++) {
      const promptIdx = promptOrder[i];
      const prompt = prompts[promptIdx];
      const hash = prompt?.hash || prompt?.prompt_hash;

      if (hash) {
        const imgUrl = `${gridState.patternsBaseUrl}/${modelName}/prompts/${hash}/L${layer}/H${head}/attn.png`;

        // Create clickable link to pattern viewer
        const link = document.createElement("a");
        link.href = getPatternViewerUrl(headId, hash);
        link.target = "_blank";
        link.className = "pattern-link";
        link.title = "Open in pattern viewer";

        const img = document.createElement("img");
        img.className = "pattern-image";
        img.src = imgUrl;
        img.alt = `Pattern for ${headId}`;
        img.loading = "lazy";
        img.style.width = gridState.patternSize + "px";
        img.style.height = gridState.patternSize + "px";
        img.onerror = () => {
          link.style.display = "none";
        };

        link.appendChild(img);
        patternRow.appendChild(link);
      }
    }

    if (maxPatterns === 0) {
      const placeholder = document.createElement("div");
      placeholder.className = "pattern-image-placeholder";
      placeholder.textContent = "No patterns";
      patternRow.appendChild(placeholder);
    }

    section.appendChild(patternRow);
    patternImages.appendChild(section);
  }
}

/**
 * Show tooltip
 */
function showTooltip(event, headId) {
  const tooltip = gridState.tooltip;
  const clusterId = window.CLUSTER_STATE.getClusterId(headId);
  const label =
    clusterId !== undefined ? gridState.resolvedLabels[clusterId] : null;
  const nameStr = label && label.name ? `: ${label.name}` : "";
  const descStr =
    label && label.desc ? `<div class="cluster-desc">${label.desc}</div>` : "";

  tooltip.innerHTML = `
    <div class="head-id">${headId}</div>
    <div class="cluster-info">Cluster ${clusterId !== undefined ? clusterId : "?"}${nameStr}</div>
    ${descStr}
  `;
  tooltip.style.display = "block";
  moveTooltip(event);
}

/**
 * Move tooltip to follow cursor
 */
function moveTooltip(event) {
  const tooltip = gridState.tooltip;
  tooltip.style.left = event.clientX + 12 + "px";
  tooltip.style.top = event.clientY + 12 + "px";
}

/**
 * Hide tooltip
 */
function hideTooltip() {
  gridState.tooltip.style.display = "none";
}

/**
 * Update statistics display with sparkline
 */
function updateStats(assignments, nClusters, smallClusters = new Set()) {
  // Store for re-rendering when toggling log scale
  gridState.currentAssignments = assignments;
  gridState.currentNClusters = nClusters;

  const sizes = window.CLUSTER_STATE.getClusterSizes();
  const sortedSizes = Object.values(sizes).sort((a, b) => b - a);

  const statsEl = document.getElementById("stats");
  statsEl.innerHTML = `
    <div class="stats-text">
      <span><strong>Total Heads:</strong> ${gridState.nHeads}</span>
      <span><strong>Clusters:</strong> ${nClusters}</span>
      <span><strong>Largest:</strong> ${sortedSizes[0]}</span>
      <span><strong>Smallest:</strong> ${sortedSizes[sortedSizes.length - 1]}</span>
    </div>
    <div class="stats-sparkline" id="stats-sparkline">
      <span class="stats-sparkline-label">Distribution:</span>
      <button class="scale-toggle toggle-btn" id="scale-toggle">${gridState.logScale ? "Log" : "Linear"}</button>
    </div>
  `;

  // Add sparkline bar chart
  if (sortedSizes.length > 0) {
    const sparklineContainer = document.getElementById("stats-sparkline");
    const svgString = sparkbars(sortedSizes, null, {
      width: 200,
      height: 40,
      color: "#1565c0",
      yAxis: { ticks: true },
      logScale: gridState.logScale,
    });
    sparklineContainer.insertAdjacentHTML("beforeend", svgString);

    // Add toggle handler
    document.getElementById("scale-toggle").addEventListener("click", () => {
      gridState.logScale = !gridState.logScale;
      updateStats(gridState.currentAssignments, gridState.currentNClusters);
    });
  }

  // Resolve and render labels
  updateResolvedLabels();

  // Render top clusters list
  renderTopClusters();

  // Render cluster labels editor
  renderClusterLabels();

  // Render slider ticks for labeled cut heights
  renderSliderTicks();

  // Render mini dendrogram
  renderMiniDendrogram();
}

/**
 * Build a simplified tree from linkage matrix, collapsed at cluster boundaries
 * Returns a tree where leaves are clusters (not individual heads)
 */
function buildClusterTree() {
  const linkage = gridState.linkage;
  const n = gridState.nHeads;
  const assignments = window.CLUSTER_STATE.getAssignments();

  // Build full tree first
  const nodes = [];

  // Leaf nodes (individual heads)
  for (let i = 0; i < n; i++) {
    const headId = gridState.clsValues[i];
    nodes.push({
      id: i,
      isLeaf: true,
      height: 0,
      clusterId: assignments[headId],
      count: 1,
    });
  }

  // Internal nodes from linkage
  for (let i = 0; i < linkage.length; i++) {
    const [idx1, idx2, distance, count] = linkage[i];
    const left = nodes[Math.floor(idx1)];
    const right = nodes[Math.floor(idx2)];

    // If both children have the same cluster, this node has that cluster
    // If different clusters, this is a merge point (clusterId = null)
    const clusterId =
      left.clusterId === right.clusterId ? left.clusterId : null;

    nodes.push({
      id: n + i,
      isLeaf: false,
      height: distance,
      clusterId: clusterId,
      count: count,
      children: [left, right],
    });
  }

  const root = nodes[nodes.length - 1];

  // Now collapse: for any node where all descendants have the same cluster,
  // replace it with a leaf representing that cluster
  function collapse(node) {
    if (node.isLeaf) {
      return {
        isLeaf: true,
        clusterId: node.clusterId,
        count: node.count,
        height: node.height,
      };
    }

    // If this entire subtree is one cluster, collapse to leaf
    if (node.clusterId !== null) {
      return {
        isLeaf: true,
        clusterId: node.clusterId,
        count: node.count,
        height: node.height,
      };
    }

    // Otherwise, recurse
    return {
      isLeaf: false,
      clusterId: null,
      height: node.height,
      count: node.count,
      children: [collapse(node.children[0]), collapse(node.children[1])],
    };
  }

  return collapse(root);
}

/**
 * Truncate a cluster tree to at most maxLeaves leaf nodes.
 * Collapses subtrees containing only small clusters into aggregate nodes.
 * @param {Object} tree - Collapsed cluster tree from buildClusterTree()
 * @param {number} maxLeaves - Maximum number of leaf nodes to show
 * @returns {Object} Truncated tree
 */
function truncateTree(tree, maxLeaves) {
  // Count current leaves
  function countLeaves(node) {
    if (node.isLeaf || node.isAggregate) return 1;
    return countLeaves(node.children[0]) + countLeaves(node.children[1]);
  }

  if (countLeaves(tree) <= maxLeaves) return tree;

  const sizes = window.CLUSTER_STATE.getClusterSizes();

  // Collect all leaves with their sizes
  /** @type {Array<{clusterId: number, size: number}>} */
  const allLeaves = [];
  function collectLeaves(node) {
    if (node.isLeaf) {
      allLeaves.push({
        clusterId: node.clusterId,
        size: sizes[node.clusterId] || node.count,
      });
      return;
    }
    collectLeaves(node.children[0]);
    collectLeaves(node.children[1]);
  }
  collectLeaves(tree);

  // Sort by size descending, keep top (maxLeaves - 1) as "big"
  allLeaves.sort((a, b) => b.size - a.size);
  const bigClusterIds = new Set(
    allLeaves.slice(0, maxLeaves - 1).map((l) => l.clusterId),
  );

  // Walk tree: collapse subtrees where ALL leaves are small
  function truncateNode(node) {
    if (node.isLeaf) {
      if (bigClusterIds.has(node.clusterId)) return node;
      // Small leaf - return as-is, parent will aggregate
      return node;
    }

    const left = truncateNode(node.children[0]);
    const right = truncateNode(node.children[1]);

    // Collect all cluster IDs in each subtree
    function getClusterIds(n) {
      if (n.isAggregate) return [...n.clusterIds];
      if (n.isLeaf) return [n.clusterId];
      return [...getClusterIds(n.children[0]), ...getClusterIds(n.children[1])];
    }

    const leftIds = getClusterIds(left);
    const rightIds = getClusterIds(right);
    const allIds = [...leftIds, ...rightIds];
    const allSmall = allIds.every((id) => !bigClusterIds.has(id));

    if (allSmall) {
      // Collapse entire subtree into aggregate
      return {
        isAggregate: true,
        isLeaf: false,
        clusterIds: allIds,
        totalCount: allIds.reduce((sum, id) => sum + (sizes[id] || 0), 0),
        height: node.height,
        count: node.count,
      };
    }

    return {
      isLeaf: false,
      clusterId: null,
      height: node.height,
      count: node.count,
      children: [left, right],
    };
  }

  return truncateNode(tree);
}

/**
 * Render a simple horizontal dendrogram showing cluster structure
 */
function renderMiniDendrogram() {
  const svg = document.getElementById("mini-dendrogram");
  if (!svg) return;

  const fullTree = buildClusterTree();
  const tree = gridState.dendrogramFull ? fullTree : truncateTree(fullTree, 10);
  const sizes = window.CLUSTER_STATE.getClusterSizes();

  // Update toggle button text
  const toggleBtn = document.getElementById("dendrogram-toggle");
  if (toggleBtn) {
    const fullLeafCount = (function countL(n) {
      return n.isLeaf ? 1 : countL(n.children[0]) + countL(n.children[1]);
    })(fullTree);
    toggleBtn.textContent = gridState.dendrogramFull ? "Truncate" : "Show All";
    toggleBtn.style.display = fullLeafCount <= 10 ? "none" : "";
    toggleBtn.onclick = () => {
      gridState.dendrogramFull = !gridState.dendrogramFull;
      renderMiniDendrogram();
    };
  }

  // Get dimensions
  const rect = svg.getBoundingClientRect();
  const width = rect.width || 380;
  const height = rect.height || 160;
  const margin = { top: 5, right: 60, bottom: 5, left: 5 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Count leaves and assign y positions
  let leafCount = 0;
  function countLeaves(node) {
    if (node.isLeaf || node.isAggregate) {
      node.leafIndex = leafCount++;
      return 1;
    }
    return countLeaves(node.children[0]) + countLeaves(node.children[1]);
  }
  const totalLeaves = countLeaves(tree);

  // Get min and max heights for x scaling
  function getHeightRange(node) {
    if (node.isLeaf || node.isAggregate) return { min: Infinity, max: 0 };
    const left = getHeightRange(node.children[0]);
    const right = getHeightRange(node.children[1]);
    return {
      min: Math.min(node.height, left.min, right.min),
      max: Math.max(node.height, left.max, right.max),
    };
  }
  const { min: minHeight, max: maxHeight } = getHeightRange(tree);
  const heightRange = maxHeight - minHeight || 1;

  // Position nodes - compact root line
  function positionNodes(node) {
    if (node.isLeaf || node.isAggregate) {
      node.x = innerWidth;
      node.y = (node.leafIndex + 0.5) * (innerHeight / totalLeaves);
      return node.y;
    }

    const y0 = positionNodes(node.children[0]);
    const y1 = positionNodes(node.children[1]);
    const normalizedHeight = (maxHeight - node.height) / heightRange;
    node.x = innerWidth * (0.02 + normalizedHeight * 0.98);
    node.y = (y0 + y1) / 2;
    return node.y;
  }
  positionNodes(tree);

  // Clear and rebuild SVG
  svg.innerHTML = "";

  // Create SVG group
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");

  function renderNode(node) {
    if (node.isAggregate) {
      // Aggregate node: gray circle with "(N clusters)" label
      const circle = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "circle",
      );
      circle.setAttribute("cx", margin.left + node.x);
      circle.setAttribute("cy", margin.top + node.y);
      circle.setAttribute("r", 6);
      circle.setAttribute("fill", "#bbb");
      circle.setAttribute("stroke", "white");
      circle.setAttribute("stroke-width", 1);
      circle.setAttribute("cursor", "pointer");

      // Hover: highlight corresponding cluster chips
      circle.addEventListener("mouseenter", () => {
        const chipContainer = document.getElementById("top-clusters");
        if (!chipContainer) return;
        const aggregateIds = new Set(node.clusterIds);
        chipContainer.querySelectorAll(".cluster-chip").forEach((chip) => {
          const cid = parseInt(chip.dataset.clusterId);
          if (aggregateIds.has(cid)) {
            chip.style.outline = "2px solid #333";
          }
        });
      });
      circle.addEventListener("mouseleave", () => {
        const chipContainer = document.getElementById("top-clusters");
        if (!chipContainer) return;
        chipContainer.querySelectorAll(".cluster-chip").forEach((chip) => {
          chip.style.outline = "";
        });
      });
      g.appendChild(circle);

      // Label
      const text = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "text",
      );
      text.setAttribute("x", margin.left + node.x + 10);
      text.setAttribute("y", margin.top + node.y + 4);
      text.setAttribute("font-size", 10);
      text.setAttribute("fill", "#888");
      text.textContent = `(${node.clusterIds.length} clusters)`;
      g.appendChild(text);
      return;
    }

    if (node.isLeaf) {
      const displayColor =
        node.clusterId === -1 ? "#666" : clusterColor(node.clusterId);
      const size = sizes[node.clusterId] || node.count;

      // Create circle
      const circle = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "circle",
      );
      circle.setAttribute("cx", margin.left + node.x);
      circle.setAttribute("cy", margin.top + node.y);
      circle.setAttribute("r", 6);
      circle.setAttribute("fill", displayColor);
      circle.setAttribute("stroke", "white");
      circle.setAttribute("stroke-width", 1);
      circle.setAttribute("cursor", "pointer");
      circle.addEventListener("click", () => {
        selectCluster(node.clusterId);
        renderTopClusters();
      });
      g.appendChild(circle);

      // Create label
      const text = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "text",
      );
      text.setAttribute("x", margin.left + node.x + 10);
      text.setAttribute("y", margin.top + node.y + 4);
      text.setAttribute("font-size", 11);
      text.setAttribute("fill", "#333");
      text.textContent = size;
      g.appendChild(text);
      return;
    }

    // Draw elbow paths to children
    for (const child of node.children) {
      const x1 = margin.left + node.x;
      const y1 = margin.top + node.y;
      const x2 = margin.left + child.x;
      const y2 = margin.top + child.y;

      const path = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "path",
      );
      path.setAttribute("d", `M${x1},${y1} H${x2} V${y2}`);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "#999");
      path.setAttribute("stroke-width", 1.5);
      g.appendChild(path);

      renderNode(child);
    }
  }

  renderNode(tree);
  svg.appendChild(g);
}

/**
 * Render the cluster labels editor section
 */
function renderClusterLabels() {
  const container = document.getElementById("cluster-labels");
  if (!container) return;

  // Only show clusters for currently selected heads
  const selectedClusterIds = new Set(
    gridState.selectedHeads
      .map((h) => window.CLUSTER_STATE.getClusterId(h))
      .filter((id) => id !== undefined),
  );

  if (selectedClusterIds.size === 0) {
    container.innerHTML =
      '<div class="cluster-labels-hint" style="color: #888; font-style: italic; padding: 4px 0;">Select heads to edit cluster labels</div>';
    return;
  }

  const sizes = window.CLUSTER_STATE.getClusterSizes();
  const resolved = gridState.resolvedLabels;

  // Show only clusters that contain selected heads, sorted by size
  const sortedClusters = Object.entries(sizes)
    .map(([clusterId, size]) => ({ clusterId: parseInt(clusterId), size }))
    .filter(({ clusterId }) => selectedClusterIds.has(clusterId))
    .sort((a, b) => b.size - a.size);

  container.innerHTML = `
    <div class="cluster-labels-header">
      <span class="cluster-labels-title">Cluster Labels:</span>
      <button class="toggle-btn" id="export-labels-btn">Export Labels</button>
    </div>
    <div class="cluster-labels-grid">
      ${sortedClusters
        .map(({ clusterId, size }) => {
          const color = clusterId === -1 ? "#666" : clusterColor(clusterId);
          const label = resolved[clusterId];
          const name = (label && label.name) || "";
          const desc = (label && label.desc) || "";
          return `
            <div class="cluster-label-row" data-cluster-id="${clusterId}">
              <span class="cluster-label-color" style="background-color: ${color}"></span>
              <span class="cluster-label-id">${clusterId === -1 ? "misc" : clusterId}</span>
              <span class="cluster-label-size">(${size})</span>
              <input type="text" class="cluster-label-name"
                value="${name.replace(/"/g, "&quot;")}"
                placeholder="Name"
                data-cluster-id="${clusterId}" />
              <input type="text" class="cluster-label-desc"
                value="${desc.replace(/"/g, "&quot;")}"
                placeholder="Description"
                data-cluster-id="${clusterId}" />
            </div>
          `;
        })
        .join("")}
    </div>
  `;

  // Wire up input handlers
  container.querySelectorAll(".cluster-label-row").forEach((row) => {
    const clusterId = parseInt(row.dataset.clusterId);
    const nameInput = row.querySelector(".cluster-label-name");
    const descInput = row.querySelector(".cluster-label-desc");
    const onChange = () => {
      setClusterLabel(clusterId, nameInput.value, descInput.value);
      renderTopClusters();
    };
    nameInput.addEventListener("change", onChange);
    descInput.addEventListener("change", onChange);
  });

  // Wire up export button
  document
    .getElementById("export-labels-btn")
    .addEventListener("click", exportClusterLabels);
}

/**
 * Render the top 20 largest clusters as clickable chips
 */
function renderTopClusters() {
  const container = document.getElementById("top-clusters");
  if (!container) return;

  const sizes = window.CLUSTER_STATE.getClusterSizes();
  const totalClusters = Object.keys(sizes).length;

  // Sort clusters by size descending, take top 20
  const sortedClusters = Object.entries(sizes)
    .map(([clusterId, size]) => ({ clusterId: parseInt(clusterId), size }))
    .sort((a, b) => b.size - a.size)
    .slice(0, 20);

  const hasMore = totalClusters > 20;

  const resolved = gridState.resolvedLabels;

  container.innerHTML = `
    <span class="top-clusters-label">Top ${sortedClusters.length} clusters:</span>
    ${sortedClusters
      .map(({ clusterId, size }) => {
        const color = clusterColor(clusterId);
        const heads = window.CLUSTER_STATE.getHeadsInCluster(clusterId);
        const allSelected =
          heads.length > 0 &&
          heads.every((h) => gridState.selectedHeads.includes(h));
        const selectedClass = allSelected ? "selected" : "";
        const label = resolved[clusterId];
        const name = (label && label.name) || "";
        const desc = (label && label.desc) || "";
        const labelHtml = name
          ? `<span class="cluster-chip-label" title="${desc.replace(/"/g, "&quot;")}">${name}</span>`
          : "";
        return `
          <div class="cluster-chip ${selectedClass}" data-cluster-id="${clusterId}">
            <span class="cluster-chip-color" style="background-color: ${color}"></span>
            <span class="cluster-chip-size">${size}</span>
            ${labelHtml}
          </div>
        `;
      })
      .join("")}
    ${hasMore ? `<span class="top-clusters-ellipsis">... +${totalClusters - 20} more</span>` : ""}
  `;

  // Add click handlers
  container.querySelectorAll(".cluster-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const clusterId = parseInt(chip.dataset.clusterId);
      selectCluster(clusterId);
      renderTopClusters(); // Re-render to update selection state
    });
  });
}

/**
 * Select all heads in a cluster
 * @param {number} clusterId - Cluster ID to select
 */
function selectCluster(clusterId) {
  const heads = window.CLUSTER_STATE.getHeadsInCluster(clusterId);
  // Toggle: if all are selected, deselect; otherwise select all
  const allSelected = heads.every((h) => gridState.selectedHeads.includes(h));

  if (allSelected) {
    // Deselect all in this cluster
    gridState.selectedHeads = gridState.selectedHeads.filter(
      (h) => !heads.includes(h),
    );
  } else {
    // Add all from this cluster
    for (const h of heads) {
      if (!gridState.selectedHeads.includes(h)) {
        gridState.selectedHeads.push(h);
      }
    }
  }

  updateSelectedCells();
  updateSidePane();
}

/**
 * Select heads in a cluster belonging to a specific model only
 * @param {number} clusterId - Cluster ID to select
 * @param {string} modelName - Model name to filter by
 */
function selectClusterForModel(clusterId, modelName) {
  const allHeads = window.CLUSTER_STATE.getHeadsInCluster(clusterId);
  const heads = allHeads.filter((h) => h.split(":")[0] === modelName);
  if (heads.length === 0) return;

  // Toggle: if all are selected, deselect; otherwise select all
  const allSelected = heads.every((h) => gridState.selectedHeads.includes(h));

  if (allSelected) {
    gridState.selectedHeads = gridState.selectedHeads.filter(
      (h) => !heads.includes(h),
    );
  } else {
    for (const h of heads) {
      if (!gridState.selectedHeads.includes(h)) {
        gridState.selectedHeads.push(h);
      }
    }
  }

  updateSelectedCells();
  updateSidePane();
}
