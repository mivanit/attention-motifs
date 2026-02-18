/**
 * Grid View visualization for clustering
 *
 * Renders model grids with cells colored by cluster assignment.
 * Supports multi-selection of heads with pattern display in side pane.
 */

let gridState = {
  linkage: null,
  clsValues: null,
  nHeads: 0,
  modelConfigs: {},
  selectedHeads: [],
  prompts: {},
  scale: 1.0,
  patternsBaseUrl: "",
  tooltip: null,
  sortByCluster: false,
  patternSize: 100,
  promptIndices: {},
  logScale: false,
  currentAssignments: null,
  currentNClusters: 0,
  modelSizes: {},
  modelOrder: "data",
  originalModelOrder: [],
  modelDataFrame: null,
};

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
        gridState.modelConfigs[cfg.model_name] = {
          n_layers: cfg.n_layers,
          n_heads: cfg.n_heads,
        };
      }
    }

    // Store original model order
    gridState.originalModelOrder = Object.keys(gridState.modelConfigs);

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

    // Set up controls
    setupControls(config.defaultNClusters);

    // Set up tooltip
    setupTooltip();

    // Set up side pane
    setupSidePane();

    // Set up resizable divider
    setupResizableDivider();

    // Set up help tooltip with model data
    setupHelpTooltip();

    // Initial render with cut height = 5
    const initialCutHeight = 5;
    document.getElementById("cut-height").value = initialCutHeight;
    document.getElementById("cut-height-value").textContent =
      initialCutHeight.toFixed(1);
    updateClustersByHeight(initialCutHeight);
  } catch (error) {
    console.error("Error initializing grid view:", error);
    container.innerHTML = `<div class="error">Error loading data: ${error.message}</div>`;
  }
}

/**
 * Set up control elements
 * @param {number} defaultNClusters - Default number of clusters
 */
function setupControls(defaultNClusters) {
  const nClustersSlider = document.getElementById("n-clusters");
  const nClustersValue = document.getElementById("n-clusters-value");
  const cutHeightSlider = document.getElementById("cut-height");
  const cutHeightValue = document.getElementById("cut-height-value");
  const scaleSlider = document.getElementById("scale");
  const scaleValue = document.getElementById("scale-value");

  // Set max clusters to number of heads
  nClustersSlider.max = Math.min(gridState.nHeads, 100);
  nClustersSlider.value = defaultNClusters;
  nClustersValue.textContent = defaultNClusters;

  // Get max height from linkage, capped at 20
  const maxHeight = Math.min(
    Math.max(...gridState.linkage.map((row) => row[2])),
    20,
  );
  cutHeightSlider.max = maxHeight;
  cutHeightSlider.step = maxHeight / 1000;

  // n-clusters slider
  nClustersSlider.addEventListener("input", (e) => {
    const n = parseInt(e.target.value);
    nClustersValue.textContent = n;
    updateClusters(n);
  });

  // cut-height slider
  cutHeightSlider.addEventListener("input", (e) => {
    const height = parseFloat(e.target.value);
    cutHeightValue.textContent = height.toFixed(3);
    updateClustersByHeight(height);
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
 * Compute cluster assignments by cutting at a specific number of clusters
 * @param {number} nClusters - Number of clusters
 * @returns {Object.<string, number>} Map of head ID to cluster ID
 */
function computeClusters(nClusters) {
  const linkage = gridState.linkage;
  const clsValues = gridState.clsValues;
  const n = clsValues.length;

  if (nClusters >= n) {
    const assignments = {};
    clsValues.forEach((cls, i) => {
      assignments[cls] = i;
    });
    return assignments;
  }

  // Find the cut height that gives us the desired number of clusters
  const heights = linkage.map((row) => row[2]).sort((a, b) => b - a);
  const cutHeight = heights[n - nClusters - 1] + 1e-10;

  return computeClustersByHeightInternal(cutHeight);
}

/**
 * Compute cluster assignments by cutting at a specific height
 * @param {number} cutHeight - Height at which to cut
 * @returns {Object.<string, number>} Map of head ID to cluster ID
 */
function computeClustersByHeightInternal(cutHeight) {
  const linkage = gridState.linkage;
  const clsValues = gridState.clsValues;
  const n = clsValues.length;

  // Union-find
  const parent = Array.from({ length: 2 * n - 1 }, (_, i) => i);

  function find(x) {
    if (parent[x] !== x) {
      parent[x] = find(parent[x]);
    }
    return parent[x];
  }

  function union(x, y, newParent) {
    parent[find(x)] = newParent;
    parent[find(y)] = newParent;
  }

  for (let i = 0; i < linkage.length; i++) {
    const [idx1, idx2, distance] = linkage[i];
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
 * Update visualization for a given number of clusters
 * @param {number} nClusters - Number of clusters
 */
function updateClusters(nClusters) {
  const assignments = computeClusters(nClusters);
  window.CLUSTER_STATE.setAssignments(assignments, nClusters);
  renderModelGrids();
  updateStats(assignments, nClusters);
}

/**
 * Update visualization for a given cut height
 * @param {number} cutHeight - Height at which to cut
 */
function updateClustersByHeight(cutHeight) {
  const assignments = computeClustersByHeightInternal(cutHeight);
  const nClusters = new Set(Object.values(assignments)).size;

  document.getElementById("n-clusters").value = nClusters;
  document.getElementById("n-clusters-value").textContent = nClusters;

  window.CLUSTER_STATE.setAssignments(assignments, nClusters);
  renderModelGrids();
  updateStats(assignments, nClusters);
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

      // Click handler - shift-click selects all heads in cluster
      cell.addEventListener("click", (e) => {
        if (e.shiftKey) {
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
      const clusterLabel = document.createElement("span");
      clusterLabel.className = "pattern-section-cluster";
      clusterLabel.textContent = `Cluster ${clusterId}`;
      header.appendChild(clusterLabel);
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
          gridState.prompts[modelName] = text
            .trim()
            .split("\n")
            .filter((line) => line.trim())
            .map((line) => JSON.parse(line));
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
        const img = document.createElement("img");
        img.className = "pattern-image";
        img.src = imgUrl;
        img.alt = `Pattern for ${headId}`;
        img.loading = "lazy";
        img.style.width = gridState.patternSize + "px";
        img.style.height = gridState.patternSize + "px";
        img.onerror = () => {
          img.style.display = "none";
        };
        patternRow.appendChild(img);
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

  tooltip.innerHTML = `
    <div class="head-id">${headId}</div>
    <div class="cluster-info">Cluster ${clusterId !== undefined ? clusterId : "?"}</div>
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
function updateStats(assignments, nClusters) {
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

  // Render top clusters list
  renderTopClusters();
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

  container.innerHTML = `
    <span class="top-clusters-label">Top ${sortedClusters.length} clusters:</span>
    ${sortedClusters
      .map(({ clusterId, size }) => {
        const color =
          window.CLUSTER_STATE.colors[
            clusterId % window.CLUSTER_STATE.colors.length
          ];
        const heads = window.CLUSTER_STATE.getHeadsInCluster(clusterId);
        const allSelected =
          heads.length > 0 &&
          heads.every((h) => gridState.selectedHeads.includes(h));
        const selectedClass = allSelected ? "selected" : "";
        return `
          <div class="cluster-chip ${selectedClass}" data-cluster-id="${clusterId}">
            <span class="cluster-chip-color" style="background-color: ${color}"></span>
            <span class="cluster-chip-size">${size}</span>
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
