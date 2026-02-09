/**
 * Dendrogram visualization using D3.js
 *
 * Renders hierarchical clustering as an interactive dendrogram with:
 * - Adjustable number of clusters via slider
 * - Color-coded branches by cluster
 * - Zoom and pan
 * - Click to highlight cluster members
 */

let dendrogramState = {
  linkage: null,
  clsValues: null,
  nHeads: 0,
  tree: null,
  svg: null,
  g: null,
  zoom: null,
  width: 0,
  height: 0,
  margin: { top: 20, right: 200, bottom: 20, left: 40 },
};

/**
 * Initialize the dendrogram visualization
 * @param {Object} config - Configuration object
 */
async function initDendrogram(config) {
  const container = document.getElementById("dendrogram-container");
  container.innerHTML = '<div class="loading">Loading clustering data...</div>';

  try {
    // Load metadata and linkage data
    const [metaResponse, linkageResponse] = await Promise.all([
      fetch(config.clusteringMetaUrl),
      fetch(config.linkageUrl),
    ]);

    if (!metaResponse.ok || !linkageResponse.ok) {
      throw new Error("Failed to load clustering data");
    }

    const meta = await metaResponse.json();
    const linkage = await linkageResponse.json();

    dendrogramState.clsValues = meta.cls_values;
    dendrogramState.linkage = linkage;
    dendrogramState.nHeads = meta.cls_values.length;

    // Build tree structure from linkage matrix
    dendrogramState.tree = buildTree(linkage, meta.cls_values);

    // Set up the SVG
    container.innerHTML = "";
    setupSVG();

    // Set up controls
    setupControls(config.defaultNClusters);

    // Initial render
    updateClusters(config.defaultNClusters);
  } catch (error) {
    console.error("Error initializing dendrogram:", error);
    container.innerHTML = `<div class="error">Error loading clustering data: ${error.message}</div>`;
  }
}

/**
 * Build a tree structure from scipy's linkage matrix format
 * @param {number[][]} linkage - Linkage matrix from scipy
 * @param {string[]} clsValues - Head IDs
 * @returns {Object} Root node of the tree
 */
function buildTree(linkage, clsValues) {
  const n = clsValues.length;

  // Create leaf nodes
  const nodes = clsValues.map((name, i) => ({
    id: i,
    name: name,
    isLeaf: true,
    height: 0,
    children: null,
  }));

  // Add internal nodes from linkage matrix
  // Each row: [idx1, idx2, distance, count]
  for (let i = 0; i < linkage.length; i++) {
    const [idx1, idx2, distance, count] = linkage[i];
    const nodeId = n + i;

    nodes.push({
      id: nodeId,
      name: `cluster_${i}`,
      isLeaf: false,
      height: distance,
      count: count,
      children: [nodes[Math.floor(idx1)], nodes[Math.floor(idx2)]],
    });
  }

  // Root is the last node
  return nodes[nodes.length - 1];
}

/**
 * Set up the SVG element with zoom behavior
 */
function setupSVG() {
  const container = document.getElementById("dendrogram-container");
  const containerRect = container.getBoundingClientRect();

  // Calculate dimensions based on number of heads
  const minHeight = 600;
  const heightPerHead = 15;
  dendrogramState.height = Math.max(
    minHeight,
    dendrogramState.nHeads * heightPerHead + dendrogramState.margin.top + dendrogramState.margin.bottom
  );
  dendrogramState.width = containerRect.width;

  // Create SVG
  const svg = d3
    .select("#dendrogram-container")
    .append("svg")
    .attr("id", "dendrogram")
    .attr("width", dendrogramState.width)
    .attr("height", dendrogramState.height);

  // Add zoom behavior
  dendrogramState.zoom = d3
    .zoom()
    .scaleExtent([0.5, 5])
    .on("zoom", (event) => {
      dendrogramState.g.attr("transform", event.transform);
    });

  svg.call(dendrogramState.zoom);

  // Create main group for content
  dendrogramState.g = svg
    .append("g")
    .attr("transform", `translate(${dendrogramState.margin.left},${dendrogramState.margin.top})`);

  dendrogramState.svg = svg;
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
  const resetZoomBtn = document.getElementById("reset-zoom");
  const exportSvgBtn = document.getElementById("export-svg");

  // Set max clusters to number of heads
  nClustersSlider.max = Math.min(dendrogramState.nHeads, 100);
  nClustersSlider.value = defaultNClusters;
  nClustersValue.textContent = defaultNClusters;

  // Get max height from tree
  const maxHeight = dendrogramState.tree.height;
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

  // Reset zoom
  resetZoomBtn.addEventListener("click", () => {
    dendrogramState.svg
      .transition()
      .duration(500)
      .call(dendrogramState.zoom.transform, d3.zoomIdentity);
  });

  // Export SVG
  exportSvgBtn.addEventListener("click", exportSVG);
}

/**
 * Compute cluster assignments by cutting at a specific number of clusters
 * This mimics scipy's fcluster with criterion='maxclust'
 * @param {number} nClusters - Number of clusters
 * @returns {Object.<string, number>} Map of head ID to cluster ID
 */
function computeClusters(nClusters) {
  const linkage = dendrogramState.linkage;
  const clsValues = dendrogramState.clsValues;
  const n = clsValues.length;

  if (nClusters >= n) {
    // Each head is its own cluster
    const assignments = {};
    clsValues.forEach((cls, i) => {
      assignments[cls] = i;
    });
    return assignments;
  }

  // Find the cut height that gives us the desired number of clusters
  // Sort merge distances
  const heights = linkage.map((row) => row[2]).sort((a, b) => b - a);

  // We need (n - nClusters) merges to go from n clusters to nClusters
  // So we cut just above the (n - nClusters)th highest merge
  const cutHeight = heights[n - nClusters - 1] + 1e-10;

  return computeClustersByHeight(cutHeight);
}

/**
 * Compute cluster assignments by cutting at a specific height
 * @param {number} cutHeight - Height at which to cut
 * @returns {Object.<string, number>} Map of head ID to cluster ID
 */
function computeClustersByHeight(cutHeight) {
  const linkage = dendrogramState.linkage;
  const clsValues = dendrogramState.clsValues;
  const n = clsValues.length;

  // Union-find to track cluster membership
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

  // Process merges below the cut height
  for (let i = 0; i < linkage.length; i++) {
    const [idx1, idx2, distance] = linkage[i];
    if (distance <= cutHeight) {
      union(Math.floor(idx1), Math.floor(idx2), n + i);
    }
  }

  // Assign cluster IDs
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
  renderDendrogram(assignments);
  updateStats(assignments, nClusters);
  updateLegend(assignments);
}

/**
 * Update visualization for a given cut height
 * @param {number} cutHeight - Height at which to cut
 */
function updateClustersByHeight(cutHeight) {
  const assignments = computeClustersByHeight(cutHeight);
  const nClusters = new Set(Object.values(assignments)).size;

  // Update n-clusters slider to match
  document.getElementById("n-clusters").value = nClusters;
  document.getElementById("n-clusters-value").textContent = nClusters;

  window.CLUSTER_STATE.setAssignments(assignments, nClusters);
  renderDendrogram(assignments, cutHeight);
  updateStats(assignments, nClusters);
  updateLegend(assignments);
}

/**
 * Render the dendrogram with current cluster assignments
 * @param {Object.<string, number>} assignments - Cluster assignments
 * @param {number} [cutHeight] - Optional cut height to draw
 */
function renderDendrogram(assignments, cutHeight = null) {
  const g = dendrogramState.g;
  const tree = dendrogramState.tree;
  const margin = dendrogramState.margin;

  const innerWidth = dendrogramState.width - margin.left - margin.right;
  const innerHeight = dendrogramState.height - margin.top - margin.bottom;

  // Clear previous content
  g.selectAll("*").remove();

  // Create D3 hierarchy
  const root = d3.hierarchy(tree, (d) => d.children);

  // Use cluster layout
  const cluster = d3.cluster().size([innerHeight, innerWidth - 100]);
  cluster(root);

  // Scale x by height for proper dendrogram proportions
  const maxHeight = tree.height;
  const xScale = d3.scaleLinear().domain([0, maxHeight]).range([0, innerWidth - 100]);

  // Assign x position based on height
  root.each((d) => {
    d.y = xScale(d.data.height);
  });

  // Draw links (elbow connectors)
  g.selectAll(".dendrogram-link")
    .data(root.links())
    .enter()
    .append("path")
    .attr("class", "dendrogram-link")
    .attr("d", elbow)
    .attr("stroke", (d) => getLinkColor(d.target, assignments));

  // Draw cut line if specified
  if (cutHeight !== null) {
    const cutX = xScale(cutHeight);
    g.append("line")
      .attr("class", "cut-line")
      .attr("x1", cutX)
      .attr("y1", 0)
      .attr("x2", cutX)
      .attr("y2", innerHeight)
      .attr("stroke", "#e53935")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5");
  }

  // Draw leaf labels
  const leaves = root.leaves();
  g.selectAll(".dendrogram-label")
    .data(leaves)
    .enter()
    .append("text")
    .attr("class", "dendrogram-label")
    .attr("x", (d) => d.y + 5)
    .attr("y", (d) => d.x)
    .attr("dy", "0.35em")
    .text((d) => formatHeadId(d.data.name))
    .style("fill", (d) => window.CLUSTER_STATE.getColor(d.data.name))
    .on("click", (event, d) => {
      highlightCluster(assignments[d.data.name]);
    })
    .append("title")
    .text((d) => d.data.name);
}

/**
 * Generate elbow path for dendrogram links
 */
function elbow(d) {
  return `M${d.source.y},${d.source.x}H${d.target.y}V${d.target.x}`;
}

/**
 * Get link color based on cluster of target node
 */
function getLinkColor(node, assignments) {
  // For internal nodes, get cluster of any descendant leaf
  if (!node.data.isLeaf) {
    const leaf = node.leaves()[0];
    if (leaf) {
      return window.CLUSTER_STATE.getColor(leaf.data.name);
    }
  }
  return window.CLUSTER_STATE.getColor(node.data.name);
}

/**
 * Format head ID for display (shorter version)
 */
function formatHeadId(headId) {
  // e.g., "gpt2-small:L5:H3" -> "gpt2-sm L5H3"
  const parts = headId.split(":");
  if (parts.length === 3) {
    const model = parts[0].replace("-small", "-sm").replace("-medium", "-md");
    return `${model} ${parts[1]}${parts[2]}`;
  }
  return headId;
}

/**
 * Highlight all heads in a cluster
 */
function highlightCluster(clusterId) {
  const heads = window.CLUSTER_STATE.getHeadsInCluster(clusterId);
  console.log(`Cluster ${clusterId}:`, heads);

  // Highlight labels
  dendrogramState.g
    .selectAll(".dendrogram-label")
    .classed("highlighted", (d) => heads.includes(d.data.name));
}

/**
 * Update statistics display
 */
function updateStats(assignments, nClusters) {
  const sizes = window.CLUSTER_STATE.getClusterSizes();
  const sortedSizes = Object.values(sizes).sort((a, b) => b - a);

  const statsEl = document.getElementById("stats");
  statsEl.innerHTML = `
    <span><strong>Total Heads:</strong> ${dendrogramState.nHeads}</span>
    <span><strong>Clusters:</strong> ${nClusters}</span>
    <span><strong>Largest Cluster:</strong> ${sortedSizes[0]} heads</span>
    <span><strong>Smallest Cluster:</strong> ${sortedSizes[sortedSizes.length - 1]} heads</span>
  `;
}

/**
 * Update legend display
 */
function updateLegend(assignments) {
  const sizes = window.CLUSTER_STATE.getClusterSizes();
  const legendEl = document.getElementById("legend");

  // Sort clusters by size (descending)
  const sortedClusters = Object.entries(sizes)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20); // Show top 20 clusters

  legendEl.innerHTML = sortedClusters
    .map(
      ([clusterId, count]) => `
      <div class="legend-item" onclick="highlightCluster(${clusterId})">
        <div class="legend-swatch" style="background: ${window.CLUSTER_STATE.colors[clusterId]}"></div>
        <span>Cluster ${clusterId}</span>
        <span class="legend-count">(${count})</span>
      </div>
    `
    )
    .join("");
}

/**
 * Export the current SVG as a file
 */
function exportSVG() {
  const svgEl = document.getElementById("dendrogram");
  const serializer = new XMLSerializer();
  const svgStr = serializer.serializeToString(svgEl);

  const blob = new Blob([svgStr], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "dendrogram.svg";
  a.click();

  URL.revokeObjectURL(url);
}
