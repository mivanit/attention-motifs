/**
 * Shared clustering visualization utilities.
 *
 * Used by cluster_trends and potentially other frontends that need
 * cluster coloring and toggle buttons.
 */

// ── Cluster colors (golden angle) ──────────────────────────────

const GOLDEN_ANGLE = 137.508;

/**
 * Generate a distinct opaque color for a cluster index.
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
 * Generate a distinct color with alpha for a cluster index.
 * @param {number} idx
 * @param {number} alpha - opacity 0..1
 * @returns {string} CSS hsla color
 */
function clusterColorAlpha(idx, alpha) {
  const hue = (idx * GOLDEN_ANGLE) % 360;
  const sat = 65 + (idx % 3) * 10;
  const lit = 50 + (idx % 2) * 8;
  return `hsla(${hue}, ${sat}%, ${lit}%, ${alpha})`;
}

// ── Toggle buttons ─────────────────────────────────────────────

/**
 * Build toggle buttons inside a container element.
 * Each button toggles a boolean in the shared enabledState object.
 *
 * @param {string} containerId - DOM element ID for the container
 * @param {string[]} items - sorted list of item labels
 * @param {Object<string, boolean>} enabledState - shared state (mutated in place)
 * @param {string} cssClass - CSS class for each button (e.g. "family-toggle")
 * @param {() => void} onChange - callback when any toggle changes
 */
function buildToggleButtons(
  containerId,
  items,
  enabledState,
  cssClass,
  onChange,
) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  for (const item of items) {
    const btn = document.createElement("span");
    btn.className = cssClass;
    btn.textContent = item;
    if (!enabledState[item]) btn.classList.add("disabled");
    btn.addEventListener("click", () => {
      enabledState[item] = !enabledState[item];
      btn.classList.toggle("disabled", !enabledState[item]);
      onChange();
    });
    container.appendChild(btn);
  }
}

// ── Cluster labels ──────────────────────────────────────────────

const CLUSTER_LABELS_STORAGE_KEY = "clustering_labels";

/**
 * Load cluster labels from localStorage and optionally from a server URL.
 * localStorage labels override server labels on merge.
 *
 * @param {string|null} [serverUrl] - URL to fetch server-side labels JSON
 * @returns {Promise<Object>} Merged labels: { cutHeightKey: { clusterIdx: { name, desc, heads } } }
 */
async function loadClusterLabels(serverUrl) {
  // Load from localStorage
  let localLabels = {};
  try {
    const raw = localStorage.getItem(CLUSTER_LABELS_STORAGE_KEY);
    if (raw) localLabels = JSON.parse(raw);
  } catch (e) {
    console.warn("Failed to load cluster labels from localStorage:", e);
  }

  // Load from server (optional)
  let serverLabels = {};
  if (serverUrl) {
    try {
      const resp = await fetch(serverUrl);
      if (resp.ok) {
        serverLabels = await resp.json();
      } else {
        console.warn(
          `Cluster labels not found at ${serverUrl} (${resp.status})`,
        );
      }
    } catch (e) {
      console.warn("Failed to fetch cluster labels:", e);
    }
  }

  // Merge: server as base, localStorage overrides
  const merged = {};
  const allKeys = new Set([
    ...Object.keys(serverLabels),
    ...Object.keys(localLabels),
  ]);
  for (const heightKey of allKeys) {
    merged[heightKey] = {
      ...(serverLabels[heightKey] || {}),
      ...(localLabels[heightKey] || {}),
    };
  }
  return merged;
}

/**
 * Resolve stored labels against current cluster assignments.
 * Matches each label's heads list to the current cluster with the
 * highest overlap, returning a map of clusterId -> {name, desc}.
 *
 * @param {Object} allLabels - Full labels object keyed by cut height
 * @param {number} cutHeight - Current cut height
 * @param {Object<string, number>} assignments - headId -> clusterId
 * @returns {Object<number, {name: string, desc: string|null}>} clusterId -> label info
 */
function resolveClusterLabels(allLabels, cutHeight, assignments) {
  const heightKey = cutHeight.toFixed(3);
  const labelsForHeight = allLabels[heightKey];
  if (!labelsForHeight) return {};

  const resolvedCounts = {};
  const resolved = {};

  for (const [_origIdx, entry] of Object.entries(labelsForHeight)) {
    if (!entry || !entry.name) continue;
    const heads = entry.heads || [];

    // Count overlap with each current cluster
    const clusterOverlap = {};
    for (const headId of heads) {
      const cid = assignments[headId];
      if (cid !== undefined) {
        clusterOverlap[cid] = (clusterOverlap[cid] || 0) + 1;
      }
    }

    // Find cluster with most overlap
    let bestCluster = null;
    let bestCount = 0;
    for (const [cid, count] of Object.entries(clusterOverlap)) {
      if (count > bestCount) {
        bestCount = count;
        bestCluster = parseInt(cid);
      }
    }

    if (bestCluster !== null && bestCount > 0) {
      if (!resolved[bestCluster] || bestCount > resolvedCounts[bestCluster]) {
        resolved[bestCluster] = { name: entry.name, desc: entry.desc || null };
        resolvedCounts[bestCluster] = bestCount;
      }
    }
  }

  return resolved;
}

/**
 * Get a short display name for a cluster, using name if available.
 *
 * @param {number} clusterId
 * @param {Object<number, {name: string, desc: string|null}>} resolvedLabels
 * @returns {string} e.g. "Cluster 3: Induction" or "Cluster 3"
 */
function clusterDisplayName(clusterId, resolvedLabels) {
  if (clusterId === -1) return "misc";
  const label = resolvedLabels[clusterId];
  if (label && label.name) return `Cluster ${clusterId}: ${label.name}`;
  return `Cluster ${clusterId}`;
}
