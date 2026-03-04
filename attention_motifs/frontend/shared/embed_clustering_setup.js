/**
 * Embedding clustering hook setup.
 *
 * Assumes ClusteringLoader is already defined (concatenated before this file).
 * Sets HOOKS.onReady to wire clustering controls into the js-embedding-vis
 * custom panel: cut height slider, min cluster size slider, enable checkbox.
 */

HOOKS.onReady = async (pointCloud, uiManager) => {
  const clustering = new ClusteringLoader();
  const available = await clustering.isAvailable();
  if (!available) {
    console.warn("Clustering data not available, disabling clustering panel");
    return;
  }

  // State
  let enabled = true;

  // Grab DOM elements from the custom panel
  const cutSlider = document.getElementById("clusterCutHeight");
  const cutValue = document.getElementById("clusterCutHeightValue");
  const minSizeSlider = document.getElementById("clusterMinSize");
  const minSizeValue = document.getElementById("clusterMinSizeValue");
  const statsEl = document.getElementById("clusterStats");
  const enabledCheckbox = document.getElementById("clusterEnabled");

  // Configure cut height slider (capped at 10, default 5)
  const maxHeight = Math.min(clustering.getMaxCutHeight(), 10);
  cutSlider.max = maxHeight;
  cutSlider.step = maxHeight / 1000;
  cutSlider.value = 5;
  cutValue.textContent = "5.00";
  clustering._computeAssignmentsByCutHeight(5);

  // Update stats display
  function updateStats() {
    const n = clustering.getNClustersActual();
    const misc = clustering.getUnclusteredCount();
    statsEl.textContent =
      `${n} clusters` + (misc > 0 ? `, ${misc} unclustered` : "");
  }

  // Rebuild color override from current assignments and refresh
  function rebuildAndRefresh() {
    updateStats();
    pointCloud._updateColors();
  }

  // Color override hook: return cluster RGB for each point
  HOOKS.colorOverrideFn = (rowIndex, row) => {
    if (!enabled) return null;
    const cid = clustering._assignments[row.cls];
    if (cid === undefined) return null;
    return clustering.getClusterColorRGB(cid);
  };

  // Hover extension hook: show cluster ID, name, and desc
  HOOKS.hoverExtendFn = (rowIndex, row) => {
    if (!enabled) return null;
    const cid = clustering._assignments[row.cls];
    if (cid === undefined) return null;
    const color = clustering.getClusterColor(cid);
    const label = clustering.getClusterLabel(cid);
    let text = `<b>Cluster</b>: ${cid === -1 ? "misc" : cid} <span style="color:${color}">\u25cf</span>`;
    if (label && label.name) text += ` <b>${label.name}</b>`;
    if (label && label.desc) text += `<br><i>${label.desc}</i>`;
    return text;
  };

  // Bind cut height slider
  cutSlider.addEventListener("input", () => {
    const h = parseFloat(cutSlider.value);
    cutValue.textContent = h.toFixed(2);
    clustering._computeAssignmentsByCutHeight(h);
    rebuildAndRefresh();
  });

  // Bind min cluster size slider
  minSizeSlider.addEventListener("input", () => {
    const n = parseInt(minSizeSlider.value);
    minSizeValue.textContent = n;
    clustering._minClusterSize = n;
    clustering._assignments = clustering._applyMinSizeFilter(
      clustering._rawAssignments,
    );
    clustering._resolveLabels();
    rebuildAndRefresh();
  });

  // Bind enable/disable checkbox
  enabledCheckbox.addEventListener("change", () => {
    enabled = enabledCheckbox.checked;
    pointCloud._updateColors();
  });

  // Initial render
  updateStats();
  rebuildAndRefresh();
};
