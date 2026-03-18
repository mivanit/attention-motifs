/**
 * Embedding clustering hook setup.
 *
 * Assumes ClusteringLoader is already defined (concatenated before this file).
 * Sets HOOKS.onReady to wire clustering controls into the js-embedding-vis
 * custom panel: method selector, cut height slider, param selector,
 * min cluster size slider, enable checkbox.
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
  const methodSelect = document.getElementById("clusterMethod");
  const cutHeightRow = document.getElementById("clusterCutHeightRow");
  const cutSlider = document.getElementById("clusterCutHeight");
  const cutValue = document.getElementById("clusterCutHeightValue");
  const paramRow = document.getElementById("clusterParamRow");
  const paramSelect = document.getElementById("clusterParamSelect");
  const paramLabel = document.getElementById("clusterParamLabel");
  const minSizeSlider = document.getElementById("clusterMinSize");
  const minSizeValue = document.getElementById("clusterMinSizeValue");
  const statsEl = document.getElementById("clusterStats");
  const enabledCheckbox = document.getElementById("clusterEnabled");

  // Populate method dropdown
  const methods = clustering.getAvailableMethods();
  if (methodSelect) {
    methodSelect.innerHTML = "";
    for (const m of methods) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      methodSelect.appendChild(opt);
    }
    methodSelect.value = clustering.getMethod();
  }

  // Show/hide controls based on method
  function updateControlVisibility() {
    const isHier = clustering.isHierarchical();
    if (cutHeightRow) cutHeightRow.style.display = isHier ? "" : "none";
    if (paramRow) paramRow.style.display = isHier ? "none" : "";

    if (!isHier) {
      const info = clustering.getFlatParamInfo();
      if (info && paramSelect && paramLabel) {
        paramLabel.textContent = info.paramName + ":";
        paramSelect.innerHTML = "";
        for (const pk of info.paramKeys) {
          const opt = document.createElement("option");
          opt.value = pk;
          const meta = info.meta[pk];
          const extra = meta
            ? ` (${meta.n_clusters} clusters${meta.n_outliers ? `, ${meta.n_outliers} outliers` : ""})`
            : "";
          opt.textContent = pk + extra;
          paramSelect.appendChild(opt);
        }
        paramSelect.value = clustering.getParamKey() || info.paramKeys[0];
      }
    }
  }

  // Configure cut height slider (capped at 10, default 5)
  if (clustering.isHierarchical() && clustering.getMaxCutHeight()) {
    const maxHeight = Math.min(clustering.getMaxCutHeight(), 10);
    cutSlider.max = maxHeight;
    cutSlider.step = maxHeight / 1000;
    cutSlider.value = 5;
    cutValue.textContent = "5.00";
    clustering._computeAssignmentsByCutHeight(5);
  }

  updateControlVisibility();

  // Update stats display
  function updateStats() {
    const n = clustering.getNClustersActual();
    const misc = clustering.getUnclusteredCount();
    let text = `${n} clusters`;
    if (misc > 0) text += `, ${misc} unclustered`;
    text += ` [${clustering.getMethod()}]`;
    statsEl.textContent = text;
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

  // Bind method selector
  if (methodSelect) {
    methodSelect.addEventListener("change", () => {
      clustering.setMethod(methodSelect.value);
      updateControlVisibility();
      rebuildAndRefresh();
    });
  }

  // Bind param selector (for flat methods)
  if (paramSelect) {
    paramSelect.addEventListener("change", async () => {
      await clustering.setParamKey(paramSelect.value);
      rebuildAndRefresh();
    });
  }

  // Bind cut height slider
  cutSlider.addEventListener("input", () => {
    const h = parseFloat(cutSlider.value);
    cutValue.textContent = h.toFixed(2);
    clustering._computeAssignmentsByCutHeight(h);
    rebuildAndRefresh();
  });

  // Bind min cluster size slider
  minSizeSlider.addEventListener("input", async () => {
    const n = parseInt(minSizeSlider.value);
    minSizeValue.textContent = n;
    await clustering.setMinClusterSize(n);
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

  // --- Middle-click: show patterns + links for head ---

  // Determine a model to fetch prompts from (all models share the same prompts)
  const firstModel = pointCloud.model.row(0).model;
  const patternsBase = "../../../patterns/";
  const promptsUrl = `${patternsBase}${firstModel}/prompts.jsonl`;
  const N_PATTERNS = 6;

  try {
    const resp = await fetch(promptsUrl);
    if (!resp.ok) throw new Error(`${resp.status}`);
    const lines = (await resp.text()).trim().split("\n");
    const hashes = lines.slice(0, N_PATTERNS).map((l) => JSON.parse(l).hash);

    const imgs = hashes
      .map(
        (h) =>
          `<img src="${patternsBase}{model}/prompts/${h}/L{layer}/H{head}/attn.png" ` +
          `style="width:80px;height:80px;image-rendering:pixelated;background:#111;" />`,
      )
      .join("");

    CONFIG.middleClick.content =
      `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2px;">` +
      imgs +
      `</div>` +
      `<div style="margin-top:6px;font-size:11px;">` +
      `<span style="color:#aaa;">{type.group}</span><br>` +
      `<a href="../../attnpedia/index.html?head_viewing={cls}" target="_blank" style="color:#0af;">attentionpedia</a> · ` +
      `<a href="../../clustering/index.html?heads={cls}" target="_blank" style="color:#0af;">clustering</a>` +
      `</div>`;
  } catch (e) {
    console.warn("Could not load prompts for middle-click patterns:", e);
  }
};
