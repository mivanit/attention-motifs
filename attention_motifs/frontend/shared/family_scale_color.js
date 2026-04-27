/**
 * Family × Scale coloring for head embedding visualization.
 *
 * Colors each point by model family (hue) and model size within that
 * family (lightness). Designed to be initialized from embed_clustering_setup.js
 * after the clustering module.
 *
 * Exports (via global): initFamilyScaleColor(pointCloud)
 * Returns: { enabled, getColor(row) }
 */

// eslint-disable-next-line no-unused-vars
function initFamilyScaleColor(pointCloud) {
  let enabled = false;

  // --- Discover families and per-family size ranges from data ---

  const nRows = pointCloud.model.numRows;
  const familySet = new Map(); // family -> { sizes: Set, models: Map<model, size> }

  for (let i = 0; i < nRows; i++) {
    const row = pointCloud.model.row(i);
    const family = row.model_family;
    const size = row.model_size;
    if (!family || size == null) continue;

    if (!familySet.has(family)) {
      familySet.set(family, { sizes: new Set(), models: new Map() });
    }
    const entry = familySet.get(family);
    entry.sizes.add(size);
    entry.models.set(row.model, size);
  }

  if (familySet.size === 0) {
    console.warn(
      "Family×Scale: no model_family/model_size data found. " +
        "Re-run pipeline step s5 to generate these columns.",
    );
    // Hide the panel if data is missing
    const panel = document.getElementById("customPanel-familyScale");
    if (panel) panel.style.display = "none";
    return { enabled: false, getColor: () => null };
  }

  // Sort families alphabetically, assign each a hue
  const families = [...familySet.keys()].sort();
  const GOLDEN_ANGLE = 137.508;
  const familyHues = new Map();
  families.forEach((f, i) => {
    familyHues.set(f, (i * GOLDEN_ANGLE) % 360);
  });

  // Per-family: sort sizes and build rank map (0..n-1)
  const familySizeRanks = new Map(); // family -> Map<size, rank_normalized>
  for (const [family, entry] of familySet) {
    const sortedSizes = [...entry.sizes].sort((a, b) => a - b);
    const rankMap = new Map();
    sortedSizes.forEach((s, i) => {
      // Normalize to [0, 1] where 0 = smallest, 1 = largest
      const t = sortedSizes.length > 1 ? i / (sortedSizes.length - 1) : 0.5;
      rankMap.set(s, t);
    });
    familySizeRanks.set(family, rankMap);
  }

  // --- Color computation ---

  // Cache: model_name -> {r, g, b}
  const colorCache = new Map();

  function hslToRgb01(h, s, l) {
    // h in [0,360], s,l in [0,1] -> {r,g,b} in [0,1]
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = l - c / 2;
    let r, g, b;
    if (h < 60) {
      r = c;
      g = x;
      b = 0;
    } else if (h < 120) {
      r = x;
      g = c;
      b = 0;
    } else if (h < 180) {
      r = 0;
      g = c;
      b = x;
    } else if (h < 240) {
      r = 0;
      g = x;
      b = c;
    } else if (h < 300) {
      r = x;
      g = 0;
      b = c;
    } else {
      r = c;
      g = 0;
      b = x;
    }
    return { r: r + m, g: g + m, b: b + m };
  }

  function computeColor(family, size) {
    const hue = familyHues.get(family);
    if (hue === undefined) return null;

    const ranks = familySizeRanks.get(family);
    const t = ranks ? (ranks.get(size) ?? 0.5) : 0.5;

    // Map scale to lightness: small=light (0.75), large=dark (0.35)
    const lightness = 0.75 - t * 0.4;
    const saturation = 0.7;

    return hslToRgb01(hue, saturation, lightness);
  }

  function getColor(row) {
    if (!enabled) return null;

    const model = row.model;
    if (colorCache.has(model)) return colorCache.get(model);

    const family = row.model_family;
    const size = row.model_size;
    if (!family || size == null) return null;

    const color = computeColor(family, size);
    colorCache.set(model, color);
    return color;
  }

  // --- Legend ---

  const legendEl = document.getElementById("familyScaleLegend");
  if (legendEl) {
    let html = "";
    for (const family of families) {
      const entry = familySet.get(family);
      // Sort models by size
      const models = [...entry.models.entries()].sort((a, b) => a[1] - b[1]);
      html += `<div style="margin-bottom:6px;"><b style="color:#0ff;">${family}</b><br>`;
      for (const [modelName, size] of models) {
        const color = computeColor(family, size);
        const cssColor = `rgb(${Math.round(color.r * 255)},${Math.round(color.g * 255)},${Math.round(color.b * 255)})`;
        html += `<span style="color:${cssColor};">\u25cf</span> ${modelName}<br>`;
      }
      html += "</div>";
    }
    legendEl.innerHTML = html;
  }

  // --- Checkbox binding ---

  const checkbox = document.getElementById("familyScaleEnabled");
  if (checkbox) {
    checkbox.addEventListener("change", () => {
      enabled = checkbox.checked;
      pointCloud._updateColors();
    });
  }

  return {
    get enabled() {
      return enabled;
    },
    getColor,
  };
}
