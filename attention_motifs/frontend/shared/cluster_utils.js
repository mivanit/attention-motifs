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
