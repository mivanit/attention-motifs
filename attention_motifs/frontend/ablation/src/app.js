// === Data ===

let allData = null;
let currentModel = null;
let sortColumn = "loss_increase";
let sortDescending = true;
let methodFilters = new Set();

// Column definitions grouped by metric family
const METRIC_GROUPS = {
  Loss: ["baseline_repeated_loss", "ablated_repeated_loss", "loss_increase"],
  "Prefix (induction)": [
    "baseline_prefix_score",
    "ablated_prefix_score",
    "prefix_score_decrease",
  ],
  "Prefix (legacy)": [
    "baseline_prefix_score_legacy",
    "ablated_prefix_score_legacy",
    "prefix_score_decrease_legacy",
  ],
  ICL: ["baseline_icl_score", "ablated_icl_score", "icl_degradation"],
  Copying: ["copying_score", "ablated_copying_score", "copying_score_decrease"],
  "OV Copying": [
    "ov_copying_score",
    "ablated_ov_copying_score",
    "ov_copying_score_decrease",
  ],
};

// Which groups are visible by default
const DEFAULT_VISIBLE_GROUPS = new Set([
  "Loss",
  "Prefix (induction)",
  "ICL",
  "Copying",
]);

let visibleGroups = new Set(DEFAULT_VISIBLE_GROUPS);

// Delta columns and their color semantics
// "higher_is_worse" means a large positive value gets red coloring
const DELTA_COLUMNS = {
  loss_increase: { higher_is_worse: true, thresholds: [0.1, 0.5] },
  prefix_score_decrease: { higher_is_worse: false, thresholds: [0.03, 0.1] },
  prefix_score_decrease_legacy: {
    higher_is_worse: false,
    thresholds: [0.03, 0.1],
  },
  icl_degradation: { higher_is_worse: true, thresholds: [0.05, 0.2] },
  copying_score_decrease: { higher_is_worse: false, thresholds: [0.05, 0.2] },
  ov_copying_score_decrease: {
    higher_is_worse: false,
    thresholds: [0.05, 0.2],
  },
};

// Human-readable short labels for column headers
const COLUMN_LABELS = {
  head: "Head",
  ablation_method: "Method",
  baseline_repeated_loss: "Base Loss",
  ablated_repeated_loss: "Abl. Loss",
  loss_increase: "\u0394 Loss",
  baseline_prefix_score: "Base Prefix",
  ablated_prefix_score: "Abl. Prefix",
  prefix_score_decrease: "\u0394 Prefix",
  baseline_prefix_score_legacy: "Base Pfx Leg.",
  ablated_prefix_score_legacy: "Abl. Pfx Leg.",
  prefix_score_decrease_legacy: "\u0394 Pfx Leg.",
  baseline_icl_score: "Base ICL",
  ablated_icl_score: "Abl. ICL",
  icl_degradation: "\u0394 ICL",
  copying_score: "Base Copy",
  ablated_copying_score: "Abl. Copy",
  copying_score_decrease: "\u0394 Copy",
  ov_copying_score: "Base OV",
  ablated_ov_copying_score: "Abl. OV",
  ov_copying_score_decrease: "\u0394 OV",
};

// === Init ===

function init() {
  const dataEl = document.getElementById("ablation-data");
  if (!dataEl || dataEl.textContent.trim() === "__ABLATION_DATA__") {
    document.querySelector(".container").innerHTML =
      '<p style="color:#999;text-align:center;padding:60px;">No data embedded. This template is populated by the ablation pipeline.</p>';
    return;
  }
  allData = JSON.parse(dataEl.textContent);
  const modelNames = Object.keys(allData.models);
  if (modelNames.length === 0) return;

  currentModel = modelNames[0];

  // Collect all methods across all models
  for (const name of modelNames) {
    for (const r of allData.models[name].results) {
      methodFilters.add(r.ablation_method);
    }
  }

  renderModelTabs(modelNames);
  renderControls();
  renderCurrentModel();
}

// === Rendering ===

function renderCurrentModel() {
  const m = allData.models[currentModel];
  renderConfigSummary(m);
  renderModelSummary(m);
  renderTable(m);
}

function renderModelTabs(modelNames) {
  const container = document.getElementById("model-tabs");
  if (modelNames.length <= 1) {
    container.style.display = "none";
    return;
  }
  container.innerHTML = modelNames
    .map(
      (name) =>
        `<button class="model-tab${name === currentModel ? " active" : ""}" onclick="switchModel('${name}')">${name}</button>`,
    )
    .join("");
}

function switchModel(name) {
  currentModel = name;
  renderModelTabs(Object.keys(allData.models));
  renderCurrentModel();
}

function renderConfigSummary(modelData) {
  const cfg = modelData.config;
  const items = [
    ["sequences", cfg.n_sequences],
    ["seq length", cfg.seq_length],
    ["repetitions", cfg.n_repetitions],
    ["methods", cfg.ablation_methods.join(", ")],
    ["calibration", cfg.n_calibration_prompts],
    ["seed", cfg.seed],
  ];
  document.getElementById("config-summary").innerHTML = items
    .map(
      ([k, v]) =>
        `<span class="config-item"><span class="config-key">${k}:</span> ${v}</span>`,
    )
    .join("");
}

function renderModelSummary(modelData) {
  const stats = [
    ["Baseline Loss", modelData.baseline_loss],
    ["Baseline ICL", modelData.baseline_icl],
    ["Heads", new Set(modelData.results.map((r) => r.head)).size],
  ];
  document.getElementById("model-summary").innerHTML = stats
    .map(
      ([label, value]) =>
        `<div class="summary-stat"><span class="summary-label">${label}</span><span class="summary-value">${typeof value === "number" ? value.toFixed(4) : value}</span></div>`,
    )
    .join("");
}

function renderControls() {
  // Method filters
  const methods = [...methodFilters].sort();
  document.getElementById("method-filters").innerHTML = methods
    .map(
      (m) =>
        `<label class="chip"><input type="checkbox" checked onchange="toggleMethod('${m}')"> ${m}</label>`,
    )
    .join("");

  // Column group toggles
  const groups = Object.keys(METRIC_GROUPS);
  document.getElementById("column-toggles").innerHTML = groups
    .map(
      (g) =>
        `<label class="chip"><input type="checkbox" ${visibleGroups.has(g) ? "checked" : ""} onchange="toggleGroup('${g}')"> ${g}</label>`,
    )
    .join("");
}

function toggleMethod(method) {
  if (methodFilters.has(method)) {
    methodFilters.delete(method);
  } else {
    methodFilters.add(method);
  }
  renderTable(allData.models[currentModel]);
}

function toggleGroup(group) {
  if (visibleGroups.has(group)) {
    visibleGroups.delete(group);
  } else {
    visibleGroups.add(group);
  }
  renderTable(allData.models[currentModel]);
}

// === Table ===

function getVisibleColumns() {
  const cols = ["head", "ablation_method"];
  for (const [group, fields] of Object.entries(METRIC_GROUPS)) {
    if (visibleGroups.has(group)) {
      cols.push(...fields);
    }
  }
  return cols;
}

function renderTable(modelData) {
  const columns = getVisibleColumns();

  // Filter by method
  let rows = modelData.results.filter((r) =>
    methodFilters.has(r.ablation_method),
  );

  // Sort
  rows = sortResults(rows, sortColumn, sortDescending);

  // Thead
  const thead = document.getElementById("results-thead");
  thead.innerHTML =
    "<tr>" +
    columns
      .map((col) => {
        const label = COLUMN_LABELS[col] || col;
        const isSorted = col === sortColumn;
        const arrow = isSorted ? (sortDescending ? " \u25BC" : " \u25B2") : "";
        const cls = isSorted ? ' class="sorted"' : "";
        return `<th${cls} onclick="sortBy('${col}')">${label}<span class="sort-arrow">${arrow}</span></th>`;
      })
      .join("") +
    "</tr>";

  // Tbody
  const tbody = document.getElementById("results-tbody");
  if (rows.length === 0) {
    tbody.innerHTML =
      '<tr><td colspan="' +
      columns.length +
      '" style="text-align:center;color:#999;padding:30px;">No results match filters.</td></tr>';
    return;
  }

  tbody.innerHTML = rows
    .map(
      (row) =>
        "<tr>" +
        columns
          .map((col) => {
            const val = row[col];
            const cls = getCellClass(col, val);
            const display =
              typeof val === "number" ? formatNumber(val) : val || "";
            return `<td class="${cls}">${display}</td>`;
          })
          .join("") +
        "</tr>",
    )
    .join("");
}

function sortBy(column) {
  if (sortColumn === column) {
    sortDescending = !sortDescending;
  } else {
    sortColumn = column;
    sortDescending = true;
  }
  renderTable(allData.models[currentModel]);
}

function sortResults(results, column, descending) {
  return [...results].sort((a, b) => {
    const va = a[column];
    const vb = b[column];
    if (typeof va === "string" && typeof vb === "string") {
      return descending ? vb.localeCompare(va) : va.localeCompare(vb);
    }
    const diff = (va || 0) - (vb || 0);
    return descending ? -diff : diff;
  });
}

// === Formatting ===

function formatNumber(val) {
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
  // For "higher_is_worse" columns: positive values are "bad" (red)
  // For other columns: positive values are "good" (green)
  const isPositive = value > 0;
  const isBad = spec.higher_is_worse ? isPositive : !isPositive;

  if (isBad) {
    return isStrong ? "cell-bad-strong" : "cell-bad-mild";
  }
  return isStrong ? "cell-good-strong" : "cell-good-mild";
}

// === Boot ===
document.addEventListener("DOMContentLoaded", init);
