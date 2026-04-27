/**
 * Shared D3.js chart utilities.
 *
 * Provides reusable helpers for creating SVG charts with D3,
 * including axes, legends, tooltips, boxplots, and SVG export.
 *
 * Requires D3.js v7 to be loaded before this script.
 */

// ── Style constants ─────────────────────────────────────────────

const CHART_STYLES = {
  tickColor: "#888",
  labelColor: "#555",
  gridColor: "rgba(0,0,0,0.08)",
  labelFontSize: 11,
  titleFontSize: 13,
  legendFontSize: 11,
  tooltipBg: "rgba(0,0,0,0.82)",
  tooltipColor: "#fff",
  tooltipFontSize: "12px",
  tooltipPadding: "6px 10px",
  tooltipRadius: "4px",
};

// ── SVG container ───────────────────────────────────────────────

/**
 * Create an SVG chart container inside a DOM element.
 * Clears any existing SVG in the container.
 *
 * @param {string} containerId - DOM element ID
 * @param {{top: number, right: number, bottom: number, left: number}} margin
 * @returns {{svg: d3.Selection, g: d3.Selection, width: number, height: number, innerWidth: number, innerHeight: number}}
 */
function createChartSVG(containerId, margin) {
  const container = document.getElementById(containerId);
  // Remove existing SVG and export button
  const existing = container.querySelector("svg");
  if (existing) existing.remove();
  const existingBtn = container.querySelector(".export-svg-btn");
  if (existingBtn) existingBtn.remove();

  const rect = container.getBoundingClientRect();
  const cs = getComputedStyle(container);
  const width =
    rect.width - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const height =
    rect.height - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const svg = d3
    .select(container)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("xmlns", "http://www.w3.org/2000/svg")
    .style("font-family", "sans-serif");

  const g = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  return { svg, g, width, height, innerWidth, innerHeight };
}

// ── Axes ────────────────────────────────────────────────────────

/**
 * Render a bottom (X) axis with optional grid lines and label.
 *
 * @param {d3.Selection} g - Chart inner group
 * @param {d3.Scale} scale - D3 scale for the axis
 * @param {{height: number, label?: string, tickFormat?: Function, gridHeight?: number, ticks?: number}} opts
 */
function createXAxis(g, scale, opts) {
  const axis = d3.axisBottom(scale);
  if (opts.tickFormat) axis.tickFormat(opts.tickFormat);
  if (opts.ticks) axis.ticks(opts.ticks);

  const axisG = g
    .append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${opts.height})`)
    .call(axis);

  axisG.selectAll("text").attr("fill", CHART_STYLES.tickColor);
  axisG.selectAll("line").attr("stroke", CHART_STYLES.tickColor);
  axisG.select(".domain").attr("stroke", CHART_STYLES.tickColor);

  // Grid lines
  if (opts.gridHeight) {
    const gridAxis = d3
      .axisBottom(scale)
      .tickSize(-opts.gridHeight)
      .tickFormat("");
    if (opts.ticks) gridAxis.ticks(opts.ticks);
    g.append("g")
      .attr("class", "x-grid")
      .attr("transform", `translate(0,${opts.height})`)
      .call(gridAxis)
      .selectAll("line")
      .attr("stroke", CHART_STYLES.gridColor);
    g.select(".x-grid .domain").remove();
  }

  // Label
  if (opts.label) {
    g.append("text")
      .attr("class", "x-label")
      .attr("x", scale.range()[1] / 2 + (scale.range()[0] || 0) / 2)
      .attr("y", opts.height + 36)
      .attr("text-anchor", "middle")
      .attr("fill", CHART_STYLES.labelColor)
      .attr("font-size", CHART_STYLES.labelFontSize)
      .text(opts.label);
  }
}

/**
 * Render a left (Y) axis with optional grid lines and label.
 *
 * @param {d3.Selection} g - Chart inner group
 * @param {d3.Scale} scale - D3 scale for the axis
 * @param {{label?: string, gridWidth?: number, ticks?: number}} opts
 */
function createYAxis(g, scale, opts) {
  const axis = d3.axisLeft(scale);
  if (opts.ticks) axis.ticks(opts.ticks);

  const axisG = g.append("g").attr("class", "y-axis").call(axis);

  axisG.selectAll("text").attr("fill", CHART_STYLES.tickColor);
  axisG.selectAll("line").attr("stroke", CHART_STYLES.tickColor);
  axisG.select(".domain").attr("stroke", CHART_STYLES.tickColor);

  // Grid lines
  if (opts.gridWidth) {
    const gridAxis = d3
      .axisLeft(scale)
      .tickSize(-opts.gridWidth)
      .tickFormat("");
    if (opts.ticks) gridAxis.ticks(opts.ticks);
    g.append("g")
      .attr("class", "y-grid")
      .call(gridAxis)
      .selectAll("line")
      .attr("stroke", CHART_STYLES.gridColor);
    g.select(".y-grid .domain").remove();
  }

  // Label
  if (opts.label) {
    g.append("text")
      .attr("class", "y-label")
      .attr("transform", "rotate(-90)")
      .attr("x", -(scale.range()[0] + scale.range()[1]) / 2)
      .attr("y", -40)
      .attr("text-anchor", "middle")
      .attr("fill", CHART_STYLES.labelColor)
      .attr("font-size", CHART_STYLES.labelFontSize)
      .text(opts.label);
  }
}

// ── Legend ───────────────────────────────────────────────────────

/**
 * Render an interactive legend inside the SVG.
 *
 * @param {d3.Selection} svg - The root SVG selection
 * @param {{items: Array<{label: string, color: string, dashed?: boolean}>, x: number, y: number, onClick?: (label: string, index: number) => void}} opts
 */
function createLegend(svg, opts) {
  const items = opts.items.filter((d) => d.label && d.label !== "");
  const legendG = svg
    .append("g")
    .attr("class", "chart-legend")
    .attr("transform", `translate(${opts.x},${opts.y})`);

  const itemGs = legendG
    .selectAll(".legend-item")
    .data(items)
    .enter()
    .append("g")
    .attr("class", "legend-item")
    .attr("transform", (_, i) => `translate(0,${i * 18})`)
    .style("cursor", opts.onClick ? "pointer" : "default");

  // Color swatch
  itemGs.each(function (d) {
    const el = d3.select(this);
    if (d.dashed) {
      el.append("line")
        .attr("x1", 0)
        .attr("y1", 5)
        .attr("x2", 14)
        .attr("y2", 5)
        .attr("stroke", d.color)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "4,2");
    } else {
      el.append("rect")
        .attr("width", 12)
        .attr("height", 12)
        .attr("rx", 2)
        .attr("fill", d.color);
    }
  });

  // Label text
  itemGs
    .append("text")
    .attr("x", 18)
    .attr("y", 10)
    .attr("fill", CHART_STYLES.labelColor)
    .attr("font-size", CHART_STYLES.legendFontSize)
    .text((d) => d.label);

  // Click handler
  if (opts.onClick) {
    itemGs.on("click", (event, d) => {
      const idx = items.indexOf(d);
      opts.onClick(d.label, idx, d);
    });
  }

  return legendG;
}

// ── Tooltip ─────────────────────────────────────────────────────

/**
 * Create a tooltip div attached to a container.
 * Returns show/hide functions.
 *
 * @param {string} containerId - DOM element ID
 * @returns {{show: (event: MouseEvent, html: string) => void, hide: () => void, el: HTMLElement}}
 */
function createTooltip(containerId) {
  const container = document.getElementById(containerId);

  // Remove existing tooltip
  const existing = container.querySelector(".d3-tooltip");
  if (existing) existing.remove();

  const el = document.createElement("div");
  el.className = "d3-tooltip";
  el.style.cssText = `
    position: absolute;
    background: ${CHART_STYLES.tooltipBg};
    color: ${CHART_STYLES.tooltipColor};
    padding: ${CHART_STYLES.tooltipPadding};
    border-radius: ${CHART_STYLES.tooltipRadius};
    font-size: ${CHART_STYLES.tooltipFontSize};
    pointer-events: none;
    z-index: 100;
    white-space: nowrap;
    opacity: 0;
    transition: opacity 0.15s;
  `;
  container.style.position = "relative";
  container.appendChild(el);

  function show(event, html) {
    el.innerHTML = html;
    el.style.opacity = "1";
    const containerRect = container.getBoundingClientRect();
    let left = event.clientX - containerRect.left + 12;
    let top = event.clientY - containerRect.top - 10;
    // Keep within bounds
    const elRect = el.getBoundingClientRect();
    if (left + elRect.width > containerRect.width) {
      left = event.clientX - containerRect.left - elRect.width - 12;
    }
    if (top < 0) top = 4;
    el.style.left = left + "px";
    el.style.top = top + "px";
  }

  function hide() {
    el.style.opacity = "0";
  }

  return { show, hide, el };
}

// ── SVG Export ───────────────────────────────────────────────────

/**
 * Add an "Export SVG" button below a chart container.
 *
 * @param {string} containerId - DOM element ID containing the SVG
 * @param {string} filename - Download filename (without extension)
 */
function addExportButton(containerId, filename) {
  const container = document.getElementById(containerId);
  // Remove existing button
  const existing = container.querySelector(".export-svg-btn");
  if (existing) existing.remove();

  const btn = document.createElement("button");
  btn.className = "export-svg-btn";
  btn.textContent = "Export SVG";
  btn.style.cssText = `
    position: absolute;
    top: 4px;
    right: 4px;
    padding: 3px 8px;
    font-size: 11px;
    border: 1px solid #ccc;
    border-radius: 3px;
    background: #f8f9fa;
    color: #555;
    cursor: pointer;
    z-index: 10;
    opacity: 0.6;
    transition: opacity 0.15s;
  `;
  btn.addEventListener("mouseenter", () => (btn.style.opacity = "1"));
  btn.addEventListener("mouseleave", () => (btn.style.opacity = "0.6"));
  btn.addEventListener("click", () => {
    const svgEl = container.querySelector("svg");
    if (!svgEl) return;
    const clone = svgEl.cloneNode(true);
    clone.querySelectorAll(".no-export").forEach((el) => el.remove());
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(clone);
    const blob = new Blob([svgStr], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${filename}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  });
  container.appendChild(btn);

  // Remove existing PNG button
  const existingPng = container.querySelector(".export-png-btn");
  if (existingPng) existingPng.remove();

  const pngBtn = document.createElement("button");
  pngBtn.className = "export-png-btn";
  pngBtn.textContent = "Export PNG";
  pngBtn.style.cssText = `
    position: absolute;
    top: 4px;
    right: 80px;
    padding: 3px 8px;
    font-size: 11px;
    border: 1px solid #ccc;
    border-radius: 3px;
    background: #f8f9fa;
    color: #555;
    cursor: pointer;
    z-index: 10;
    opacity: 0.6;
    transition: opacity 0.15s;
  `;
  pngBtn.addEventListener("mouseenter", () => (pngBtn.style.opacity = "1"));
  pngBtn.addEventListener("mouseleave", () => (pngBtn.style.opacity = "0.6"));
  pngBtn.addEventListener("click", () => {
    const svgEl = container.querySelector("svg");
    if (!svgEl) return;
    const clone = svgEl.cloneNode(true);
    clone.querySelectorAll(".no-export").forEach((el) => el.remove());
    const w = parseFloat(svgEl.getAttribute("width"));
    const h = parseFloat(svgEl.getAttribute("height"));
    const scale = 8;
    const canvas = document.createElement("canvas");
    canvas.width = w * scale;
    canvas.height = h * scale;
    const ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    const svgStr = new XMLSerializer().serializeToString(clone);
    const blob = new Blob([svgStr], {
      type: "image/svg+xml;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, w, h);
      URL.revokeObjectURL(url);
      canvas.toBlob((pngBlob) => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(pngBlob);
        a.download = `${filename}.png`;
        a.click();
        URL.revokeObjectURL(a.href);
      });
    };
    img.src = url;
  });
  container.appendChild(pngBtn);
}

// ── KDE ─────────────────────────────────────────────────────────

/**
 * Gaussian kernel density estimator.
 * Returns a function that evaluates density at any point x.
 *
 * @param {number[]} values - Sample data
 * @param {number} [bandwidth] - Bandwidth (uses Silverman's rule if omitted)
 * @returns {(x: number) => number} Density function
 */
function createKDE(values, bandwidth) {
  const n = values.length;
  if (n === 0) return () => 0;
  const sorted = [...values].sort((a, b) => a - b);
  if (!bandwidth || bandwidth <= 0) {
    const mean = d3.mean(sorted);
    const std = Math.sqrt(d3.mean(sorted.map((v) => (v - mean) ** 2)));
    const q1 = d3.quantile(sorted, 0.25);
    const q3 = d3.quantile(sorted, 0.75);
    const iqr = q3 - q1;
    bandwidth = 0.9 * Math.min(std, iqr / 1.34) * n ** -0.2;
    if (bandwidth <= 0) bandwidth = std * 0.5 || 1;
  }
  const factor = 1 / (n * bandwidth * Math.sqrt(2 * Math.PI));
  return function (x) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const u = (x - sorted[i]) / bandwidth;
      sum += Math.exp(-0.5 * u * u);
    }
    return sum * factor;
  };
}

// ── Letter-Value Plot ───────────────────────────────────────────

/**
 * Compute letter-value levels from sorted values.
 * Returns array of {lower, upper, depth} from outermost to innermost.
 *
 * @param {number[]} sorted - Pre-sorted values
 * @returns {Array<{lower: number, upper: number, depth: number}>}
 */
function computeLetterValues(sorted) {
  const n = sorted.length;
  if (n === 0) return [];

  function quantile(arr, q) {
    const pos = (arr.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (base + 1 < arr.length)
      return arr[base] + rest * (arr[base + 1] - arr[base]);
    return arr[base];
  }

  const levels = [];
  const maxDepth = Math.max(1, Math.floor(Math.log2(n)) - 1);

  for (let k = 1; k <= maxDepth; k++) {
    const p = 0.5 ** k;
    const lower = quantile(sorted, p);
    const upper = quantile(sorted, 1 - p);
    levels.push({ lower, upper, depth: k });
    // Stop if the level covers fewer than ~4 data points on each tail
    if (n * p < 4) break;
  }
  return levels;
}

/**
 * Draw a letter-value (boxen) plot as SVG elements.
 *
 * @param {d3.Selection} g - Parent group
 * @param {{values: number[], x: number, width: number, yScale: d3.Scale, fillColor: string, strokeColor: string}} opts
 */
function drawLetterValuePlot(g, opts) {
  const { values, x, width, yScale, fillColor, strokeColor } = opts;
  if (values.length === 0) return;
  const sorted = [...values].sort((a, b) => a - b);
  const levels = computeLetterValues(sorted);
  const nLevels = levels.length;
  const lvG = g.append("g").attr("class", "lettervalue");

  // Parse base color for alpha blending
  const baseColor = fillColor;

  // Draw levels from outermost (narrowest box) to innermost (widest box)
  for (let i = 0; i < nLevels; i++) {
    const lv = levels[i];
    const boxW = width * ((i + 1) / nLevels);
    const xOff = x + (width - boxW) / 2;
    const alpha = 0.2 + 0.6 * ((i + 1) / nLevels);

    lvG
      .append("rect")
      .attr("x", xOff)
      .attr("y", yScale(lv.upper))
      .attr("width", boxW)
      .attr("height", Math.max(0, yScale(lv.lower) - yScale(lv.upper)))
      .attr("fill", fillColor)
      .attr("fill-opacity", alpha)
      .attr("stroke", strokeColor)
      .attr("stroke-opacity", 0.4)
      .attr("stroke-width", 0.5);
  }

  // Median line
  const median = d3.quantile(sorted, 0.5);
  lvG
    .append("line")
    .attr("x1", x)
    .attr("y1", yScale(median))
    .attr("x2", x + width)
    .attr("y2", yScale(median))
    .attr("stroke", strokeColor)
    .attr("stroke-width", 2);

  // Whisker line spanning full data range
  lvG
    .append("line")
    .attr("x1", x + width / 2)
    .attr("y1", yScale(sorted[0]))
    .attr("x2", x + width / 2)
    .attr("y2", yScale(sorted[sorted.length - 1]))
    .attr("stroke", strokeColor)
    .attr("stroke-width", 0.5);

  return lvG;
}

// ── Boxplot ─────────────────────────────────────────────────────

/**
 * Compute box plot statistics from an array of values.
 *
 * @param {number[]} values
 * @returns {{min: number, q1: number, median: number, q3: number, max: number, whiskerLow: number, whiskerHigh: number, outliers: number[]}}
 */
function computeBoxplotStats(values) {
  if (values.length === 0) {
    return {
      min: 0,
      q1: 0,
      median: 0,
      q3: 0,
      max: 0,
      whiskerLow: 0,
      whiskerHigh: 0,
      outliers: [],
    };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;

  function quantile(arr, q) {
    const pos = (arr.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (base + 1 < arr.length) {
      return arr[base] + rest * (arr[base + 1] - arr[base]);
    }
    return arr[base];
  }

  const q1 = quantile(sorted, 0.25);
  const median = quantile(sorted, 0.5);
  const q3 = quantile(sorted, 0.75);
  const iqr = q3 - q1;
  const whiskerLow = Math.max(sorted[0], q1 - 1.5 * iqr);
  const whiskerHigh = Math.min(sorted[n - 1], q3 + 1.5 * iqr);
  const outliers = sorted.filter((v) => v < whiskerLow || v > whiskerHigh);

  return {
    min: sorted[0],
    q1,
    median,
    q3,
    max: sorted[n - 1],
    whiskerLow,
    whiskerHigh,
    outliers,
  };
}

/**
 * Draw a single box-and-whisker plot as SVG elements.
 *
 * @param {d3.Selection} g - Parent group
 * @param {{stats: Object, x: number, width: number, yScale: d3.Scale, fillColor: string, strokeColor: string}} opts
 */
function drawBoxplot(g, opts) {
  const { stats, x, width, yScale, fillColor, strokeColor } = opts;
  const boxG = g.append("g").attr("class", "boxplot");

  // Whisker line (vertical)
  boxG
    .append("line")
    .attr("x1", x + width / 2)
    .attr("y1", yScale(stats.whiskerHigh))
    .attr("x2", x + width / 2)
    .attr("y2", yScale(stats.whiskerLow))
    .attr("stroke", strokeColor)
    .attr("stroke-width", 1);

  // Whisker caps
  const capW = width * 0.5;
  for (const val of [stats.whiskerLow, stats.whiskerHigh]) {
    boxG
      .append("line")
      .attr("x1", x + width / 2 - capW / 2)
      .attr("y1", yScale(val))
      .attr("x2", x + width / 2 + capW / 2)
      .attr("y2", yScale(val))
      .attr("stroke", strokeColor)
      .attr("stroke-width", 1);
  }

  // Box (Q1 to Q3)
  boxG
    .append("rect")
    .attr("x", x)
    .attr("y", yScale(stats.q3))
    .attr("width", width)
    .attr("height", Math.max(0, yScale(stats.q1) - yScale(stats.q3)))
    .attr("fill", fillColor)
    .attr("stroke", strokeColor)
    .attr("stroke-width", 1);

  // Median line
  boxG
    .append("line")
    .attr("x1", x)
    .attr("y1", yScale(stats.median))
    .attr("x2", x + width)
    .attr("y2", yScale(stats.median))
    .attr("stroke", strokeColor)
    .attr("stroke-width", 2);

  // Outliers
  for (const val of stats.outliers) {
    boxG
      .append("circle")
      .attr("cx", x + width / 2)
      .attr("cy", yScale(val))
      .attr("r", 2)
      .attr("fill", strokeColor)
      .attr("opacity", 0.6);
  }

  return boxG;
}
