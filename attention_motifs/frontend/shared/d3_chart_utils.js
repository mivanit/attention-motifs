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
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(svgEl);
    const blob = new Blob([svgStr], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${filename}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  });
  container.appendChild(btn);
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
