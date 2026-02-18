// sparklines.js - Generate SVG sparklines and sparkbars from data arrays
// origin: https://github.com/mivanit/js-dev-toolkit
// license: GPLv3

const _PLOT_OPTS_DEFAULT = {
  width: 160,
  height: 80,
  color: "#4169E1",
  shading: 0.3,
  lineWidth: 2,
  markers: null,
  margin: 5,
  style: "line",
  barWidthRatio: 1,
  logScale: false,

  xAxis: {
    line: false,
    ticks: false,
    text_offset: 10,
    text_y_position: 0,
    label_margin: 5,
    limits_length: 2,
  },

  yAxis: {
    line: false,
    ticks: false,
    text_offset: 2,
    text_y_offset: 0,
    label_margin: 10,
    limits_length: 2,
  },

  xlims: null,
  ylims: null,

  bar: {
    opacity: 0.8,
    y_axis_shift_ratio: 0.6,
    y_axis_reduction_factor: 1,
  },

  axis_style: {
    color: "#ccc",
    width: 1,
    font_size: 10,
    text_color: "#666",
  },

  gradient: {
    start_offset: "0%",
    end_offset: "100%",
    end_opacity: 0,
  },

  min_range: 1,
};

function merge_options(options = {}) {
  return {
    ..._PLOT_OPTS_DEFAULT,
    ...options,
    xAxis: { ..._PLOT_OPTS_DEFAULT.xAxis, ...(options.xAxis || {}) },
    yAxis: { ..._PLOT_OPTS_DEFAULT.yAxis, ...(options.yAxis || {}) },
    bar: { ..._PLOT_OPTS_DEFAULT.bar, ...(options.bar || {}) },
    axis_style: {
      ..._PLOT_OPTS_DEFAULT.axis_style,
      ...(options.axis_style || {}),
    },
    gradient: {
      ..._PLOT_OPTS_DEFAULT.gradient,
      ...(options.gradient || {}),
    },
  };
}

function process_values(values, yvalues = null) {
  if (!Array.isArray(values)) {
    throw new Error(`First parameter must be an array, got: ${typeof values}`);
  }

  if (values.length === 0) {
    throw new Error("Values array cannot be empty");
  }

  let xvals, yvals;

  if (yvalues === null) {
    yvals = values;
    xvals = Array.from({ length: values.length }, (_, i) => i);
  } else {
    if (!Array.isArray(yvalues)) {
      throw new Error(
        `Second parameter must be an array or null, got: ${typeof yvalues}`,
      );
    }
    xvals = values;
    yvals = yvalues;
  }

  if (xvals.length !== yvals.length) {
    throw new Error(
      `x-values length (${xvals.length}) must match y-values length (${yvals.length})`,
    );
  }

  return { xvalues: xvals, yvalues: yvals };
}

function plot(values, yvalues = null, options = {}) {
  const opts = merge_options(options);
  const { xvalues, yvalues: yvals } = process_values(values, yvalues);

  const needsLeftMargin = opts.yAxis.ticks;
  const needsBottomMargin = opts.xAxis.ticks;
  const barChartExtraMargin =
    opts.style === "bar" && (opts.yAxis.line || opts.yAxis.ticks)
      ? ((opts.width - 2 * opts.margin) / yvals.length) *
        opts.bar.y_axis_shift_ratio
      : 0;
  const leftMargin =
    (needsLeftMargin ? opts.margin + opts.yAxis.label_margin : opts.margin) +
    barChartExtraMargin;
  const bottomMargin = needsBottomMargin
    ? opts.margin + opts.xAxis.label_margin
    : opts.margin;

  const dataYmin = Math.min(...yvals);
  const dataYmax = Math.max(...yvals);
  let ymin = opts.ylims && opts.ylims[0] !== null ? opts.ylims[0] : dataYmin;
  let ymax = opts.ylims && opts.ylims[1] !== null ? opts.ylims[1] : dataYmax;

  if (opts.logScale) {
    if (yvals.some((v) => v < 0) || ymin < 0 || ymax < 0) {
      throw new Error("Log scale requires all values >= 0");
    }
  }

  const transformY = opts.logScale
    ? (y) => (y > 0 ? Math.log10(y) : 0)
    : (y) => y;

  const positiveYvals = opts.logScale ? yvals.filter((v) => v > 0) : yvals;
  const yminForScale =
    opts.logScale && positiveYvals.length > 0
      ? Math.min(...positiveYvals)
      : ymin;
  const ymaxForScale =
    opts.logScale && positiveYvals.length > 0
      ? Math.max(...positiveYvals)
      : ymax;

  const yminTransformed = transformY(yminForScale);
  const ymaxTransformed = transformY(ymaxForScale);
  const yrange = ymaxTransformed - yminTransformed || opts.min_range;

  const dataXmin = Math.min(...xvalues);
  const dataXmax = Math.max(...xvalues);
  const xmin = opts.xlims && opts.xlims[0] !== null ? opts.xlims[0] : dataXmin;
  const xmax = opts.xlims && opts.xlims[1] !== null ? opts.xlims[1] : dataXmax;
  const xrange = xmax - xmin || opts.min_range;

  const chartWidth = opts.width - leftMargin - opts.margin;
  const chartHeight = opts.height - opts.margin - bottomMargin;

  let svg = "<svg " + `width="${opts.width}" height="${opts.height}"` + ">";

  if (opts.style === "bar") {
    const baseY = opts.height - bottomMargin;
    const adjustedChartWidth = chartWidth - chartWidth / yvals.length / 2;

    let barSpacing;
    if (yvals.length === 1) {
      barSpacing = adjustedChartWidth;
    } else {
      const sortedIndices = [...Array(xvalues.length).keys()].sort(
        (a, b) => xvalues[a] - xvalues[b],
      );
      let minSpacing = Infinity;
      for (let j = 1; j < sortedIndices.length; j++) {
        const prevX = xvalues[sortedIndices[j - 1]];
        const currX = xvalues[sortedIndices[j]];
        const spacing = ((currX - prevX) / xrange) * adjustedChartWidth;
        if (spacing < minSpacing) minSpacing = spacing;
      }
      barSpacing =
        minSpacing === Infinity
          ? adjustedChartWidth / yvals.length
          : minSpacing;
    }

    const actualBarWidth = barSpacing * opts.barWidthRatio;

    yvals.forEach((yval, i) => {
      if (opts.logScale && yval === 0) {
        return;
      }

      const xval = xvalues[i];
      const x = leftMargin + ((xval - xmin) / xrange) * adjustedChartWidth;
      const yvalTransformed = transformY(yval);
      const barHeight =
        Math.abs((yvalTransformed - yminTransformed) / yrange) * chartHeight;
      const barY = baseY - barHeight;

      const barX = x - actualBarWidth / 2;

      svg += `<rect x="${barX}" y="${barY}" width="${actualBarWidth}" height="${barHeight}" fill="${opts.color}" opacity="${opts.bar.opacity}"/>`;
    });
  } else {
    let path = "";
    let dots = "";
    const points = [];

    yvals.forEach((yval, i) => {
      const xval = xvalues[i];
      const x = leftMargin + ((xval - xmin) / xrange) * chartWidth;

      if (opts.logScale && yval === 0) {
        return;
      }

      const yvalTransformed = transformY(yval);
      const y =
        opts.margin +
        (1 - (yvalTransformed - yminTransformed) / yrange) * chartHeight;
      points.push({ x, y, xval, yval });

      const moveCmd =
        i === 0 || (opts.logScale && i > 0 && yvals[i - 1] === 0) ? "M" : "L";
      path += `${moveCmd} ${x} ${y} `;

      if (opts.markers !== null && opts.markers > 0) {
        dots += `<circle cx="${x}" cy="${y}" r="${opts.markers}" fill="${opts.color}"/>`;
      }
    });

    if (opts.shading !== false) {
      if (opts.shading === true) {
        for (let i = 0; i < points.length - 1; i++) {
          const p1 = points[i];
          const p2 = points[i + 1];
          const segmentPath = `M ${p1.x} ${p1.y} L ${p2.x} ${p2.y} L ${p2.x} ${opts.height - bottomMargin} L ${p1.x} ${opts.height - bottomMargin} Z`;
          svg += `<path d="${segmentPath}" fill="${opts.color}"/>`;
        }
      } else {
        const gradId = `grad${Date.now()}${Math.random()}`;
        const opacity = parseFloat(opts.shading);
        svg += `
          <defs>
            <linearGradient id="${gradId}" x1="${opts.gradient.start_offset}" y1="${opts.gradient.start_offset}" x2="${opts.gradient.start_offset}" y2="${opts.gradient.end_offset}">
              <stop offset="${opts.gradient.start_offset}" style="stop-color:${opts.color};stop-opacity:${opacity}"/>
              <stop offset="${opts.gradient.end_offset}" style="stop-color:${opts.color};stop-opacity:${opts.gradient.end_opacity}"/>
            </linearGradient>
          </defs>`;

        for (let i = 0; i < points.length - 1; i++) {
          const p1 = points[i];
          const p2 = points[i + 1];
          const segmentPath = `M ${p1.x} ${p1.y} L ${p2.x} ${p2.y} L ${p2.x} ${opts.height - bottomMargin} L ${p1.x} ${opts.height - bottomMargin} Z`;
          svg += `<path d="${segmentPath}" fill="url(#${gradId})"/>`;
        }
      }
    }

    svg += `<path d="${path}" stroke="${opts.color}" stroke-width="${opts.lineWidth}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`;

    svg += dots;
  }

  const formatYTick = (value) => {
    if (!opts.logScale) {
      return value;
    }
    if (value === 0) {
      return "0";
    }
    return value.toExponential(1);
  };

  if (opts.yAxis.line || opts.yAxis.ticks) {
    const barShift =
      opts.style === "bar"
        ? (chartWidth / yvals.length) * opts.bar.y_axis_shift_ratio
        : 0;
    const yAxisX = leftMargin - barShift;

    if (opts.yAxis.line) {
      svg += `<line x1="${yAxisX}" y1="${opts.margin}" x2="${yAxisX}" y2="${opts.height - bottomMargin}"
                    stroke="${opts.axis_style.color}" stroke-width="${opts.axis_style.width}"/>`;
    }
    if (opts.yAxis.ticks) {
      svg += `<text x="${yAxisX - opts.yAxis.text_offset}" y="${opts.margin + opts.yAxis.text_y_offset + opts.axis_style.font_size}" font-size="${opts.axis_style.font_size}" fill="${opts.axis_style.text_color}" text-anchor="end">${formatYTick(ymaxForScale)}</text>`;
      const lowerLabel =
        opts.logScale && ymin === 0 ? "0" : formatYTick(yminForScale);
      svg += `<text x="${yAxisX - opts.yAxis.text_offset}" y="${opts.height - bottomMargin + opts.yAxis.text_offset}" font-size="${opts.axis_style.font_size}" fill="${opts.axis_style.text_color}" text-anchor="end">${lowerLabel}</text>`;
    }
  }

  if (opts.xAxis.line) {
    const xAxisEnd =
      opts.style === "bar"
        ? leftMargin + chartWidth - chartWidth / yvals.length / 2
        : opts.width - opts.margin;
    svg += `<line x1="${leftMargin}" y1="${opts.height - bottomMargin}" x2="${xAxisEnd}" y2="${opts.height - bottomMargin}"
                    stroke="${opts.axis_style.color}" stroke-width="${opts.axis_style.width}"/>`;
  }
  if (opts.xAxis.ticks && yvals.length > 0) {
    const xTickEnd =
      opts.style === "bar"
        ? leftMargin + chartWidth - chartWidth / yvals.length / 2
        : opts.width - opts.margin;
    const xLabelY =
      opts.height -
      bottomMargin +
      opts.xAxis.text_offset +
      opts.xAxis.text_y_position;
    svg += `<text x="${leftMargin}" y="${xLabelY}" font-size="${opts.axis_style.font_size}" fill="${opts.axis_style.text_color}" text-anchor="middle">${xmin}</text>`;
    svg += `<text x="${xTickEnd}" y="${xLabelY}" font-size="${opts.axis_style.font_size}" fill="${opts.axis_style.text_color}" text-anchor="middle">${xmax}</text>`;
  }

  svg += "</svg>";

  return svg;
}

function sparkline(values, yvalues = null, options = {}) {
  return plot(values, yvalues, { ...options, style: "line" });
}

function sparkbars(values, yvalues = null, options = {}) {
  return plot(values, yvalues, { ...options, style: "bar" });
}
