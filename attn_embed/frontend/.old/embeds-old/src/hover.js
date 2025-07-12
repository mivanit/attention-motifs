/**
 * Creates a hover template based on provided hover columns
 * @param {Array<string>} hoverColumns - Columns to show in hover text
 * @returns {string} - Formatted hover template string
 */
function createHoverTemplate(hoverColumns) {
  let template = "<b>Point Info</b><br>";

  // Add each hover column to the template
  hoverColumns.forEach((col, index) => {
    // Display just the last part of the column name after the last dot
    const displayName = col.split(".").pop();
    template += `${displayName}: %{customdata[${index}]}<br>`;
  });

  // Add coordinates at the end
  template += "coord: [%{x:.2f}, %{y:.2f}, %{z:.2f}]<extra></extra>";

  return template;
}

/**
 * Creates a trace configuration for a Plotly 3D scatter plot with dynamic hover template
 * @param {Array<string>} hoverColumns - Columns to display in hover text
 * @returns {Object} - Trace configuration object
 */
function createDynamicTraceConfig(hoverColumns) {
  return {
    mode: "markers",
    type: "scatter3d",
    hovertemplate: createHoverTemplate(hoverColumns),
  };
}

/**
 * Generate customdata for points based on their indices
 *
 * @param {Object} plotData - The plot data object with all data columns
 * @param {Array<number>} indices - Array of point indices to generate customdata for
 * @param {string} selectionColumn - Column used for selection (will be included if not one of the hover fields)
 * @param {Array<string>} hoverColumns - Columns to show in hover text
 * @returns {Array<Array>} - Array of customdata arrays for each point
 */
function generateCustomdata(
  plotData,
  indices,
  selectionColumn = null,
  hoverColumns = ["activation.cls", "activation.prompt"],
) {
  // Generate customdata for hover and selection
  return indices.map((i) => {
    // existing hover data
    const data = hoverColumns.map((col) => plotData[col]?.[i] ?? "N/A");

    // if we're including the selection column
    if (selectionColumn && !hoverColumns.includes(selectionColumn)) {
      data.push(plotData[selectionColumn][i]);
    }

    // also store the global index at the end
    data.push(i);
    return data;
  });
}
