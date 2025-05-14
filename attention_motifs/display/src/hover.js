/**
 * Create dynamic hover template based on hover columns configuration
 */
function createHoverTemplate(hoverColumns) {
	let template = '<b>Point Info</b><br>';

	// Add each hover column to the template
	hoverColumns.forEach((col, index) => {
		// Display just the last part of the column name after the last dot
		const displayName = col.split('.').pop();
		template += `${displayName}: %{customdata[${index}]}<br>`;
	});

	// Add coordinates at the end
	template += 'coord: [%{x:.2f}, %{y:.2f}, %{z:.2f}]<extra></extra>';

	return template;
}

/**
 * Creates a default trace configuration for a Plotly 3D scatter plot with dynamic hover template
 */
function createDynamicTraceConfig(hoverColumns) {
	return {
		mode: 'markers',
		type: 'scatter3d',
		hovertemplate: createHoverTemplate(hoverColumns)
	};
}

/**
 * Updated generateCustomdata function to use configurable hover columns
 */
function generateCustomdata(plotData, indices, selectionColumn = null, hoverColumns = ['activation.cls', 'activation.prompt']) {
	return indices.map(i => {
		// Start with an array for the hover columns
		const data = hoverColumns.map(col => plotData[col]?.[i] ?? 'N/A');

		// Add the selection column as an additional element if it's not already in the hover columns
		if (selectionColumn && !hoverColumns.includes(selectionColumn)) {
			data.push(plotData[selectionColumn][i]);
		}

		return data;
	});
}