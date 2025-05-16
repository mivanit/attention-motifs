/**
 * Creates a layout configuration for a Plotly 3D scatter plot
 * @param {string} title - Title of the plot
 * @param {Object|null} cameraPosition - Optional camera position to preserve
 * @param {Object} axes - Object with x, y, z indices of PCA components to use
 * @param {Object|null} axisLabels - Optional object with labels for each axis
 * @returns {Object} Layout configuration object for Plotly
 */
function createPlotLayout(title, cameraPosition = null, axes = { x: 0, y: 1, z: 2 }, axisLabels = null) {
	const layout = {
		title: title,
		scene: {
			xaxis: { title: axisLabels ? axisLabels.x : `PC${axes.x}` },
			yaxis: { title: axisLabels ? axisLabels.y : `PC${axes.y}` },
			zaxis: { title: axisLabels ? axisLabels.z : `PC${axes.z}` },
			camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
		},
		// Set larger legend font size and constant marker sizing
		legend: {
			font: {
				size: 16     // Increase text size
			},
			itemsizing: 'constant',  // Keeps marker size constant
			itemwidth: 200            // Increase space for the legend markers
		},
		margin: { l: 0, r: 0, b: 0, t: 50 },
		hovermode: 'closest'
	};

	// Preserve saved camera position if available
	if (cameraPosition) {
		layout.scene.camera = cameraPosition;
	}

	return layout;
}


/**
 * Creates a default trace configuration for a Plotly 3D scatter plot
 * @returns {Object} Base trace configuration
 */
function createDefaultTraceConfig() {
	return {
		mode: 'markers',
		type: 'scatter3d',
		// The hover template that matches the Python format
		hovertemplate: '<b>Point Info</b><br>' +
			'head: %{customdata[0]}<br>' +
			'prompt: %{customdata[1]}<br>' +
			'coord: [%{x:.2f}, %{y:.2f}, %{z:.2f}]<extra></extra>',
	};
}

/**
 * Gets the current camera position from a Plotly plot
 * @param {Object} plotContainer - The Plotly plot container reference
 * @returns {Object|null} - The current camera position or null if it cannot be retrieved
 */
function getCurrentCameraPosition(plotContainer) {
	if (plotContainer && plotContainer._fullLayout) {
		try {
			// Deep clone to avoid reference issues
			return JSON.parse(JSON.stringify(
				plotContainer._fullLayout.scene.camera
			));
		} catch (e) {
			console.warn("Could not get camera position:", e);
			return null;
		}
	}
	return null;
}


/**
 * Cache object for storing generated customdata
 */
const customdataCache = {
	data: {},

	/**
	 * Get cached customdata if available
	 * 
	 * @param {string} cacheKey - Cache key
	 * @returns {Array|null} - Cached customdata or null if not found
	 */
	get(cacheKey) {
		return this.data[cacheKey] || null;
	},

	/**
	 * Store customdata in cache
	 * 
	 * @param {string} cacheKey - Cache key
	 * @param {Array} customdata - Customdata to store
	 */
	set(cacheKey, customdata) {
		this.data[cacheKey] = customdata;
	},

	/**
	 * Clear the entire cache or a specific key
	 * 
	 * @param {string|null} cacheKey - Optional specific key to clear
	 */
	clear(cacheKey = null) {
		if (cacheKey) {
			delete this.data[cacheKey];
		} else {
			this.data = {};
		}
	}
};

/**
 * Generate customdata with caching support
 * 
 * @param {Object} plotData - The plot data object with all data columns
 * @param {Array<number>} indices - Array of point indices to generate customdata for
 * @param {string} selectionColumn - Column used for selection
 * @param {boolean} useCache - Whether to use the cache
 * @returns {Array<Array>} - Array of customdata arrays for each point
 */
function getCustomdata(plotData, indices, selectionColumn = null, hoverColumns = ['activation.cls', 'activation.prompt'], useCache = true) {
	// Only use cache if instructed and we have indices
	if (useCache && indices && indices.length > 0) {
		// Create a cache key based on indices, selection column, and hover columns
		const hoverColumnsKey = hoverColumns.join(',');
		const cacheKey = `${selectionColumn || 'default'}_${hoverColumnsKey}_${indices[0]}_${indices.length}`;

		// Try to get from cache first
		const cachedData = customdataCache.get(cacheKey);
		if (cachedData) {
			return cachedData;
		}

		// Generate new data and store in cache
		const customdata = generateCustomdata(plotData, indices, selectionColumn, hoverColumns);
		customdataCache.set(cacheKey, customdata);
		return customdata;
	}

	// If not using cache, generate directly
	return generateCustomdata(plotData, indices, selectionColumn, hoverColumns);
}

/**
 * Clear the customdata cache when data changes
 */
function clearCustomdataCache() {
	customdataCache.clear();
}



/**
 * Create traces grouped by categorical value (for initial view).
 * Points with a value of null, "None" or "unknown" in selectedColumn
 * get merged into the "Other Points" trace and use the non-selected style.
 * @param {Object} plotData - The plot data object with coordinates and metadata
 * @param {string} selectedColumn - The column to group by
 * @param {Object} dataFrame - The DataFrame containing the data
 * @param {Object} options - Configuration options
 * @param {Object} options.defaultTraceConfig - The default trace configuration
 * @param {number} options.selectedSize - The point size for selected points
 * @param {number} options.selectedOpacity - The opacity for selected points
 * @param {string} options.selectionColumn - The column used for selection (optional)
 * @returns {Array} - Array of trace objects for Plotly
 */
function createTracesByCategory(
	plotData,
	selectedColumn,
	dataFrame,
	options
) {
	console.time('Create Category Traces');
	if (!plotData || !selectedColumn) return [];

	const {
		// Defaults if not provided
		defaultTraceConfig = createDefaultTraceConfig(),
		selectedSize = 6,
		selectedOpacity = 1.0,

		// The non-selected style
		nonSelectedSize = 4,
		nonSelectedOpacity = 0.4,
		nonSelectedColor = '#969696',

		// For customdata / hover
		selectionColumn = selectedColumn,
		hoverColumns = ['activation.cls', 'activation.prompt'],
		useCache = true,
	} = options || {};

	// -----------------------------------------------------
	// 1) Check if this column is numeric.
	//    We'll do a continuous colorscale if numeric.
	// -----------------------------------------------------
	const columnData = plotData[selectedColumn];

	// Quick test for numeric: every value is a number and not NaN.
	// (Adjust logic to handle nulls or missing data as you prefer.)
	const isNumeric = columnData.every(
		(val) => typeof val === 'number' && !isNaN(val)
	);

	if (isNumeric) {
		// Single trace with continuous colormap
		// Filter out invalid values (null, undefined, NaN) when computing min/max
		const validValues = columnData.filter((v) => v != null && !isNaN(v));
		const cmin = Math.min(...validValues);
		const cmax = Math.max(...validValues);

		// We'll just include *all* points in a single trace
		const indices = columnData.map((_, i) => i);

		// Build customdata for all points
		const customdata = getCustomdata(
			plotData,
			indices,
			selectionColumn,
			hoverColumns,
			useCache
		);

		const trace = {
			...defaultTraceConfig,
			x: indices.map((i) => plotData.x[i]),
			y: indices.map((i) => plotData.y[i]),
			z: indices.map((i) => plotData.z[i]),
			name: selectedColumn,
			marker: {
				size: selectedSize,
				opacity: selectedOpacity,
				color: columnData,
				colorscale: 'Viridis',  // or another Plotly colormap
				cmin: cmin,
				cmax: cmax,
				showscale: true,        // show the color scale legend
			},
			customdata,
		};

		console.timeEnd('Create Category Traces');
		return [trace];
	}

	// -----------------------------------------------------
	// 2) Otherwise, treat it as categorical as before.
	//    (Below is your existing logic, unchanged.)
	// -----------------------------------------------------
	// Identify distinct values in this column
	const allValues = [...dataFrame.col_unique(selectedColumn)];

	// Filter out known "invalid" entries
	const knownValues = allValues.filter(
		(v) => v !== null && v !== 'None' && v !== 'unknown'
	);

	// Collect indices for all unknown (null/None/unknown)
	const unknownIndices = [];
	for (let i = 0; i < plotData[selectedColumn].length; i++) {
		const val = plotData[selectedColumn][i];
		if (val === null || val === 'None' || val === 'unknown') {
			unknownIndices.push(i);
		}
	}

	// Prepare result array of Plotly trace objects
	const traces = [];

	// Create one "Other Points" trace for unknown/null values
	if (unknownIndices.length > 0) {
		const customdata = getCustomdata(
			plotData,
			unknownIndices,
			selectionColumn,
			hoverColumns,
			useCache
		);

		traces.push({
			...defaultTraceConfig,
			x: unknownIndices.map((i) => plotData.x[i]),
			y: unknownIndices.map((i) => plotData.y[i]),
			z: unknownIndices.map((i) => plotData.z[i]),
			name: 'Other Points',
			marker: {
				size: nonSelectedSize,
				color: nonSelectedColor,
				opacity: nonSelectedOpacity,
			},
			customdata,
		});
	}

	// Make one trace per valid category
	const colors = [
		'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
		'#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
	];

	knownValues.forEach((value, index) => {
		// Gather row indices for this category
		const indices = [];
		for (let i = 0; i < plotData[selectedColumn].length; i++) {
			if (plotData[selectedColumn][i] === value) {
				indices.push(i);
			}
		}

		// Build customdata for hover text
		const customdata = getCustomdata(
			plotData,
			indices,
			selectionColumn,
			hoverColumns,
			useCache
		);

		traces.push({
			...defaultTraceConfig,
			x: indices.map((i) => plotData.x[i]),
			y: indices.map((i) => plotData.y[i]),
			z: indices.map((i) => plotData.z[i]),
			name: String(value),
			marker: {
				size: selectedSize,
				color: colors[index % colors.length],
				opacity: selectedOpacity,
			},
			customdata,
		});
	});

	console.timeEnd('Create Category Traces');
	return traces;
}


/**
 * Create traces with selection highlighting
 * @param {Object} plotData - The plot data object with coordinates and metadata
 * @param {string} selectedColumn - The column to filter by
 * @param {Array} selectedValues - Array of selected values
 * @param {Object} options - Configuration options
 * @param {Object} options.defaultTraceConfig - The default trace configuration
 * @param {number} options.selectedSize - The point size for selected points
 * @param {number} options.nonSelectedSize - The point size for non-selected points
 * @param {number} options.selectedOpacity - The opacity for selected points
 * @param {number} options.nonSelectedOpacity - The opacity for non-selected points
 * @param {string} options.nonSelectedColor - The color for non-selected points
 * @param {Function} options.getSelectionColor - Function to get color for a selection index
 * @param {string} options.colorByColumn - The column used for coloring (optional)
 * @returns {Array} - Array of trace objects for Plotly
 */
function createTracesWithSelection(plotData, selectedColumn, selectedValues, options) {
	console.time('Create Selection Traces');

	if (!plotData) return [];

	const {
		defaultTraceConfig = createDefaultTraceConfig(),
		selectedSize = 6,
		nonSelectedSize = 4,
		selectedOpacity = 1.0,
		nonSelectedOpacity = 0.4,
		nonSelectedColor = '#969696',
		getSelectionColor = (index) => ['#ff7f0e', '#2ca02c', '#d62728'][index % 3], // Default if not provided
		colorByColumn = selectedColumn, // Default to selectedColumn if not provided
		useCache = true
	} = options || {};

	// First, separate points into selected and non-selected
	const selectedIndices = {};
	const nonSelectedIndices = [];

	// Group points by selection status
	for (let i = 0; i < plotData[selectedColumn].length; i++) {
		const value = plotData[selectedColumn][i];

		if (selectedValues.includes(value)) {
			if (!selectedIndices[value]) {
				selectedIndices[value] = [];
			}
			selectedIndices[value].push(i);
		} else {
			nonSelectedIndices.push(i);
		}
	}

	console.log(`Selection counts: ${Object.keys(selectedIndices).length} selected values, ${nonSelectedIndices.length} non-selected points`);

	const traces = [];

	// Add non-selected points (with configurable color/opacity)
	if (nonSelectedIndices.length > 0) {
		console.log(`Creating trace for ${nonSelectedIndices.length} non-selected points...`);

		// Get customdata with caching for better performance
		const customdata = getCustomdata(plotData, nonSelectedIndices, selectedColumn, options.hoverColumns, useCache);

		traces.push({
			...defaultTraceConfig,
			x: nonSelectedIndices.map(i => plotData.x[i]),
			y: nonSelectedIndices.map(i => plotData.y[i]),
			z: nonSelectedIndices.map(i => plotData.z[i]),
			name: 'Other Points',
			marker: {
				size: nonSelectedSize,
				color: nonSelectedColor,
				opacity: nonSelectedOpacity
			},
			customdata: customdata
		});
	}

	// Add a trace for each selected value
	const selectedValuesArray = Object.keys(selectedIndices);
	selectedValuesArray.forEach((value, index) => {
		const indices = selectedIndices[value];
		console.log(`Creating trace for selected value "${value}" with ${indices.length} points...`);

		// Get customdata with caching for better performance
		const customdata = getCustomdata(plotData, indices, selectedColumn, options.hoverColumns, useCache);

		// Use the getSelectionColor function passed as an option
		const colorIndex = selectedValues.indexOf(value);
		const color = getSelectionColor(colorIndex);

		traces.push({
			...defaultTraceConfig,
			x: indices.map(i => plotData.x[i]),
			y: indices.map(i => plotData.y[i]),
			z: indices.map(i => plotData.z[i]),
			name: `Selected: ${value}`,
			marker: {
				size: selectedSize,
				color: color,
				opacity: selectedOpacity
			},
			customdata: customdata
		});
	});

	console.timeEnd('Create Selection Traces');
	return traces;
}