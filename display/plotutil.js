/**
 * Creates a layout configuration for a Plotly 3D scatter plot
 * @param {string} title - Title of the plot
 * @param {Object|null} cameraPosition - Optional camera position to preserve
 * @param {Object} axes - Object with x, y, z indices of PCA components to use
 * @returns {Object} Layout configuration object for Plotly
 */
function createPlotLayout(title, cameraPosition = null, axes = { x: 0, y: 1, z: 2 }) {
	const layout = {
		title: title,
		scene: {
			xaxis: { title: `PC${axes.x}` },
			yaxis: { title: `PC${axes.y}` },
			zaxis: { title: `PC${axes.z}` },
			camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
		},
		margin: { l: 0, r: 0, b: 0, t: 50 },
		hovermode: 'closest'
	};

	// Apply saved camera position if provided
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
		// hovertemplate: '%{customdata}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
		hovertemplate: '%{customdata}<br>coord: [%{x:.2f}, %{y:.2f}, %{z:.2f}]',
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
 * Create traces grouped by categorical value (for initial view)
 * @param {Object} plotData - The plot data object with coordinates and metadata
 * @param {string} selectedColumn - The column to group by
 * @param {Object} dataFrame - The DataFrame containing the data
 * @param {Object} options - Configuration options
 * @param {Object} options.defaultTraceConfig - The default trace configuration
 * @param {number} options.selectedSize - The point size for selected points
 * @param {number} options.selectedOpacity - The opacity for selected points
 * @returns {Array} - Array of trace objects for Plotly
 */
function createTracesByCategory(plotData, selectedColumn, dataFrame, options) {
	console.time('Create Category Traces');
	
	if (!plotData || !selectedColumn) return [];
	
	const {
		defaultTraceConfig = createDefaultTraceConfig(),
		selectedSize = 6,
		selectedOpacity = 1.0
	} = options || {};

	const uniqueValues = [...dataFrame.col_unique(selectedColumn)];
	console.log(`Found ${uniqueValues.length} unique values for ${selectedColumn}`);

	const traces = [];

	// Colors for category differentiation
	const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'];

	// Create a trace for each unique value
	uniqueValues.forEach((value, index) => {
		console.log(`Creating trace for ${value}...`);
		const indices = [];

		// Find all points with this value
		for (let i = 0; i < plotData[selectedColumn].length; i++) {
			if (plotData[selectedColumn][i] === value) {
				indices.push(i);
			}
		}

		console.log(`Found ${indices.length} points for ${value}`);

		traces.push({
			...defaultTraceConfig,
			x: indices.map(i => plotData.x[i]),
			y: indices.map(i => plotData.y[i]),
			z: indices.map(i => plotData.z[i]),
			name: String(value),
			marker: {
				size: selectedSize,
				color: colors[index % colors.length],
				opacity: selectedOpacity // Use the selected opacity for better visibility
			},
			customdata: indices.map(i => `${selectedColumn}: ${plotData[selectedColumn][i]}`)
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
		getSelectionColor = (index) => ['#ff7f0e', '#2ca02c', '#d62728'][index % 3]
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
			customdata: nonSelectedIndices.map(i => `${selectedColumn}: ${plotData[selectedColumn][i]}`)
		});
	}

	// Add a trace for each selected value
	Object.keys(selectedIndices).forEach((value, index) => {
		const indices = selectedIndices[value];
		console.log(`Creating trace for selected value "${value}" with ${indices.length} points...`);

		traces.push({
			...defaultTraceConfig,
			x: indices.map(i => plotData.x[i]),
			y: indices.map(i => plotData.y[i]),
			z: indices.map(i => plotData.z[i]),
			name: `Selected: ${value}`,
			marker: {
				size: selectedSize,
				color: getSelectionColor(index),
				opacity: selectedOpacity
			},
			customdata: indices.map(i => `${selectedColumn}: ${plotData[selectedColumn][i]}`)
		});
	});

	console.timeEnd('Create Selection Traces');
	return traces;
}