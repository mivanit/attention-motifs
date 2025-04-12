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
