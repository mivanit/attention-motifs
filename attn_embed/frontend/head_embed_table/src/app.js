let allPlots = [];
let filters = {
	methods: [],
	dimensions: [],
	neighbors: []
};

// Load and display the embedding data
async function loadEmbeddingData() {
	try {
		const response = await fetch('head_embed_plots.json');
		const data = await response.json();
		allPlots = data.plots;
		initializeFilters();
		displayEmbeddingPlots(allPlots);
	} catch (error) {
		console.error('Error loading embedding data:', error);
		document.getElementById('embedding-plots').innerHTML =
			'<p style="color: red; text-align: center;">Error loading embedding data. Please check that head_embed_plots.json exists.</p>';
	}
}

function initializeFilters() {
	// Extract unique values from data
	const uniqueMethods = [...new Set(allPlots.map(p => p.method))].sort();
	const uniqueDimensions = [...new Set(allPlots.map(p => p.n_components))].sort();
	const uniqueNeighbors = [...new Set(allPlots.map(p => parseInt(p.n_neighbors)))].sort((a, b) => a - b);

	// Initialize all filters as selected
	filters.methods = [...uniqueMethods];
	filters.dimensions = [...uniqueDimensions];
	filters.neighbors = [...uniqueNeighbors];

	// Build filter UI
	const filterContainer = document.getElementById('filter-controls');
	let html = '';

	// Methods checkboxes
	html += '<div class="filter-group">';
	html += '<label class="filter-label">Methods:</label>';
	html += '<div class="checkbox-group">';
	uniqueMethods.forEach(method => {
		html += `
                    <div class="checkbox-item">
                        <input type="checkbox" id="method-${method}" value="${method}" checked>
                        <label for="method-${method}">${method.toUpperCase()}</label>
                    </div>
                `;
	});
	html += '</div></div>';


	// Neighbors checkboxes
	html += '<div class="filter-group">';
	html += '<label class="filter-label" style="cursor: pointer;" onclick="toggleNeighborCheckboxes()">Number of Neighbors:</label>';
	html += '<div class="checkbox-group">';
	uniqueNeighbors.forEach(neighbors => {
		html += `
                    <div class="checkbox-item">
                        <input type="checkbox" id="neighbors-${neighbors}" value="${neighbors}" checked>
                        <label for="neighbors-${neighbors}">${neighbors}</label>
                    </div>
                `;
	});
	html += '</div></div>';

	// Buttons
	html += '<div class="filter-buttons">';
	html += '<button class="btn-apply" onclick="applyFilters()">Apply Filters</button>';
	html += '<button class="btn-reset" onclick="resetFilters()">Reset All</button>';
	html += '</div>';

	filterContainer.innerHTML = html;
}

function toggleNeighborCheckboxes() {
	const checkboxes = document.querySelectorAll('input[id^="neighbors-"]');
	const allChecked = Array.from(checkboxes).every(cb => cb.checked);
	checkboxes.forEach(cb => {
		cb.checked = !allChecked;
	});
}

function applyFilters() {
	// Get selected methods
	filters.methods = [];
	document.querySelectorAll('input[id^="method-"]:checked').forEach(cb => {
		filters.methods.push(cb.value);
	});

	// Only use 2D plots
	filters.dimensions = ['2'];

	// Get selected neighbors
	filters.neighbors = [];
	document.querySelectorAll('input[id^="neighbors-"]:checked').forEach(cb => {
		filters.neighbors.push(parseInt(cb.value));
	});

	// Filter plots (only 2D plots for display, but keep allPlots for 3D button lookup)
	const filteredPlots = allPlots.filter(plot => {
		const neighbors = parseInt(plot.n_neighbors);
		return filters.methods.includes(plot.method) &&
			plot.n_components === '2' &&
			filters.neighbors.includes(neighbors);
	});

	displayEmbeddingPlots(filteredPlots);
}

function resetFilters() {
	// Check all checkboxes
	document.querySelectorAll('input[type="checkbox"]').forEach(cb => {
		cb.checked = true;
	});

	// Apply filters
	applyFilters();
}

function displayEmbeddingPlots(plots) {
	const container = document.getElementById('embedding-plots');

	if (plots.length === 0) {
		container.innerHTML = '<p style="text-align: center; color: #666; padding: 40px;">No visualizations match the selected filters.</p>';
		return;
	}

	let html = '';

	// Group plots by method, only showing 2D
	const methodGroups = {};
	plots.filter(plot => plot.n_components === '2').forEach(plot => {
		if (!methodGroups[plot.method]) {
			methodGroups[plot.method] = [];
		}
		methodGroups[plot.method].push(plot);
	});

	// Generate HTML for each method
	Object.keys(methodGroups).sort().forEach(method => {
		html += `<div class="method-section">`;
		html += `<div class="method-title">${method.toUpperCase()}</div>`;
		html += `<div class="plots-grid">`;

		// Sort by n_neighbors and find corresponding 3D plot for button
		methodGroups[method]
			.sort((a, b) => parseInt(a.n_neighbors) - parseInt(b.n_neighbors))
			.forEach(plot => {
				// Find the corresponding 3D plot for the button
				const plot3d = allPlots.find(p =>
					p.method === plot.method &&
					p.n_neighbors === plot.n_neighbors &&
					p.n_components === '3'
				);

				html += `
                            <div class="plot-item">
                                <div class="plot-title">${plot.n_neighbors} neighbors</div>
                                <div class="plot-info">${plot.prefix}</div>
                                ${plot3d ? `<a href="${plot3d.embed_url}" target="_blank" class="plot-3d-btn">3D</a>` : ''}
                                <img src="${plot.svg_filename}" alt="${plot.prefix}" loading="lazy">
                            </div>
                        `;
			});

		html += `</div></div>`;
	});

	container.innerHTML = html;
}