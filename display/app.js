// Create Vue app
const app = Vue.createApp({
	data() {
		return {
			title: 'Interactive 3D PCA Plot',
			statusMessage: 'Loading data...',
			isLoading: true,
			loadingMessage: 'Loading data...',
			loadingDetail: 'Initializing...',
			loadingProgress: false,
			loadingProgressPercent: 0,
			processingSelection: false,
			selectedValues: [],
			plotData: null,
			pcaArray: null,
			dataFrame: null,
			// Column selection dropdowns
			colorByColumn: '', // Column for default coloring
			selectionColumn: '', // Column for selection coloring
			pendingColorByColumn: '', // Temp storage for color selection before applying
			pendingSelectionColumn: '', // Temp storage for selection column before applying
			showColorDropdown: false, // Control color dropdown visibility
			showSelectionDropdown: false, // Control selection dropdown visibility
			isUpdatingPlot: false, // Flag to show loading during plot updates
			availableColumns: [],
			pcaAxes: { x: 0, y: 1, z: 2 },
			availablePcaAxes: [], // Will be populated with available components
			selectionColors: [
				'#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
				'#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
			],
			initialSelectionColors: [
				'#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
				'#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
			],
			nonSelectedColor: '#969696', // Gray for non-selected points
			nonSelectedOpacity: 0.05,
			nonSelectedSize: 1,
			selectedOpacity: 0.2,
			selectedSize: 2,
			baseUrl: 'https://miv.name/pattern-lens/demo/?',
			dropdownTimeout: null,
			configCollapsed: false,
			// Default trace configuration
			defaultTraceConfig: createDefaultTraceConfig(),
			// Store current camera position
			currentCameraPosition: null,
			// Store selection timing information
			lastSelectionTime: 0,
			// Debounce timeout
			updatePlotTimeout: null
		};
	},

	computed: {
		// Compute combined URL with all selected values
		combinedUrl() {
			if (!this.selectedValues.length) return this.baseUrl;
			return this.baseUrl + this.selectedValues.map(val => encodeURIComponent(val)).join('&');
		}
	},

	methods: {
		// Load PCA data and metadata
		async loadData() {
			this.isLoading = true;
			this.loadingMessage = 'Loading PCA data...';
			this.loadingDetail = 'Requesting NPY file';
			this.statusMessage = 'Loading PCA data...';
			console.log('Starting data loading process');

			try {
				// Load PCA data using the function from dataLoader.js
				this.loadingDetail = 'Downloading NPY file...';
				this.pcaArray = await loadPcaData();

				this.loadingProgressPercent = 30;
				this.loadingProgress = true;

				// Setup available PCA axes
				this.availablePcaAxes = new Array(this.pcaArray.shape[1]).fill(0).map((_, i) => i);
				console.log('Available PCA components:', this.availablePcaAxes.length);

				// Load metadata using the function from dataLoader.js
				this.loadingMessage = 'Loading metadata...';
				this.loadingDetail = 'Requesting JSONL file';
				this.statusMessage = 'Loading feature metadata...';

				this.loadingProgressPercent = 50;
				this.loadingDetail = 'Processing JSONL file (takes a while)...';
				this.dataFrame = await loadMetadata();

				this.loadingProgressPercent = 70;

				// Process data using the function from dataLoader.js
				this.loadingMessage = 'Processing data...';
				this.loadingDetail = 'Preparing plot data';
				this.loadingProgressPercent = 80;
				console.time('Process Data');
				this.plotData = processData(this.pcaArray, this.dataFrame);
				// Update coordinates based on initial PCA axes selection
				updatePlotCoordinates(this.plotData, this.pcaAxes);
				console.timeEnd('Process Data');

				this.loadingProgressPercent = 90;

				// Find categorical columns using the function from dataLoader.js
				console.log('Finding categorical columns...');
				this.availableColumns = findCategoricalColumns(this.dataFrame);
				console.log('Available categorical columns:', this.availableColumns);

				// Set default selected columns
				this.colorByColumn = 'activation.model';
				this.pendingColorByColumn = this.colorByColumn;
				this.selectionColumn = 'activation.model'; // Initialize with the same default
				this.pendingSelectionColumn = this.selectionColumn;
				console.log('Initial color column:', this.colorByColumn);
				console.log('Initial selection column:', this.selectionColumn);

				this.loadingProgressPercent = 95;
				this.statusMessage = 'Data loaded successfully';
				this.loadingMessage = 'Rendering plot...';
				this.loadingDetail = 'Creating visualization';

				// Initialize the plot after data is loaded
				this.initPlot();

				this.loadingProgressPercent = 100;
				this.isLoading = false;
				console.log('Data loading complete');
			} catch (error) {
				console.error("Error loading data:", error);
				this.statusMessage = `Error loading data: ${error.message}`;
				this.loadingMessage = 'Error loading data';
				this.loadingDetail = error.message;
				this.isLoading = false;
				this.loadingProgress = false;
			}
		},

		// Process loaded data into a format suitable for plotting
		processData() {
			if (!this.pcaArray || !this.dataFrame) {
				throw new Error("PCA data or metadata not loaded");
			}

			// Use the processData function from dataLoader.js
			this.plotData = processData(this.pcaArray, this.dataFrame);
			// Initialize with current axes selection
			this.updatePlotCoordinates();
		},

		// Update x, y, z arrays based on current PCA axis selection
		updatePlotCoordinates() {
			// Use the updatePlotCoordinates function from dataLoader.js
			updatePlotCoordinates(this.plotData, this.pcaAxes);
		},

		// Handle change of PCA axes
		updatePlotAxes() {
			console.log('PCA axes changed', this.pcaAxes);
			this.updatePlotCoordinates();
			this.updatePlot();
		},

		// Get current camera position from the plot (now using function from plotutil.js)
		getCurrentCameraPosition() {
			return getCurrentCameraPosition(this.$refs.plotContainer);
		},

		// Select a colorByColumn from the dropdown
		selectColorByColumn(column) {
			console.log(`Selected color-by column: ${column}`);
			this.pendingColorByColumn = column;
			this.showColorDropdown = false;
		},

		// Select a selectionColumn from the dropdown
		selectSelectionColumn(column) {
			console.log(`Selected selection column: ${column}`);
			this.pendingSelectionColumn = column;
			this.showSelectionDropdown = false;
		},

		// Apply colorByColumn change
		applyColorByColumn() {
			if (!this.pendingColorByColumn) return;

			console.log(`Applying color-by column change from ${this.colorByColumn} to ${this.pendingColorByColumn}`);

			// Store current camera position before update
			this.currentCameraPosition = this.getCurrentCameraPosition();

			// Show plot updating indicator
			this.isUpdatingPlot = true;

			// Update the column and clear selection if needed
			if (this.colorByColumn !== this.pendingColorByColumn) {
				this.colorByColumn = this.pendingColorByColumn;
				this.selectedValues = []; // Clear selection when changing default color column
				this.updatePlot();
			}
		},

		// Apply selectionColumn change
		applySelectionColumn() {
			if (!this.pendingSelectionColumn) return;

			console.log(`Applying selection column change from ${this.selectionColumn} to ${this.pendingSelectionColumn}`);

			// Store current camera position before update
			this.currentCameraPosition = this.getCurrentCameraPosition();

			// Show plot updating indicator
			this.isUpdatingPlot = true;

			// Update the column and clear selection
			if (this.selectionColumn !== this.pendingSelectionColumn) {
				this.selectionColumn = this.pendingSelectionColumn;
				this.selectedValues = []; // Clear selection when changing selection column
				this.updatePlot();
			}
		},

		// Hide dropdowns with delay to allow for click
		hideColorDropdownDelayed() {
			this.dropdownTimeout = setTimeout(() => {
				this.showColorDropdown = false;
			}, 200);
		},

		hideSelectionDropdownDelayed() {
			this.dropdownTimeout = setTimeout(() => {
				this.showSelectionDropdown = false;
			}, 200);
		},

		// Create traces grouped by categorical value (for initial view) - now using function from plotutil.js
		createTracesByCategory() {
			console.log('Creating traces by category...');

			// Use the createTracesByCategory function from plotutil.js with colorByColumn
			return createTracesByCategory(
				this.plotData,
				this.colorByColumn,
				this.dataFrame,
				{
					defaultTraceConfig: this.defaultTraceConfig,
					selectedSize: this.selectedSize,
					selectedOpacity: this.selectedOpacity
				}
			);
		},

		// Create traces with selection highlighting - now using function from plotutil.js
		createTracesWithSelection() {
			console.log('Creating traces with selection highlighting...');

			// Use the createTracesWithSelection function from plotutil.js with selectionColumn
			return createTracesWithSelection(
				this.plotData,
				this.selectionColumn,
				this.selectedValues,
				{
					defaultTraceConfig: this.defaultTraceConfig,
					selectedSize: this.selectedSize,
					nonSelectedSize: this.nonSelectedSize,
					selectedOpacity: this.selectedOpacity,
					nonSelectedOpacity: this.nonSelectedOpacity,
					nonSelectedColor: this.nonSelectedColor,
					getSelectionColor: this.getSelectionColor.bind(this)
				}
			);
		},

		// Initialize the plot
		initPlot() {
			this.statusMessage = 'Creating plot...';
			console.log('Initializing plot...');

			if (!this.plotData) {
				console.error("Plot data not available");
				this.statusMessage = 'Error: Plot data not available';
				return;
			}

			// Create initial traces by the selected categorical column
			const traces = this.createTracesByCategory();

			// Create the plot
			console.log('Rendering initial plot...');
			Plotly.newPlot(
				this.$refs.plotContainer,
				traces,
				createPlotLayout(this.title, null, this.pcaAxes)
			);

			// Add click handler with proper binding
			const boundHandlePointClick = this.handlePointClick.bind(this);
			this.$refs.plotContainer.on('plotly_click', boundHandlePointClick);

			// Add relayout event handler to capture camera position changes
			this.$refs.plotContainer.on('plotly_relayout', (eventData) => {
				// Update our stored camera position when user interacts with the plot
				this.currentCameraPosition = this.getCurrentCameraPosition();
			});

			this.statusMessage = 'Plot ready - click points to select values';
			console.log('Plot initialization complete');
		},

		// Handle point clicks
		handlePointClick(data) {
			if (data.points && data.points.length > 0) {
				const point = data.points[0];
				console.log("Clicked point:", point);

				// Avoid processing clicks too rapidly
				const now = Date.now();
				if (now - this.lastSelectionTime < 300) { // 300ms debounce
					console.log("Click ignored - too soon after previous click");
					return;
				}
				this.lastSelectionTime = now;

				if (point.customdata) {
					// Extract value from customdata (format: "column: value")
					const customData = point.customdata;
					const match = customData.match(new RegExp(`${this.selectionColumn}: (.*)`));

					if (match && match[1]) {
						const value = match[1];
						console.log(`Clicked ${this.selectionColumn}:`, value);

						// Show processing indicator
						this.processingSelection = true;

						// Store current camera position before update
						this.currentCameraPosition = this.getCurrentCameraPosition();

						// Use setTimeout to avoid blocking the UI
						setTimeout(() => {
							console.time('Process Selection');

							// Toggle selection
							const index = this.selectedValues.indexOf(value);

							if (index >= 0) {
								// Remove if already selected
								this.selectedValues.splice(index, 1);
								console.log("Removed value:", value);
							} else {
								// Add if not already selected
								this.selectedValues.push(value);
								console.log("Added value:", value);
							}

							console.timeEnd('Process Selection');

							// Update the visualization with some delay to prevent UI blocking
							this.debounceUpdatePlot();
						}, 10);
					}
				}
			}
		},

		// Debounce the updatePlot call to prevent too frequent updates
		debounceUpdatePlot() {
			if (this.updatePlotTimeout) {
				clearTimeout(this.updatePlotTimeout);
			}

			this.updatePlotTimeout = setTimeout(() => {
				this.updatePlot();
			}, 100);
		},

		// Update the plot based on selection
		updatePlot() {
			console.time('Update Plot');
			try {
				console.log('Updating plot...');

				// Show updating indicator
				this.isUpdatingPlot = true;

				// Store current camera position before update if not already stored
				if (!this.currentCameraPosition) {
					this.currentCameraPosition = this.getCurrentCameraPosition();
				}

				// Create new traces based on selection state
				console.log('Creating new traces...');
				const traces = this.selectedValues.length > 0
					? this.createTracesWithSelection()
					: this.createTracesByCategory();

				// First remove old click handlers to prevent duplicates
				if (this.$refs.plotContainer && this.$refs.plotContainer.removeAllListeners) {
					this.$refs.plotContainer.removeAllListeners('plotly_click');
				}

				// Use complete redraw with newPlot to avoid potential issues
				// Apply saved camera position in the layout
				console.log('Rendering plot with new data...');
				Plotly.newPlot(
					this.$refs.plotContainer,
					traces,
					createPlotLayout(this.title, this.currentCameraPosition, this.pcaAxes)
				);

				// Add click handler again after plot is redrawn with proper binding
				const boundHandlePointClick = this.handlePointClick.bind(this);
				this.$refs.plotContainer.on('plotly_click', boundHandlePointClick);

				// Re-add the relayout event handler
				this.$refs.plotContainer.on('plotly_relayout', (eventData) => {
					this.currentCameraPosition = this.getCurrentCameraPosition();
				});

				// Hide processing indicator if it was shown
				this.processingSelection = false;
				// Hide updating indicator
				this.isUpdatingPlot = false;
				this.statusMessage = 'Plot updated';
				console.log('Plot update complete');
			} catch (error) {
				console.error("Error updating plot:", error);
				this.statusMessage = `Error updating plot: ${error.message}`;
				this.processingSelection = false;
				// Hide updating indicator
				this.isUpdatingPlot = false;
			}
			console.timeEnd('Update Plot');
		},

		// Toggle config section collapse
		toggleConfigCollapse() {
			this.configCollapsed = !this.configCollapsed;
			console.log(`Configuration ${this.configCollapsed ? 'collapsed' : 'expanded'}`);
		},

		// Clear selection
		clearSelection() {
			console.log('Clearing all selections');
			// Store current camera position before clearing
			this.currentCameraPosition = this.getCurrentCameraPosition();

			this.selectedValues = [];
			// Reset colors back to initial set
			this.selectionColors = [...this.initialSelectionColors];
			this.updatePlot();
		},

		// Remove a specific value from selection
		removeSelectedValue(value) {
			console.log(`Removing value from selection: ${value}`);
			// Store current camera position before removing value
			this.currentCameraPosition = this.getCurrentCameraPosition();

			const index = this.selectedValues.indexOf(value);
			if (index >= 0) {
				this.selectedValues.splice(index, 1);
				this.updatePlot();
			}
		},

		// Get selection color for display
		getSelectionColor(index) {
			return this.selectionColors[index % this.selectionColors.length];
		},

		// Generate new random colors that are visually distinct
		regenerateColors() {
			console.log('Regenerating selection colors');
			// Store current camera position before changing colors
			this.currentCameraPosition = this.getCurrentCameraPosition();
			// Using the imported generateDistinctColors function from colorutil.js
			this.selectionColors = generateDistinctColors(9);
			// Save the new colors as the initial set too
			this.initialSelectionColors = [...this.selectionColors];
			this.statusMessage = 'Colors regenerated';
			// Always update the plot
			this.updatePlot();
		},

		// Show performance information in console for debugging
		logPerformanceInfo() {
			console.log('Performance Information:');
			console.log(`Total points: ${this.plotData?.x?.length || 'N/A'}`);
			console.log(`Color by: ${this.colorByColumn}, Select by: ${this.selectionColumn}`);
			console.log(`Total unique ${this.colorByColumn} values: ${this.availableColumns.length > 0 ?
				this.dataFrame.col_unique(this.colorByColumn).size : 'N/A'}`);
			console.log(`Current selection: ${this.selectedValues.length} values`);
			console.log(`Available PCA components: ${this.availablePcaAxes.length}`);
			console.log(`Current axes: PC${this.pcaAxes.x + 1}, PC${this.pcaAxes.y + 1}, PC${this.pcaAxes.z + 1}`);

			// Log memory usage if available
			if (window.performance && window.performance.memory) {
				const memory = window.performance.memory;
				console.log(`Used JS heap: ${(memory.usedJSHeapSize / (1024 * 1024)).toFixed(2)} MB`);
				console.log(`Total JS heap: ${(memory.totalJSHeapSize / (1024 * 1024)).toFixed(2)} MB`);
			}
		}
	},

	mounted() {
		// Load data when component is mounted
		this.loadData();

		// Set up keyboard shortcuts for debugging
		window.addEventListener('keydown', (e) => {
			// Ctrl+Shift+P to log performance info
			if (e.ctrlKey && e.shiftKey && e.key === 'P') {
				this.logPerformanceInfo();
			}
		});
	}
});