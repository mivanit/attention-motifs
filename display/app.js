// Create Vue app
const app = Vue.createApp({
	// ==================================================
	// CHUNK: App Data
	// ==================================================
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
			showConfigPanel: false,
			// Default trace configuration
			defaultTraceConfig: createDefaultTraceConfig(),
			// Store current camera position
			currentCameraPosition: null,
			// Store selection timing information
			lastSelectionTime: 0,
			// Debounce timeout
			updatePlotTimeout: null,
			// Flag to enable/disable customdata caching
			useCustomdataCache: true,
			// Last hover data for clipboard copy
			lastHoverData: null,
			// Flag to track if right mouse button is pressed
			isRightMouseDown: false,
			// Configuration for data loading
			dataConfig: {
				filePath: 'data/features/pca.jsonl', // Default path to the data file
				numericalPrefix: 'pc.', // Prefix for numerical columns (PCA components)
				defaultColorColumn: 'activation.model', // Default column for coloring
				defaultSelectionColumn: 'activation.model', // Default column for selection
			},
			// URL params configuration
			urlParams: {
				filePath: 'dataPath',
				numericalPrefix: 'numPrefix',
				defaultColorColumn: 'colorBy',
				defaultSelectionColumn: 'selectBy'
			}
		};
	},
	// ==================================================

	computed: {
		// Compute combined URL with all selected values
		combinedUrl() {
			return computeCombinedUrl(this.baseUrl, this.selectedValues, this.selectionColumn);
		}
	},

	methods: {
		// ==================================================
		// CHUNK: config
		// ==================================================
		// Toggle configuration panel
		toggleConfigPanel() {
			this.showConfigPanel = !this.showConfigPanel;
		},

		// Apply configuration changes and reload data
		applyConfigChanges() {
			// Show loading indicator
			loading.showLoading(this, 'Applying configuration...', 'Reloading data', true);

			// Update URL with new configuration
			this.baseUrl = updateUrlWithConfig(this.dataConfig, this.urlParams);

			// Reload data with new configuration
			this.loadData();
		},
		// ==================================================
		// CHUNK: data loading
		// ==================================================
		// Load data from a single JSONL file
		async loadData() {
			loading.showLoading(this, 'Loading data...', 'Initializing', true);
			logger.log('Starting data loading process');

			try {
				// Clear any existing customdata cache
				clearCustomdataCache();

				// Parse URL parameters to override default configuration
				this.dataConfig = parseUrlParams(this.dataConfig, this.urlParams);

				// Load and process data using the function from dataloader.js
				const result = await loadJsonlData(this.dataConfig, (percent, detail) => {
					loading.updateProgress(this, percent, detail);
				});

				// Update app data with results
				this.dataFrame = result.dataFrame;
				this.pcaArray = result.pcaArray;
				this.plotData = result.plotData;
				this.availableColumns = result.availableColumns;
				this.availablePcaAxes = new Array(this.pcaArray.shape[1]).fill(0).map((_, i) => i);

				// Set default selected columns from configuration
				this.colorByColumn = this.dataConfig.defaultColorColumn;
				this.pendingColorByColumn = this.colorByColumn;
				this.selectionColumn = this.dataConfig.defaultSelectionColumn;
				this.pendingSelectionColumn = this.selectionColumn;

				loading.updateProgress(this, 95, 'Rendering plot...');
				this.statusMessage = 'Data loaded successfully';

				// Initialize the plot after data is loaded
				this.initPlot();

				loading.updateProgress(this, 100, 'Completing setup...');

				loading.hideLoading(this);
				logger.log('Data loading complete');
			} catch (error) {
				logger.error("Error loading data:", error);
				this.statusMessage = `Error loading data: ${error.message}`;

				loading.showLoading(this, 'Error loading data', error.message, false);
				loading.hideLoading(this);
			}
		},

		// Process loaded data into a format suitable for plotting
		processData() {
			return processDataFromDataFrame(this.dataFrame, this.pcaArray, this.pcaAxes);
		},

		// ==================================================
		// CHUNK: selection
		// ==================================================

		// Handle change of PCA axes
		updatePlotAxes() {
			logger.log('PCA axes changed', this.pcaAxes);
			updatePlotCoordinates(this.plotData, this.pcaAxes);
			this.updatePlot();
		},

		// Get current camera position from the plot (now using function from plotutil.js)
		getCurrentCameraPosition() {
			return getCurrentCameraPosition(this.$refs.plotContainer);
		},

		// Select a colorByColumn from the dropdown
		selectColorByColumn(column) {
			this.pendingColorByColumn = column;
			this.showColorDropdown = false;
		},

		// Select a selectionColumn from the dropdown
		selectSelectionColumn(column) {
			this.pendingSelectionColumn = column;
			this.showSelectionDropdown = false;
		},

		// Unified function to apply column changes
		applyColumnChange(columnType) {
			const isPendingColor = columnType === 'color';
			const currentColumn = isPendingColor ? this.colorByColumn : this.selectionColumn;
			const pendingColumn = isPendingColor ? this.pendingColorByColumn : this.pendingSelectionColumn;

			if (!pendingColumn) return;

			logger.log(`Changing ${columnType}-by column: ${currentColumn} → ${pendingColumn}`);

			// Store current camera position before update
			this.currentCameraPosition = this.getCurrentCameraPosition();

			// Show plot updating indicator
			loading.showUpdating(this);

			// Update the column and clear selection if needed
			if (currentColumn !== pendingColumn) {
				// Clear customdata cache when column changes
				clearCustomdataCache();

				if (isPendingColor) {
					this.colorByColumn = pendingColumn;
				} else {
					this.selectionColumn = pendingColumn;
				}

				this.selectedValues = []; // Clear selection when changing columns
				this.updatePlot();
			} else {
				// Hide updating indicator if no change
				loading.hideUpdating(this);
			}
		},

		// Apply colorByColumn change
		applyColorByColumn() {
			this.applyColumnChange('color');
		},

		// Apply selectionColumn change
		applySelectionColumn() {
			this.applyColumnChange('selection');
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

		// ==================================================
		// CHUNK: traces
		// ==================================================
		// Create traces grouped by categorical value (for initial view) - now using function from plotutil.js
		createTracesByCategory() {
			logger.log('Creating traces by category');

			// Use the createTracesByCategory function from plotutil.js with colorByColumn
			return createTracesByCategory(
				this.plotData,
				this.colorByColumn,
				this.dataFrame,
				{
					defaultTraceConfig: this.defaultTraceConfig,
					selectedSize: this.selectedSize,
					selectedOpacity: this.selectedOpacity,
					// Add the selectionColumn so it's included in customdata
					selectionColumn: this.selectionColumn,
					useCache: this.useCustomdataCache
				}
			);
		},

		// Create traces with selection highlighting - now using function from plotutil.js
		createTracesWithSelection() {
			logger.log('Creating traces with selection highlighting');

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
					getSelectionColor: this.getSelectionColor.bind(this),
					// Add colorByColumn so it's included in customdata
					colorByColumn: this.colorByColumn,
					useCache: this.useCustomdataCache
				}
			);
		},

		// Copy hover data to clipboard using the function from clipboard.js
		copyHoverDataToClipboard() {
			if (!this.lastHoverData) return;
			copyHoverDataToClipboard(
				this.lastHoverData,
				this.selectionColumn,
				(statusMsg) => { this.statusMessage = statusMsg; }
			);
		},


		// ==================================================
		// CHUNK: listeners
		// ==================================================

		// Setup event handlers for the plot
		setupPlotEventHandlers() {
			// Add click handler with proper binding
			const boundHandlePointClick = this.handlePointClick.bind(this);
			this.$refs.plotContainer.on('plotly_click', boundHandlePointClick);

			// Add hover event handler to capture hover data
			this.$refs.plotContainer.on('plotly_hover', (hoverData) => {
				this.lastHoverData = hoverData;
			});

			// Add mousedown event listener to detect right clicks
			this.$refs.plotContainer.addEventListener('mousedown', (e) => {
				if (e.button === 2) { // Right mouse button
					this.isRightMouseDown = true;
					// Only prevent default and copy if we have hover data (meaning we're over a point)
					if (this.lastHoverData && this.lastHoverData.points && this.lastHoverData.points.length > 0) {
						e.preventDefault();
						this.copyHoverDataToClipboard();
					}
				}
			});

			// Add mouseup event listener to reset right click flag
			this.$refs.plotContainer.addEventListener('mouseup', (e) => {
				if (e.button === 2) { // Right mouse button
					this.isRightMouseDown = false;
				}
			});

			// Add contextmenu event listener to prevent default context menu only when over points
			this.$refs.plotContainer.addEventListener('contextmenu', (e) => {
				if (this.lastHoverData && this.lastHoverData.points && this.lastHoverData.points.length > 0) {
					e.preventDefault(); // Only prevent context menu if over a point
				}
			});

			// Add relayout event handler to capture camera position changes
			this.$refs.plotContainer.on('plotly_relayout', (eventData) => {
				// Update our stored camera position when user interacts with the plot
				this.currentCameraPosition = this.getCurrentCameraPosition();
			});
		},

		// ==================================================
		// CHUNK: initPlot
		// ==================================================

		// Initialize the plot
		initPlot() {
			this.statusMessage = 'Creating plot...';
			logger.log('Initializing plot');

			if (!this.plotData) {
				logger.error("Plot data not available");
				this.statusMessage = 'Error: Plot data not available';
				return;
			}

			// Create initial traces by the selected categorical column
			const traces = this.createTracesByCategory();

			// Create the plot
			logger.time('Initial plot render');
			Plotly.newPlot(
				this.$refs.plotContainer,
				traces,
				createPlotLayout(this.title, null, this.pcaAxes)
			);
			logger.timeEnd('Initial plot render');

			// Setup event handlers
			this.setupPlotEventHandlers();

			this.statusMessage = 'Plot ready - click points to select values, right-click to copy hover text';
		},

		// ==================================================
		// CHUNK: handlePointClick
		// ==================================================

		// Handle point clicks
		handlePointClick(data) {
			// Don't process clicks if the right mouse button is down
			if (this.isRightMouseDown) {
				return;
			}

			if (data.points && data.points.length > 0) {
				const point = data.points[0];

				// Avoid processing clicks too rapidly
				const now = Date.now();
				if (now - this.lastSelectionTime < 300) { // 300ms debounce
					return;
				}
				this.lastSelectionTime = now;

				if (point.customdata) {
					// Define the mapping of columns to customdata indices
					const customDataIndices = {
						'activation.cls': 0,
						'activation.prompt': 1
					};

					// If the selection column is not one of the first two, it will be the third element
					if (this.selectionColumn !== 'activation.cls' && this.selectionColumn !== 'activation.prompt') {
						customDataIndices[this.selectionColumn] = 2;
					}

					// Check if the selection column is in our customdata indices mapping
					if (this.selectionColumn in customDataIndices) {
						const index = customDataIndices[this.selectionColumn];
						const value = point.customdata[index];

						if (value !== undefined) {
							logger.log(`Selected ${this.selectionColumn}: ${value}`);

							// Show processing indicator
							loading.showUpdating(this, true);

							// Store current camera position before update
							this.currentCameraPosition = this.getCurrentCameraPosition();

							// Use setTimeout to avoid blocking the UI
							setTimeout(() => {
								logger.time('Process Selection');

								// Toggle selection
								const valueIndex = this.selectedValues.indexOf(value);

								if (valueIndex >= 0) {
									// Remove if already selected
									this.selectedValues.splice(valueIndex, 1);
								} else {
									// Add if not already selected
									this.selectedValues.push(value);
								}

								logger.timeEnd('Process Selection');

								// Update the visualization with some delay to prevent UI blocking
								this.debounceUpdatePlot();
							}, 10);
						} else {
							logger.error(`Could not find ${this.selectionColumn} value in customdata at index ${index}`);
						}
					} else {
						logger.error(`Selection column ${this.selectionColumn} not found in customdata structure`);
					}
				}
			}
		},

		// ==================================================
		// CHUNK: update plot
		// ==================================================

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
			logger.time('Update Plot');
			try {
				// Show updating indicator
				loading.showUpdating(this);

				// Store current camera position before update if not already stored
				if (!this.currentCameraPosition) {
					this.currentCameraPosition = this.getCurrentCameraPosition();
				}

				// Create new traces based on selection state
				const traces = this.selectedValues.length > 0
					? this.createTracesWithSelection()
					: this.createTracesByCategory();

				// First remove old event handlers to prevent duplicates
				if (this.$refs.plotContainer && this.$refs.plotContainer.removeAllListeners) {
					this.$refs.plotContainer.removeAllListeners('plotly_click');
					this.$refs.plotContainer.removeAllListeners('plotly_hover');
					this.$refs.plotContainer.removeAllListeners('plotly_relayout');
				}

				// Use complete redraw with newPlot to avoid potential issues
				// Apply saved camera position in the layout
				logger.time('Plot Render');
				Plotly.newPlot(
					this.$refs.plotContainer,
					traces,
					createPlotLayout(this.title, this.currentCameraPosition, this.pcaAxes)
				);
				logger.timeEnd('Plot Render');

				// Setup event handlers again
				this.setupPlotEventHandlers();

				// Hide updating indicators
				loading.hideUpdating(this);
				this.statusMessage = 'Plot updated - right-click on points to copy hover text';
			} catch (error) {
				logger.error("Error updating plot:", error);
				this.statusMessage = `Error updating plot: ${error.message}`;
				loading.hideUpdating(this);
			}
			logger.timeEnd('Update Plot');
		},

		// Toggle config section collapse
		toggleConfigCollapse() {
			this.configCollapsed = !this.configCollapsed;
		},

		// Clear selection
		clearSelection() {
			logger.log('Clearing all selections');
			// Store current camera position before clearing
			this.currentCameraPosition = this.getCurrentCameraPosition();

			this.selectedValues = [];
			// Reset colors back to initial set
			this.selectionColors = [...this.initialSelectionColors];
			this.updatePlot();
		},

		// Remove a specific value from selection
		removeSelectedValue(value) {
			logger.log(`Removing from selection: ${value}`);
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
			logger.log('Regenerating selection colors');
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


		// ==================================================
		// CHUNK: performance
		// ==================================================
		// Show performance information in console for debugging
		logPerformanceInfo() {
			logger.log('Performance Information:');
			logger.log(`Total points: ${this.plotData?.x?.length || 'N/A'}`);
			logger.log(`Color by: ${this.colorByColumn}, Select by: ${this.selectionColumn}`);
			logger.log(`Total unique ${this.colorByColumn} values: ${this.availableColumns.length > 0 ?
				this.dataFrame.col_unique(this.colorByColumn).size : 'N/A'}`);
			logger.log(`Current selection: ${this.selectedValues.length} values`);
			logger.log(`Available PCA components: ${this.availablePcaAxes.length}`);
			logger.log(`Current axes: PC${this.pcaAxes.x + 1}, PC${this.pcaAxes.y + 1}, PC${this.pcaAxes.z + 1}`);
			logger.log(`Customdata caching: ${this.useCustomdataCache ? 'Enabled' : 'Disabled'}`);

			// Log cache info if available
			if (customdataCache && customdataCache.data) {
				const cacheKeys = Object.keys(customdataCache.data);
				logger.log(`Customdata cache entries: ${cacheKeys.length}`);

				if (cacheKeys.length > 0) {
					const sampleKey = cacheKeys[0];
					const sampleSize = customdataCache.data[sampleKey].length;
					logger.log(`Sample cache entry (${sampleKey}): ${sampleSize} points`);
				}
			}

			// Log memory usage if available
			if (window.performance && window.performance.memory) {
				const memory = window.performance.memory;
				logger.log(`Used JS heap: ${(memory.usedJSHeapSize / (1024 * 1024)).toFixed(2)} MB`);
				logger.log(`Total JS heap: ${(memory.totalJSHeapSize / (1024 * 1024)).toFixed(2)} MB`);
			}
		},

		// Toggle customdata caching
		toggleCustomdataCache() {
			this.useCustomdataCache = !this.useCustomdataCache;
			if (!this.useCustomdataCache) {
				// Clear the cache when disabling
				clearCustomdataCache();
			}
			logger.log(`Customdata caching ${this.useCustomdataCache ? 'enabled' : 'disabled'}`);
			this.statusMessage = `Customdata caching ${this.useCustomdataCache ? 'enabled' : 'disabled'}`;
		}
	},

	// ==================================================
	// CHUNK: mounted
	// ==================================================
	mounted() {
		// Load data when component is mounted
		this.loadData();

		// Set up keyboard shortcuts for debugging
		window.addEventListener('keydown', (e) => {
			// Ctrl+Shift+P to log performance info
			if (e.ctrlKey && e.shiftKey && e.key === 'P') {
				this.logPerformanceInfo();
			}

			// Ctrl+Shift+C to toggle customdata caching
			if (e.ctrlKey && e.shiftKey && e.key === 'C') {
				this.toggleCustomdataCache();
			}
		});
	}
});