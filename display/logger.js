/**
 * Simple logging utility for the application
 */
const logger = {
	/**
	 * Whether logging is enabled
	 */
	enabled: true,

	/**
	 * Log information message
	 */
	log(message, ...args) {
		if (this.enabled) {
			console.log(message, ...args);
		}
	},

	/**
	 * Log error message
	 */
	error(message, ...args) {
		console.error(message, ...args);
	},

	/**
	 * Start a performance timer
	 */
	time(label) {
		if (this.enabled) {
			console.time(label);
		}
	},

	/**
	 * End a performance timer
	 */
	timeEnd(label) {
		if (this.enabled) {
			console.timeEnd(label);
		}
	}
};

// Enable or disable logging via console
window.toggleLogging = function (enable = true) {
	logger.enabled = enable;
	console.log(`Logging ${enable ? 'enabled' : 'disabled'}`);
};

/**
* Simple utilities for showing/updating loading indicators
*/
const loading = {
	/**
	 * Show main loading state with optional progress
	 */
	showLoading(vm, message, detail = 'Please wait...', showProgress = false) {
		vm.isLoading = true;
		vm.loadingMessage = message;
		vm.loadingDetail = detail;
		vm.loadingProgress = showProgress;
		if (!showProgress) {
			vm.loadingProgressPercent = 0;
		}
		vm.statusMessage = message;
	},

	/**
	 * Update loading progress
	 */
	updateProgress(vm, percent, detail = null) {
		vm.loadingProgressPercent = percent;
		if (detail) {
			vm.loadingDetail = detail;
		}
	},

	/**
	 * Hide loading indicator
	 */
	hideLoading(vm) {
		vm.isLoading = false;
		vm.loadingProgress = false;
	},

	/**
	 * Show plot updating indicator
	 */
	showUpdating(vm, isProcessingSelection = false) {
		vm.isUpdatingPlot = true;
		vm.processingSelection = isProcessingSelection;
	},

	/**
	 * Hide updating indicators
	 */
	hideUpdating(vm) {
		vm.isUpdatingPlot = false;
		vm.processingSelection = false;
	}
};