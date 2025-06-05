/* global, mutable CONFIG + helper to merge an optional config.json */
function getDefaultConfig() {
	return {
		// Data loading
		dataFile: "pca.jsonl",
		numericalPrefix: "pc.",
		defaultColorColumn: "activation.model",
		defaultSelectionColumn: "activation.model",
		hoverColumns: ["activation.cls", "activation.prompt"],

		// Selected values
		selectedValues: [],

		// UI panel visibility
		panels: {
			help: false,
			menu: false,
			info: false,
			legend: true,
			navbar: true,
			stats: false
		},

		// Default axis configuration
		axes: {
			x: 0,
			y: 1,
			z: 2
		},

		// Point appearance - selected points
		selectedPoints: {
			size: 6,
			sizeMin: 1,
			sizeMax: 20,
			sizeStep: 1,
			opacity: 1.0,
			opacityMin: 0.1,
			opacityMax: 1.0,
			opacityStep: 0.1
		},

		// Point appearance - non-selected points
		nonSelectedPoints: {
			size: 4,
			sizeMin: 1,
			sizeMax: 20,
			sizeStep: 1,
			opacity: 0.25,
			opacityMin: 0.01,
			opacityMax: 1.0,
			opacityStep: 0.01,
			color: "#666666"
		},

		// Movement settings
		movement: {
			speed: 50,
			speedMin: 1,
			speedMax: 200,
			speedStep: 1,
			rollSpeed: 0.02,
			mouseSensitivity: 0.002,
			sprintMultiplier: 3
		},

		// Interaction settings
		interaction: {
			hoverActive: true,
			selectOnClick: true,
			rightClickActive: true,
			raycastThreshold: 0.15,
			raycastThresholdMultiplier: 3
		},

		// Performance settings
		performance: {
			fpsThreshold: 15,
			performanceCheckInterval: 2000,
			performanceWarningDuration: 4000,
			performanceOptimizationCooldown: 30000
		},

		// Rendering settings
		rendering: {
			clearColor: 0x000011,
			cameraFov: 75,
			cameraNear: 0.1,
			cameraFar: 2000,
			antialiasing: true
		},

		// Cross-hair settings
		crosshair: {
			color: 0xffff00,
			opacity: 0.4,
			length: 1000
		},

		// Navball settings
		navball: {
			size: 150,
			sensitivity: 0.01,
			sphereDetail: { widthSegments: 12, heightSegments: 8 },
			axisLength: 1.3,
			arrowLength: 0.1,
			arrowRadius: 0.02,
			labelScale: 0.3
		},

		// Color palette settings
		colors: {
			paletteSize: 128,
			nullValueColor: "#4d4d4d", // Dark gray for null values
			viridisColors: {
				// Viridis colormap coefficients for better performance
				r: [0.267004, 0.105010, 0.330010, 2.437600, -5.179800, 2.066100],
				g: [0.004874, 0.406910, 1.193600, -1.375200, 0.813500, -0.073200],
				b: [0.329415, 0.718080, -0.724400, 0.063300, 0.016700, 0.000000]
			}
		},

		// Legend and info panel settings
		ui: {
			maxCategoricalDisplay: 15,
			maxSelectedDisplay: 20,
			hoverOffset: { x: 15, y: 15 },
			shortcutStatusUpdateDelay: 100,
			menuWidth: 350
		},
		// Right-click behavior configuration
		rightClick: {
			mode: "content", // "content" or "url"

			// For content mode - creates a new tab with plain text content
			content: {
				title: "Point Data: {activation.cls}",	
				template: "Point Information\n===================\n\nClass: {activation.cls}\nModel: {activation.model}\nPrompt: {activation.prompt}\nCoordinates:\n  X ({axis.x.name}): {coord.x}\n  Y ({axis.y.name}): {coord.y}\n  Z ({axis.z.name}): {coord.z}\nConfiguration:\n  Data File: {config.dataFile}\n  Color By: {config.defaultColorColumn}\n  Select By: {config.defaultSelectionColumn}"
			},
			// For URL mode - opens a URL constructed from template
			url: {
				template: "https://example.com/{activation.cls}/{activation.model}?prompt={activation.prompt}&x={coord.x}&y={coord.y}&z={coord.z}"
			}
		},

		info: {
			title: "Embedding Explorer",
			help: "This tool lets you visualize high-dimensional data in 3D space."
		},
	}
}

let CONFIG = getDefaultConfig();
let LOADED_CONFIG = null; // Store the config as loaded from file for comparison
let URL_UPDATE_TIMEOUT = null;

/**
 * Load config.json (if present) and merge into CONFIG.
 * Also parse URL parameters and apply them to CONFIG.
 * Priority: URL params > config.json > defaults
 * @returns {Promise<object>} resolved CONFIG object
 */
async function getConfig() {
	try {
		// First, try to load config.json
		const r = await fetch("config.json");
		if (r.ok) {
			const loaded = await r.json();
			// Deep merge loaded config into CONFIG
			deepMerge(CONFIG, loaded);
			// Store a deep copy of the loaded config for URL comparison
			LOADED_CONFIG = JSON.parse(JSON.stringify(CONFIG));
			console.log("Loaded config.json");
		} else {
			console.warn("config.json not found, using defaults");
			// If no config.json, use defaults for comparison
			LOADED_CONFIG = JSON.parse(JSON.stringify(CONFIG));
		}
	} catch (e) {
		console.error("Config load error:", e);
		// On error, use defaults for comparison
		LOADED_CONFIG = JSON.parse(JSON.stringify(CONFIG));
	}

	// Parse URL parameters and override CONFIG values (highest priority)
	parseURLParams();

	return CONFIG;
}

/**
 * Deep merge source object into target object
 */
function deepMerge(target, source) {
	for (const key in source) {
		if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
			if (!target[key]) target[key] = {};
			deepMerge(target[key], source[key]);
		} else {
			target[key] = source[key];
		}
	}
}

/**
 * Parse URL parameters and update CONFIG
 * Supports nested paths like: ?axes.x=2&selectedPoints.size=8&panels.menu=true
 * Also supports arrays like: ?selectedValues=value1,value2,value3
 */
function parseURLParams() {
	const params = new URLSearchParams(window.location.search);

	for (const [key, value] of params) {
		setNestedConfigValue(CONFIG, key, parseConfigValue(value));
	}
}

/**
 * Set a nested configuration value using dot notation
 * Example: setNestedConfigValue(CONFIG, "axes.x", 2)
 */
function setNestedConfigValue(obj, path, value) {
	const keys = path.split('.');
	let current = obj;

	for (let i = 0; i < keys.length - 1; i++) {
		const key = keys[i];
		if (!(key in current) || typeof current[key] !== 'object') {
			current[key] = {};
		}
		current = current[key];
	}

	const finalKey = keys[keys.length - 1];
	current[finalKey] = value;
	console.log(`URL param override: ${path} = ${value}`);
}

/**
 * Parse a string value from URL params into appropriate type
 * Handles arrays (comma-separated values)
 */
function parseConfigValue(value) {
	// Boolean
	if (value === 'true') return true;
	if (value === 'false') return false;

	// Array (comma-separated) - but handle single values too
	if (value.includes(',')) {
		return value.split(',').map(v => v.trim()).filter(v => v.length > 0);
	}

	// Number
	if (!isNaN(value) && !isNaN(parseFloat(value))) {
		return parseFloat(value);
	}

	// String (including hex colors) - treat single values as arrays for selectedValues
	return value;
}

/**
 * Update the URL with current CONFIG state
 * Debounced to avoid excessive URL updates
 */
function updateURL() {
	if (URL_UPDATE_TIMEOUT) {
		clearTimeout(URL_UPDATE_TIMEOUT);
	}

	URL_UPDATE_TIMEOUT = setTimeout(() => {
		const params = generateURLParams();
		const newURL = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
		window.history.replaceState({}, '', newURL);
		URL_UPDATE_TIMEOUT = null;
	}, 500); // 500ms debounce
}

/**
 * Generate URL search params from current CONFIG state
 * Only includes values that differ from the loaded config (not defaults)
 */
function generateURLParams() {
	if (!LOADED_CONFIG) {
		// Fallback to default config if loaded config not available
		return new URLSearchParams();
	}

	const params = new URLSearchParams();
	const differences = findConfigDifferences(CONFIG, LOADED_CONFIG);

	for (const [path, value] of differences) {
		// Special handling for arrays
		if (Array.isArray(value)) {
			if (value.length > 0) {
				params.set(path, value.join(','));
			}
		} else {
			params.set(path, value.toString());
		}
	}

	return params;
}

/**
 * Find differences between current config and loaded config
 * Returns array of [path, value] tuples
 * Uses epsilon comparison for floats
 */
function findConfigDifferences(current, base, prefix = '') {
	const differences = [];
	const EPSILON = 0.001;

	for (const key in current) {
		const currentPath = prefix ? `${prefix}.${key}` : key;
		const currentValue = current[key];
		const baseValue = base[key];

		if (Array.isArray(currentValue)) {
			// Special handling for arrays
			if (!Array.isArray(baseValue) || !arraysEqual(currentValue, baseValue)) {
				differences.push([currentPath, currentValue]);
			}
		} else if (typeof currentValue === 'object' && currentValue !== null) {
			if (typeof baseValue === 'object' && !Array.isArray(baseValue) && baseValue !== null) {
				differences.push(...findConfigDifferences(currentValue, baseValue, currentPath));
			} else {
				// Base doesn't have this object, include all of current
				differences.push([currentPath, JSON.stringify(currentValue)]);
			}
		} else {
			// Compare primitive values with epsilon for floats
			let valuesEqual = false;

			if (typeof currentValue === 'number' && typeof baseValue === 'number') {
				// Use epsilon comparison for floats
				valuesEqual = Math.abs(currentValue - baseValue) < EPSILON;
			} else {
				// Direct comparison for other types
				valuesEqual = currentValue === baseValue;
			}

			if (!valuesEqual) {
				differences.push([currentPath, currentValue]);
			}
		}
	}

	return differences;
}

/**
 * Helper function to compare arrays
 */
function arraysEqual(arr1, arr2) {
	if (arr1.length !== arr2.length) return false;
	for (let i = 0; i < arr1.length; i++) {
		if (arr1[i] !== arr2[i]) return false;
	}
	return true;
}

/**
 * Get the current configuration as a formatted JSON string
 */
function getConfigAsJSON() {
	return JSON.stringify(CONFIG, null, 2);
}

/**
 * Open a new tab with the current configuration
 */
function exportConfigToNewTab() {
	const configText = getConfigAsJSON();
	const blob = new Blob([configText], { type: 'application/json' });
	const url = URL.createObjectURL(blob);
	const newWindow = window.open(url, '_blank');

	// Clean up the object URL after a delay
	setTimeout(() => {
		URL.revokeObjectURL(url);
	}, 1000);
}


/**
 * Reset CONFIG to the loaded config.json state and clear URL parameters
 */
function resetConfigToLoaded() {
    if (!LOADED_CONFIG) {
        console.warn("No loaded config available, resetting to defaults");
        CONFIG = getDefaultConfig();
    } else {
        // Deep copy the loaded config back to CONFIG
        CONFIG = JSON.parse(JSON.stringify(LOADED_CONFIG));
    }
    
    // Clear URL parameters by navigating to clean URL
    const cleanURL = window.location.pathname;
    window.history.replaceState({}, '', cleanURL);
    
    // Clear the URL update timeout if it exists
    if (URL_UPDATE_TIMEOUT) {
        clearTimeout(URL_UPDATE_TIMEOUT);
        URL_UPDATE_TIMEOUT = null;
    }
    
    console.log("Config reset to loaded state and URL cleared");
}