/* global, mutable CONFIG + helper to merge an optional config.json */
let CONFIG = {
	// Data loading
	dataFile: "pca.jsonl",
	numericalPrefix: "pc.",
	defaultColorColumn: "activation.model",
	defaultSelectionColumn: "activation.model",
	hoverColumns: ["activation.cls", "activation.prompt"],

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
		shortcutStatusUpdateDelay: 100
	}
};

/**
 * Load config.json (if present) and merge into CONFIG.
 * Also parse URL parameters and apply them to CONFIG.
 * Missing keys fall back to the defaults above.
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
		} else {
			console.warn("config.json not found, using defaults");
		}
	} catch (e) {
		console.error("Config load error:", e);
	}

	// Parse URL parameters and override CONFIG values
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
 */
function parseConfigValue(value) {
	// Boolean
	if (value === 'true') return true;
	if (value === 'false') return false;

	// Number
	if (!isNaN(value) && !isNaN(parseFloat(value))) {
		return parseFloat(value);
	}

	// String (including hex colors)
	return value;
}

/**
 * Generate URL search params from current CONFIG state
 * Only includes values that differ from defaults
 */
function generateURLParams(baseConfig = null) {
	if (!baseConfig) {
		// Create a fresh default config for comparison
		baseConfig = createDefaultConfig();
	}

	const params = new URLSearchParams();
	const differences = findConfigDifferences(CONFIG, baseConfig);

	for (const [path, value] of differences) {
		params.set(path, value.toString());
	}

	return params;
}

/**
 * Create a fresh default CONFIG object for comparison
 */
function createDefaultConfig() {
	// Return the same structure as the initial CONFIG
	// This is a bit redundant but ensures we can compare against true defaults
	return {
		dataFile: "pca.jsonl",
		numericalPrefix: "pc.",
		defaultColorColumn: "activation.model",
		defaultSelectionColumn: "activation.model",
		hoverColumns: ["activation.cls", "activation.prompt"],
		panels: { help: false, menu: false, info: false, legend: true, navbar: true, stats: false },
		axes: { x: 0, y: 1, z: 2 },
		selectedPoints: { size: 6, sizeMin: 1, sizeMax: 20, sizeStep: 1, opacity: 1.0, opacityMin: 0.1, opacityMax: 1.0, opacityStep: 0.1 },
		nonSelectedPoints: { size: 4, sizeMin: 1, sizeMax: 20, sizeStep: 1, opacity: 0.25, opacityMin: 0.01, opacityMax: 1.0, opacityStep: 0.01, color: "#666666" },
		movement: { speed: 50, speedMin: 1, speedMax: 200, speedStep: 1, rollSpeed: 0.02, mouseSensitivity: 0.002, sprintMultiplier: 3 },
		interaction: { hoverActive: true, selectOnClick: true, raycastThreshold: 0.15, raycastThresholdMultiplier: 3 },
		performance: { fpsThreshold: 15, performanceCheckInterval: 2000, performanceWarningDuration: 4000, performanceOptimizationCooldown: 30000 },
		rendering: { clearColor: 0x000011, cameraFov: 75, cameraNear: 0.1, cameraFar: 2000, antialiasing: true },
		crosshair: { color: 0xffff00, opacity: 0.4, length: 1000 },
		navball: { size: 150, sensitivity: 0.01, sphereDetail: { widthSegments: 12, heightSegments: 8 }, axisLength: 1.3, arrowLength: 0.1, arrowRadius: 0.02, labelScale: 0.3 },
		colors: { paletteSize: 128, nullValueColor: "#4d4d4d", viridisColors: { r: [0.267004, 0.105010, 0.330010, 2.437600, -5.179800, 2.066100], g: [0.004874, 0.406910, 1.193600, -1.375200, 0.813500, -0.073200], b: [0.329415, 0.718080, -0.724400, 0.063300, 0.016700, 0.000000] } },
		ui: { maxCategoricalDisplay: 15, maxSelectedDisplay: 20, hoverOffset: { x: 15, y: 15 }, shortcutStatusUpdateDelay: 100 }
	};
}

/**
 * Find differences between current config and base config
 * Returns array of [path, value] tuples
 */
function findConfigDifferences(current, base, prefix = '') {
	const differences = [];

	for (const key in current) {
		const currentPath = prefix ? `${prefix}.${key}` : key;
		const currentValue = current[key];
		const baseValue = base[key];

		if (typeof currentValue === 'object' && !Array.isArray(currentValue) && currentValue !== null) {
			if (typeof baseValue === 'object' && !Array.isArray(baseValue) && baseValue !== null) {
				differences.push(...findConfigDifferences(currentValue, baseValue, currentPath));
			} else {
				// Base doesn't have this object, include all of current
				differences.push([currentPath, JSON.stringify(currentValue)]);
			}
		} else {
			// Compare primitive values
			if (currentValue !== baseValue) {
				differences.push([currentPath, currentValue]);
			}
		}
	}

	return differences;
}