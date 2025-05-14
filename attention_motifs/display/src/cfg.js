/**
 * Loads configuration from config.json file in the current directory
 * @returns {Promise<Object|null>} Configuration object or null if not found
 */
async function loadConfigFile() {
	try {
		logger.log('Attempting to load config.json...');
		const response = await fetch('config.json');

		// If the file doesn't exist or can't be loaded, return null
		if (!response.ok) {
			logger.log('No config.json found or unable to load.');
			return null;
		}

		const config = await response.json();
		logger.log('Loaded config.json:', config);
		return config;
	} catch (error) {
		logger.error('Error loading config.json:', error);
		return null;
	}
}

/**
 * Applies configuration from config file to app data
 * @param {Vue} vm - Vue instance (this)
 * @param {Object} config - Configuration object from config.json
 */
function applyConfigFromFile(vm, config) {
	if (!config) return;

	logger.log('Applying configuration from config.json');

	// Process all keys in the config object
	Object.keys(config).forEach(key => {
		const configValue = config[key];

		// Handle special case for nested objects that need merging
		if (key === 'dataConfig' && configValue && typeof configValue === 'object') {
			// Merge dataConfig object instead of replacing
			vm.dataConfig = vm.dataConfig || {};
			Object.keys(configValue).forEach(subKey => {
				logger.log(`Applying config.dataConfig.${subKey} = ${JSON.stringify(configValue[subKey])}`);
				vm.dataConfig[subKey] = configValue[subKey];
			});
		}
		// Handle special case for pcaAxes object
		else if (key === 'pcaAxes' && configValue && typeof configValue === 'object') {
			// Merge pcaAxes object
			vm.pcaAxes = vm.pcaAxes || { x: 0, y: 1, z: 2 };
			Object.keys(configValue).forEach(axisKey => {
				logger.log(`Applying config.pcaAxes.${axisKey} = ${configValue[axisKey]}`);
				vm.pcaAxes[axisKey] = configValue[axisKey];
			});
		}
		// Apply other top-level properties directly
		else if (typeof configValue !== 'undefined') {
			logger.log(`Applying config.${key} = ${JSON.stringify(configValue)}`);
			vm[key] = configValue;
		}
	});

	// Sync pending values with their primary values
	if (typeof vm.colorByColumn !== 'undefined') {
		vm.pendingColorByColumn = vm.colorByColumn;
	}

	if (typeof vm.selectionColumn !== 'undefined') {
		vm.pendingSelectionColumn = vm.selectionColumn;
	}

	logger.log('Configuration applied from config.json');
}