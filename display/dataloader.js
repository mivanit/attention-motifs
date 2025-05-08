/**
 * Functions for loading and processing data for the PCA visualization
 */

/**
 * Load PCA data from NPY file
 * @returns {Promise<NDArray>} The loaded PCA array
 */
function loadPcaData() {
    logger.time('Load PCA NPY');
    return NDArray.load("data/features/pca.npy").then(pcaArray => {
        logger.timeEnd('Load PCA NPY');
        logger.log('PCA data shape:', pcaArray.shape);
        return pcaArray;
    });
}

/**
 * Load metadata from JSONL file
 * @returns {Promise<DataFrame>} The loaded DataFrame
 */
function loadMetadata() {
    logger.time('Load JSONL');

    return fetch("data/features/meta.jsonl")
        .then(response => response.text())
        .then(text => {
            logger.log('JSONL text length:', text.length);
            const dataFrame = DataFrame.from_jsonl(text);
            logger.timeEnd('Load JSONL');
            logger.log('DataFrame rows:', dataFrame.length);
            return dataFrame;
        });
}

/**
 * Process loaded PCA data and metadata into a format suitable for plotting
 * @param {NDArray} pcaArray - The PCA array
 * @param {DataFrame} dataFrame - The metadata DataFrame
 * @returns {Object} The processed plot data
 */
function processData(pcaArray, dataFrame) {
    if (!pcaArray || !dataFrame) {
        throw new Error("PCA data or metadata not loaded");
    }

    const pcaData = pcaArray;

    // Create the plot data object with arrays for all available PCA components
    const plotData = {
        // Initialize empty arrays for all PCA components
        pcaComponents: []
    };

    // Initialize arrays for each PCA component
    for (let i = 0; i < pcaData.shape[1]; i++) {
        // Use the get method with null to get all values for a specific component
        plotData.pcaComponents[i] = Array.from(pcaData.get(null, i).data);
    }

    // Add metadata columns from the DataFrame to plotData
    for (const column of dataFrame.columns) {
        plotData[column] = dataFrame.col(column);
    }

    return plotData;
}

/**
 * Update x, y, z arrays in plotData based on PCA axis selection
 * @param {Object} plotData - The plot data object 
 * @param {Object} pcaAxes - The selected PCA axes {x, y, z}
 */
function updatePlotCoordinates(plotData, pcaAxes) {
    // Set x, y, z from the selected PCA components
    plotData.x = plotData.pcaComponents[pcaAxes.x];
    plotData.y = plotData.pcaComponents[pcaAxes.y];
    plotData.z = plotData.pcaComponents[pcaAxes.z];
}

/**
 * Find categorical columns in the DataFrame (those with fewer than 50 unique values)
 * @param {DataFrame} dataFrame - The metadata DataFrame
 * @returns {Array<string>} Array of column names that have categorical values
 */
function findCategoricalColumns(dataFrame) {
    return dataFrame.columns;
    // return dataFrame.columns.filter(col => {
    //     try {
    //         const uniqueCount = dataFrame.col_unique(col).size;
    //         // At least 2 unique values but not too many (fewer than 50)
    //         return uniqueCount < 50;
    //     } catch (e) {
    //         logger.error(`Error checking column ${col}:`, e);
    //         return false;
    //     }
    // });
}

/**
 * Functions for loading and processing data for the PCA visualization
 */

/**
 * Load data from a JSONL file and process it
 * @param {Object} config - Configuration object with data loading parameters
 * @param {Function} updateProgressCallback - Callback function for updating loading progress
 * @returns {Promise<Object>} Object containing the data frame, pca array, and plot data
 */
async function loadJsonlData(config, updateProgressCallback = () => { }) {
    const { filePath, numericalPrefix } = config;

    logger.time('Load Data');

    // Update progress to 10%
    updateProgressCallback(10, 'Downloading data file...');

    // Load data from the configured file path
    const response = await fetch(filePath);
    const text = await response.text();
    logger.log('JSONL text length:', text.length);
    const dataFrame = DataFrame.from_jsonl(text);
    logger.timeEnd('Load Data');

    // Update progress to 40%
    updateProgressCallback(40, 'Processing data...');

    // Extract numerical columns (PCA components) based on prefix
    logger.time('Extract PCA Components');
    const pcaArray = extractPcaComponents(dataFrame, numericalPrefix);
    logger.timeEnd('Extract PCA Components');

    // Update progress to 70%
    updateProgressCallback(70, 'Preparing plot data');

    // Process data
    logger.time('Process Data');
    const plotData = processDataFromDataFrame(dataFrame, pcaArray);
    logger.timeEnd('Process Data');

    // Update progress to 90%
    updateProgressCallback(90, 'Finding available columns...');

    // Find available columns for selection
    const availableColumns = findAvailableColumns(dataFrame, numericalPrefix);
    logger.log('Available categorical columns:', availableColumns);

    return {
        dataFrame,
        pcaArray,
        plotData,
        availableColumns
    };
}

/**
 * Extract PCA components from the DataFrame based on the numerical prefix
 * @param {DataFrame} dataFrame - The DataFrame containing the data
 * @param {string} numericalPrefix - Prefix for numerical columns
 * @returns {Object} PCA array-like object
 */
function extractPcaComponents(dataFrame, numericalPrefix) {
    const numericalColumns = dataFrame.columns.filter(col =>
        col.startsWith(numericalPrefix)
    ).sort();

    logger.log('Found numerical columns:', numericalColumns);

    if (numericalColumns.length === 0) {
        throw new Error(`No numerical columns found with prefix "${numericalPrefix}"`);
    }

    // Extract the numerical data into a virtual PCA array
    const numRows = dataFrame.length;
    const numCols = numericalColumns.length;

    // Create a Float32Array to store the PCA data
    const pcaData = new Float32Array(numRows * numCols);

    // Fill the array with data from the DataFrame
    for (let i = 0; i < numRows; i++) {
        const row = dataFrame.row(i);
        for (let j = 0; j < numCols; j++) {
            const colName = numericalColumns[j];
            pcaData[i * numCols + j] = row[colName] || 0;
        }
    }

    // Create an NDArray-like object from the data
    return {
        data: pcaData,
        shape: [numRows, numCols],
        get: function (rowIdx, colIdx) {
            if (rowIdx === null) {
                // Extract entire column
                const colData = new Float32Array(numRows);
                for (let i = 0; i < numRows; i++) {
                    colData[i] = this.data[i * numCols + colIdx];
                }
                return { data: colData };
            } else if (colIdx === null) {
                // Extract entire row
                const rowData = new Float32Array(numCols);
                for (let j = 0; j < numCols; j++) {
                    rowData[j] = this.data[rowIdx * numCols + j];
                }
                return { data: rowData };
            } else {
                // Return a single value
                return this.data[rowIdx * numCols + colIdx];
            }
        }
    };
}

/**
 * Find available columns for selection (non-numerical columns)
 * @param {DataFrame} dataFrame - The DataFrame containing the data
 * @param {string} numericalPrefix - Prefix for numerical columns
 * @returns {Array<string>} Array of available column names
 */
function findAvailableColumns(dataFrame, numericalPrefix) {
    return dataFrame.columns.filter(col =>
        !col.startsWith(numericalPrefix)
    );
}

/**
 * Process loaded data into a format suitable for plotting
 * @param {DataFrame} dataFrame - The DataFrame containing the data
 * @param {Object} pcaArray - The PCA array-like object
 * @param {Object} pcaAxes - Optional axes selection {x, y, z}
 * @returns {Object} The processed plot data
 */
function processDataFromDataFrame(dataFrame, pcaArray, pcaAxes = { x: 0, y: 1, z: 2 }) {
    if (!pcaArray || !dataFrame) {
        throw new Error("PCA data or metadata not available");
    }

    // Create the plot data object with arrays for all available PCA components
    const plotData = {
        // Initialize empty arrays for all PCA components
        pcaComponents: []
    };

    // Initialize arrays for each PCA component
    for (let i = 0; i < pcaArray.shape[1]; i++) {
        // Use the get method with null to get all values for a specific component
        plotData.pcaComponents[i] = Array.from(pcaArray.get(null, i).data);
    }

    // Add metadata columns from the DataFrame to plotData
    for (const column of dataFrame.columns) {
        plotData[column] = dataFrame.col(column);
    }

    // Initialize x, y, z with the selected components
    updatePlotCoordinates(plotData, pcaAxes);

    return plotData;
}

/**
 * Parse URL parameters to override default configuration
 * @param {Object} dataConfig - Default data configuration
 * @param {Object} urlParams - Mapping of config keys to URL parameter names
 * @returns {Object} Updated configuration object
 */
function parseUrlParams(dataConfig, urlParams) {
    const urlSearchParams = new URLSearchParams(window.location.search);
    const updatedConfig = { ...dataConfig };

    // Update config from URL parameters if they exist
    for (const [configKey, paramName] of Object.entries(urlParams)) {
        const paramValue = urlSearchParams.get(paramName);
        if (paramValue) {
            updatedConfig[configKey] = paramValue;
            logger.log(`Using URL parameter ${paramName}=${paramValue} for ${configKey}`);
        }
    }

    return updatedConfig;
}

/**
 * Update URL with current configuration
 * @param {Object} dataConfig - Current data configuration
 * @param {Object} urlParams - Mapping of config keys to URL parameter names
 * @returns {string} Updated base URL
 */
function updateUrlWithConfig(dataConfig, urlParams) {
    const url = new URL(window.location.href);
    const params = url.searchParams;

    // Set parameters from current configuration
    for (const [configKey, paramName] of Object.entries(urlParams)) {
        params.set(paramName, dataConfig[configKey]);
    }

    // Update browser URL without reloading the page
    window.history.replaceState({}, '', url.toString());

    // Return the updated base URL
    return url.toString().split('?')[0] + '?';
}