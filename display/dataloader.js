/**
 * Functions for loading and processing data for the PCA visualization
 */

/**
 * Load PCA data from NPY file
 * @returns {Promise<NDArray>} The loaded PCA array
 */
function loadPcaData() {
    logger.time('Load PCA NPY');
    return NDArray.load("../data/features/pca_data.npy").then(pcaArray => {
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
    
    return fetch("../data/features/features_meta.jsonl")
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
    return dataFrame.columns.filter(col => {
        try {
            const uniqueCount = dataFrame.col_unique(col).size;
            // At least 2 unique values but not too many (fewer than 50)
            return uniqueCount < 50;
        } catch (e) {
            logger.error(`Error checking column ${col}:`, e);
            return false;
        }
    });
}