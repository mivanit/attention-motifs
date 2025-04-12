/**
 * Functions for loading and processing data for the PCA visualization
 */

/**
 * Load PCA data from NPY file
 * @returns {Promise<NDArray>} The loaded PCA array
 */
function loadPcaData() {
    console.time('Load PCA NPY');
    console.log('Requesting PCA data file');
    return NDArray.load("../data/features/pca_data.npy").then(pcaArray => {
        console.timeEnd('Load PCA NPY');
        console.log('PCA data loaded:', pcaArray);
        console.log('PCA data shape:', pcaArray.shape);
        console.log('PCA data type:', pcaArray.dtype);
        return pcaArray;
    });
}

/**
 * Load metadata from JSONL file
 * @returns {Promise<DataFrame>} The loaded DataFrame
 */
function loadMetadata() {
    console.time('Load JSONL');
    console.log('Requesting metadata file');
    
    return fetch("../data/features/features_scaled.jsonl")
        .then(response => {
            console.log('Metadata file received, processing text...');
            return response.text();
        })
        .then(text => {
            console.log('JSONL text length:', text.length);
            console.log('Parsing JSONL into DataFrame...');
            const dataFrame = DataFrame.from_jsonl(text);
            console.timeEnd('Load JSONL');
            console.log('DataFrame loaded:', dataFrame);
            console.log('DataFrame columns:', dataFrame.columns);
            console.log('DataFrame rows:', dataFrame.length);
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

    console.log('Processing data for plotting...');
    const pcaData = pcaArray;

    // Create the plot data object with arrays for all available PCA components
    const plotData = {
        // Initialize empty arrays for all PCA components
        pcaComponents: []
    };

    // Initialize arrays for each PCA component
    for (let i = 0; i < pcaData.shape[1]; i++) {
        console.log(`Extracting PCA component ${i + 1}...`);
        // Use the get method with null to get all values for a specific component
        plotData.pcaComponents[i] = Array.from(pcaData.get(null, i).data);
    }

    // Add metadata columns from the DataFrame to plotData
    console.log('Adding metadata columns...');
    for (const column of dataFrame.columns) {
        plotData[column] = dataFrame.col(column);
    }

    console.log('Data processing complete');
    return plotData;
}

/**
 * Update x, y, z arrays in plotData based on PCA axis selection
 * @param {Object} plotData - The plot data object 
 * @param {Object} pcaAxes - The selected PCA axes {x, y, z}
 */
function updatePlotCoordinates(plotData, pcaAxes) {
    console.log(`Updating coordinates to PC${pcaAxes.x}, PC${pcaAxes.y}, PC${pcaAxes.z}`);

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
            console.error(`Error checking column ${col}:`, e);
            return false;
        }
    });
}