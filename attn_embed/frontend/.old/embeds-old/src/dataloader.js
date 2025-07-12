/**
 * Functions for loading and processing data for the PCA visualization
 */

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
    pcaComponents: [],
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
 * @param {Array<string>} pcaColumnNames - Array of PCA column names
 */
function updatePlotCoordinates(plotData, pcaAxes, pcaColumnNames = null) {
  if (pcaColumnNames && pcaColumnNames.length > 0) {
    // Set x, y, z from the actual columns in the data
    plotData.x = plotData[pcaColumnNames[pcaAxes.x]];
    plotData.y = plotData[pcaColumnNames[pcaAxes.y]];
    plotData.z = plotData[pcaColumnNames[pcaAxes.z]];
  } else {
    // Fall back to using pcaComponents
    plotData.x = plotData.pcaComponents[pcaAxes.x];
    plotData.y = plotData.pcaComponents[pcaAxes.y];
    plotData.z = plotData.pcaComponents[pcaAxes.z];
  }
}

/**
 * Load data from a JSONL file and process it
 * @param {Object} config - Configuration object with data loading parameters
 * @param {Function} updateProgressCallback - Callback function for updating loading progress
 * @returns {Promise<Object>} Object containing the data frame, pca array, and plot data
 */
async function loadJsonlData(
  filePath,
  numericalPrefix,
  updateProgressCallback = () => {},
) {
  logger.time("Load Data");

  // Update progress to 10%
  updateProgressCallback(10, "Downloading data file...");

  // Load data from the configured file path
  const response = await fetch(filePath);
  const text = await response.text();
  logger.log("JSONL text length:", filePath, text.length);
  const dataFrame = DataFrame.from_jsonl(text);
  logger.timeEnd("Load Data");

  // Update progress to 40%
  updateProgressCallback(40, "Processing data...");

  // Extract numerical columns (PCA components) based on prefix
  logger.time("Extract PCA Components");
  const pcaColumnNames = dataFrame.columns
    .filter((col) => col.startsWith(numericalPrefix))
    .sort((a, b) => {
      // Try to extract numbers from the column names (e.g., "pc.5" -> 5)
      const aStr = a.split(".").pop();
      const bStr = b.split(".").pop();

      // Try to parse as integers
      const numA = parseInt(aStr);
      const numB = parseInt(bStr);

      // If both can be parsed as valid numbers, sort numerically
      if (!isNaN(numA) && !isNaN(numB)) {
        return numA - numB;
      }

      // Otherwise, fall back to lexicographical (string) sorting
      return aStr.localeCompare(bStr);
    });
  logger.log("Found PCA columns:", pcaColumnNames);
  const pcaArray = extractPcaComponents(
    dataFrame,
    numericalPrefix,
    pcaColumnNames,
  );
  logger.timeEnd("Extract PCA Components");

  // Update progress to 70%
  updateProgressCallback(70, "Preparing plot data");

  // Process data
  logger.time("Process Data");
  const plotData = processDataFromDataFrame(
    dataFrame,
    pcaArray,
    { x: 0, y: 1, z: 2 },
    pcaColumnNames,
  );
  logger.timeEnd("Process Data");

  // Update progress to 90%
  updateProgressCallback(90, "Finding available columns...");

  // Find available columns for selection
  const availableColumns = findAvailableColumns(dataFrame, numericalPrefix);
  logger.log("Available categorical columns:", availableColumns);

  return {
    dataFrame,
    pcaArray,
    plotData,
    availableColumns,
    pcaColumnNames,
  };
}

/**
 * Extract PCA components from the DataFrame based on the numerical prefix
 * @param {DataFrame} dataFrame - The DataFrame containing the data
 * @param {string} numericalPrefix - Prefix for numerical columns
 * @param {Array<string>} pcaColumnNames - Optional array of PCA column names
 * @returns {Object} PCA array-like object
 */
function extractPcaComponents(
  dataFrame,
  numericalPrefix,
  pcaColumnNames = null,
) {
  // Use provided column names or find them
  const numericalColumns =
    pcaColumnNames ||
    dataFrame.columns.filter((col) => col.startsWith(numericalPrefix)).sort();

  logger.log("Using numerical columns:", numericalColumns);

  if (numericalColumns.length === 0) {
    throw new Error(
      `No numerical columns found with prefix "${numericalPrefix}"`,
    );
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
    },
  };
}

/**
 * Find available columns for selection (non-numerical columns)
 * @param {DataFrame} dataFrame - The DataFrame containing the data
 * @param {string} numericalPrefix - Prefix for numerical columns
 * @returns {Array<string>} Array of available column names
 */
function findAvailableColumns(dataFrame, numericalPrefix) {
  return dataFrame.columns.filter((col) => !col.startsWith(numericalPrefix));
}

/**
 * Process loaded data into a format suitable for plotting
 * @param {DataFrame} dataFrame - The DataFrame containing the data
 * @param {Object} pcaArray - The PCA array-like object
 * @param {Object} pcaAxes - Optional axes selection {x, y, z}
 * @param {Array<string>} pcaColumnNames - Optional array of PCA column names
 * @returns {Object} The processed plot data
 */
function processDataFromDataFrame(
  dataFrame,
  pcaArray,
  pcaAxes = { x: 0, y: 1, z: 2 },
  pcaColumnNames = null,
) {
  if (!pcaArray || !dataFrame) {
    throw new Error("PCA data or metadata not available");
  }

  // Create the plot data object with arrays for all available PCA components
  const plotData = {
    // Initialize empty arrays for all PCA components
    pcaComponents: [],
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
  if (pcaColumnNames && pcaColumnNames.length > 0) {
    // Use column names directly if available
    plotData.x = plotData[pcaColumnNames[pcaAxes.x]];
    plotData.y = plotData[pcaColumnNames[pcaAxes.y]];
    plotData.z = plotData[pcaColumnNames[pcaAxes.z]];
  } else {
    // Fall back to using pcaComponents
    updatePlotCoordinates(plotData, pcaAxes);
  }

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
      // Handle array parameters (comma or tilde separated)
      if (paramName === "selected") {
        updatedConfig.selectedValues = paramValue.includes("~")
          ? paramValue.split("~")
          : paramValue.split(",");
      }
      // Handle numeric parameters
      else if (
        [
          "xAxis",
          "yAxis",
          "zAxis",
          "selOpacity",
          "nonSelOpacity",
          "selSize",
          "nonSelSize",
        ].includes(paramName)
      ) {
        updatedConfig[configKey] = Number(paramValue);
      }
      // Handle regular string parameters
      else {
        updatedConfig[configKey] = paramValue;
      }
      logger.log(
        `Using URL parameter ${paramName}=${paramValue} for ${configKey}`,
      );
    }
  }

  return updatedConfig;
}

/**
 * Update URL with current configuration
 * @param {Object} dataConfig - Current data configuration
 * @param {Object} urlParams - Mapping of config keys to URL parameter names
 * @param {Object} appState - Optional additional app state to include in URL
 * @returns {string} Updated base URL
 */
function updateUrlWithConfig(dataConfig, urlParams, appState = {}) {
  const url = new URL(window.location.href);
  const params = url.searchParams;

  // Set parameters from current configuration
  for (const [configKey, paramName] of Object.entries(urlParams)) {
    if (configKey in dataConfig) {
      params.set(paramName, dataConfig[configKey]);
    }
  }

  // Add additional state parameters if provided
  if (Object.keys(appState).length > 0) {
    // Handle selected values
    if (appState.selectedValues && appState.selectedValues.length > 0) {
      params.set(urlParams.selectedValues, appState.selectedValues.join("~"));
    } else {
      params.delete(urlParams.selectedValues);
    }

    // Handle axis settings
    if (appState.pcaAxes) {
      params.set(urlParams.xAxis, appState.pcaAxes.x);
      params.set(urlParams.yAxis, appState.pcaAxes.y);
      params.set(urlParams.zAxis, appState.pcaAxes.z);
    }

    // Handle appearance settings
    if (appState.selectedOpacity !== undefined)
      params.set(urlParams.selectedOpacity, appState.selectedOpacity);
    if (appState.nonSelectedOpacity !== undefined)
      params.set(urlParams.nonSelectedOpacity, appState.nonSelectedOpacity);
    if (appState.selectedSize !== undefined)
      params.set(urlParams.selectedSize, appState.selectedSize);
    if (appState.nonSelectedSize !== undefined)
      params.set(urlParams.nonSelectedSize, appState.nonSelectedSize);
  }

  // Update browser URL without reloading the page
  window.history.replaceState({}, "", url.toString());

  // Return the updated base URL
  return url.toString().split("?")[0] + "?";
}
