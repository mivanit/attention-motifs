class DataModel {
	constructor(df, numericCols) {
		this.df = df;
		this.numericCols = numericCols;                // ordered ["pc.0", …]
		this.rowCount = df.data.length;
		this._pcaFlat = this._buildFlatArray();        // Float32Array row-major
	}

	_buildFlatArray() {
		const out = new Float32Array(this.rowCount * this.numericCols.length);
		for (let i = 0; i < this.rowCount; i++) {
			const row = this.df.data[i];
			for (let j = 0; j < this.numericCols.length; j++) {
				out[i * this.numericCols.length + j] = row[this.numericCols[j]];
			}
		}
		return out;
	}

	/** fast accessor */
	getCoord(rowIdx, axisIdx) {
		return this._pcaFlat[rowIdx * this.numericCols.length + axisIdx];
	}

	row(idx) { return this.df.data[idx]; }

	static async load(filename, numericalPrefix) {
		const spinner = NOTIF.spinner('Processing data...');

		try {
			let df;

			// Check if dataFile is null - use embedded data
			if (filename === null || filename === undefined) {
				if (!CONFIG.data || !Array.isArray(CONFIG.data)) {
					const errorMsg = 'No dataFile specified and no embedded data found in CONFIG.data';
					spinner.complete();
					NOTIF.error(errorMsg, new Error(errorMsg));
					throw new Error(errorMsg);
				}

				// Use embedded data directly
				const pbar = NOTIF.pbar('Processing embedded data...');
				pbar.progress(0.2);

				// Extract all unique column names from all rows
				const allColumns = new Set();
				for (const row of CONFIG.data) {
					Object.keys(row).forEach(key => allColumns.add(key));
				}

				pbar.progress(0.6);
				df = new DataFrame(CONFIG.data, Array.from(allColumns));
				pbar.progress(0.9);

			} else {
				// Original file loading logic
				spinner.complete();
				const downloadSpinner = NOTIF.spinner('Downloading data...');

				const resp = await fetch(filename);
				if (!resp.ok) {
					const errorMsg = `Failed to load data: ${resp.status} ${resp.statusText}`;
					downloadSpinner.complete();
					NOTIF.error(errorMsg, new Error(errorMsg));
					throw new Error(errorMsg);
				}

				downloadSpinner.complete();
				const pbar = NOTIF.pbar('Processing data...');

				pbar.progress(0.1);
				const text = await resp.text();

				pbar.progress(0.3);
				df = DataFrame.from_jsonl(text);
				pbar.progress(0.6);
			}

			const numeric = df.columns
				.filter(c => c.startsWith(numericalPrefix))
				.sort((a, b) => {
					// Extract the part after the prefix
					const aSuffix = a.substring(numericalPrefix.length);
					const bSuffix = b.substring(numericalPrefix.length);

					// Check if both suffixes are integers
					const aNum = parseInt(aSuffix, 10);
					const bNum = parseInt(bSuffix, 10);

					// If both are valid integers, sort numerically
					if (!isNaN(aNum) && !isNaN(bNum) &&
						aNum.toString() === aSuffix && bNum.toString() === bSuffix) {
						return aNum - bNum;
					}

					// Otherwise, sort lexicographically
					return a.localeCompare(b);
				});

			const result = new DataModel(df, numeric);

			if (filename === null || filename === undefined) {
				spinner.complete();
				NOTIF.success(`Loaded ${df.data.length} embedded data points with ${numeric.length} dimensions`);
			} else {
				const pbar = NOTIF.pbar('Processing data...');
				pbar.progress(1.0);
				pbar.complete();
				NOTIF.success(`Loaded ${df.data.length} data points with ${numeric.length} dimensions`);
			}

			return result;
		} catch (error) {
			spinner.complete();
			NOTIF.error('Failed to load data', error, 99999999999);
			console.error("DataModel load error:", error);
			throw error;
		}
	}
}