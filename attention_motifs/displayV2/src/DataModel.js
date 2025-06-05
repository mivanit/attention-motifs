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

	getCoord(rowIdx, axisIdx) {
		return this._pcaFlat[rowIdx * this.numericCols.length + axisIdx];
	}

	row(idx) { return this.df.data[idx]; }

	static async load(filename, numericalPrefix, progressCallback = (msg, err = null) => {}) {
		progressCallback('Downloading data...');

		const resp = await fetch(filename);
		if (!resp.ok) {
			progressCallback(`Failed to load data: ${resp.status} ${resp.statusText}`, Error);
		}

		progressCallback('Parsing data...');

		const text = await resp.text();
		const df = DataFrame.from_jsonl(text);

		progressCallback('Processing columns...');

		const numeric = df.columns
			.filter(c => c.startsWith(numericalPrefix))
			.sort((a, b) => {
				const aSuffix = a.substring(numericalPrefix.length);
				const bSuffix = b.substring(numericalPrefix.length);
				const aNum = parseInt(aSuffix, 10);
				const bNum = parseInt(bSuffix, 10);
				if (!isNaN(aNum) && !isNaN(bNum) &&
					aNum.toString() === aSuffix && bNum.toString() === bSuffix) {
					return aNum - bNum;
				}
				return a.localeCompare(b);
			});

		progressCallback('Finalizing...');

		return new DataModel(df, numeric);
	}
}