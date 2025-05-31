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
		const resp = await fetch(filename);
		const text = await resp.text();
		const df = DataFrame.from_jsonl(text);
		const numeric = df.columns
			.filter(c => c.startsWith(numericalPrefix))
			.sort();
		return new DataModel(df, numeric);
	}
}
