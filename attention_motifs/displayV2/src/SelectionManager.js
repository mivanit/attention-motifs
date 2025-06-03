/* SelectionManager.js  -- optimized version using CONFIG */
class SelectionManager {
	constructor(model, state) {
		this.model = model;
		this.state = state;
		this.palette = generateDistinctColors(CONFIG.colors.paletteSize);

		// Cache for expensive computations
		this._colorCache = new Map();
		this._numericRangeCache = new Map();
	}

	/** colour & selection attributes for a given row id */
	attrs(rowId) {
		const row = this.model.row(rowId);

		// Check if this point should be treated as selected
		const selectValue = row[this.state.selectBy];
		const isExplicitlySelected = this.state.selection.has(selectValue);
		const isValidValue = selectValue !== null && selectValue !== 'null' && selectValue !== 'unknown';

		// If no explicit selection, treat all valid points as selected
		const treatAsSelected = isExplicitlySelected || (this.state.selection.size === 0 && isValidValue);

		// Get base color from colorBy column (with caching)
		const colorValue = row[this.state.colorBy];
		const baseColor = this._getColorCached(colorValue);

		// Apply selection highlighting
		if (treatAsSelected) {
			return {
				r: baseColor.r,
				g: baseColor.g,
				b: baseColor.b,
				size: this.state.selSize,
				opacity: this.state.selOp
			};
		} else {
			// Non-selected styling
			const nonSelColor = new THREE.Color(this.state.nonSelColor);
			return {
				r: nonSelColor.r,
				g: nonSelColor.g,
				b: nonSelColor.b,
				size: this.state.nonSelSize,
				opacity: this.state.nonSelOp
			};
		}
	}

	_getColorCached(value) {
		const cacheKey = `${this.state.colorBy}:${value}`;

		if (!this._colorCache.has(cacheKey)) {
			let color;
			if (this.state.isNumericColumn(this.state.colorBy)) {
				color = this._getViridisColor(value, this.state.colorBy);
			} else {
				color = this._getCategoricalColor(value);
			}
			this._colorCache.set(cacheKey, color);
		}

		return this._colorCache.get(cacheKey);
	}

	_getViridisColor(value, column) {
		if (value === null || value === undefined || isNaN(value)) {
			return new THREE.Color(CONFIG.colors.nullValueColor);
		}

		// Get cached min/max for the column
		if (!this._numericRangeCache.has(column)) {
			const values = this.model.df.col(column).filter(v => typeof v === 'number' && !isNaN(v));
			const min = Math.min(...values);
			const max = Math.max(...values);
			this._numericRangeCache.set(column, { min, max });
		}

		const { min, max } = this._numericRangeCache.get(column);

		// Normalize to 0-1
		const t = max > min ? (value - min) / (max - min) : 0;

		// Viridis colormap approximation using CONFIG coefficients
		return this._viridis(t);
	}

	_getCategoricalColor(value) {
		if (value === null || value === 'null' || value === 'unknown') {
			return new THREE.Color(CONFIG.colors.nullValueColor);
		}

		// Use a simple hash for categorical values to avoid expensive sorting
		const hash = typeof value === 'string'
			? [...value].reduce((s, c) => s + c.charCodeAt(0), 0)
			: (value | 0);

		return new THREE.Color(this.palette[hash % this.palette.length]);
	}

	_viridis(t) {
		// Viridis colormap approximation using CONFIG coefficients
		t = Math.max(0, Math.min(1, t));

		const { r: rCoeff, g: gCoeff, b: bCoeff } = CONFIG.colors.viridisColors;

		const r = rCoeff[0] + t * (rCoeff[1] + t * (rCoeff[2] + t * (rCoeff[3] + t * (rCoeff[4] + t * rCoeff[5]))));
		const g = gCoeff[0] + t * (gCoeff[1] + t * (gCoeff[2] + t * (gCoeff[3] + t * (gCoeff[4] + t * gCoeff[5]))));
		const b = bCoeff[0] + t * (bCoeff[1] + t * (bCoeff[2] + t * (bCoeff[3] + t * (bCoeff[4] + t * bCoeff[5]))));

		return new THREE.Color(r, g, b);
	}

	randomizeColors() {
		this.palette = generateDistinctColors(CONFIG.colors.paletteSize);
		// Clear color cache when palette changes
		this._colorCache.clear();
	}

	// Clear caches when column changes
	clearCaches() {
		this._colorCache.clear();
		this._numericRangeCache.clear();
	}
}