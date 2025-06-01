/* SelectionManager.js  -- optimized version */
class SelectionManager {
	constructor(model, state) {
		this.model = model;
		this.state = state;
		this.palette = generateDistinctColors(128);

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
			return new THREE.Color(0.5, 0.5, 0.5); // Gray for invalid values
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

		// Viridis colormap approximation
		return this._viridis(t);
	}

	_getCategoricalColor(value) {
		if (value === null || value === 'null' || value === 'unknown') {
			return new THREE.Color(0.3, 0.3, 0.3); // Dark gray for null values
		}

		// Use a simple hash for categorical values to avoid expensive sorting
		const hash = typeof value === 'string'
			? [...value].reduce((s, c) => s + c.charCodeAt(0), 0)
			: (value | 0);

		return new THREE.Color(this.palette[hash % this.palette.length]);
	}

	_viridis(t) {
		// Viridis colormap approximation
		t = Math.max(0, Math.min(1, t));

		const r = 0.267004 + t * (0.105010 + t * (0.330010 + t * (2.437600 + t * (-5.179800 + t * 2.066100))));
		const g = 0.004874 + t * (0.406910 + t * (1.193600 + t * (-1.375200 + t * (0.813500 + t * (-0.073200)))));
		const b = 0.329415 + t * (0.718080 + t * (-0.724400 + t * (0.063300 + t * 0.016700)));

		return new THREE.Color(r, g, b);
	}

	randomizeColors() {
		this.palette = generateDistinctColors(128);
		// Clear color cache when palette changes
		this._colorCache.clear();
	}

	// Clear caches when column changes
	clearCaches() {
		this._colorCache.clear();
		this._numericRangeCache.clear();
	}
}