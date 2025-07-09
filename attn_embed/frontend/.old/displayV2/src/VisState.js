/* VisState.js  -- replaces previous version with URL sync */
class VisState extends EventTarget {
	constructor(model) {
		super();
		this.model = model;

		/* view / meta options - use CONFIG defaults */
		this.axis = { ...CONFIG.axes };

		/* run-time configurable keys */
		this.colorBy = CONFIG.defaultColorColumn;
		this.selectBy = CONFIG.defaultSelectionColumn;

		/* appearance - use CONFIG defaults */
		this.nonSelSize = CONFIG.nonSelectedPoints.size;
		this.selSize = CONFIG.selectedPoints.size;
		this.nonSelOp = CONFIG.nonSelectedPoints.opacity;
		this.selOp = CONFIG.selectedPoints.opacity;
		this.nonSelColor = CONFIG.nonSelectedPoints.color;

		/* store *values* (categories) now, not row indices */
		this.selection = new Set();

		// Initialize selection from CONFIG if specified
		// Handle both array and single value cases from URL parsing
		if (CONFIG.selectedValues) {
			if (Array.isArray(CONFIG.selectedValues)) {
				CONFIG.selectedValues.forEach(value => this.selection.add(value));
			} else if (typeof CONFIG.selectedValues === 'string') {
				// Single value from URL (not comma-separated)
				this.selection.add(CONFIG.selectedValues);
				// Update CONFIG to be consistent array format
				CONFIG.selectedValues = [CONFIG.selectedValues];
			}
		}
	}

	isNumericColumn(column) {
		if (!this.model.df.columns.includes(column)) return false;

		const values = this.model.df.col(column);
		// Check if most values are numbers
		const numericCount = values.filter(v => typeof v === 'number' && !isNaN(v)).length;
		return numericCount > values.length * 0.8; // 80% threshold
	}

	/* ---------- helpers ---------- */
	setAxis(dim, val) {
		this.axis[dim] = val;
		// Update CONFIG to keep it in sync
		CONFIG.axes[dim] = val;
		this._syncToURL();
		this._fire('axis');
	}

	setColorBy(col) {
		this.colorBy = col;
		CONFIG.defaultColorColumn = col;
		this._syncToURL();
		this._fire('vis');
	}

	setSelectBy(col) {
		this.selectBy = col;
		CONFIG.defaultSelectionColumn = col;
		this.clearSel();
		this._syncToURL();
		this._fire('vis');
	}

	setVisParam(k, v) {
		this[k] = v;

		// Update CONFIG to keep it in sync
		switch (k) {
			case 'selSize':
				CONFIG.selectedPoints.size = v;
				break;
			case 'selOp':
				CONFIG.selectedPoints.opacity = v;
				break;
			case 'nonSelSize':
				CONFIG.nonSelectedPoints.size = v;
				break;
			case 'nonSelOp':
				CONFIG.nonSelectedPoints.opacity = v;
				break;
			case 'nonSelColor':
				CONFIG.nonSelectedPoints.color = v;
				break;
		}

		this._syncToURL();
		this._fire('vis');
	}

	/** toggle category value */
	toggleValue(v) {
		if (v == null) return;
		this.selection.has(v) ? this.selection.delete(v)
			: this.selection.add(v);

		// Update CONFIG with current selection
		CONFIG.selectedValues = Array.from(this.selection);
		this._syncToURL();
		this._fire('selection');
	}

	clearSel() {
		this.selection.clear();
		CONFIG.selectedValues = [];
		this._syncToURL();
		this._fire('selection');
	}

	/**
	 * Sync current state to URL parameters
	 */
	_syncToURL() {
		updateURL();
	}

	_fire(type) { this.dispatchEvent(new Event(type)); }
}