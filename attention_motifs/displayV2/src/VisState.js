/* VisState.js  -- replaces previous version */
class VisState extends EventTarget {
	constructor(model) {
		super();
		this.model = model;

		/* view / meta options */
		this.axis = { x: 0, y: 1, z: 2 };

		/* run-time configurable keys */
		this.colorBy = CONFIG.defaultColorColumn;
		this.selectBy = CONFIG.defaultSelectionColumn;

		/* appearance */
		this.nonSelSize = 4; this.selSize = 6;
		this.nonSelOp = 0.25; this.selOp = 1.0;
		this.nonSelColor = '#666666';


		/* store *values* (categories) now, not row indices */
		this.selection = new Set();
	}

	isNumericColumn(column) {
		if (!this.model.df.columns.includes(column)) return false;

		const values = this.model.df.col(column);
		// Check if most values are numbers
		const numericCount = values.filter(v => typeof v === 'number' && !isNaN(v)).length;
		return numericCount > values.length * 0.8; // 80% threshold
	}

	/* ---------- helpers ---------- */
	setAxis(dim, val) { this.axis[dim] = val; this._fire('axis'); }
	setColorBy(col) { this.colorBy = col; this._fire('vis'); }
	setSelectBy(col) { this.selectBy = col; this.clearSel(); this._fire('vis'); }
	setVisParam(k, v) { this[k] = v; this._fire('vis'); }

	/** toggle category value */
	toggleValue(v) {
		if (v == null) return;
		this.selection.has(v) ? this.selection.delete(v)
			: this.selection.add(v);
		this._fire('selection');
	}

	clearSel() { this.selection.clear(); this._fire('selection'); }

	_fire(type) { this.dispatchEvent(new Event(type)); }
}
