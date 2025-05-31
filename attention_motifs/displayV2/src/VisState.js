class VisState extends EventTarget {
	constructor(model) {
		super();
		this.model = model;

		/* view params */
		this.axis = { x: 0, y: 1, z: 2 };

		this.colorBy = CONFIG.defaultColorColumn;
		this.selectBy = CONFIG.defaultSelectionColumn;

		this.nonSelSize = 4; this.selSize = 6;
		this.nonSelOp = 0.25; this.selOp = 1.0;

		this.selection = new Set();          // store row-ids
	}

	/* helpers ------------------------------------------------------ */
	setAxis(dim, val) { this.axis[dim] = val; this._fire("axis"); }
	setColorBy(c) { this.colorBy = c; this._fire("vis"); }
	setSelectBy(c) { this.selectBy = c; this.selection.clear(); this._fire("vis"); }
	setVisParam(k, v) { this[k] = v; this._fire("vis"); }

	toggle(id) {
		this.selection.has(id) ? this.selection.delete(id)
			: this.selection.add(id);
		this._fire("selection");
	}
	clearSel() { this.selection.clear(); this._fire("selection"); }

	_fire(t) { this.dispatchEvent(new Event(t)); }
}