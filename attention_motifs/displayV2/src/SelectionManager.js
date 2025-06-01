/* SelectionManager.js  -- replaces previous version */
class SelectionManager {
	constructor(model, state) {
		this.model = model;
		this.state = state;
		this.palette = generateDistinctColors(128);
	}

	/** colour & selection attributes for a given row id */
	attrs(rowId) {
		const row = this.model.row(rowId);

		/* base colour comes from current colour-by column */
		const base = row[this.state.colorBy];
		const hash = typeof base === 'string'
			? [...base].reduce((s, c) => s + c.charCodeAt(0), 0)
			: (base | 0);
		const col = new THREE.Color(this.palette[hash % this.palette.length]);

		/* highlighted if its “selectBy” value is currently selected */
		const isSel = this.state.selection.has(row[this.state.selectBy]);

		return {
			r: isSel ? 1.0 : col.r,
			g: isSel ? 0.2 : col.g,
			b: isSel ? 0.2 : col.b
		};
	}
	randomizeColors() {
		this.palette = generateDistinctColors(128);
	}
}
