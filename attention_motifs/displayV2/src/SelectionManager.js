class SelectionManager {
	constructor(model, state) {
		this.model = model; this.state = state;
		this.palette = generateDistinctColors(128);
	}

	attrs(rowId) {
		const row = this.model.row(rowId);
		const sel = this.state.selection.has(rowId);
		const base = row[this.state.colorBy];
		const idx = typeof base === "string"
			? [...base].reduce((a, c) => a + c.charCodeAt(0), 0)
			: (base | 0);
		const color = new THREE.Color(this.palette[idx % this.palette.length]);

		return {
			r: sel ? 1.0 : color.r,
			g: sel ? 0.2 : color.g,
			b: sel ? 0.2 : color.b,
			size: sel ? this.state.selSize : this.state.nonSelSize,
			alpha: sel ? this.state.selOp : this.state.nonSelOp,
			isSel: sel
		};
	}
}