class AttentionPedia {
	constructor() {
		this.heads = new Map();
		this.classifications = new Map();
		this._initializeMockData();
	}

	getHead(id) {
		return this.heads.get(id);
	}

	getHeadsByClassification(classification) {
		return this.classifications.get(classification) || [];
	}

	getNearestHeads(headId, n = 5) {
		const targetHead = this.heads.get(headId);
		if (!targetHead) return [];

		const distances = [];
		this.heads.forEach((head, id) => {
			if (id !== headId && head.model === targetHead.model) {
				distances.push({
					head: head,
					distance: targetHead.distanceTo(head)
				});
			}
		});

		return distances
			.sort((a, b) => a.distance - b.distance)
			.slice(0, n)
			.map(item => ({ ...item.head, distance: item.distance }));
	}
}