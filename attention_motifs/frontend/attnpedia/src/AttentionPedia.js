class AttentionPedia {
	constructor() {
		this.heads = new Map();
		this.classifications = new Map();
		this._initializeMockData();
	}

	_initializeMockData() {
		const models = ['gpt2-small', 'gpt2-medium', 'gpt2-large'];
		const classificationTypes = ['induction', 'duplicate', 'previous', 'name-mover', 'subject', 'copy'];
		const layers = 12;
		const headsPerLayer = 12;

		// Generate mock heads
		models.forEach(model => {
			const paramCount = model === 'gpt2-small' ? 124 : model === 'gpt2-medium' ? 355 : 774;

			for (let layer = 0; layer < layers; layer++) {
				for (let head = 0; head < headsPerLayer; head++) {
					const id = `${model}:L${layer}:H${head}`;
					const headInfo = new HeadInfo(id, model, layer, head, paramCount);

					// Add random classifications
					if (Math.random() > 0.3) {
						const classType = classificationTypes[Math.floor(Math.random() * classificationTypes.length)];
						headInfo.classifications.push({
							name: classType,
							type: classType,
							source: Math.random() > 0.5 ? 'IOI' : 'SAE-survey'
						});

						// Add to classification map
						if (!this.classifications.has(classType)) {
							this.classifications.set(classType, []);
						}
						this.classifications.get(classType).push(headInfo);
					}

					// Add mock citations
					if (Math.random() > 0.7) {
						headInfo.citations.push({
							title: 'IOI: indirect object identification',
							authors: 'Wang et al. 2022',
							url: 'https://arxiv.org/pdf/2211.00593',
							description: `Model: ${model}`
						});
					}

					this.heads.set(id, headInfo);
				}
			}
		});
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