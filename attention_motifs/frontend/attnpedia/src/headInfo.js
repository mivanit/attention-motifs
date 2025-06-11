class HeadInfo {
	constructor(id, model, layer, head) {
		this.id = id;
		this.model = model;
		this.layer = layer;
		this.head = head;
		this.classifications = [];
	}


	static from_id(id) {
		// split by ":", should have 3 components
		// `{model}:L{layer}:H{head}`
		
	}

	async getPatterns(promptHash) {
		// Mock pattern data
		await new Promise(resolve => setTimeout(resolve, 100));
		const size = Math.floor(Math.random() * 8) + 8;
		return {
			size: size,
			data: Array(size * size).fill(0).map(() => Math.random())
		};
	}

	distanceTo(otherHead, metric = 'L2') {
		// Mock distance calculation
		const hash1 = this.id.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
		const hash2 = otherHead.id.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
		return Math.abs(hash1 - hash2) / 10000;
	}
}	