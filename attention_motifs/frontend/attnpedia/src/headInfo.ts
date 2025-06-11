interface PatternData {
	size: number;
	data: number[];
}

type DistanceMetric = 'L2' | 'L1' | 'cosine';

class HeadInfo {
	id: string;
	model: string;
	layer: number;
	head: number;
	classifications: string[];

	constructor(id: string, model: string, layer: number, head: number) {
		this.id = id;
		this.model = model;
		this.layer = layer;
		this.head = head;
		this.classifications = [];
	}

	static from_id(id: string): HeadInfo {
		// split by ":", should have 3 components
		// `{model}:L{layer}:H{head}`
		const parts = id.split(':');
		if (parts.length !== 3) {
			throw new Error(`Invalid id format: ${id}. Expected format: {model}:L{layer}:H{head}`);
		}

		const [model, layerPart, headPart] = parts;

		if (!layerPart.startsWith('L')) {
			throw new Error(`Invalid layer format: ${layerPart}. Expected format: L{number}`);
		}

		if (!headPart.startsWith('H')) {
			throw new Error(`Invalid head format: ${headPart}. Expected format: H{number}`);
		}

		const layer = parseInt(layerPart.slice(1), 10);
		const head = parseInt(headPart.slice(1), 10);

		if (isNaN(layer) || isNaN(head)) {
			throw new Error(`Invalid numeric values in id: ${id}`);
		}

		return new HeadInfo(id, model, layer, head);
	}

	async getPatterns(promptHash: string): Promise<PatternData> {
		// Mock pattern data
		await new Promise(resolve => setTimeout(resolve, 100));
		const size = Math.floor(Math.random() * 8) + 8;
		return {
			size: size,
			data: Array(size * size).fill(0).map(() => Math.random())
		};
	}

	distanceTo(otherHead: HeadInfo, metric: DistanceMetric = 'L2'): number {
		// Mock distance calculation
		const hash1 = this.id.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
		const hash2 = otherHead.id.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
		return Math.abs(hash1 - hash2) / 10000;
	}
}