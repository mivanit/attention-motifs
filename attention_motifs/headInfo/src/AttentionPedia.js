function generateRandomEmbedding() {
	return Array.from({ length: 64 }, () => Math.random() * 2 - 1);
}

async function loadAttentionPattern(headId, promptHash) {
	// Simulate API call
	await new Promise(resolve => setTimeout(resolve, 100));
	return promptHashes.map(hash => ({
		hash: hash,
		dims: [Math.floor(Math.random() * 20) + 5, Math.floor(Math.random() * 20) + 5],
		data: null // Would contain actual attention matrix
	}));
}

class HeadInfo {
	constructor(id, classifications = [], embeddingLocation = null) {
		this.id = id;
		this.classifications = classifications;
		this.embeddingLocation = embeddingLocation;
	}

	async getAttentionPatterns(promptHashes) {
		const cacheKey = promptHashes.join(',');
		if (this.patternCache.has(cacheKey)) {
			return this.patternCache.get(cacheKey);
		}

		const patterns = await loadAttentionPatterns(this.id, promptHashes);
		this.patternCache.set(cacheKey, patterns);
		return patterns;
	}

	computeDistance(otherHead, metric = 'l2') {
		const diff = this.embeddingLocation.map((val, i) =>
			val - otherHead.embeddingLocation[i]
		);

		if (metric === 'l1') {
			return diff.reduce((sum, d) => sum + Math.abs(d), 0);
		} else {
			return Math.sqrt(diff.reduce((sum, d) => sum + d * d, 0));
		}
	}
}

class AttentionPedia {
	constructor() {
		this.heads = new Map();
		this.classifications = [
			'IOI: Induction', 'SAE-survey: Induction', 'Copy',
			'Previous Token', 'Duplicate Token', 'Position'
		];
		this.initializeDummyData();
	}

	initializeDummyData() {
		// Generate dummy heads for gpt2-small (12 layers, 12 heads each)
		for (let layer = 0; layer < 12; layer++) {
			for (let head = 0; head < 12; head++) {
				const id = `gpt2-small:L${layer}:H${head}`;
				const classifications = this.generateRandomClassifications();
				this.heads.set(id, new HeadInfo(id, classifications));
			}
		}
	}

	generateRandomClassifications() {
		const numClassifications = Math.floor(Math.random() * 3) + 1;
		const selected = [];

		for (let i = 0; i < numClassifications; i++) {
			const classType = this.classifications[Math.floor(Math.random() * this.classifications.length)];
			if (!selected.find(c => c.name === classType)) {
				selected.push({
					name: classType,
					type: classType.toLowerCase().includes('induction') ? 'induction' :
						classType.toLowerCase().includes('copy') ? 'copy' : 'duplicate',
					fullName: `Full name: ${classType}`,
					authors: `Author et al. ${2020 + Math.floor(Math.random() * 5)}`,
					url: `https://arxiv.org/pdf/example${Math.floor(Math.random() * 1000)}`,
					description: `Description of ${classType} behavior in transformer models.`
				});
			}
		}
		return selected;
	}

	getHeadInfo(headId) {
		return this.heads.get(headId);
	}

	getNearestNeighbors(headId, count = 5) {
		const targetHead = this.heads.get(headId);
		if (!targetHead) return [];

		const neighbors = Array.from(this.heads.values())
			.filter(head => head.id !== headId)
			.map(head => ({
				...head,
				distance: targetHead.computeDistance(head)
			}))
			.sort((a, b) => a.distance - b.distance)
			.slice(0, count);

		return neighbors;
	}

	getSameClassificationHeads(headId, count = 5) {
		const targetHead = this.heads.get(headId);
		if (!targetHead || !targetHead.classifications.length) return [];

		const targetClassNames = targetHead.classifications.map(c => c.name);

		const sameClass = Array.from(this.heads.values())
			.filter(head => {
				if (head.id === headId) return false;
				return head.classifications.some(c =>
					targetClassNames.includes(c.name)
				);
			})
			.slice(0, count);

		return sameClass;
	}
}

function headAnalysisApp() {
	return {
		attentionPedia: null,
		currentHeadId: 'gpt2-small:L5:H5',
		currentHead: null,
		nearestNeighbors: [],
		sameClassHeads: [],
		prompts: [
			"The cat sat on the mat",
			"When John and Mary went to the store",
			"The quick brown fox jumps over",
			"In the beginning was the word"
		],
		promptHashes: [],
		patterns: new Map(),
		loading: true,

		init() {
			console.log('Initializing app...');
			this.attentionPedia = new AttentionPedia();
			this.generatePromptHashes();
			this.loadCurrentHead();
			this.loadNeighbors();
			this.loadPatterns().then(() => {
				this.loading = false;
				console.log('App initialized successfully');
			});
		},

		generatePromptHashes() {
			this.promptHashes = this.prompts.map(prompt =>
				'hash_' + Math.random().toString(36).substr(2, 9)
			);
		},

		loadCurrentHead() {
			this.currentHead = this.attentionPedia.getHeadInfo(this.currentHeadId);
			console.log('Current head loaded:', this.currentHead);
		},

		loadNeighbors() {
			this.nearestNeighbors = this.attentionPedia.getNearestNeighbors(this.currentHeadId, 3);
			this.sameClassHeads = this.attentionPedia.getSameClassificationHeads(this.currentHeadId, 4);
			console.log('Neighbors loaded:', this.nearestNeighbors.length, this.sameClassHeads.length);
		},

		async loadPatterns() {
			const allHeads = [this.currentHead, ...this.nearestNeighbors, ...this.sameClassHeads];

			for (const head of allHeads) {
				if (head) {
					const patterns = await head.getAttentionPatterns(this.promptHashes);
					this.patterns.set(head.id, patterns);
				}
			}
			console.log('Patterns loaded for', this.patterns.size, 'heads');
		},

		getModelInfo() {
			if (!this.currentHead) return 'Loading...';
			const parts = this.currentHead.id.split(':');
			const layer = parts[1];
			const head = parts[2];
			return `${layer}, ${head} • GPT-2 Small (124M parameters)`;
		},

		getPositionText() {
			if (!this.currentHead) return 'Loading...';
			const parts = this.currentHead.id.split(':');
			const layer = parseInt(parts[1].substring(1));
			const head = parseInt(parts[2].substring(1));
			return `Layer ${layer}, Head ${head} of 12×12 grid:`;
		},

		isCurrentPosition(index) {
			if (!this.currentHead) return false;
			const parts = this.currentHead.id.split(':');
			const layer = parseInt(parts[1].substring(1));
			const head = parseInt(parts[2].substring(1));
			const targetIndex = layer * 12 + head + 1;
			return index === targetIndex;
		},

		getTagClass(type) {
			return type;
		},

		getClassificationSummary(head) {
			if (!head || !head.classifications?.length) return 'No classification';
			return head.classifications.map(c => c.name).join(', ');
		},

		getPatternDisplay(head, promptIndex) {
			if (!head) return 'Loading...';
			const patterns = this.patterns.get(head.id);
			if (!patterns || !patterns[promptIndex]) {
				return 'Loading...';
			}
			const dims = patterns[promptIndex].dims;
			return `${dims[0]}×${dims[1]}`;
		},

		randomizePrompts() {
			const newPrompts = [
				"After the rain stopped, we decided",
				"The ancient library contained many",
				"Sarah walked through the garden and",
				"During the winter months, the lake"
			];
			this.prompts = newPrompts;
			this.generatePromptHashes();
			this.loadPatterns();
		}
	}
}