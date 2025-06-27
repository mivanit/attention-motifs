document.addEventListener('alpine:init', () => {
	Alpine.data('attentionHeadApp', () => ({
		// State
		loading: true,
		error: null,
		currentHead: null,
		nearestNeighbors: [],
		sameClassHeads: [],
		allModelHeads: [],
		modelInfo: { layers: 12, headsPerLayer: 12 },
		currentPrompts: [],

		// Data instances
		attentionPedia: new AttentionPedia(),

		// Initialization
		async initialize() {
			try {
				await this.loadFromURL();
				this.generatePrompts();
			} catch (err) {
				this.error = err.message;
			} finally {
				this.loading = false;
			}
		},

		async loadFromURL() {
			const urlParams = new URLSearchParams(window.location.search);
			const headId = urlParams.get('head') || 'gpt2-small:L5:H5';

			if (!this.validateHeadId(headId)) {
				throw new Error(`Invalid head ID format: ${headId}`);
			}

			this.currentHead = this.attentionPedia.getHead(headId);
			if (!this.currentHead) {
				throw new Error(`Head not found: ${headId}`);
			}

			// Load related data
			this.nearestNeighbors = this.attentionPedia.getNearestHeads(headId, 3);

			if (this.currentHead.classifications.length > 0) {
				const classification = this.currentHead.classifications[0].name;
				this.sameClassHeads = this.attentionPedia.getHeadsByClassification(classification)
					.filter(head => head.id !== headId)
					.slice(0, 3);
			}

			this.generateModelGrid();
		},

		validateHeadId(id) {
			return /^[^:]+:L\d+:H\d+$/.test(id);
		},

		generateModelGrid() {
			this.allModelHeads = [];
			const currentClassification = this.currentHead.classifications[0]?.name;
			const neighborIds = new Set(this.nearestNeighbors.map(n => n.id));

			for (let layer = 0; layer < this.modelInfo.layers; layer++) {
				for (let head = 0; head < this.modelInfo.headsPerLayer; head++) {
					const id = `${this.currentHead.model}:L${layer}:H${head}`;
					const headInfo = this.attentionPedia.getHead(id);

					this.allModelHeads.push({
						id,
						layer,
						head,
						isCurrent: id === this.currentHead.id,
						isSameClass: headInfo?.classifications.some(c => c.name === currentClassification) || false,
						isNeighbor: neighborIds.has(id),
						classification: headInfo?.classifications[0]?.name
					});
				}
			}
		},

		generatePrompts() {
			const prompts = [
				"The cat sat on the mat",
				"When John and Mary went to the store",
				"The quick brown fox jumps over",
				"In the beginning was the word"
			];

			this.currentPrompts = prompts.map((text, idx) => ({
				text,
				hash: `prompt_${idx}_${Date.now()}`,
				patternSize: Math.floor(Math.random() * 8) + 8
			}));
		},

		// Navigation
		async navigateToHead(headId) {
			const url = new URL(window.location);
			url.searchParams.set('head', headId);
			window.history.pushState({}, '', url);

			this.loading = true;
			this.error = null;

			try {
				await this.loadFromURL();
			} catch (err) {
				this.error = err.message;
			} finally {
				this.loading = false;
			}
		},

		// Actions
		openPatternLens(mode) {
			let headIds = [this.currentHead.id];

			if (mode === 'neighbors') {
				headIds = headIds.concat(this.nearestNeighbors.map(n => n.id));
			} else if (mode === 'classification') {
				headIds = headIds.concat(this.sameClassHeads.map(h => h.id));
			}

			const url = `/pattern-lens?heads=${headIds.join(',')}`;
			window.open(url, '_blank');
		},

		openEmbeddingSpace() {
			const url = `/embedding-space?highlight=${this.currentHead.id}`;
			window.open(url, '_blank');
		},

		async viewPattern(headId, promptHash) {
			const head = this.attentionPedia.getHead(headId);
			if (head) {
				const pattern = await head.getPatterns(promptHash);
				alert(`Viewing ${pattern.size}×${pattern.size} pattern for ${headId}`);
			}
		},

		viewAveragePattern(classification, promptHash) {
			alert(`Viewing average pattern for ${classification} heads`);
		},

		randomizePrompts() {
			const newPrompts = [
				"The weather today is quite",
				"Alice and Bob decided to",
				"Machine learning models can",
				"Once upon a time there"
			];

			this.currentPrompts = newPrompts.map((text, idx) => ({
				text,
				hash: `prompt_${idx}_${Date.now()}`,
				patternSize: Math.floor(Math.random() * 8) + 8
			}));
		},

		configurePrompts() {
			alert('Prompt configuration dialog would open here');
		},

		downloadPatterns() {
			alert('Downloading patterns for ' + this.currentHead.id);
		},

		exportAnalysis() {
			const analysis = {
				head: this.currentHead,
				neighbors: this.nearestNeighbors,
				sameClass: this.sameClassHeads,
				timestamp: new Date().toISOString()
			};

			const blob = new Blob([JSON.stringify(analysis, null, 2)],
				{ type: 'application/json' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = `${this.currentHead.id}_analysis.json`;
			a.click();
			URL.revokeObjectURL(url);
		},

		compareClassification() {
			const classification = this.currentHead.classifications[0]?.name;
			alert(`Comparing all ${classification} heads`);
		}
	}));
});