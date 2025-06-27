document.addEventListener('alpine:init', () => {
	Alpine.data('attentionApp', () => ({
		loading: true,
		error: null,
		prompts: [],
		heads: [],

		async init() {
			try {
				// Simulate loading delay
				await new Promise(resolve => setTimeout(resolve, 500));

				// In a real app, you'd do: const config = await this.loadJson('config.json');
				const config = mockConfig;

				this.prompts = config.prompts;
				this.heads = config.heads;
			} catch (err) {
				this.error = `Failed to load config: ${err.message}`;
			} finally {
				this.loading = false;
			}
		},

		async loadJson(filename) {
			const response = await fetch(filename);
			if (!response.ok) {
				throw new Error(`Failed to fetch ${filename}: ${response.status}`);
			}
			return await response.json();
		},

		getPatternSize() {
			return Math.floor(Math.random() * 8) + 8;
		},

		viewPattern(headId, promptHash) {
			alert(`Viewing pattern for ${headId} on prompt ${promptHash}`);
		}
	}));
});