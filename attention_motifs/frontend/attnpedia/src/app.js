document.addEventListener('alpine:init', () => {
	Alpine.data('attentionApp', () => ({
		loading: true,
		error: null,
		prompts: [],
		heads_display: [],

		async init() {
			try {
				const config = mockConfig;
				this.prompts = config.prompts;
				this.heads_display = config.heads_display;
			} catch (err) {
				this.error = `Failed to load config: ${err.message}`;
			} finally {
				this.loading = false;
			}
		},
	}));
});