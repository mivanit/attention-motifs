document.addEventListener('alpine:init', () => {
	Alpine.data('attentionApp', () => ({
		loading: true,
		error: null,
		prompts: [],
		heads_display: [],

		async init() {
			const config = mockConfig;
			this.prompts = config.prompts;
			this.heads_display = config.heads_display;
			this.loading = false;
		},

		patternComponent(headId, promptHash) {
			return {
				loading: true,
				imageUrl: null,
				error: null,

				async init() {
					try {
						const headInfo = HeadInfo.from_id(headId);
						this.imageUrl = await headInfo.get_pattern_url(promptHash);
						this.loading = false;
					} catch (error) {
						console.error(`Failed to get pattern URL for ${headId}:${promptHash}:`, error);
						this.error = error.message;
						this.loading = false;
					}
				}
			};
		}
	}));
});