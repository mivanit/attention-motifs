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
		},
	}));
});