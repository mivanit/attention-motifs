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
		},

		classificationComponent(headId) {
			return {
				loading: true,
				classifications: [],
				typesMetadata: {},
				error: null,

				async init() {
					try {
						const types = await ATTENTION_PEDIA.get_head_types(headId);
						this.classifications = types;

						// Load metadata for each type
						for (const type of types) {
							this.typesMetadata[type] = await ATTENTION_PEDIA.get_type_meta(type);
						}

						this.loading = false;
					} catch (error) {
						console.error(`Failed to get classifications for ${headId}:`, error);
						this.error = error.message;
						this.loading = false;
					}
				},

				showTooltip(event, type) {
					const meta = this.typesMetadata[type];
					if (!meta || Object.keys(meta).length === 0) {
						return;
					}

					let content = type;
					if (meta.description) {
						content += ` - ${meta.description}`;
					}
					if (meta.frequency) {
						content += ` (freq: ${meta.frequency})`;
					}

					// Create or update tooltip
					let tooltip = document.getElementById('tooltip');
					if (!tooltip) {
						tooltip = document.createElement('div');
						tooltip.id = 'tooltip';
						tooltip.className = 'tooltip';
						document.body.appendChild(tooltip);
					}

					tooltip.textContent = content;

					// Position tooltip
					const rect = event.target.getBoundingClientRect();
					tooltip.style.left = (rect.left + window.scrollX) + 'px';
					tooltip.style.top = (rect.bottom + window.scrollY + 5) + 'px';
					tooltip.classList.add('show');
				},

				hideTooltip() {
					const tooltip = document.getElementById('tooltip');
					if (tooltip) {
						tooltip.classList.remove('show');
					}
				}
			};
		}
	}));
});