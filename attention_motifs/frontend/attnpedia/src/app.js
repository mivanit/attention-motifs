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
				classifications: [],
				typesMetadata: {},

				async loadData() {
					const types = await ATTENTION_PEDIA.get_head_types(headId);
					this.classifications = types;

					for (const type of types) {
						this.typesMetadata[type] = await ATTENTION_PEDIA.get_type_meta(type);
					}
				},

				showTooltip(event, type) {
					const meta = this.typesMetadata[type];
					const content = `${type}<br>${meta.notes}<br>${meta.n_heads} heads<br><a href="${meta.url}" target="_blank">${meta.url}</a>`;

					let tooltip = document.getElementById('tooltip');
					if (!tooltip) {
						tooltip = document.createElement('div');
						tooltip.id = 'tooltip';
						tooltip.className = 'tooltip';
						tooltip.style.pointerEvents = 'auto';
						document.body.appendChild(tooltip);
					}

					clearTimeout(this.hideTimeout);
					tooltip.innerHTML = content;
					tooltip.style.left = (event.target.getBoundingClientRect().left + window.scrollX) + 'px';
					tooltip.style.top = (event.target.getBoundingClientRect().bottom + window.scrollY + 5) + 'px';
					tooltip.classList.add('show');
				},

				hideTooltip() {
					this.hideTimeout = setTimeout(() => {
						const tooltip = document.getElementById('tooltip');
						if (tooltip) tooltip.classList.remove('show');
					}, 500);
				}
			};
		}
	}));
});