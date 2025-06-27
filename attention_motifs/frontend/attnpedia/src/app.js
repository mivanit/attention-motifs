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
				currentTooltip: null,
				hideTimeout: null,

				async loadData() {
					const types = await ATTENTION_PEDIA.get_head_types(headId);
					this.classifications = types;

					for (const type of types) {
						this.typesMetadata[type] = await ATTENTION_PEDIA.get_type_meta(type);
					}
				},

				showTooltip(event, type) {
					// Clear any existing timeout
					if (this.hideTimeout) {
						clearTimeout(this.hideTimeout);
						this.hideTimeout = null;
					}

					// Hide any existing tooltip
					this.hideCurrentTooltip();

					const meta = this.typesMetadata[type];
					const content = `${type}<br>${meta.notes}<br>${meta.n_heads} heads<br><a href="${meta.url}" target="_blank">${meta.url}</a>`;

					const tooltip = document.createElement('div');
					tooltip.className = 'tooltip show';
					tooltip.innerHTML = content;
					document.body.appendChild(tooltip);

					// Position to the right of the element
					const rect = event.target.getBoundingClientRect();
					tooltip.style.left = (rect.right + window.scrollX + 10) + 'px';
					tooltip.style.top = (rect.top + window.scrollY) + 'px';

					this.currentTooltip = tooltip;

					// Add hover listeners to keep tooltip open
					tooltip.addEventListener('mouseenter', () => {
						if (this.hideTimeout) {
							clearTimeout(this.hideTimeout);
							this.hideTimeout = null;
						}
					});

					tooltip.addEventListener('mouseleave', () => {
						this.startHideTimer();
					});
				},

				hideTooltip() {
					this.startHideTimer();
				},

				startHideTimer() {
					if (this.hideTimeout) {
						clearTimeout(this.hideTimeout);
					}
					this.hideTimeout = setTimeout(() => {
						this.hideCurrentTooltip();
					}, 300);
				},

				hideCurrentTooltip() {
					if (this.currentTooltip) {
						this.currentTooltip.remove();
						this.currentTooltip = null;
					}
				}
			};
		}
	}));
});