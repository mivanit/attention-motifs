document.addEventListener('alpine:init', () => {
	Alpine.data('attentionApp', () => ({
		loading: true,
		error: null,
		prompts: [],
		heads_display: [],
		heads_display_with_distances: [],
		current_head: null,

		async init() {
			this.attention_pedia = new AttentionPedia();
			this.head_distances = new HeadDistances();
			this.prompts_loader = new PromptsLoader();
			const allPrompts = await this.prompts_loader.getPrompts();
			const n_prompts = CONFIG.n_prompts || 10;
			this.prompts = allPrompts.slice(0, n_prompts);
			this.heads_display = CONFIG.heads_display;
			this.current_head = CONFIG.head_viewing;
			this.promptTooltip = null;
			await this.updateHeadsWithDistances();
			this.loading = false;
		},

		async updateHeadsWithDistances() {
			if (!this.current_head) {
				this.heads_display_with_distances = [];
				return;
			}

			// If heads_display is explicitly set, use it
			if (this.heads_display && Array.isArray(this.heads_display)) {
				const headsWithDistances = await this.head_distances.getHeadDistances(this.current_head, this.heads_display);
				const maxDistance = Math.max(...headsWithDistances.map(h => h.distance || 0));
				
				// Get current head classifications for matching
				const currentHeadClassifications = await this.attention_pedia.get_head_types(this.current_head);
				
				this.heads_display_with_distances = [];
				for (const item of headsWithDistances.sort((a, b) => (a.distance || 0) - (b.distance || 0))) {
					const distance = item.distance !== undefined ? item.distance : 0;
					
					// Check if this head has matching classifications
					const headClassifications = await this.attention_pedia.get_head_types(item.head_name);
					const hasMatchingClassification = headClassifications.some(cls => 
						currentHeadClassifications.includes(cls)
					);
					
					this.heads_display_with_distances.push({
						headId: item.head_name,
						distance: distance,
						distanceText: distance === 0 ? 'Current' : distance.toFixed(3),
						distanceColor: this.getDistanceColor(distance, maxDistance),
						hasMatchingClassification: hasMatchingClassification && item.head_name !== this.current_head
					});
				}
				return;
			}

			// Auto-populate table with current head + nearby + same class heads
			const headsToShow = new Set();
			headsToShow.add(this.current_head);

			// Add nearest heads
			const n_nearby = CONFIG.table?.n_nearby || 3;
			const nearestHeads = await this.head_distances.getNearestHeads(this.current_head, n_nearby);
			nearestHeads.head_names.forEach(head => headsToShow.add(head));

			// Add same class heads
			const n_share_class = CONFIG.table?.n_share_class || 2;
			const currentHeadTypes = await this.attention_pedia.get_head_types(this.current_head);
			for (const type of currentHeadTypes) {
				const sameTypeHeads = await this.attention_pedia.get_type_heads(type);
				sameTypeHeads.slice(0, n_share_class).forEach(head => headsToShow.add(head));
			}

			const allHeads = Array.from(headsToShow);
			const headsWithDistances = await this.head_distances.getHeadDistances(this.current_head, allHeads);

			const maxDistance = Math.max(...headsWithDistances.map(h => h.distance || 0));

			// Get current head classifications for matching
			const currentHeadClassifications = await this.attention_pedia.get_head_types(this.current_head);
			
			this.heads_display_with_distances = [];
			for (const item of headsWithDistances.sort((a, b) => (a.distance || 0) - (b.distance || 0))) {
				const distance = item.distance !== undefined ? item.distance : 0;
				
				// Check if this head has matching classifications
				const headClassifications = await this.attention_pedia.get_head_types(item.head_name);
				const hasMatchingClassification = headClassifications.some(cls => 
					currentHeadClassifications.includes(cls)
				);
				
				this.heads_display_with_distances.push({
					headId: item.head_name,
					distance: distance,
					distanceText: distance === 0 ? 'Current' : distance.toFixed(3),
					distanceColor: this.getDistanceColor(distance, maxDistance),
					hasMatchingClassification: hasMatchingClassification && item.head_name !== this.current_head
				});
			}
		},

		getDistanceColor(distance, maxDistance) {
			if (distance === 0) return 'rgba(0, 123, 255, 0.1)';
			const intensity = Math.min(distance / maxDistance, 1);
			const blue = Math.floor(255 * (1 - intensity * 0.7));
			return `rgba(0, 123, ${blue}, ${0.2 + intensity * 0.6})`;
		},

		showPromptTooltip(event, prompt) {
			// Hide existing tooltip
			this.hidePromptTooltip();

			const tooltip = document.createElement('div');
			tooltip.className = 'prompt-tooltip show';
			tooltip.innerHTML = `<strong>Text:</strong><br>${prompt.text}<br><br><strong>Source:</strong> ${prompt.meta?.pile_set_name || 'Unknown'}`;
			document.body.appendChild(tooltip);

			// Position tooltip
			const rect = event.target.getBoundingClientRect();
			tooltip.style.left = (rect.left + window.scrollX) + 'px';
			tooltip.style.top = (rect.bottom + window.scrollY + 5) + 'px';

			this.promptTooltip = tooltip;
		},

		hidePromptTooltip() {
			if (this.promptTooltip) {
				this.promptTooltip.remove();
				this.promptTooltip = null;
			}
		},

		patternComponent(headId, promptHash) {
			return {
				loading: true,
				imageUrl: null,
				error: null,

				async init() {
					try {
						const headInfo = HeadInfo.from_id(headId);
						imageUrl_rel = await headInfo.get_pattern_url(promptHash);
						this.imageUrl = `${CONFIG.patterns_path}/${imageUrl_rel}`;
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
					const types = await this.attention_pedia.get_head_types(headId);
					this.classifications = types;

					for (const type of types) {
						this.typesMetadata[type] = await this.attention_pedia.get_type_meta(type);
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