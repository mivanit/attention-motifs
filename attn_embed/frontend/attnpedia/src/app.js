document.addEventListener('alpine:init', () => {
	Alpine.data('attentionApp', () => ({
		loading: true,
		error: null,
		prompts: [],
		heads_display: [],
		heads_display_with_distances: [],
		current_head: null,
		
		// Error tracking for aggregated notifications
		failureTracker: {
			patterns: { failed: 0, total: 0 },
			classifications: { failed: 0, total: 0 }
		},

		async init() {
			try {
				this.attention_pedia = new AttentionPedia();
				this.head_distances = new HeadDistances();
				this.prompts_loader = new PromptsLoader();
				const allPrompts = await this.prompts_loader.getPrompts();
				const n_prompts = CONFIG.n_prompts || 10;
				this.prompts = allPrompts.slice(0, n_prompts);
				this.heads_display = CONFIG.heads_display;
				this.current_head = CONFIG.head_viewing;
				
				// Debug logging
				console.log('Patterns we are looking at:', this.prompts.map(p => p.hash));
				console.log('Current head viewing:', this.current_head);
				console.log('Heads display config:', this.heads_display);
				this.promptTooltip = null;
				
				// Set pattern size CSS variable
				const patternSize = CONFIG.pattern_size || 120;
				document.documentElement.style.setProperty('--pattern-size', `${patternSize}px`);
				
				await this.updateHeadsWithDistances();
				this.loading = false;
				
				// Check for aggregated errors after a short delay to let components load
				setTimeout(() => {
					this.showAggregatedErrors();
					this.resetFailureTracker();
				}, 2000);
			} catch (error) {
				NOTIF.error('Failed to initialize application data', error);
				this.loading = false;
				this.error = error.message;
			}
		},

		async updateHeadsWithDistances() {
			if (!this.current_head) {
				console.log('No current head set, table will be empty');
				this.heads_display_with_distances = [];
				return;
			}

			// If heads_display is explicitly set, use it
			if (this.heads_display && Array.isArray(this.heads_display)) {
				const headsWithDistances = await this.head_distances.getHeadDistances(this.current_head, this.heads_display);
				const maxDistance = Math.max(...headsWithDistances.map(h => h.distance || 0));
				
				// Get current head classifications for matching
				const currentHeadClassifications = await this.attention_pedia.get_head_types(this.current_head);
				
				// Build the entire array at once instead of pushing items
				const newHeadsArray = [];
				for (const item of headsWithDistances.sort((a, b) => (a.distance || 0) - (b.distance || 0))) {
					const distance = item.distance !== undefined ? item.distance : 0;
					
					// Check if this head has matching classifications
					const headClassifications = await this.attention_pedia.get_head_types(item.head_name);
					const hasMatchingClassification = headClassifications.some(cls => 
						currentHeadClassifications.includes(cls)
					);
					
					newHeadsArray.push({
						headId: item.head_name,
						distance: distance,
						distanceText: distance === 0 ? 'Current' : distance.toFixed(3),
						distanceColor: this.getDistanceColor(distance, maxDistance),
						hasMatchingClassification: hasMatchingClassification && item.head_name !== this.current_head,
						isGap: false
					});
				}
				
				// Assign the complete array at once to trigger Alpine.js reactivity
				this.heads_display_with_distances = newHeadsArray;
				console.log('Final heads list (explicit config):', this.heads_display_with_distances.map(h => h.headId || 'GAP'));
				return;
			}

			// Auto-populate table with current head + nearby + same class heads
			const headsToShow = new Set();
			headsToShow.add(this.current_head);

			// Add nearest heads
			const n_nearby = CONFIG.table?.n_nearby || 3;
			const nearestHeads = await this.head_distances.getNearestHeads(this.current_head, n_nearby);
			nearestHeads.head_names.forEach(head => headsToShow.add(head));

			// Get all heads with distances to identify gaps
			const allHeadsWithDistances = await this.head_distances.getHeadDistances(this.current_head, Array.from(headsToShow));
			const sortedByDistance = allHeadsWithDistances.sort((a, b) => a.distance - b.distance);

			// Add same class heads and track where they fit in distance order
			const n_share_class = CONFIG.table?.n_share_class || 2;
			const currentHeadTypes = await this.attention_pedia.get_head_types(this.current_head);
			const sameClassHeads = new Set();
			
			for (const type of currentHeadTypes) {
				const sameTypeHeads = await this.attention_pedia.get_type_heads(type);
				sameTypeHeads.slice(0, n_share_class).forEach(head => {
					if (!headsToShow.has(head)) {
						headsToShow.add(head);
						sameClassHeads.add(head);
					}
				});
			}

			const finalHeadsWithDistances = await this.head_distances.getHeadDistances(this.current_head, Array.from(headsToShow));

			// Create ordered list with gap indicators
			const sortedFinal = finalHeadsWithDistances.sort((a, b) => a.distance - b.distance);
			const rowsWithGaps = this.insertGapRows(sortedFinal, nearestHeads, sameClassHeads);

			const maxDistance = Math.max(...finalHeadsWithDistances.map(h => h.distance || 0));

			// Get current head classifications for matching
			const currentHeadClassifications = await this.attention_pedia.get_head_types(this.current_head);
			
			// Build the entire array at once instead of pushing items
			const newHeadsArray = [];
			for (const item of rowsWithGaps) {
				if (item.isGap) {
					// Add gap row
					newHeadsArray.push({
						isGap: true,
						gapCount: item.gapCount,
						nextDistance: item.nextDistance,
						expandable: true,
						expanded: false
					});
				} else {
					const distance = item.distance !== undefined ? item.distance : 0;
					
					// Check if this head has matching classifications
					const headClassifications = await this.attention_pedia.get_head_types(item.head_name);
					const hasMatchingClassification = headClassifications.some(cls => 
						currentHeadClassifications.includes(cls)
					);
					
					newHeadsArray.push({
						headId: item.head_name,
						distance: distance,
						distanceText: distance === 0 ? 'Current' : distance.toFixed(3),
						distanceColor: this.getDistanceColor(distance, maxDistance),
						hasMatchingClassification: hasMatchingClassification && item.head_name !== this.current_head,
						isGap: false
					});
				}
			}
			
			// Assign the complete array at once to trigger Alpine.js reactivity
			this.heads_display_with_distances = newHeadsArray;
			console.log('Final heads list (auto-populated):', this.heads_display_with_distances.map(h => h.headId || 'GAP'));
			console.log('Full heads_display_with_distances structure:', this.heads_display_with_distances);
			console.log('Number of items in heads_display_with_distances:', this.heads_display_with_distances.length);
			console.log('First few items detailed:', this.heads_display_with_distances.slice(0, 3));
			console.log('First item properties:', JSON.stringify(this.heads_display_with_distances[0], null, 2));
			console.log('Is first item a gap?', this.heads_display_with_distances[0].isGap);
			console.log('Loading state:', this.loading);
			console.log('All items isGap status:', this.heads_display_with_distances.map((item, i) => `${i}: ${item.isGap} (${item.headId || 'GAP'})`));
			
			// Force Alpine.js reactivity update
			this.$nextTick(() => {
				console.log('After $nextTick, length:', this.heads_display_with_distances.length);
			});
		},

		insertGapRows(sortedHeads, nearestHeads, sameClassHeads) {
			const result = [];
			let lastNearbyIndex = -1;
			
			// Find the last index of nearby heads in the sorted list
			for (let i = 0; i < sortedHeads.length; i++) {
				if (nearestHeads.head_names.includes(sortedHeads[i].head_name) || sortedHeads[i].head_name === this.current_head) {
					lastNearbyIndex = i;
				}
			}
			
			// Add all heads up to and including the last nearby head
			for (let i = 0; i <= lastNearbyIndex; i++) {
				result.push(sortedHeads[i]);
			}
			
			// Check for gaps when adding same-class heads
			let nextNearbyIndex = lastNearbyIndex + 1;
			for (let i = lastNearbyIndex + 1; i < sortedHeads.length; i++) {
				if (sameClassHeads.has(sortedHeads[i].head_name)) {
					// Check if there's a gap
					const gapSize = i - nextNearbyIndex;
					if (gapSize > 0) {
						result.push({
							isGap: true,
							gapCount: gapSize,
							nextDistance: sortedHeads[i].distance
						});
					}
					result.push(sortedHeads[i]);
					nextNearbyIndex = i + 1;
				}
			}
			
			return result;
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

		async expandGap(index) {
			// TODO: Implement gap expansion logic
			// This would fetch the skipped heads and insert them into the table
			console.log('Expanding gap at index:', index);
			// For now, just mark as expanded
			this.heads_display_with_distances[index].expanded = true;
			this.heads_display_with_distances[index].expandable = false;
		},

		showAggregatedErrors() {
			const { patterns, classifications } = this.failureTracker;
			
			if (patterns.failed > 0) {
				NOTIF.error(`Failed to load ${patterns.failed}/${patterns.total} attention patterns`);
			}
			
			if (classifications.failed > 0) {
				NOTIF.error(`Failed to load ${classifications.failed}/${classifications.total} classification sets`);
			}
		},

		resetFailureTracker() {
			this.failureTracker.patterns = { failed: 0, total: 0 };
			this.failureTracker.classifications = { failed: 0, total: 0 };
		},

		patternComponent(headId, promptHash) {
			const app = this;
			return {
				loading: true,
				imageUrl: null,
				error: null,

				async init() {
					app.failureTracker.patterns.total++;
					try {
						const headInfo = HeadInfo.from_id(headId);
						imageUrl_rel = await headInfo.get_pattern_url(promptHash);
						this.imageUrl = `${CONFIG.patterns_path}/${imageUrl_rel}`;
						this.loading = false;
					} catch (error) {
						app.failureTracker.patterns.failed++;
						this.error = error.message;
						this.loading = false;
					}
				}
			};
		},

		classificationComponent(headId) {
			const app = this;
			return {
				classifications: [],
				typesMetadata: {},
				currentTooltip: null,
				hideTimeout: null,

				async loadData() {
					app.failureTracker.classifications.total++;
					try {
						const types = await this.attention_pedia.get_head_types(headId);
						this.classifications = types;

						for (const type of types) {
							this.typesMetadata[type] = await this.attention_pedia.get_type_meta(type);
						}
					} catch (error) {
						app.failureTracker.classifications.failed++;
						this.classifications = [];
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