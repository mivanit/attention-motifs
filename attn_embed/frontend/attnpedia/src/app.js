document.addEventListener('alpine:init', () => {
	Alpine.data('attentionApp', () => ({
		loading: true,
		error: null,
		prompts: [],
		allPrompts: [],
		allPromptsCount: 0,
		heads_display: [],
		heads_display_with_distances: [],
		current_head: null,
		classification_mode: false,
		current_classification: null,
		n_prompts: 5,
		table: {
			n_nearby: 2,
			n_share_class: 2,
			n_distant: 0,
			n_random: 0
		},
		
		// Error tracking for aggregated notifications
		failureTracker: {
			patterns: { failed: 0, total: 0 },
			classifications: { failed: 0, total: 0 }
		},

		async init() {
			await getConfig();
			try {
				this.attention_pedia = new AttentionPedia();
				this.head_distances = new HeadDistances();
				this.prompts_loader = new PromptsLoader();
				this.allPrompts = await this.prompts_loader.get_all();
				this.allPromptsCount = this.allPrompts.length;
				this.n_prompts = CONFIG.n_prompts || 5;
				
				// Check if specific prompts are selected via URL
				if (CONFIG.selected_prompts) {
					let selectedHashes;
					if (Array.isArray(CONFIG.selected_prompts)) {
						selectedHashes = CONFIG.selected_prompts;
					} else {
						selectedHashes = CONFIG.selected_prompts.split(',');
					}
					this.prompts = this.allPrompts.filter(p => selectedHashes.includes(p.hash));
					// If we couldn't find all the selected prompts, fall back to slice
					if (this.prompts.length === 0) {
						this.prompts = this.allPrompts.slice(0, this.n_prompts);
					}
				} else {
					this.prompts = this.allPrompts.slice(0, this.n_prompts);
				}
				this.heads_display = CONFIG.heads_display;
				this.current_head = CONFIG.head_viewing;
				this.classification_mode = CONFIG.classification_mode || false;
				this.current_classification = CONFIG.current_classification || null;
				this.table.n_nearby = CONFIG.table?.n_nearby || 2;
				this.table.n_share_class = CONFIG.table?.n_share_class || 2;
				this.table.n_distant = CONFIG.table?.n_distant || 0;
				this.table.n_random = CONFIG.table?.n_random || 0;
				
				// Handle classification mode
				if (this.classification_mode && this.current_classification) {
					await this.setupClassificationMode(this.current_classification);
				}
				
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

		async setupClassificationMode(classificationType) {
			// Get all heads with this classification
			const headsOfType = await this.attention_pedia.get_type_heads(classificationType);
			if (!headsOfType || headsOfType.length === 0) return;
			
			// Find the best head for this classification (prefer heads with only this classification)
			let bestHead = null;
			let minClassifications = Infinity;
			
			for (const head of headsOfType) {
				const headTypes = await this.attention_pedia.get_head_types(head);
				if (headTypes.length === 1 && headTypes[0] === classificationType) {
					// Perfect match - only has this classification
					bestHead = head;
					break;
				} else if (headTypes.length < minClassifications) {
					// Keep track of head with fewest classifications
					minClassifications = headTypes.length;
					bestHead = head;
				}
			}
			
			if (bestHead) {
				this.current_head = bestHead;
			}
			
			// Set heads_display to up to 99 heads of this classification
			this.heads_display = headsOfType.slice(0, 99);
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
				
				// Get all possible heads for ranking
				const allPossibleHeads = await this.head_distances.getNearestHeads(this.current_head, 999999);
				const allHeadsSorted = allPossibleHeads.head_names.map((head, idx) => ({
					head_name: head,
					distance: allPossibleHeads.distances[idx]
				})).sort((a, b) => a.distance - b.distance);
				
				// Create a map of head names to their rank among all heads
				const headRankMap = new Map();
				const totalHeads = allHeadsSorted.length;
				allHeadsSorted.forEach((head, index) => {
					headRankMap.set(head.head_name, index + 1);
				});
				
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
						distanceText: distance === 0 ? '0.000\n(Current)' : distance.toFixed(3),
						distanceColor: this.getDistanceColor(distance, maxDistance),
						hasMatchingClassification: hasMatchingClassification && item.head_name !== this.current_head,
						rank: headRankMap.get(item.head_name) || 0,
						totalHeads: totalHeads
					});
				}
				
				// Assign the complete array at once to trigger Alpine.js reactivity
				this.heads_display_with_distances = newHeadsArray;
				return;
			}

			// Get all possible heads and their distances first (efficient single call)
			const allPossibleHeads = await this.head_distances.getNearestHeads(this.current_head, 999999); // Get all heads
			const allHeadsSorted = allPossibleHeads.head_names.map((head, idx) => ({
				head_name: head,
				distance: allPossibleHeads.distances[idx]
			})).sort((a, b) => a.distance - b.distance);
			
			// Create a map of head names to their rank among all heads
			const headRankMap = new Map();
			const totalHeads = allHeadsSorted.length;
			allHeadsSorted.forEach((head, index) => {
				headRankMap.set(head.head_name, index + 1);
			});

			// Now efficiently select what we need
			const headsToShow = new Set();
			headsToShow.add(this.current_head);

			// Add nearest heads (from front of sorted list)
			const n_nearby = this.table.n_nearby;
			if (n_nearby > 0) {
				allHeadsSorted.slice(1, n_nearby + 1).forEach(item => headsToShow.add(item.head_name));
			}

			// Add most distant heads (from back of sorted list)
			const n_distant = this.table.n_distant;
			if (n_distant > 0) {
				allHeadsSorted.slice(-n_distant).forEach(item => headsToShow.add(item.head_name));
			}

			// Add random heads (randomly selected from middle portion)
			const n_random = this.table.n_random;
			if (n_random > 0) {
				const middleHeads = allHeadsSorted.slice(1, -1); // Exclude current and most distant
				const shuffled = [...middleHeads].sort(() => Math.random() - 0.5);
				shuffled.slice(0, n_random).forEach(item => headsToShow.add(item.head_name));
			}

			// Add same class heads
			const n_share_class = this.table.n_share_class;
			const sameClassHeads = new Set();
			
			if (n_share_class > 0) {
				const currentHeadTypes = await this.attention_pedia.get_head_types(this.current_head);
				
				for (const type of currentHeadTypes) {
					const sameTypeHeads = await this.attention_pedia.get_type_heads(type);
					sameTypeHeads.slice(0, n_share_class).forEach(head => {
						if (!headsToShow.has(head)) {
							headsToShow.add(head);
							sameClassHeads.add(head);
						}
					});
				}
			}

			// Get final distances for selected heads
			const finalHeadsWithDistances = await this.head_distances.getHeadDistances(this.current_head, Array.from(headsToShow));

			// Sort by distance
			const sortedFinal = finalHeadsWithDistances.sort((a, b) => a.distance - b.distance);

			const maxDistance = Math.max(...finalHeadsWithDistances.map(h => h.distance || 0));

			// Get current head classifications for matching
			const currentHeadClassifications = await this.attention_pedia.get_head_types(this.current_head);
			
			// Build the entire array at once instead of pushing items
			const newHeadsArray = [];
			for (const item of sortedFinal) {
				const distance = item.distance !== undefined ? item.distance : 0;
				
				// Check if this head has matching classifications
				const headClassifications = await this.attention_pedia.get_head_types(item.head_name);
				const hasMatchingClassification = headClassifications.some(cls => 
					currentHeadClassifications.includes(cls)
				);
				
				newHeadsArray.push({
					headId: item.head_name,
					distance: distance,
					distanceText: distance === 0 ? '0.000\n(Current)' : distance.toFixed(3),
					distanceColor: this.getDistanceColor(distance, maxDistance),
					hasMatchingClassification: hasMatchingClassification && item.head_name !== this.current_head,
					rank: headRankMap.get(item.head_name) || 0,
					totalHeads: totalHeads
				});
			}
			
			// Assign the complete array at once to trigger Alpine.js reactivity
			this.heads_display_with_distances = newHeadsArray;
		},


		getDistanceColor(distance, maxDistance) {
			if (distance === 0) return 'rgba(0, 123, 255, 0.1)';
			const intensity = Math.min(distance / maxDistance, 1);
			const blue = Math.floor(255 * (1 - intensity * 0.7));
			return `rgba(0, 123, ${blue}, ${0.2 + intensity * 0.6})`;
		},

		getHeadLink(headId) {
			// Create a new URL with the head_viewing parameter set to this head
			// Preserve current settings when navigating
			const url = new URL(window.location.href);
			url.searchParams.set('head_viewing', headId);
			url.searchParams.set('classification_mode', 'false');
			url.searchParams.delete('current_classification');
			url.searchParams.delete('heads_display');
			
			// Preserve current settings
			url.searchParams.set('n_prompts', this.n_prompts.toString());
			url.searchParams.set('table.n_nearby', this.table.n_nearby.toString());
			url.searchParams.set('table.n_share_class', this.table.n_share_class.toString());
			url.searchParams.set('table.n_distant', this.table.n_distant.toString());
			url.searchParams.set('table.n_random', this.table.n_random.toString());
			
			// Preserve current prompts selection
			const promptHashes = this.prompts.map(p => p.hash).join(',');
			url.searchParams.set('selected_prompts', promptHashes);
			
			return url.toString();
		},

		getClassificationLink(classificationType) {
			// Create URL for classification mode
			const url = new URL(window.location.href);
			url.searchParams.set('classification_mode', 'true');
			url.searchParams.set('current_classification', classificationType);
			url.searchParams.delete('heads_display'); // Let setupClassificationMode handle this
			
			// Preserve current settings
			url.searchParams.set('n_prompts', this.n_prompts.toString());
			
			// Preserve current prompts selection
			const promptHashes = this.prompts.map(p => p.hash).join(',');
			url.searchParams.set('selected_prompts', promptHashes);
			
			return url.toString();
		},

		getPatternLink(headId, promptHash) {
			// Generate pattern URL from template
			const template = CONFIG.pattern_url_template;
			if (!template) return '#';
			
			// Parse headId to get model, layer, and head number
			const headInfo = HeadInfo.from_id(headId);
			
			// Replace placeholders in template
			return template
				.replace('{prompt_hash}', promptHash)
				.replace('{model}', headInfo.model)
				.replace('{layer}', headInfo.layer)
				.replace('{head}', headInfo.head);
		},

		getPatternLensCurrentHeadLink() {
			// Generate pattern lens URL for current head only
			const template = CONFIG.patternlens_url_template;
			if (!template || !this.current_head) return '#';
			
			// Parse current head to get model, layer, and head number
			const headInfo = HeadInfo.from_id(this.current_head);
			
			// Get all prompt hashes and join with ~
			const promptHashes = this.prompts.map(p => p.hash).join('~');
			
			// Format head selection as L{layer}H{head}
			const headSelection = `L${headInfo.layer}H${headInfo.head}`;
			
			// Build URL with proper model and head parameters
			const baseUrl = template.split('?')[0];
			const params = new URLSearchParams();
			
			// Add models parameter
			params.set('models', headInfo.model);
			
			// Add prompts parameter
			params.set('prompts', promptHashes);
			
			// Add heads parameter for the model
			params.set(`heads-${headInfo.model}`, headSelection);
			
			return `${baseUrl}?${params.toString()}`;
		},

		getPatternLensAllHeadsLink() {
			// Generate pattern lens URL for all displayed heads
			const template = CONFIG.patternlens_url_template;
			if (!template || !this.current_head) return '#';
			
			// Get all prompt hashes and join with ~
			const promptHashes = this.prompts.map(p => p.hash).join('~');
			
			// Group heads by model
			const headsByModel = {};
			this.heads_display_with_distances.forEach(h => {
				const hInfo = HeadInfo.from_id(h.headId);
				if (!headsByModel[hInfo.model]) {
					headsByModel[hInfo.model] = [];
				}
				headsByModel[hInfo.model].push(`L${hInfo.layer}H${hInfo.head}`);
			});
			
			// Build URL with proper model and head parameters
			const baseUrl = template.split('?')[0];
			const params = new URLSearchParams();
			
			// Add models parameter
			const models = Object.keys(headsByModel);
			params.set('models', models.join('~'));
			
			// Add prompts parameter
			params.set('prompts', promptHashes);
			
			// Add heads parameter for each model
			models.forEach(model => {
				params.set(`heads-${model}`, headsByModel[model].join('~'));
			});
			
			return `${baseUrl}?${params.toString()}`;
		},

		getPatternLensLinkForClassification() {
			// Generate pattern lens URL for classification pages (all visible heads only)
			const template = CONFIG.patternlens_url_template;
			if (!template) return '#';
			
			// For classification pages, we don't have a single current head
			if (!this.heads_display_with_distances.length) return '#';
			
			// Get all prompt hashes and join with ~
			const promptHashes = this.prompts.map(p => p.hash).join('~');
			
			// Group heads by model
			const headsByModel = {};
			this.heads_display_with_distances.forEach(h => {
				const hInfo = HeadInfo.from_id(h.headId);
				if (!headsByModel[hInfo.model]) {
					headsByModel[hInfo.model] = [];
				}
				headsByModel[hInfo.model].push(`L${hInfo.layer}H${hInfo.head}`);
			});
			
			// Build URL with proper model and head parameters
			const baseUrl = template.split('?')[0];
			const params = new URLSearchParams();
			
			// Add models parameter
			const models = Object.keys(headsByModel);
			params.set('models', models.join('~'));
			
			// Add prompts parameter
			params.set('prompts', promptHashes);
			
			// Add heads parameter for each model
			models.forEach(model => {
				params.set(`heads-${model}`, headsByModel[model].join('~'));
			});
			
			return `${baseUrl}?${params.toString()}`;
		},

		getPatternLensTextForClassification() {
			// Generate text for Pattern Lens link on classification pages
			const allVisibleHeads = this.heads_display_with_distances.map(h => h.headId).join(', ');
			return `Pattern Lens: ${allVisibleHeads}`;
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
				patternLink: null,

				async init() {
					app.failureTracker.patterns.total++;
					try {
						const headInfo = HeadInfo.from_id(headId);
						const imageUrl_rel = await headInfo.get_pattern_url(promptHash);
						this.imageUrl = `${CONFIG.patterns_path}/${imageUrl_rel}`;
						this.patternLink = app.getPatternLink(headId, promptHash);
						this.loading = false;
					} catch (error) {
						app.failureTracker.patterns.failed++;
						this.loading = false;
						throw new Error(`Failed to load pattern for ${headId}/${promptHash}: ${error.message}`);
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
						const types = await app.attention_pedia.get_head_types(headId);
						this.classifications = types;

						for (const type of types) {
							this.typesMetadata[type] = await app.attention_pedia.get_type_meta(type);
						}
					} catch (error) {
						app.failureTracker.classifications.failed++;
						throw new Error(`Failed to load classifications for ${headId}: ${error.message}`);
					}
				},

				getClassificationLink(type) {
					return app.getClassificationLink(type);
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
		},

		classificationInfoComponent(classificationType) {
			const app = this;
			return {
				metadata: {},

				async loadMetadata() {
					try {
						this.metadata = await app.attention_pedia.get_type_meta(classificationType);
					} catch (error) {
						this.metadata = {};
					}
				}
			};
		},
		
		async updatePrompts() {
			// Update the number of prompts displayed - only ensure positive value
			this.n_prompts = Math.max(1, this.n_prompts);
			this.prompts = this.allPrompts.slice(0, this.n_prompts);
			
			// Update URL
			setConfigValue('n_prompts', this.n_prompts);
			
			// Reset failure tracker for new prompts
			this.resetFailureTracker();
		},
		
		async randomizePrompts() {
			// Shuffle array using Fisher-Yates algorithm
			const shuffled = [...this.allPrompts];
			for (let i = shuffled.length - 1; i > 0; i--) {
				const j = Math.floor(Math.random() * (i + 1));
				[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
			}
			
			// Update prompts with random selection
			this.prompts = shuffled.slice(0, this.n_prompts);
			
			// Reset failure tracker for new prompts
			this.resetFailureTracker();
		},
		
		async updateHeadsDisplay() {
			// Update table configuration - only ensure non-negative values
			this.table.n_nearby = Math.max(0, this.table.n_nearby);
			this.table.n_share_class = Math.max(0, this.table.n_share_class);
			this.table.n_distant = Math.max(0, this.table.n_distant);
			this.table.n_random = Math.max(0, this.table.n_random);
			
			// Update URL
			setConfigValue('table.n_nearby', this.table.n_nearby);
			setConfigValue('table.n_share_class', this.table.n_share_class);
			setConfigValue('table.n_distant', this.table.n_distant);
			setConfigValue('table.n_random', this.table.n_random);
			
			// Refresh the heads display
			await this.updateHeadsWithDistances();
			
			// Reset failure tracker
			this.resetFailureTracker();
		}
	}));
});