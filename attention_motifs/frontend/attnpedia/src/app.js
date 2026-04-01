document.addEventListener("alpine:init", () => {
  Alpine.data("attentionApp", () => ({
    loading: true,
    error: null,
    classificationError: null,
    prompts: [],
    allPrompts: [],
    allPromptsCount: 0,
    renderedPromptsCount: null, // null = all rendered, number = subset
    heads_display: [],
    heads_display_with_distances: [],
    current_head: null,
    classification_mode: false,
    current_classification: null,
    n_prompts: 5,
    table: {
      n_nearby: 2,
      n_share_class: 2,
      n_same_cluster: 2,
      n_distant: 0,
      n_random: 0,
    },

    // Filtering state
    models_data: [],
    available_models: [],
    available_layers: [],
    selected_models: new Set(),
    selected_layers: new Set(),
    model_filter_open: false,
    layer_filter_open: false,
    head_search_query: "",
    matching_heads: [],
    autocomplete_suggestions: [],
    autocomplete_visible: false,
    selected_suggestion_index: -1,
    head_autocomplete: null,
    search_mode: "switch", // 'switch' or 'add'

    // Error tracking for aggregated notifications
    failureTracker: {
      patterns: { failed: 0, total: 0 },
      classifications: { failed: 0, total: 0 },
    },

    // Column visibility
    column_visibility: {
      distance: true,
      classifications: true,
      cluster: true,
    },
    settings_open: false,
    max_ctx: 0, // 0 = no limit

    // Clustering state
    clustering: null,
    clustering_available: false,
    clustering_method: "leiden",
    clustering_methods: [],
    clustering_param_label: "",
    clustering_param_key: null,
    clustering_param_keys: [],
    n_clusters: 10,
    cluster_stats: null,
    cluster_cut_height: null,
    cluster_max_height: null,
    cluster_min_size: 0,
    top_clusters: [],
    current_head_cluster: null, // {id, size, color}
    _updateGeneration: 0, // guards against stale async updates

    async init() {
      await getConfig();
      try {
        this.attention_pedia = new AttentionPedia();
        this.head_distances = new HeadDistances();
        this.clustering = new ClusteringLoader();
        this.prompts_loader = new PromptsLoader();
        this.allPrompts = await this.prompts_loader.get_all();
        this.allPromptsCount = this.allPrompts.length;

        // Filter to only rendered prompts if rendered_prompts.jsonl exists
        const renderedHashes = await this.prompts_loader.loadRenderedHashes();
        if (renderedHashes) {
          this.allPrompts = this.allPrompts.filter((p) =>
            renderedHashes.has(p.hash),
          );
          this.renderedPromptsCount = this.allPrompts.length;
          console.log(
            `Filtered to ${this.renderedPromptsCount}/${this.allPromptsCount} rendered prompts`,
          );
        }

        this.n_prompts = CONFIG.n_prompts || 5;

        // Load model data for filtering
        await this.loadModelData();

        // Initialize autocomplete
        this.head_autocomplete = new HeadAutocomplete(
          this.models_data,
          this.head_distances,
        );

        // Check if specific prompts are selected via URL
        if (CONFIG.selected_prompts) {
          let selectedHashes;
          if (Array.isArray(CONFIG.selected_prompts)) {
            selectedHashes = CONFIG.selected_prompts;
          } else {
            selectedHashes = CONFIG.selected_prompts.split("~");
          }
          this.prompts = this.allPrompts.filter((p) =>
            selectedHashes.includes(p.hash),
          );
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
        this.table.n_same_cluster = CONFIG.table?.n_same_cluster || 2;
        this.table.n_distant = CONFIG.table?.n_distant || 0;
        this.table.n_random = CONFIG.table?.n_random || 0;

        // Load column visibility from config
        if (CONFIG.column_visibility) {
          const cv = CONFIG.column_visibility;
          if (cv.distance !== undefined)
            this.column_visibility.distance = cv.distance;
          if (cv.classifications !== undefined)
            this.column_visibility.classifications = cv.classifications;
          if (cv.cluster !== undefined)
            this.column_visibility.cluster = cv.cluster;
        }

        // Load max context window size
        this.max_ctx = CONFIG.max_ctx || 0;

        // Handle classification mode
        if (this.classification_mode && this.current_classification) {
          await this.setupClassificationMode(this.current_classification);
        }

        this.promptTooltip = null;

        // Set pattern size CSS variable
        const patternSize = CONFIG.pattern_size || 120;
        document.documentElement.style.setProperty(
          "--pattern-size",
          `${patternSize}px`,
        );

        // Add click outside listener for dropdowns
        document.addEventListener("click", (e) => {
          if (!e.target.closest(".filter-dropdown")) {
            this.model_filter_open = false;
            this.layer_filter_open = false;
          }
          if (
            !e.target.closest(".head-search-container") &&
            !e.target.closest(".search-mode-toggle")
          ) {
            this.matching_heads = [];
            this.autocomplete_visible = false;
            this.selected_suggestion_index = -1;
          }
        });

        // Initialize clustering (non-blocking)
        this.clustering_available = await this.clustering.isAvailable();
        if (this.clustering_available) {
          // ClusteringLoader already picked the best default method
          const methods = this.clustering.getAvailableMethods();
          this.clustering_methods = methods;
          this.clustering_method = this.clustering.getMethod();

          if (this.clustering.isHierarchical()) {
            this.n_clusters = CONFIG.default_n_clusters || 10;
            await this.clustering.setNClusters(this.n_clusters);
            this.cluster_cut_height = this.clustering.getCutHeight();
            this.cluster_max_height = this.clustering.getMaxCutHeight();
          } else {
            this._updateFlatParamUI();
          }
          await this.updateClusterInfo();
        }

        await this.updateHeadsWithDistances();
        this.loading = false;

        // Check for aggregated errors after a short delay to let components load
        setTimeout(() => {
          this.showAggregatedErrors();
          this.resetFailureTracker();
        }, 2000);
      } catch (error) {
        NOTIF.error("Failed to initialize application data", error);
        this.loading = false;
        this.error = error.message;
      }
    },

    async setupClassificationMode(classificationType) {
      // Get all heads with this classification
      const headsOfType =
        await this.attention_pedia.get_type_heads(classificationType);
      if (!headsOfType || headsOfType.length === 0) {
        // Classification doesn't exist - show error
        this.classificationError = classificationType;
        this.loading = false;
        return;
      }

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

      // Set heads_display to up to 99 heads of this classification (filtered)
      const filteredHeadsOfType = headsOfType.filter((head) =>
        this.shouldShowHead(head),
      );
      this.heads_display = filteredHeadsOfType.slice(0, 99);
    },

    async updateHeadsWithDistances() {
      const generation = ++this._updateGeneration;

      if (!this.current_head) {
        this.heads_display_with_distances = [];
        return;
      }

      // Immediately show the current head while loading distances
      this.heads_display_with_distances = [
        {
          headId: this.current_head,
          distance: 0,
          distanceText: "0.000\n(Current)",
          distanceColor: "#4caf50",
          hasMatchingClassification: false,
          rank: 1,
          totalHeads: 1,
          isLoadingOthers: true,
          clusterInfo: await this.getClusterInfo(this.current_head),
        },
      ];

      // If heads_display is explicitly set, use it
      if (this.heads_display && Array.isArray(this.heads_display)) {
        // Apply filtering to explicitly set heads
        const filteredHeads = this.heads_display.filter((headId) =>
          this.shouldShowHead(headId),
        );
        const headsWithDistances = await this.head_distances.getHeadDistances(
          this.current_head,
          filteredHeads,
        );
        const maxDistance = Math.max(
          ...headsWithDistances.map((h) => h.distance || 0),
        );

        // Get all possible heads for ranking
        const allPossibleHeads = await this.head_distances.getNearestHeads(
          this.current_head,
          999999,
        );
        const allHeadsSorted = allPossibleHeads.head_names
          .map((head, idx) => ({
            head_name: head,
            distance: allPossibleHeads.distances[idx],
          }))
          .sort((a, b) => a.distance - b.distance);

        // Create a map of head names to their rank among all heads
        const headRankMap = new Map();
        const totalHeads = allHeadsSorted.length;
        allHeadsSorted.forEach((head, index) => {
          headRankMap.set(head.head_name, index + 1);
        });

        // Get current head classifications for matching
        const currentHeadClassifications =
          await this.attention_pedia.get_head_types(this.current_head);

        // Build the entire array at once instead of pushing items
        const newHeadsArray = [];
        for (const item of headsWithDistances.sort(
          (a, b) => (a.distance || 0) - (b.distance || 0),
        )) {
          const distance = item.distance !== undefined ? item.distance : 0;

          // Check if this head has matching classifications
          const headClassifications = await this.attention_pedia.get_head_types(
            item.head_name,
          );
          const hasMatchingClassification = headClassifications.some((cls) =>
            currentHeadClassifications.includes(cls),
          );

          // For current head, highlight if it has any classifications at all
          const shouldHighlight =
            item.head_name === this.current_head
              ? headClassifications.length > 0
              : hasMatchingClassification;

          newHeadsArray.push({
            headId: item.head_name,
            distance: distance,
            distanceText:
              distance === 0 ? "0.000\n(Current)" : distance.toFixed(3),
            distanceColor: this.getDistanceColor(distance, maxDistance),
            hasMatchingClassification: shouldHighlight,
            rank: headRankMap.get(item.head_name) || 0,
            totalHeads: totalHeads,
            clusterInfo: await this.getClusterInfo(item.head_name),
          });
        }

        // Discard if a newer update was started while we were computing
        if (generation !== this._updateGeneration) return;
        // Assign the complete array at once to trigger Alpine.js reactivity
        this.heads_display_with_distances = newHeadsArray;
        return;
      }

      // Get all possible heads and their distances first (efficient single call)
      const allPossibleHeads = await this.head_distances.getNearestHeads(
        this.current_head,
        999999,
      ); // Get all heads
      const allHeadsSorted = allPossibleHeads.head_names
        .map((head, idx) => ({
          head_name: head,
          distance: allPossibleHeads.distances[idx],
        }))
        .filter((item) => this.shouldShowHead(item.head_name))
        .sort((a, b) => a.distance - b.distance);

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
        allHeadsSorted
          .slice(1, n_nearby + 1)
          .forEach((item) => headsToShow.add(item.head_name));
      }

      // Add most distant heads (from back of sorted list)
      const n_distant = this.table.n_distant;
      if (n_distant > 0) {
        allHeadsSorted
          .slice(-n_distant)
          .forEach((item) => headsToShow.add(item.head_name));
      }

      // Add random heads (randomly selected from middle portion)
      const n_random = this.table.n_random;
      if (n_random > 0) {
        const middleHeads = allHeadsSorted.slice(1, -1); // Exclude current and most distant
        const shuffled = [...middleHeads].sort(() => Math.random() - 0.5);
        shuffled
          .slice(0, n_random)
          .forEach((item) => headsToShow.add(item.head_name));
      }

      // Add same class heads
      const n_share_class = this.table.n_share_class;
      const sameClassHeads = new Set();

      if (n_share_class > 0) {
        const currentHeadTypes = await this.attention_pedia.get_head_types(
          this.current_head,
        );

        for (const type of currentHeadTypes) {
          const sameTypeHeads = await this.attention_pedia.get_type_heads(type);
          const filteredSameTypeHeads = sameTypeHeads.filter((head) =>
            this.shouldShowHead(head),
          );
          filteredSameTypeHeads.slice(0, n_share_class).forEach((head) => {
            if (!headsToShow.has(head)) {
              headsToShow.add(head);
              sameClassHeads.add(head);
            }
          });
        }
      }

      // Add same-cluster heads
      const n_same_cluster = this.table.n_same_cluster;
      if (n_same_cluster > 0 && this.clustering_available) {
        const currentClusterId = await this.clustering.getClusterId(
          this.current_head,
        );
        if (currentClusterId !== undefined) {
          const sameClusterHeads =
            await this.clustering.getHeadsInCluster(currentClusterId);
          const filteredClusterHeads = sameClusterHeads.filter(
            (head) => head !== this.current_head && this.shouldShowHead(head),
          );
          filteredClusterHeads.slice(0, n_same_cluster).forEach((head) => {
            if (!headsToShow.has(head)) {
              headsToShow.add(head);
            }
          });
        }
      }

      // Get final distances for selected heads
      const finalHeadsWithDistances =
        await this.head_distances.getHeadDistances(
          this.current_head,
          Array.from(headsToShow),
        );

      // Sort by distance
      const sortedFinal = finalHeadsWithDistances.sort(
        (a, b) => a.distance - b.distance,
      );

      const maxDistance = Math.max(
        ...finalHeadsWithDistances.map((h) => h.distance || 0),
      );

      // Get current head classifications for matching
      const currentHeadClassifications =
        await this.attention_pedia.get_head_types(this.current_head);

      // Build the entire array at once instead of pushing items
      const newHeadsArray = [];
      for (const item of sortedFinal) {
        const distance = item.distance !== undefined ? item.distance : 0;

        // Check if this head has matching classifications
        const headClassifications = await this.attention_pedia.get_head_types(
          item.head_name,
        );
        const hasMatchingClassification = headClassifications.some((cls) =>
          currentHeadClassifications.includes(cls),
        );

        // For current head, highlight if it has any classifications at all
        const shouldHighlight =
          item.head_name === this.current_head
            ? headClassifications.length > 0
            : hasMatchingClassification;

        newHeadsArray.push({
          headId: item.head_name,
          distance: distance,
          distanceText:
            distance === 0 ? "0.000\n(Current)" : distance.toFixed(3),
          distanceColor: this.getDistanceColor(distance, maxDistance),
          hasMatchingClassification: shouldHighlight,
          rank: headRankMap.get(item.head_name) || 0,
          totalHeads: totalHeads,
          clusterInfo: await this.getClusterInfo(item.head_name),
        });
      }

      // Discard if a newer update was started while we were computing
      if (generation !== this._updateGeneration) return;
      // Assign the complete array at once to trigger Alpine.js reactivity
      this.heads_display_with_distances = newHeadsArray;
    },

    getDistanceColor(distance, maxDistance) {
      if (distance === 0) return "rgba(0, 123, 255, 0.1)";
      const intensity = Math.min(distance / maxDistance, 1);
      const blue = Math.floor(255 * (1 - intensity * 0.7));
      return `rgba(0, 123, ${blue}, ${0.2 + intensity * 0.6})`;
    },

    getHeadLink(headId) {
      // Create a new URL with the head_viewing parameter set to this head
      // Preserve current settings when navigating
      const url = new URL(window.location.href);
      url.searchParams.set("head_viewing", encodeForURL(headId));
      url.searchParams.set("classification_mode", "false");
      url.searchParams.delete("current_classification");
      url.searchParams.delete("heads_display");

      // Preserve current settings
      url.searchParams.set("n_prompts", this.n_prompts.toString());
      url.searchParams.set("table.n_nearby", this.table.n_nearby.toString());
      url.searchParams.set(
        "table.n_share_class",
        this.table.n_share_class.toString(),
      );
      url.searchParams.set(
        "table.n_same_cluster",
        this.table.n_same_cluster.toString(),
      );
      url.searchParams.set("table.n_distant", this.table.n_distant.toString());
      url.searchParams.set("table.n_random", this.table.n_random.toString());

      // Preserve current prompts selection
      const promptHashes = this.prompts.map((p) => p.hash).join("~");
      url.searchParams.set("selected_prompts", promptHashes);

      return url.toString();
    },

    getClassificationLink(classificationType) {
      // Create URL for classification mode
      const url = new URL(window.location.href);
      url.searchParams.set("classification_mode", "true");
      url.searchParams.set(
        "current_classification",
        encodeForURL(classificationType),
      );
      url.searchParams.delete("heads_display"); // Let setupClassificationMode handle this

      // Preserve current settings
      url.searchParams.set("n_prompts", this.n_prompts.toString());

      // Preserve current prompts selection
      const promptHashes = this.prompts.map((p) => p.hash).join("~");
      url.searchParams.set("selected_prompts", promptHashes);

      return url.toString();
    },

    getPatternLink(headId, promptHash) {
      // Generate pattern URL from template
      const template = CONFIG.pattern_url_template;
      if (!template) return "#";

      // Parse headId to get model, layer, and head number
      const headInfo = HeadInfo.from_id(headId);

      // Replace placeholders in template
      return template
        .replace("{prompt_hash}", promptHash)
        .replace("{model}", headInfo.model)
        .replace("{layer}", headInfo.layer)
        .replace("{head}", headInfo.head);
    },

    getPatternLensCurrentHeadLink() {
      // Generate pattern lens URL for current head only
      const template = CONFIG.patternlens_url_template;
      if (!template || !this.current_head) return "#";

      // Parse current head to get model, layer, and head number
      const headInfo = HeadInfo.from_id(this.current_head);

      // Get all prompt hashes and join with ~
      const promptHashes = this.prompts.map((p) => p.hash).join("~");

      // Format head selection as L{layer}H{head}
      const headSelection = `L${headInfo.layer}H${headInfo.head}`;

      // Build URL with proper model and head parameters
      const baseUrl = template.split("?")[0];
      const params = new URLSearchParams();

      // Add models parameter
      params.set("models", headInfo.model);

      // Add prompts parameter
      params.set("prompts", promptHashes);

      // Add heads parameter for the model
      params.set(`heads-${headInfo.model}`, headSelection);

      return `${baseUrl}?${params.toString()}`;
    },

    getPatternLensAllHeadsLink() {
      // Generate pattern lens URL for all displayed heads
      const template = CONFIG.patternlens_url_template;
      if (!template || !this.current_head) return "#";

      // Get all prompt hashes and join with ~
      const promptHashes = this.prompts.map((p) => p.hash).join("~");

      // Group heads by model
      const headsByModel = {};
      this.heads_display_with_distances.forEach((h) => {
        const hInfo = HeadInfo.from_id(h.headId);
        if (!headsByModel[hInfo.model]) {
          headsByModel[hInfo.model] = [];
        }
        headsByModel[hInfo.model].push(`L${hInfo.layer}H${hInfo.head}`);
      });

      // Build URL with proper model and head parameters
      const baseUrl = template.split("?")[0];
      const params = new URLSearchParams();

      // Add models parameter
      const models = Object.keys(headsByModel);
      params.set("models", models.join("~"));

      // Add prompts parameter
      params.set("prompts", promptHashes);

      // Add heads parameter for each model
      models.forEach((model) => {
        params.set(`heads-${model}`, headsByModel[model].join("~"));
      });

      return `${baseUrl}?${params.toString()}`;
    },

    getPatternLensLinkForClassification() {
      // Generate pattern lens URL for classification pages (all visible heads only)
      const template = CONFIG.patternlens_url_template;
      if (!template) return "#";

      // For classification pages, we don't have a single current head
      if (!this.heads_display_with_distances.length) return "#";

      // Get all prompt hashes and join with ~
      const promptHashes = this.prompts.map((p) => p.hash).join("~");

      // Group heads by model
      const headsByModel = {};
      this.heads_display_with_distances.forEach((h) => {
        const hInfo = HeadInfo.from_id(h.headId);
        if (!headsByModel[hInfo.model]) {
          headsByModel[hInfo.model] = [];
        }
        headsByModel[hInfo.model].push(`L${hInfo.layer}H${hInfo.head}`);
      });

      // Build URL with proper model and head parameters
      const baseUrl = template.split("?")[0];
      const params = new URLSearchParams();

      // Add models parameter
      const models = Object.keys(headsByModel);
      params.set("models", models.join("~"));

      // Add prompts parameter
      params.set("prompts", promptHashes);

      // Add heads parameter for each model
      models.forEach((model) => {
        params.set(`heads-${model}`, headsByModel[model].join("~"));
      });

      return `${baseUrl}?${params.toString()}`;
    },

    getPatternLensTextForClassification() {
      // Generate text for Pattern Lens link on classification pages
      const allVisibleHeads = this.heads_display_with_distances
        .map((h) => h.headId)
        .join(", ");
      return `Pattern Lens: ${allVisibleHeads}`;
    },

    showPromptTooltip(event, prompt) {
      // Hide existing tooltip
      this.hidePromptTooltip();

      const tooltip = document.createElement("div");
      tooltip.className = "prompt-tooltip show";
      tooltip.innerHTML = `<strong>Text:</strong><br>${prompt.text}<br><br><strong>Source:</strong> ${prompt.meta?.pile_set_name || "Unknown"}`;
      document.body.appendChild(tooltip);

      // Position tooltip
      const rect = event.target.getBoundingClientRect();
      tooltip.style.left = rect.left + window.scrollX + "px";
      tooltip.style.top = rect.bottom + window.scrollY + 5 + "px";

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
        NOTIF.error(
          `Failed to load ${patterns.failed}/${patterns.total} attention patterns`,
        );
      }

      if (classifications.failed > 0) {
        NOTIF.error(
          `Failed to load ${classifications.failed}/${classifications.total} classification sets`,
        );
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
        naturalSize: 0,

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
            throw new Error(
              `Failed to load pattern for ${headId}/${promptHash}: ${error.message}`,
            );
          }
        },

        onImageLoad(event) {
          this.naturalSize = event.target.naturalWidth;
        },

        get cropStyle() {
          const maxCtx = app.max_ctx;
          if (!maxCtx || maxCtx <= 0 || !this.naturalSize) return "";
          if (maxCtx >= this.naturalSize) return "";
          const scale = this.naturalSize / maxCtx;
          const size = `calc(var(--pattern-size, 120px) * ${scale})`;
          return `width: ${size}; height: ${size};`;
        },
      };
    },

    updateMaxCtx() {
      this.max_ctx = Math.max(0, this.max_ctx);
      setConfigValue("max_ctx", this.max_ctx);
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
              this.typesMetadata[type] =
                await app.attention_pedia.get_type_meta(type);
            }
          } catch (error) {
            app.failureTracker.classifications.failed++;
            throw new Error(
              `Failed to load classifications for ${headId}: ${error.message}`,
            );
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

          const tooltip = document.createElement("div");
          tooltip.className = "tooltip show";
          tooltip.innerHTML = content;
          document.body.appendChild(tooltip);

          // Position to the right of the element
          const rect = event.target.getBoundingClientRect();
          tooltip.style.left = rect.right + window.scrollX + 10 + "px";
          tooltip.style.top = rect.top + window.scrollY + "px";

          this.currentTooltip = tooltip;

          // Add hover listeners to keep tooltip open
          tooltip.addEventListener("mouseenter", () => {
            if (this.hideTimeout) {
              clearTimeout(this.hideTimeout);
              this.hideTimeout = null;
            }
          });

          tooltip.addEventListener("mouseleave", () => {
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
        },
      };
    },

    classificationInfoComponent(classificationType) {
      const app = this;
      return {
        metadata: {},

        async loadMetadata() {
          try {
            this.metadata =
              await app.attention_pedia.get_type_meta(classificationType);
          } catch (error) {
            this.metadata = {};
          }
        },
      };
    },

    async updatePrompts() {
      // Update the number of prompts displayed - only ensure positive value
      this.n_prompts = Math.max(1, this.n_prompts);
      this.prompts = this.allPrompts.slice(0, this.n_prompts);

      // Update URL
      setConfigValue("n_prompts", this.n_prompts);

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
      setConfigValue("table.n_nearby", this.table.n_nearby);
      setConfigValue("table.n_share_class", this.table.n_share_class);
      setConfigValue("table.n_same_cluster", this.table.n_same_cluster);
      setConfigValue("table.n_distant", this.table.n_distant);
      setConfigValue("table.n_random", this.table.n_random);

      // Refresh the heads display
      await this.updateHeadsWithDistances();

      // Reset failure tracker
      this.resetFailureTracker();
    },

    // Model and filtering methods
    async loadModelData() {
      try {
        const response = await fetch("../../patterns/models.jsonl");
        const text = await response.text();
        const lines = text.trim().split("\n");
        this.models_data = lines.map((line) => JSON.parse(line));

        // Extract available models and layers
        this.available_models = this.models_data
          .map((m) => m.model_name)
          .sort();

        // Initialize with all models selected
        this.selected_models = new Set(this.available_models);

        // Calculate all available layers (will be updated when models selection changes)
        this.updateAvailableLayers();

        // Initialize with all layers selected
        this.selected_layers = new Set(this.available_layers);
      } catch (error) {
        console.error("Failed to load model data:", error);
        // Fallback to empty arrays
        this.available_models = [];
        this.available_layers = [];
      }
    },

    toggleModelFilter() {
      this.model_filter_open = !this.model_filter_open;
      if (this.model_filter_open) {
        this.layer_filter_open = false;
      }
    },

    toggleLayerFilter() {
      this.layer_filter_open = !this.layer_filter_open;
      if (this.layer_filter_open) {
        this.model_filter_open = false;
      }
    },

    toggleModelSelection(model) {
      if (this.selected_models.has(model)) {
        this.selected_models.delete(model);
      } else {
        this.selected_models.add(model);
      }
      this.updateAvailableLayers();
      this.applyFilters();
    },

    toggleLayerSelection(layer) {
      if (this.selected_layers.has(layer)) {
        this.selected_layers.delete(layer);
      } else {
        this.selected_layers.add(layer);
      }
      this.applyFilters();
    },

    selectAllModels() {
      this.selected_models = new Set(this.available_models);
      this.updateAvailableLayers();
      this.applyFilters();
    },

    selectNoModels() {
      this.selected_models = new Set();
      this.updateAvailableLayers();
      this.applyFilters();
    },

    selectAllLayers() {
      this.selected_layers = new Set(this.available_layers);
      this.applyFilters();
    },

    selectNoLayers() {
      this.selected_layers = new Set();
      this.applyFilters();
    },

    shouldShowHead(headId) {
      // If models haven't loaded yet, show all heads
      if (this.available_models.length === 0) {
        return true;
      }

      try {
        const headInfo = HeadInfo.from_id(headId);
        return (
          this.selected_models.has(headInfo.model) &&
          this.selected_layers.has(headInfo.layer)
        );
      } catch (error) {
        // If there's an error parsing the head ID, show it by default
        return true;
      }
    },

    async applyFilters() {
      // Re-run the heads display logic with filtering
      await this.updateHeadsWithDistances();
    },

    async searchHeads() {
      this.matching_heads = [];
      if (!this.head_search_query.trim()) {
        this.autocomplete_visible = false;
        return;
      }

      // Get autocomplete suggestions
      if (this.head_autocomplete) {
        this.autocomplete_suggestions =
          await this.head_autocomplete.getSuggestions(this.head_search_query);
        this.autocomplete_visible = this.autocomplete_suggestions.length > 0;
        this.selected_suggestion_index = -1;
      }

      // Also search existing heads for partial matches
      await this.head_distances._ensureMetaLoaded();
      const query = this.head_search_query.toLowerCase();
      const allHeads = this.head_distances.head_dists_meta.cls_values;

      this.matching_heads = allHeads
        .filter((headId) => {
          return (
            headId.toLowerCase().includes(query) && this.shouldShowHead(headId)
          );
        })
        .slice(0, 5); // Limit to 5 results since we also have autocomplete
    },

    async handleSearchKeydown(event) {
      if (
        !this.autocomplete_visible ||
        this.autocomplete_suggestions.length === 0
      ) {
        if (event.key === "Enter" && this.head_search_query.trim()) {
          // Try to navigate to the entered head if it's a complete ID
          if (
            await this.head_autocomplete.headExists(
              this.head_search_query.trim(),
            )
          ) {
            this.navigateToHead(this.head_search_query.trim());
          }
        }
        return;
      }

      if (event.key === "ArrowDown") {
        event.preventDefault();
        this.selected_suggestion_index = Math.min(
          this.selected_suggestion_index + 1,
          this.autocomplete_suggestions.length - 1,
        );
        this.scrollToSelectedSuggestion();
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        this.selected_suggestion_index = Math.max(
          this.selected_suggestion_index - 1,
          -1,
        );
        this.scrollToSelectedSuggestion();
      } else if (event.key === "Enter") {
        event.preventDefault();
        if (this.selected_suggestion_index >= 0) {
          this.selectSuggestion(
            this.autocomplete_suggestions[this.selected_suggestion_index],
          );
        }
      } else if (event.key === "Escape") {
        this.autocomplete_visible = false;
        this.selected_suggestion_index = -1;
      }
    },

    selectSuggestion(suggestion) {
      if (suggestion.valid && suggestion.type === "head") {
        // Complete head ID - navigate to it
        this.navigateToHead(suggestion.completion);
      } else {
        // Partial completion - update search query
        this.head_search_query = suggestion.completion;
        this.searchHeads();
      }
    },

    hideAutocomplete() {
      setTimeout(() => {
        this.autocomplete_visible = false;
        this.selected_suggestion_index = -1;
      }, 150); // Small delay to allow clicks
    },

    scrollToSelectedSuggestion() {
      // Wait for next tick to ensure DOM is updated
      this.$nextTick(() => {
        const dropdown = document.querySelector(".autocomplete-dropdown");
        const selectedElement = dropdown?.querySelector(
          `.autocomplete-suggestion:nth-child(${this.selected_suggestion_index + 1})`,
        );

        if (dropdown && selectedElement) {
          const dropdownRect = dropdown.getBoundingClientRect();
          const elementRect = selectedElement.getBoundingClientRect();

          // Check if element is out of view
          if (elementRect.top < dropdownRect.top) {
            // Element is above visible area
            dropdown.scrollTop = selectedElement.offsetTop;
          } else if (elementRect.bottom > dropdownRect.bottom) {
            // Element is below visible area
            dropdown.scrollTop =
              selectedElement.offsetTop -
              dropdown.clientHeight +
              selectedElement.clientHeight;
          }
        }
      });
    },

    updateAvailableLayers() {
      // Calculate available layers based on currently selected models
      const selectedModelData = this.models_data.filter((model) =>
        this.selected_models.has(model.model_name),
      );

      if (selectedModelData.length === 0) {
        this.available_layers = [];
        this.selected_layers = new Set();
        return;
      }

      // Get maximum layer count among selected models
      const maxLayers = Math.max(
        ...selectedModelData.map((model) => model.n_layers),
      );

      // Update available layers
      const newAvailableLayers = [];
      for (let i = 0; i < maxLayers; i++) {
        newAvailableLayers.push(i);
      }
      this.available_layers = newAvailableLayers;

      // Update selected layers to only include available ones
      const newSelectedLayers = new Set();
      for (const layer of this.selected_layers) {
        if (this.available_layers.includes(layer)) {
          newSelectedLayers.add(layer);
        }
      }
      this.selected_layers = newSelectedLayers;
    },

    navigateToHead(headId) {
      if (this.search_mode === "switch") {
        // Navigate to the selected head (switch mode)
        const url = new URL(window.location.href);
        url.searchParams.set("head_viewing", encodeForURL(headId));
        url.searchParams.set("classification_mode", "false");
        url.searchParams.delete("current_classification");
        url.searchParams.delete("heads_display");
        window.location.href = url.toString();
      } else {
        // Add to current view (add mode)
        this.addHeadToCurrentView(headId);
      }
    },

    addHeadToCurrentView(headId) {
      // Add the head to the current heads display without navigating away
      if (!this.heads_display || !Array.isArray(this.heads_display)) {
        // If we don't have explicit heads_display, create one with current head plus the new one
        this.heads_display = this.current_head ? [this.current_head] : [];
      }

      // Check if head is already in display
      if (!this.heads_display.includes(headId)) {
        this.heads_display.push(headId);
        // Update URL to reflect the explicit heads_display
        const url = new URL(window.location.href);
        const encodedHeads = this.heads_display
          .map((h) => encodeForURL(h))
          .join("~");
        url.searchParams.set("heads_display", encodedHeads);
        window.history.replaceState({}, "", url.toString());

        this.updateHeadsWithDistances();
      }

      // Clear search
      this.head_search_query = "";
      this.autocomplete_visible = false;
      this.matching_heads = [];
    },

    get hasExplicitHeads() {
      return (
        this.heads_display &&
        Array.isArray(this.heads_display) &&
        this.heads_display.length > 0
      );
    },

    // Clustering methods
    async getClusterColor(headId) {
      if (!this.clustering_available) {
        return "transparent";
      }
      return await this.clustering.getColor(headId);
    },

    async getClusterInfo(headId) {
      if (!this.clustering_available) return null;
      const clusterId = await this.clustering.getClusterId(headId);
      if (clusterId === undefined || clusterId === -1) return null;
      const sizes = await this.clustering.getClusterSizes();
      return {
        id: clusterId,
        size: sizes[clusterId] || 0,
        color: this.clustering.getClusterColor(clusterId),
        label: this.clustering.getClusterLabel(clusterId),
      };
    },

    async updateNClusters() {
      if (!this.clustering_available) return;
      this.n_clusters = Math.max(2, Math.min(200, this.n_clusters));
      await this.clustering.setNClusters(this.n_clusters);
      this.cluster_cut_height = this.clustering.getCutHeight();
      await this.updateClusterInfo();
      await this.updateHeadsWithDistances();
    },

    async updateCutHeight() {
      if (!this.clustering_available) return;
      this.cluster_cut_height = Math.max(
        0,
        Math.min(this.cluster_max_height, this.cluster_cut_height),
      );
      await this.clustering.setCutHeight(this.cluster_cut_height);
      this.n_clusters = this.clustering.getNClusters();
      await this.updateClusterInfo();
      await this.updateHeadsWithDistances();
    },

    async updateMinClusterSize() {
      if (!this.clustering_available) return;
      this.cluster_min_size = Math.max(0, this.cluster_min_size);
      await this.clustering.setMinClusterSize(this.cluster_min_size);
      this.n_clusters = this.clustering.getNClustersActual();
      await this.updateClusterInfo();
      await this.updateHeadsWithDistances();
    },

    async updateClusterInfo() {
      await this.updateClusterStats();
      this.top_clusters = await this.clustering.getTopClusters(10);
      await this.updateCurrentHeadCluster();
    },

    async updateCurrentHeadCluster() {
      if (!this.clustering_available || !this.current_head) {
        this.current_head_cluster = null;
        return;
      }
      this.current_head_cluster = await this.getClusterInfo(this.current_head);
    },

    async updateClusterStats() {
      if (!this.clustering_available) {
        this.cluster_stats = null;
        return;
      }
      const sizesObj = await this.clustering.getClusterSizes();
      const sizes = Object.values(sizesObj);
      if (!sizes || sizes.length === 0) {
        this.cluster_stats = null;
        return;
      }
      const minSize = Math.min(...sizes);
      const maxSize = Math.max(...sizes);
      const avgSize = (sizes.reduce((a, b) => a + b, 0) / sizes.length).toFixed(
        1,
      );
      const unclustered = this.clustering.getUnclusteredCount();
      let statsText = `${sizes.length} clusters (min: ${minSize}, max: ${maxSize}, avg: ${avgSize})`;
      if (unclustered > 0) {
        statsText += ` · ${unclustered} unclustered`;
      }
      this.cluster_stats = statsText;
    },

    getClusterPageUrl(highlightClusterId = null) {
      if (typeof ClusteringConfig !== "undefined") {
        ClusteringConfig.setCutHeight(this.cluster_cut_height);
        if (highlightClusterId !== null) {
          ClusteringConfig.setHighlightCluster(highlightClusterId);
        }
      }
      const url = new URL("../clustering/index.html", window.location.href);
      if (highlightClusterId !== null) {
        url.searchParams.set("highlight", highlightClusterId);
      }
      return url.toString();
    },

    // --- Clustering method switching ---

    async updateClusteringMethod(method) {
      this.clustering_method = method;
      this.clustering.setMethod(method);
      ClusteringConfig.setMethod(method);

      if (method === "hierarchical") {
        this.n_clusters = CONFIG.default_n_clusters || 10;
        await this.clustering.setNClusters(this.n_clusters);
        this.cluster_cut_height = this.clustering.getCutHeight();
        this.cluster_max_height = this.clustering.getMaxCutHeight();
      } else {
        this._updateFlatParamUI();
      }
      await this.updateClusterInfo();
      await this.updateHeadsWithDistances();
    },

    async updateClusteringParam(paramKey) {
      this.clustering_param_key = paramKey;
      await this.clustering.setParamKey(paramKey);
      ClusteringConfig.setParamKey(paramKey);
      this.n_clusters = this.clustering.getNClustersActual();
      await this.updateClusterInfo();
      await this.updateHeadsWithDistances();
    },

    _updateFlatParamUI() {
      const info = this.clustering.getFlatParamInfo();
      if (!info) return;
      this.clustering_param_label = info.paramName;
      this.clustering_param_keys = info.paramKeys;
      this.clustering_param_key =
        this.clustering.getParamKey() || info.paramKeys[0];
      this.n_clusters = this.clustering.getNClustersActual();
    },

    // --- Column visibility ---

    toggleColumn(col) {
      this.column_visibility[col] = !this.column_visibility[col];
      setConfigValue(`column_visibility.${col}`, this.column_visibility[col]);
    },

    /** Number of visible fixed columns (Head always visible). */
    get visibleColCount() {
      return (
        1 +
        (this.column_visibility.distance ? 1 : 0) +
        (this.column_visibility.classifications ? 1 : 0) +
        (this.clustering_available && this.column_visibility.cluster ? 1 : 0) +
        this.prompts.length
      );
    },

    // --- Table SVG export ---

    exportTableSVG() {
      const table = document.querySelector(".pattern-table");
      if (!table) return;

      const clone = table.cloneNode(true);

      // Inline computed styles onto every element for standalone SVG
      const inlineStyles = (source, target) => {
        const cs = window.getComputedStyle(source);
        const dominated = [
          "font",
          "color",
          "background",
          "border",
          "padding",
          "margin",
          "text-align",
          "vertical-align",
          "white-space",
          "font-size",
          "font-weight",
          "font-family",
          "line-height",
        ];
        for (const prop of dominated) {
          target.style.setProperty(prop, cs.getPropertyValue(prop));
        }
        const srcChildren = source.children;
        const tgtChildren = target.children;
        for (let i = 0; i < srcChildren.length; i++) {
          if (tgtChildren[i]) inlineStyles(srcChildren[i], tgtChildren[i]);
        }
      };
      inlineStyles(table, clone);

      // XHTML namespace required for foreignObject content
      clone.setAttribute("xmlns", "http://www.w3.org/1999/xhtml");

      // Remove hidden elements (Alpine x-show sets display:none)
      clone
        .querySelectorAll("[style*='display: none']")
        .forEach((el) => el.remove());

      const rect = table.getBoundingClientRect();
      const width = Math.ceil(rect.width);
      const height = Math.ceil(rect.height);

      const svgNS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(svgNS, "svg");
      svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      svg.setAttribute("width", width);
      svg.setAttribute("height", height);

      const fo = document.createElementNS(svgNS, "foreignObject");
      fo.setAttribute("width", "100%");
      fo.setAttribute("height", "100%");
      fo.appendChild(clone);
      svg.appendChild(fo);

      const serializer = new XMLSerializer();
      const svgStr = serializer.serializeToString(svg);
      const blob = new Blob([svgStr], { type: "image/svg+xml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const headLabel = this.current_head || "table";
      a.download = `attnpedia-${headLabel}.svg`;
      a.click();
      URL.revokeObjectURL(url);
    },
  }));
});
