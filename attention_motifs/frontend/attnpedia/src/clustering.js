/**
 * ClusteringLoader - Loads clustering data for AttentionPedia
 *
 * Supports hierarchical (client-side cut-height), HDBSCAN, and Leiden
 * (precomputed flat partitions at discrete parameter values).
 */
class ClusteringLoader {
  constructor() {
    this._is_loaded = false;
    this._load_promise = null;
    this._assignments = {};
    this._rawAssignments = {}; // before min-size filtering
    this._nClusters = 10;
    this._cutHeight = null;
    this._minClusterSize = 0;

    // Cluster labels (loaded from localStorage + server)
    this._labels = {};
    this._resolvedLabels = {};

    // Current method: "hierarchical" | "hdbscan" | "leiden"
    this._method = "hierarchical";
    this._availableMethods = [];

    // Hierarchical-specific
    this._meta = null;
    this._linkage = null;
    this._maxCutHeight = null;

    // Flat method data: { method: { meta, partitions, labels } }
    this._flatData = {};
    this._currentParamKey = null;
  }

  /**
   * Ensure clustering data is loaded
   */
  async _ensureLoaded() {
    if (this._is_loaded) return;
    if (this._load_promise) {
      await this._load_promise;
      return;
    }

    this._load_promise = this._load();
    await this._load_promise;
  }

  async _load() {
    try {
      // Detect available methods from manifest
      const methodsUrl =
        CONFIG.clustering_methods_url ||
        "../../features/clustering_methods.json";
      let methods = ["hierarchical"];
      try {
        const mResp = await fetch(methodsUrl);
        if (mResp.ok) {
          const manifest = await mResp.json();
          methods = manifest.methods || ["hierarchical"];
        }
      } catch (e) {
        // Fall back to hierarchical only
      }
      this._availableMethods = methods;

      // --- Load hierarchical ---
      if (methods.includes("hierarchical")) {
        const metaUrl =
          CONFIG.clustering_meta_url ||
          "../../features/clustering/clustering_meta.json";
        const linkageUrl =
          CONFIG.clustering_linkage_url ||
          "../../features/clustering/linkage.json";

        const [metaResp, linkageResp] = await Promise.all([
          fetch(metaUrl),
          fetch(linkageUrl),
        ]);

        if (metaResp.ok && linkageResp.ok) {
          this._meta = await metaResp.json();
          this._linkage = await linkageResp.json();

          if (this._linkage && this._linkage.length > 0) {
            this._maxCutHeight = Math.max(
              ...this._linkage.map((row) => row[2]),
            );
          }

          // Load hierarchical labels
          const labelsUrl =
            CONFIG.cluster_labels_url ||
            "../../features/clustering/cluster_labels.json";
          this._labels = await loadClusterLabels(labelsUrl);
        }
      }

      // --- Load flat methods (HDBSCAN, Leiden) ---
      for (const method of ["hdbscan", "leiden"]) {
        if (!methods.includes(method)) continue;
        try {
          const flatMetaUrl =
            CONFIG[`clustering_${method}_meta_url`] ||
            `../../features/clustering_${method}/clustering_meta.json`;
          const flatPartitionsUrl =
            CONFIG[`clustering_${method}_partitions_url`] ||
            `../../features/clustering_${method}/partitions.json`;
          const flatLabelsUrl =
            CONFIG[`clustering_${method}_labels_url`] ||
            `../../features/clustering_${method}/cluster_labels.json`;

          const [fMetaResp, fPartResp] = await Promise.all([
            fetch(flatMetaUrl),
            fetch(flatPartitionsUrl),
          ]);

          if (fMetaResp.ok && fPartResp.ok) {
            const fMeta = await fMetaResp.json();
            const fPartitions = await fPartResp.json();
            const fLabels = await loadClusterLabels(flatLabelsUrl);
            this._flatData[method] = {
              meta: fMeta,
              partitions: fPartitions,
              labels: fLabels,
            };
          }
        } catch (e) {
          console.warn(`Failed to load ${method} clustering data:`, e);
        }
      }

      this._is_loaded = true;

      // Compute initial assignments: prefer saved method, then leiden, then hierarchical
      const savedMethod = ClusteringConfig.getMethod();
      if (
        savedMethod &&
        this._availableMethods.includes(savedMethod) &&
        (savedMethod === "hierarchical" ||
          this._flatData[savedMethod] !== undefined)
      ) {
        this._setMethodInternal(savedMethod);
      } else if (this._flatData["leiden"]) {
        this._setMethodInternal("leiden");
      } else if (this._linkage && this._meta) {
        this._method = "hierarchical";
        this._computeAssignmentsByNClusters(this._nClusters);
      } else if (Object.keys(this._flatData).length > 0) {
        const firstFlat = Object.keys(this._flatData)[0];
        this._setMethodInternal(firstFlat);
      }
    } catch (e) {
      console.warn("Failed to load clustering data:", e);
      this._is_loaded = true;
    }
  }

  /**
   * Check if clustering data is available
   */
  async isAvailable() {
    await this._ensureLoaded();
    return (
      (this._meta !== null && this._linkage !== null) ||
      Object.keys(this._flatData).length > 0
    );
  }

  /**
   * Get list of available clustering methods.
   * @returns {string[]}
   */
  getAvailableMethods() {
    return this._availableMethods.filter(
      (m) =>
        (m === "hierarchical" && this._linkage !== null) ||
        this._flatData[m] !== undefined,
    );
  }

  /**
   * Get current clustering method.
   * @returns {string}
   */
  getMethod() {
    return this._method;
  }

  /**
   * Check if current method is hierarchical (continuous parameter).
   * @returns {boolean}
   */
  isHierarchical() {
    return this._method === "hierarchical";
  }

  /**
   * Internal: switch to a method without persisting to config.
   * @param {string} method
   */
  _setMethodInternal(method) {
    this._method = method;

    if (method === "hierarchical") {
      if (this._cutHeight !== null) {
        this._computeAssignmentsByCutHeight(this._cutHeight);
      } else {
        this._computeAssignmentsByNClusters(this._nClusters);
      }
    } else {
      const flat = this._flatData[method];
      if (!flat || !flat.meta.param_keys.length) return;

      // Use saved param key, then method-specific default, then first available
      const savedKey = ClusteringConfig.getParamKey();
      const defaultKey = method === "leiden" ? "1.000" : null;
      const paramKey =
        savedKey && flat.meta.param_keys.includes(savedKey)
          ? savedKey
          : defaultKey && flat.meta.param_keys.includes(defaultKey)
            ? defaultKey
            : flat.meta.param_keys[0];
      this._setFlatParam(method, paramKey);
    }
  }

  /**
   * Set clustering method and recompute assignments.
   * @param {string} method - "hierarchical", "hdbscan", or "leiden"
   */
  setMethod(method) {
    ClusteringConfig.setMethod(method);
    this._setMethodInternal(method);
  }

  /**
   * Set parameter for a flat clustering method.
   * @param {string} method
   * @param {string} paramKey
   */
  _setFlatParam(method, paramKey) {
    const flat = this._flatData[method];
    if (!flat) return;

    this._currentParamKey = paramKey;
    ClusteringConfig.setParamKey(paramKey);
    this._rawAssignments = getFlatPartitionAssignments(
      flat.partitions,
      paramKey,
    );
    this._assignments = this._applyMinSizeFilter(this._rawAssignments);

    // Count clusters
    const clusterIds = new Set(
      Object.values(this._assignments).filter((c) => c !== -1),
    );
    this._nClusters = clusterIds.size;

    // Resolve labels
    this._resolvedLabels = resolveClusterLabelsFlat(
      flat.labels,
      flat.meta.method,
      flat.meta.param_name,
      paramKey,
      this._assignments,
    );
  }

  /**
   * Set flat method parameter and recompute.
   * @param {string} paramKey
   */
  async setParamKey(paramKey) {
    await this._ensureLoaded();
    if (this._method === "hierarchical") return;
    this._setFlatParam(this._method, paramKey);
  }

  /**
   * Get current parameter key for flat methods.
   * @returns {string|null}
   */
  getParamKey() {
    return this._currentParamKey;
  }

  /**
   * Get parameter metadata for the current flat method.
   * @returns {{paramName: string, paramKeys: string[], meta: Object}|null}
   */
  getFlatParamInfo() {
    if (this._method === "hierarchical") return null;
    const flat = this._flatData[this._method];
    if (!flat) return null;
    return {
      paramName: flat.meta.param_name,
      paramKeys: flat.meta.param_keys,
      meta: flat.meta.partition_meta,
    };
  }

  /**
   * Apply min-size filtering using shared engine.
   * Returns filtered assignments (small clusters get id = -1).
   */
  _applyMinSizeFilter(raw) {
    if (this._minClusterSize <= 0) return { ...raw };
    return applyMinSizeFilter(raw, this._minClusterSize).assignments;
  }

  /**
   * Compute assignments by target number of clusters (hierarchical only).
   * Uses shared computeClustersByNClusters() from cluster_engine.js.
   */
  _computeAssignmentsByNClusters(nClusters) {
    if (!this._linkage || !this._meta) return;

    const result = computeClustersByNClusters(
      this._linkage,
      this._meta.cls_values,
      nClusters,
    );
    this._cutHeight = result.cutHeight;
    this._rawAssignments = result.assignments;
    this._assignments = this._applyMinSizeFilter(this._rawAssignments);
    this._nClusters = nClusters;
    this._resolveLabels();
  }

  /**
   * Compute assignments by cut height (hierarchical only).
   * Uses shared computeClustersByHeight() from cluster_engine.js.
   */
  _computeAssignmentsByCutHeight(cutHeight) {
    if (!this._linkage || !this._meta) return;

    this._cutHeight = cutHeight;
    this._rawAssignments = computeClustersByHeight(
      this._linkage,
      this._meta.cls_values,
      cutHeight,
    );
    this._assignments = this._applyMinSizeFilter(this._rawAssignments);

    // Count actual clusters (excluding -1)
    const clusterIds = new Set(
      Object.values(this._assignments).filter((c) => c !== -1),
    );
    this._nClusters = clusterIds.size;
    this._resolveLabels();
  }

  /**
   * Resolve cluster labels against current assignments
   */
  _resolveLabels() {
    if (this._method === "hierarchical") {
      if (this._cutHeight !== null && Object.keys(this._labels).length > 0) {
        this._resolvedLabels = resolveClusterLabels(
          this._labels,
          this._cutHeight,
          this._assignments,
        );
      } else {
        this._resolvedLabels = {};
      }
    }
    // Flat methods resolve labels in _setFlatParam
  }

  /**
   * Set number of clusters and recompute assignments (hierarchical only)
   */
  async setNClusters(n) {
    await this._ensureLoaded();
    if (this._method !== "hierarchical") return;
    this._computeAssignmentsByNClusters(n);
  }

  /**
   * Set cut height and recompute assignments (hierarchical only)
   */
  async setCutHeight(h) {
    await this._ensureLoaded();
    if (this._method !== "hierarchical") return;
    this._computeAssignmentsByCutHeight(h);
  }

  /**
   * Set minimum cluster size and reapply filtering
   */
  async setMinClusterSize(n) {
    await this._ensureLoaded();
    this._minClusterSize = Math.max(0, n);
    this._assignments = this._applyMinSizeFilter(this._rawAssignments);
    // Recount actual clusters
    const clusterIds = new Set(
      Object.values(this._assignments).filter((c) => c !== -1),
    );
    this._nClusters = clusterIds.size;
    if (this._method === "hierarchical") {
      this._resolveLabels();
    } else {
      // Re-resolve flat labels with filtered assignments
      const flat = this._flatData[this._method];
      if (flat && this._currentParamKey) {
        this._resolvedLabels = resolveClusterLabelsFlat(
          flat.labels,
          flat.meta.method,
          flat.meta.param_name,
          this._currentParamKey,
          this._assignments,
        );
      }
    }
  }

  /**
   * Get current cut height (hierarchical only)
   */
  getCutHeight() {
    return this._cutHeight;
  }

  /**
   * Get max possible cut height (hierarchical only)
   */
  getMaxCutHeight() {
    return this._maxCutHeight;
  }

  /**
   * Get current min cluster size
   */
  getMinClusterSize() {
    return this._minClusterSize;
  }

  /**
   * Get cluster color for a head.
   */
  async getColor(headId) {
    await this._ensureLoaded();
    const clusterId = this._assignments[headId];
    if (clusterId === undefined || clusterId === -1) return "transparent";
    return clusterColor(clusterId);
  }

  /**
   * Get color for a cluster ID (synchronous, for use after loading).
   */
  getClusterColor(clusterId) {
    if (clusterId === undefined || clusterId === -1) return "#888888";
    return clusterColor(clusterId);
  }

  /**
   * Get cluster color as {r, g, b} in 0-1 range (for use with js-embedding-vis color override)
   */
  getClusterColorRGB(clusterId) {
    if (clusterId === undefined || clusterId === -1) return null;
    const hslStr = clusterColor(clusterId);
    // Parse "hsl(H, S%, L%)"
    const m = hslStr.match(/hsl\(([\d.]+),\s*([\d.]+)%,\s*([\d.]+)%\)/);
    if (!m) return null;
    const h = parseFloat(m[1]) / 360,
      s = parseFloat(m[2]) / 100,
      l = parseFloat(m[3]) / 100;
    // HSL to RGB conversion
    const a = s * Math.min(l, 1 - l);
    const f = (n) => {
      const k = (n + h * 12) % 12;
      return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    };
    return { r: f(0), g: f(8), b: f(4) };
  }

  /**
   * Get cluster ID for a head
   */
  async getClusterId(headId) {
    await this._ensureLoaded();
    return this._assignments[headId];
  }

  /**
   * Get all heads in a cluster
   */
  async getHeadsInCluster(clusterId) {
    await this._ensureLoaded();
    return Object.entries(this._assignments)
      .filter(([_, cid]) => cid === clusterId)
      .map(([headId, _]) => headId);
  }

  /**
   * Get current number of clusters (excluding misc/-1)
   */
  getNClusters() {
    return this._nClusters;
  }

  /**
   * Get actual cluster count (excluding -1 misc cluster)
   */
  getNClustersActual() {
    const clusterIds = new Set(
      Object.values(this._assignments).filter((c) => c !== -1),
    );
    return clusterIds.size;
  }

  /**
   * Get cluster sizes (excluding -1 misc cluster)
   */
  async getClusterSizes() {
    await this._ensureLoaded();
    const sizes = {};
    for (const clusterId of Object.values(this._assignments)) {
      if (clusterId === -1) continue;
      sizes[clusterId] = (sizes[clusterId] || 0) + 1;
    }
    return sizes;
  }

  /**
   * Get number of unclustered heads (cluster -1, from min-size filtering)
   */
  getUnclusteredCount() {
    let count = 0;
    for (const cid of Object.values(this._assignments)) {
      if (cid === -1) count++;
    }
    return count;
  }

  /**
   * Get cluster label for a cluster ID (synchronous, for use after loading)
   * @param {number} clusterId
   * @returns {{name: string, desc: string|null}|null}
   */
  getClusterLabel(clusterId) {
    return this._resolvedLabels[clusterId] || null;
  }

  /**
   * Get top N clusters sorted by size descending
   * @param {number} n - how many to return
   * @returns {Array<{id: number, size: number, color: string, label: {name: string, desc: string|null}|null}>}
   */
  async getTopClusters(n) {
    const sizes = await this.getClusterSizes();
    const sorted = Object.entries(sizes)
      .map(([id, size]) => ({
        id: parseInt(id),
        size,
        color: this.getClusterColor(parseInt(id)),
        label: this.getClusterLabel(parseInt(id)),
      }))
      .sort((a, b) => b.size - a.size);
    return sorted.slice(0, n);
  }
}
