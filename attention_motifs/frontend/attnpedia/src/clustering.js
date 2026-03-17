/**
 * ClusteringLoader - Loads hierarchical clustering data for AttentionPedia
 *
 * Loads the linkage matrix and computes cluster assignments client-side,
 * allowing dynamic adjustment of the number of clusters, cut height,
 * and minimum cluster size.
 */
class ClusteringLoader {
  constructor() {
    this._meta = null;
    this._linkage = null;
    this._is_loaded = false;
    this._load_promise = null;
    this._assignments = {};
    this._rawAssignments = {}; // before min-size filtering
    this._nClusters = 10;
    this._cutHeight = null;
    this._maxCutHeight = null;
    this._minClusterSize = 0;

    // Cluster labels (loaded from localStorage + server)
    this._labels = {};
    this._resolvedLabels = {};
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

      if (!metaResp.ok || !linkageResp.ok) {
        console.warn("Clustering data not available");
        this._is_loaded = true;
        return;
      }

      this._meta = await metaResp.json();
      this._linkage = await linkageResp.json();
      this._is_loaded = true;

      // Compute max cut height from linkage
      if (this._linkage && this._linkage.length > 0) {
        this._maxCutHeight = Math.max(...this._linkage.map((row) => row[2]));
      }

      // Load cluster labels (uses shared loadClusterLabels from cluster_utils.js)
      const labelsUrl =
        CONFIG.cluster_labels_url ||
        "../../features/clustering/cluster_labels.json";
      this._labels = await loadClusterLabels(labelsUrl);

      // Compute initial assignments
      this._computeAssignmentsByNClusters(this._nClusters);
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
    return this._meta !== null && this._linkage !== null;
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
   * Compute assignments by target number of clusters.
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
   * Compute assignments by cut height.
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

  /**
   * Set number of clusters and recompute assignments
   */
  async setNClusters(n) {
    await this._ensureLoaded();
    this._computeAssignmentsByNClusters(n);
  }

  /**
   * Set cut height and recompute assignments
   */
  async setCutHeight(h) {
    await this._ensureLoaded();
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
    this._resolveLabels();
  }

  /**
   * Get current cut height
   */
  getCutHeight() {
    return this._cutHeight;
  }

  /**
   * Get max possible cut height (max merge distance)
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
   * Uses shared clusterColor() from cluster_utils.js.
   */
  async getColor(headId) {
    await this._ensureLoaded();
    const clusterId = this._assignments[headId];
    if (clusterId === undefined || clusterId === -1) return "transparent";
    return clusterColor(clusterId);
  }

  /**
   * Get color for a cluster ID (synchronous, for use after loading).
   * Uses shared clusterColor() from cluster_utils.js.
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
